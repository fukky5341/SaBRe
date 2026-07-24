## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0012732


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0024284, 0.0049568, 0.0024284, 0.0049568, -0.0016543, 0.0016543)
1: (0.0016731, 0.0020384, 0.0016731, 0.0020384, -0.0002390, 0.0002390)
2: (0.0116194, 0.0130172, 0.0116194, 0.0130172, -0.0009146, 0.0009146)
3: (-0.0026632, -0.0012174, -0.0026632, -0.0012174, -0.0009460, 0.0009460)
4: (-0.0027190, -0.0011539, -0.0027190, -0.0011539, -0.0010241, 0.0010241)
5: (0.0052053, 0.0066865, 0.0052053, 0.0066865, -0.0009691, 0.0009691)
6: (-0.0016471, 0.0042296, -0.0016471, 0.0042296, -0.0038451, 0.0038451)
7: (-0.0083171, -0.0003135, -0.0083171, -0.0003135, -0.0052367, 0.0052367)
8: (0.9833552, 0.9889930, 0.9833552, 0.9889930, -0.0036889, 0.0036889)
9: (-0.0058959, -0.0007782, -0.0058959, -0.0007782, -0.0033485, 0.0033485)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.38 + 1.59 = 2.97 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0020807, upper bound: 0.0020806

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019667, upper bound: 0.0019816
time: 0.75 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019816, upper bound: 0.0019816
time: 0.76 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.65 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.65
Output dim: 8, lower bound: -0.0019667, upper bound: 0.0019816
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.65
Output dim: 8, lower bound: -0.0019816, upper bound: 0.0019816

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0025364, 0.0049512, 0.0024696, 0.0049548, -0.0015077, 0.0015943
1: 0.0016887, 0.0020376, 0.0016791, 0.0020381, -0.0002178, 0.0002303
2: 0.0116225, 0.0129576, 0.0116205, 0.0129945, -0.0008814, 0.0008336
3: -0.0026599, -0.0012791, -0.0026620, -0.0012409, -0.0009116, 0.0008621
4: -0.0026522, -0.0011574, -0.0026936, -0.0011552, -0.0009333, 0.0009869
5: 0.0052087, 0.0066232, 0.0052066, 0.0066624, -0.0009339, 0.0008832
6: -0.0016339, 0.0039788, -0.0016422, 0.0041340, -0.0037055, 0.0035044
7: -0.0079755, -0.0003315, -0.0081869, -0.0003201, -0.0047727, 0.0050466
8: 0.9835958, 0.9889804, 0.9834468, 0.9889884, -0.0033620, 0.0035549
9: -0.0058844, -0.0009966, -0.0058917, -0.0008614, -0.0032269, 0.0030518

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018918, upper bound: 0.0018912
time: 0.71 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018926, upper bound: 0.0019070
time: 0.78 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0025598, 0.0050648, 0.0024961, 0.0049538, -0.0015150, 0.0018350
1: 0.0016921, 0.0020540, 0.0016829, 0.0020380, -0.0002189, 0.0002651
2: 0.0115597, 0.0129446, 0.0116211, 0.0129798, -0.0010145, 0.0008376
3: -0.0027249, -0.0012925, -0.0026614, -0.0012561, -0.0010493, 0.0008663
4: -0.0026377, -0.0010871, -0.0026771, -0.0011558, -0.0009378, 0.0011359
5: 0.0051421, 0.0066095, 0.0052071, 0.0066468, -0.0010749, 0.0008875
6: -0.0018979, 0.0039244, -0.0016399, 0.0040723, -0.0042650, 0.0035213
7: -0.0079014, 0.0000281, -0.0081028, -0.0003233, -0.0047958, 0.0058085
8: 0.9836479, 0.9892336, 0.9835060, 0.9889861, -0.0033782, 0.0040917
9: -0.0061143, -0.0010440, -0.0058896, -0.0009152, -0.0037141, 0.0030665

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018911, upper bound: 0.0019061
time: 0.75 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019070, upper bound: 0.0019070
time: 0.75 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.92 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.92
Output dim: 8, lower bound: -0.0018918, upper bound: 0.0018912
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.92
Output dim: 8, lower bound: -0.0018926, upper bound: 0.0019070
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 2.92
Output dim: 8, lower bound: -0.0018911, upper bound: 0.0019061
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 2.92
Output dim: 8, lower bound: -0.0019070, upper bound: 0.0019070

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0025862, 0.0049497, 0.0025846, 0.0049413, -0.0013501, 0.0014446
1: 0.0016959, 0.0020374, 0.0016957, 0.0020362, -0.0001951, 0.0002087
2: 0.0116233, 0.0129300, 0.0116279, 0.0129309, -0.0007987, 0.0007465
3: -0.0026591, -0.0013076, -0.0026543, -0.0013067, -0.0008261, 0.0007720
4: -0.0026214, -0.0011583, -0.0026224, -0.0011635, -0.0008358, 0.0008943
5: 0.0052095, 0.0065940, 0.0052144, 0.0065950, -0.0008463, 0.0007909
6: -0.0016304, 0.0038629, -0.0016111, 0.0038668, -0.0033577, 0.0031381
7: -0.0078176, -0.0003362, -0.0078229, -0.0003626, -0.0042738, 0.0045729
8: 0.9837070, 0.9889770, 0.9837033, 0.9889584, -0.0030106, 0.0032213
9: -0.0058814, -0.0010976, -0.0058645, -0.0010942, -0.0029241, 0.0027328

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018290, upper bound: 0.0018224
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018290, upper bound: 0.0018278
time: 0.92 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0025724, 0.0049500, 0.0025529, 0.0049523, -0.0014875, 0.0013720
1: 0.0016939, 0.0020374, 0.0016911, 0.0020378, -0.0002149, 0.0001982
2: 0.0116231, 0.0129376, 0.0116219, 0.0129484, -0.0007585, 0.0008224
3: -0.0026593, -0.0012998, -0.0026606, -0.0012886, -0.0007845, 0.0008506
4: -0.0026299, -0.0011581, -0.0026420, -0.0011567, -0.0009208, 0.0008493
5: 0.0052093, 0.0066021, 0.0052080, 0.0066136, -0.0008037, 0.0008714
6: -0.0016313, 0.0038950, -0.0016365, 0.0039403, -0.0031888, 0.0034574
7: -0.0078613, -0.0003351, -0.0079231, -0.0003280, -0.0047087, 0.0043429
8: 0.9836763, 0.9889778, 0.9836327, 0.9889829, -0.0033169, 0.0030592
9: -0.0058821, -0.0010696, -0.0058866, -0.0010301, -0.0027770, 0.0030108

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018296, upper bound: 0.0018403
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018296, upper bound: 0.0018445
time: 0.89 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: 0.0026773, 0.0050429, 0.0025461, 0.0049523, -0.0013657, 0.0016848
1: 0.0017091, 0.0020508, 0.0016901, 0.0020378, -0.0001973, 0.0002434
2: 0.0115718, 0.0128796, 0.0116219, 0.0129522, -0.0009315, 0.0007551
3: -0.0027124, -0.0013597, -0.0026606, -0.0012847, -0.0009634, 0.0007809
4: -0.0025650, -0.0011007, -0.0026462, -0.0011567, -0.0008454, 0.0010429
5: 0.0051550, 0.0065407, 0.0052080, 0.0066175, -0.0009870, 0.0008000
6: -0.0018470, 0.0036511, -0.0016365, 0.0039561, -0.0039160, 0.0031743
7: -0.0075292, -0.0000413, -0.0079446, -0.0003279, -0.0043231, 0.0053333
8: 0.9839101, 0.9891849, 0.9836175, 0.9889829, -0.0030453, 0.0037569
9: -0.0060699, -0.0012820, -0.0058867, -0.0010163, -0.0034102, 0.0027643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_A1_A1

### Relational analysis result of IS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018278, upper bound: 0.0018394
time: 0.78 seconds

## Relational analysis of IS_A2_A1_A2

### Relational analysis result of IS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018278, upper bound: 0.0018436
time: 0.96 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: 0.0026429, 0.0050620, 0.0025321, 0.0049527, -0.0012947, 0.0018139
1: 0.0017041, 0.0020536, 0.0016881, 0.0020378, -0.0001870, 0.0002621
2: 0.0115612, 0.0128987, 0.0116216, 0.0129599, -0.0010028, 0.0007158
3: -0.0027233, -0.0013400, -0.0026608, -0.0012767, -0.0010372, 0.0007403
4: -0.0025863, -0.0010888, -0.0026549, -0.0011565, -0.0008014, 0.0011228
5: 0.0051437, 0.0065609, 0.0052078, 0.0066257, -0.0010626, 0.0007584
6: -0.0018915, 0.0037312, -0.0016375, 0.0039887, -0.0042159, 0.0030092
7: -0.0076383, 0.0000194, -0.0079890, -0.0003267, -0.0040982, 0.0057417
8: 0.9838333, 0.9892276, 0.9835864, 0.9889838, -0.0028869, 0.0040446
9: -0.0061088, -0.0012122, -0.0058875, -0.0009880, -0.0036714, 0.0026205

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_A2_A1

### Relational analysis result of IS_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018445, upper bound: 0.0018403
time: 0.78 seconds

## Relational analysis of IS_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018445, upper bound: 0.0018445
time: 0.77 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.94 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 8, lower bound: -0.0018290, upper bound: 0.0018224
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 8, lower bound: -0.0018290, upper bound: 0.0018278
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 8, lower bound: -0.0018296, upper bound: 0.0018403
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 8, lower bound: -0.0018296, upper bound: 0.0018445
IS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 8, lower bound: -0.0018278, upper bound: 0.0018394
IS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 8, lower bound: -0.0018278, upper bound: 0.0018436
IS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 8, lower bound: -0.0018445, upper bound: 0.0018403
IS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 8, lower bound: -0.0018445, upper bound: 0.0018445

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0025937, 0.0048875, 0.0025872, 0.0049197, -0.0013175, 0.0013698
1: 0.0016970, 0.0020284, 0.0016961, 0.0020330, -0.0001903, 0.0001979
2: 0.0116577, 0.0129259, 0.0116399, 0.0129295, -0.0007573, 0.0007284
3: -0.0026236, -0.0013119, -0.0026419, -0.0013082, -0.0007833, 0.0007534
4: -0.0026168, -0.0011968, -0.0026208, -0.0011769, -0.0008156, 0.0008479
5: 0.0052459, 0.0065897, 0.0052271, 0.0065935, -0.0008024, 0.0007718
6: -0.0014860, 0.0038456, -0.0015606, 0.0038607, -0.0031838, 0.0030622
7: -0.0077941, -0.0005329, -0.0078146, -0.0004312, -0.0041705, 0.0043361
8: 0.9837236, 0.9888384, 0.9837090, 0.9889101, -0.0029378, 0.0030544
9: -0.0057556, -0.0011126, -0.0058206, -0.0010995, -0.0027726, 0.0026667

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016944, upper bound: 0.0017427
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017617, upper bound: 0.0017603
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0025492, 0.0048687, 0.0025888, 0.0048973, -0.0014032, 0.0013760
1: 0.0016906, 0.0020257, 0.0016963, 0.0020298, -0.0002027, 0.0001988
2: 0.0116681, 0.0129505, 0.0116523, 0.0129286, -0.0007607, 0.0007758
3: -0.0026128, -0.0012864, -0.0026291, -0.0013091, -0.0007868, 0.0008024
4: -0.0026443, -0.0012084, -0.0026197, -0.0011908, -0.0008686, 0.0008517
5: 0.0052569, 0.0066158, 0.0052402, 0.0065925, -0.0008060, 0.0008220
6: -0.0014423, 0.0039491, -0.0015087, 0.0038568, -0.0031981, 0.0032614
7: -0.0079350, -0.0005924, -0.0078094, -0.0005020, -0.0044417, 0.0043556
8: 0.9836242, 0.9887967, 0.9837128, 0.9888602, -0.0031288, 0.0030682
9: -0.0057176, -0.0010225, -0.0057754, -0.0011028, -0.0027851, 0.0028402

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016944, upper bound: 0.0017474
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017617, upper bound: 0.0017668
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0025799, 0.0048879, 0.0025556, 0.0049292, -0.0014534, 0.0013030
1: 0.0016950, 0.0020285, 0.0016915, 0.0020344, -0.0002100, 0.0001882
2: 0.0116575, 0.0129335, 0.0116346, 0.0129469, -0.0007204, 0.0008036
3: -0.0026238, -0.0013041, -0.0026474, -0.0012901, -0.0007451, 0.0008311
4: -0.0026252, -0.0011966, -0.0026403, -0.0011710, -0.0008997, 0.0008066
5: 0.0052457, 0.0065977, 0.0052215, 0.0066120, -0.0007633, 0.0008514
6: -0.0014868, 0.0038775, -0.0015828, 0.0039341, -0.0030286, 0.0033782
7: -0.0078375, -0.0005318, -0.0079146, -0.0004010, -0.0046008, 0.0041247
8: 0.9836929, 0.9888393, 0.9836386, 0.9889314, -0.0032409, 0.0029055
9: -0.0057563, -0.0010848, -0.0058399, -0.0010355, -0.0026374, 0.0029419

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016944, upper bound: 0.0017645
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017623, upper bound: 0.0017826
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0025349, 0.0048691, 0.0025572, 0.0049082, -0.0015389, 0.0013018
1: 0.0016885, 0.0020257, 0.0016917, 0.0020314, -0.0002223, 0.0001881
2: 0.0116679, 0.0129584, 0.0116462, 0.0129460, -0.0007197, 0.0008508
3: -0.0026130, -0.0012783, -0.0026354, -0.0012911, -0.0007444, 0.0008800
4: -0.0026531, -0.0012082, -0.0026393, -0.0011840, -0.0009526, 0.0008059
5: 0.0052567, 0.0066241, 0.0052338, 0.0066110, -0.0007626, 0.0009015
6: -0.0014431, 0.0039822, -0.0015340, 0.0039303, -0.0030258, 0.0035769
7: -0.0079801, -0.0005913, -0.0079094, -0.0004675, -0.0048715, 0.0041209
8: 0.9835925, 0.9887974, 0.9836423, 0.9888846, -0.0034316, 0.0029028
9: -0.0057183, -0.0009936, -0.0057974, -0.0010388, -0.0026350, 0.0031150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016944, upper bound: 0.0017657
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017623, upper bound: 0.0017854
time: 0.80 seconds

## BFS IS instance: IS_A2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0026847, 0.0049798, 0.0025488, 0.0049292, -0.0013319, 0.0016186
1: 0.0017102, 0.0020417, 0.0016905, 0.0020344, -0.0001924, 0.0002338
2: 0.0116066, 0.0128756, 0.0116346, 0.0129507, -0.0008949, 0.0007364
3: -0.0026763, -0.0013639, -0.0026474, -0.0012862, -0.0009255, 0.0007616
4: -0.0025604, -0.0011397, -0.0026446, -0.0011710, -0.0008245, 0.0010019
5: 0.0051919, 0.0065364, 0.0052215, 0.0066160, -0.0009482, 0.0007802
6: -0.0017005, 0.0036341, -0.0015829, 0.0039500, -0.0037621, 0.0030958
7: -0.0075061, -0.0002407, -0.0079362, -0.0004009, -0.0042161, 0.0051236
8: 0.9839264, 0.9890443, 0.9836234, 0.9889315, -0.0029699, 0.0036092
9: -0.0059424, -0.0012968, -0.0058400, -0.0010217, -0.0032762, 0.0026959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_A1_A1_A1

### Relational analysis result of IS_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016778, upper bound: 0.0017642
time: 0.80 seconds

## Relational analysis of IS_A2_A1_A1_A2

### Relational analysis result of IS_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017669, upper bound: 0.0017817
time: 0.79 seconds

## BFS IS instance: IS_A2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0026395, 0.0049642, 0.0025504, 0.0049082, -0.0014164, 0.0016138
1: 0.0017036, 0.0020395, 0.0016908, 0.0020314, -0.0002046, 0.0002331
2: 0.0116153, 0.0129005, 0.0116462, 0.0129498, -0.0008922, 0.0007831
3: -0.0026674, -0.0013381, -0.0026354, -0.0012871, -0.0009228, 0.0008099
4: -0.0025884, -0.0011494, -0.0026435, -0.0011840, -0.0008768, 0.0009989
5: 0.0052010, 0.0065628, 0.0052338, 0.0066150, -0.0009453, 0.0008297
6: -0.0016641, 0.0037391, -0.0015341, 0.0039462, -0.0037508, 0.0032921
7: -0.0076490, -0.0002903, -0.0079311, -0.0004674, -0.0044835, 0.0051083
8: 0.9838257, 0.9890093, 0.9836270, 0.9888846, -0.0031583, 0.0035984
9: -0.0059107, -0.0012054, -0.0057975, -0.0010250, -0.0032664, 0.0028669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_A1_A2_A1

### Relational analysis result of IS_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016778, upper bound: 0.0017653
time: 1.03 seconds

## Relational analysis of IS_A2_A1_A2_A2

### Relational analysis result of IS_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017669, upper bound: 0.0017846
time: 0.82 seconds

## BFS IS instance: IS_A2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0026502, 0.0049992, 0.0025348, 0.0049296, -0.0012620, 0.0017450
1: 0.0017052, 0.0020445, 0.0016885, 0.0020345, -0.0001823, 0.0002521
2: 0.0115960, 0.0128946, 0.0116344, 0.0129585, -0.0009648, 0.0006978
3: -0.0026874, -0.0013442, -0.0026476, -0.0012782, -0.0009978, 0.0007217
4: -0.0025818, -0.0011277, -0.0026532, -0.0011708, -0.0007812, 0.0010802
5: 0.0051805, 0.0065566, 0.0052213, 0.0066242, -0.0010222, 0.0007393
6: -0.0017454, 0.0037142, -0.0015838, 0.0039825, -0.0040560, 0.0029333
7: -0.0076151, -0.0001796, -0.0079806, -0.0003997, -0.0039950, 0.0055239
8: 0.9838496, 0.9890873, 0.9835922, 0.9889323, -0.0028141, 0.0038911
9: -0.0059815, -0.0012270, -0.0058408, -0.0009934, -0.0035321, 0.0025545

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_A2_A1_A1

### Relational analysis result of IS_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016897, upper bound: 0.0017646
time: 0.82 seconds

## Relational analysis of IS_A2_A2_A1_A2

### Relational analysis result of IS_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017854, upper bound: 0.0017826
time: 0.80 seconds

## BFS IS instance: IS_A2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0026066, 0.0049815, 0.0025364, 0.0049086, -0.0013453, 0.0017412
1: 0.0016989, 0.0020420, 0.0016887, 0.0020315, -0.0001944, 0.0002515
2: 0.0116057, 0.0129188, 0.0116460, 0.0129576, -0.0009626, 0.0007438
3: -0.0026773, -0.0013193, -0.0026356, -0.0012791, -0.0009956, 0.0007693
4: -0.0026088, -0.0011386, -0.0026522, -0.0011838, -0.0008328, 0.0010778
5: 0.0051909, 0.0065821, 0.0052336, 0.0066232, -0.0010200, 0.0007881
6: -0.0017044, 0.0038156, -0.0015350, 0.0039788, -0.0040470, 0.0031269
7: -0.0077533, -0.0002355, -0.0079755, -0.0004662, -0.0042585, 0.0055116
8: 0.9837523, 0.9890480, 0.9835957, 0.9888855, -0.0029998, 0.0038825
9: -0.0059458, -0.0011387, -0.0057983, -0.0009966, -0.0035243, 0.0027230

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_A2_A2_A1

### Relational analysis result of IS_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016897, upper bound: 0.0017657
time: 1.04 seconds

## Relational analysis of IS_A2_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017853, upper bound: 0.0017853
time: 0.86 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.40 seconds
IS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 8, lower bound: -0.0016944, upper bound: 0.0017427
IS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 8, lower bound: -0.0017617, upper bound: 0.0017603
IS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 8, lower bound: -0.0016944, upper bound: 0.0017474
IS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 8, lower bound: -0.0017617, upper bound: 0.0017668
IS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 8, lower bound: -0.0016944, upper bound: 0.0017645
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 8, lower bound: -0.0017623, upper bound: 0.0017826
IS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 8, lower bound: -0.0016944, upper bound: 0.0017657
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 8, lower bound: -0.0017623, upper bound: 0.0017854
IS_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 8, lower bound: -0.0016778, upper bound: 0.0017642
IS_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 8, lower bound: -0.0017669, upper bound: 0.0017817
IS_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 8, lower bound: -0.0016778, upper bound: 0.0017653
IS_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 8, lower bound: -0.0017669, upper bound: 0.0017846
IS_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 8, lower bound: -0.0016897, upper bound: 0.0017646
IS_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 8, lower bound: -0.0017854, upper bound: 0.0017826
IS_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 8, lower bound: -0.0016897, upper bound: 0.0017657
IS_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 8, lower bound: -0.0017853, upper bound: 0.0017853

## BFS IS instance: IS_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0027686, 0.0049375, 0.0026637, 0.0049182, -0.0011144, 0.0012795
1: 0.0017223, 0.0020356, 0.0017071, 0.0020328, -0.0001610, 0.0001849
2: 0.0116300, 0.0128292, 0.0116407, 0.0128872, -0.0007074, 0.0006161
3: -0.0026521, -0.0014119, -0.0026411, -0.0013519, -0.0007316, 0.0006372
4: -0.0025085, -0.0011659, -0.0025734, -0.0011779, -0.0006898, 0.0007920
5: 0.0052167, 0.0064872, 0.0052280, 0.0065487, -0.0007495, 0.0006528
6: -0.0016021, 0.0034391, -0.0015572, 0.0036828, -0.0029739, 0.0025901
7: -0.0072405, -0.0003748, -0.0075724, -0.0004360, -0.0035275, 0.0040502
8: 0.9841136, 0.9889498, 0.9838797, 0.9889067, -0.0024848, 0.0028530
9: -0.0058567, -0.0014666, -0.0058176, -0.0012543, -0.0025898, 0.0022555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016944, upper bound: 0.0017207
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016944, upper bound: 0.0017427
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0026286, 0.0048868, 0.0025920, 0.0049196, -0.0011210, 0.0013669
1: 0.0017021, 0.0020283, 0.0016968, 0.0020330, -0.0001620, 0.0001975
2: 0.0116581, 0.0129066, 0.0116400, 0.0129268, -0.0007557, 0.0006198
3: -0.0026232, -0.0013319, -0.0026419, -0.0013109, -0.0007816, 0.0006410
4: -0.0025952, -0.0011972, -0.0026178, -0.0011770, -0.0006939, 0.0008461
5: 0.0052463, 0.0065692, 0.0052272, 0.0065907, -0.0008007, 0.0006567
6: -0.0014844, 0.0037645, -0.0015604, 0.0038496, -0.0031770, 0.0026055
7: -0.0076836, -0.0005351, -0.0077995, -0.0004316, -0.0035484, 0.0043268
8: 0.9838014, 0.9888369, 0.9837198, 0.9889099, -0.0024996, 0.0030479
9: -0.0057542, -0.0011832, -0.0058204, -0.0011091, -0.0027667, 0.0022690

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017617, upper bound: 0.0017386
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017617, upper bound: 0.0017603
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0027233, 0.0049202, 0.0026654, 0.0048958, -0.0012135, 0.0012864
1: 0.0017157, 0.0020331, 0.0017074, 0.0020296, -0.0001753, 0.0001858
2: 0.0116396, 0.0128542, 0.0116531, 0.0128862, -0.0007112, 0.0006709
3: -0.0026422, -0.0013860, -0.0026283, -0.0013529, -0.0007356, 0.0006939
4: -0.0025365, -0.0011766, -0.0025724, -0.0011917, -0.0007512, 0.0007963
5: 0.0052268, 0.0065137, 0.0052411, 0.0065477, -0.0007536, 0.0007109
6: -0.0015619, 0.0035443, -0.0015053, 0.0036789, -0.0029900, 0.0028206
7: -0.0073837, -0.0004296, -0.0075671, -0.0005067, -0.0038414, 0.0040721
8: 0.9840126, 0.9889113, 0.9838834, 0.9888570, -0.0027059, 0.0028684
9: -0.0058217, -0.0013750, -0.0057724, -0.0012577, -0.0026038, 0.0024563

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016944, upper bound: 0.0017207
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016944, upper bound: 0.0017474
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0025833, 0.0048678, 0.0025936, 0.0048972, -0.0012372, 0.0013728
1: 0.0016955, 0.0020256, 0.0016970, 0.0020298, -0.0001787, 0.0001983
2: 0.0116686, 0.0129316, 0.0116523, 0.0129259, -0.0007590, 0.0006840
3: -0.0026123, -0.0013060, -0.0026291, -0.0013119, -0.0007850, 0.0007074
4: -0.0026232, -0.0012090, -0.0026168, -0.0011908, -0.0007658, 0.0008498
5: 0.0052575, 0.0065958, 0.0052403, 0.0065897, -0.0008042, 0.0007247
6: -0.0014402, 0.0038698, -0.0015085, 0.0038457, -0.0031908, 0.0028755
7: -0.0078270, -0.0005953, -0.0077943, -0.0005023, -0.0039162, 0.0043456
8: 0.9837004, 0.9887945, 0.9837234, 0.9888600, -0.0027586, 0.0030611
9: -0.0057157, -0.0010916, -0.0057752, -0.0011125, -0.0027787, 0.0025041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017617, upper bound: 0.0017399
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017617, upper bound: 0.0017669
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0027566, 0.0049378, 0.0026333, 0.0049277, -0.0012529, 0.0011969
1: 0.0017206, 0.0020357, 0.0017027, 0.0020342, -0.0001810, 0.0001729
2: 0.0116299, 0.0128358, 0.0116354, 0.0129040, -0.0006617, 0.0006927
3: -0.0026523, -0.0014051, -0.0026465, -0.0013345, -0.0006844, 0.0007164
4: -0.0025159, -0.0011657, -0.0025922, -0.0011719, -0.0007756, 0.0007409
5: 0.0052165, 0.0064942, 0.0052224, 0.0065665, -0.0007012, 0.0007339
6: -0.0016028, 0.0034668, -0.0015794, 0.0037536, -0.0027820, 0.0029121
7: -0.0072782, -0.0003738, -0.0076687, -0.0004057, -0.0039660, 0.0037888
8: 0.9840869, 0.9889506, 0.9838119, 0.9889281, -0.0027937, 0.0026689
9: -0.0058573, -0.0014425, -0.0058369, -0.0011928, -0.0024227, 0.0025360

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016944, upper bound: 0.0017370
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016944, upper bound: 0.0017645
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0026132, 0.0048872, 0.0025602, 0.0049291, -0.0012710, 0.0013000
1: 0.0016998, 0.0020284, 0.0016922, 0.0020344, -0.0001836, 0.0001878
2: 0.0116579, 0.0129151, 0.0116347, 0.0129444, -0.0007187, 0.0007027
3: -0.0026234, -0.0013231, -0.0026473, -0.0012927, -0.0007433, 0.0007268
4: -0.0026046, -0.0011970, -0.0026375, -0.0011711, -0.0007868, 0.0008047
5: 0.0052461, 0.0065782, 0.0052216, 0.0066093, -0.0007615, 0.0007445
6: -0.0014852, 0.0038001, -0.0015826, 0.0039235, -0.0030215, 0.0029541
7: -0.0077322, -0.0005340, -0.0079001, -0.0004013, -0.0040233, 0.0041151
8: 0.9837672, 0.9888377, 0.9836489, 0.9889312, -0.0028341, 0.0028987
9: -0.0057549, -0.0011522, -0.0058397, -0.0010448, -0.0026313, 0.0025726

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017623, upper bound: 0.0017549
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017623, upper bound: 0.0017826
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0027113, 0.0049205, 0.0026350, 0.0049067, -0.0013488, 0.0012037
1: 0.0017140, 0.0020332, 0.0017030, 0.0020312, -0.0001949, 0.0001739
2: 0.0116394, 0.0128609, 0.0116471, 0.0129031, -0.0006655, 0.0007457
3: -0.0026424, -0.0013792, -0.0026345, -0.0013355, -0.0006883, 0.0007713
4: -0.0025439, -0.0011764, -0.0025912, -0.0011849, -0.0008349, 0.0007451
5: 0.0052266, 0.0065208, 0.0052347, 0.0065655, -0.0007051, 0.0007901
6: -0.0015627, 0.0035722, -0.0015306, 0.0037496, -0.0027976, 0.0031350
7: -0.0074217, -0.0004285, -0.0076634, -0.0004721, -0.0042696, 0.0038102
8: 0.9839858, 0.9889120, 0.9838156, 0.9888812, -0.0030076, 0.0026840
9: -0.0058224, -0.0013507, -0.0057945, -0.0011962, -0.0024363, 0.0027301

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016944, upper bound: 0.0017396
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016944, upper bound: 0.0017657
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0025681, 0.0048682, 0.0025618, 0.0049081, -0.0013728, 0.0012986
1: 0.0016933, 0.0020256, 0.0016924, 0.0020314, -0.0001983, 0.0001876
2: 0.0116684, 0.0129400, 0.0116463, 0.0129435, -0.0007179, 0.0007590
3: -0.0026125, -0.0012973, -0.0026353, -0.0012937, -0.0007425, 0.0007850
4: -0.0026326, -0.0012088, -0.0026365, -0.0011841, -0.0008498, 0.0008038
5: 0.0052573, 0.0066047, 0.0052339, 0.0066083, -0.0007607, 0.0008042
6: -0.0014410, 0.0039050, -0.0015338, 0.0039196, -0.0030182, 0.0031908
7: -0.0078750, -0.0005942, -0.0078949, -0.0004678, -0.0043456, 0.0041106
8: 0.9836666, 0.9887953, 0.9836525, 0.9888843, -0.0030611, 0.0028956
9: -0.0057164, -0.0010609, -0.0057972, -0.0010481, -0.0026284, 0.0027787

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017623, upper bound: 0.0017597
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017623, upper bound: 0.0017853
time: 0.98 seconds

## BFS IS instance: IS_A2_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0028669, 0.0050237, 0.0026289, 0.0049278, -0.0011255, 0.0015506
1: 0.0017365, 0.0020481, 0.0017021, 0.0020342, -0.0001626, 0.0002240
2: 0.0115824, 0.0127748, 0.0116354, 0.0129064, -0.0008573, 0.0006223
3: -0.0027014, -0.0014681, -0.0026466, -0.0013321, -0.0008866, 0.0006436
4: -0.0024476, -0.0011125, -0.0025949, -0.0011719, -0.0006967, 0.0009598
5: 0.0051662, 0.0064296, 0.0052224, 0.0065690, -0.0009083, 0.0006593
6: -0.0018025, 0.0032105, -0.0015795, 0.0037637, -0.0036040, 0.0026160
7: -0.0069291, -0.0001019, -0.0076825, -0.0004055, -0.0035628, 0.0049083
8: 0.9843329, 0.9891422, 0.9838021, 0.9889282, -0.0025097, 0.0034575
9: -0.0060312, -0.0016657, -0.0058370, -0.0011840, -0.0031385, 0.0022782

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_A1_A1_B1

### Relational analysis result of IS_A2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016778, upper bound: 0.0017367
time: 1.01 seconds

## Relational analysis of IS_A2_A1_A1_A1_B2

### Relational analysis result of IS_A2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016778, upper bound: 0.0017367
time: 0.80 seconds

## BFS IS instance: IS_A2_A1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0027163, 0.0049792, 0.0025532, 0.0049291, -0.0011475, 0.0016153
1: 0.0017147, 0.0020416, 0.0016912, 0.0020344, -0.0001658, 0.0002334
2: 0.0116070, 0.0128581, 0.0116347, 0.0129483, -0.0008930, 0.0006344
3: -0.0026759, -0.0013820, -0.0026473, -0.0012888, -0.0009236, 0.0006561
4: -0.0025408, -0.0011401, -0.0026418, -0.0011711, -0.0007103, 0.0009999
5: 0.0051923, 0.0065178, 0.0052216, 0.0066134, -0.0009462, 0.0006722
6: -0.0016989, 0.0035605, -0.0015827, 0.0039397, -0.0037543, 0.0026670
7: -0.0074058, -0.0002429, -0.0079222, -0.0004012, -0.0036323, 0.0051131
8: 0.9839971, 0.9890428, 0.9836333, 0.9889312, -0.0025586, 0.0036017
9: -0.0059410, -0.0013609, -0.0058398, -0.0010307, -0.0032694, 0.0023226

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_A1_A2_B1

### Relational analysis result of IS_A2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017669, upper bound: 0.0017541
time: 0.80 seconds

## Relational analysis of IS_A2_A1_A1_A2_B2

### Relational analysis result of IS_A2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017669, upper bound: 0.0017541
time: 0.80 seconds

## BFS IS instance: IS_A2_A1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0028237, 0.0050046, 0.0026306, 0.0049068, -0.0012267, 0.0015517
1: 0.0017302, 0.0020453, 0.0017023, 0.0020312, -0.0001772, 0.0002242
2: 0.0115930, 0.0127987, 0.0116470, 0.0129055, -0.0008579, 0.0006782
3: -0.0026905, -0.0014434, -0.0026346, -0.0013330, -0.0008873, 0.0007014
4: -0.0024744, -0.0011244, -0.0025939, -0.0011849, -0.0007593, 0.0009605
5: 0.0051774, 0.0064549, 0.0052347, 0.0065681, -0.0009090, 0.0007186
6: -0.0017580, 0.0033110, -0.0015307, 0.0037598, -0.0036065, 0.0028511
7: -0.0070660, -0.0001624, -0.0076772, -0.0004720, -0.0038829, 0.0049118
8: 0.9842364, 0.9890994, 0.9838059, 0.9888813, -0.0027352, 0.0034600
9: -0.0059925, -0.0015781, -0.0057946, -0.0011873, -0.0031407, 0.0024828

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_A2_A1_B1

### Relational analysis result of IS_A2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016778, upper bound: 0.0017395
time: 0.98 seconds

## Relational analysis of IS_A2_A1_A2_A1_B2

### Relational analysis result of IS_A2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016778, upper bound: 0.0017395
time: 1.03 seconds

## BFS IS instance: IS_A2_A1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0026698, 0.0049633, 0.0025548, 0.0049081, -0.0012547, 0.0016103
1: 0.0017080, 0.0020394, 0.0016914, 0.0020314, -0.0001813, 0.0002326
2: 0.0116158, 0.0128838, 0.0116463, 0.0129474, -0.0008903, 0.0006937
3: -0.0026669, -0.0013554, -0.0026353, -0.0012897, -0.0009208, 0.0007174
4: -0.0025696, -0.0011499, -0.0026408, -0.0011841, -0.0007767, 0.0009968
5: 0.0052015, 0.0065451, 0.0052339, 0.0066124, -0.0009433, 0.0007350
6: -0.0016621, 0.0036687, -0.0015339, 0.0039358, -0.0037427, 0.0029162
7: -0.0075532, -0.0002930, -0.0079170, -0.0004677, -0.0039716, 0.0050972
8: 0.9838932, 0.9890075, 0.9836370, 0.9888844, -0.0027977, 0.0035906
9: -0.0059090, -0.0012666, -0.0057973, -0.0010340, -0.0032593, 0.0025396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_A2_A2_B1

### Relational analysis result of IS_A2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017669, upper bound: 0.0017590
time: 0.81 seconds

## Relational analysis of IS_A2_A1_A2_A2_B2

### Relational analysis result of IS_A2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017669, upper bound: 0.0017590
time: 0.81 seconds

## BFS IS instance: IS_A2_A2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0028349, 0.0050386, 0.0026139, 0.0049282, -0.0010560, 0.0016746
1: 0.0017319, 0.0020502, 0.0016999, 0.0020343, -0.0001526, 0.0002419
2: 0.0115741, 0.0127925, 0.0116352, 0.0129147, -0.0009259, 0.0005838
3: -0.0027099, -0.0014498, -0.0026468, -0.0013235, -0.0009576, 0.0006038
4: -0.0024674, -0.0011033, -0.0026042, -0.0011717, -0.0006537, 0.0010366
5: 0.0051574, 0.0064484, 0.0052221, 0.0065778, -0.0009810, 0.0006186
6: -0.0018371, 0.0032849, -0.0015804, 0.0037986, -0.0038923, 0.0024545
7: -0.0070305, -0.0000547, -0.0077300, -0.0004043, -0.0033428, 0.0053010
8: 0.9842615, 0.9891754, 0.9837686, 0.9889292, -0.0023547, 0.0037341
9: -0.0060614, -0.0016009, -0.0058378, -0.0011536, -0.0033896, 0.0021375

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_A1_A1_B1

### Relational analysis result of IS_A2_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016897, upper bound: 0.0017370
time: 0.85 seconds

## Relational analysis of IS_A2_A2_A1_A1_B2

### Relational analysis result of IS_A2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016897, upper bound: 0.0017370
time: 0.88 seconds

## BFS IS instance: IS_A2_A2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0026820, 0.0049985, 0.0025391, 0.0049295, -0.0010695, 0.0017414
1: 0.0017098, 0.0020444, 0.0016891, 0.0020345, -0.0001545, 0.0002516
2: 0.0115963, 0.0128771, 0.0116345, 0.0129561, -0.0009628, 0.0005913
3: -0.0026870, -0.0013624, -0.0026476, -0.0012807, -0.0009958, 0.0006116
4: -0.0025621, -0.0011281, -0.0026505, -0.0011708, -0.0006621, 0.0010780
5: 0.0051809, 0.0065379, 0.0052213, 0.0066216, -0.0010201, 0.0006265
6: -0.0017439, 0.0036403, -0.0015836, 0.0039724, -0.0040475, 0.0024859
7: -0.0075145, -0.0001817, -0.0079668, -0.0004000, -0.0033855, 0.0055123
8: 0.9839206, 0.9890859, 0.9836019, 0.9889321, -0.0023848, 0.0038830
9: -0.0059802, -0.0012914, -0.0058406, -0.0010022, -0.0035247, 0.0021648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_A1_A2_B1

### Relational analysis result of IS_A2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017853, upper bound: 0.0017549
time: 0.81 seconds

## Relational analysis of IS_A2_A2_A1_A2_B2

### Relational analysis result of IS_A2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017853, upper bound: 0.0017549
time: 0.80 seconds

## BFS IS instance: IS_A2_A2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0027883, 0.0050203, 0.0026156, 0.0049072, -0.0011558, 0.0016741
1: 0.0017251, 0.0020476, 0.0017002, 0.0020312, -0.0001670, 0.0002419
2: 0.0115843, 0.0128183, 0.0116468, 0.0129138, -0.0009255, 0.0006390
3: -0.0026995, -0.0014232, -0.0026348, -0.0013244, -0.0009572, 0.0006609
4: -0.0024962, -0.0011146, -0.0026032, -0.0011846, -0.0007155, 0.0010363
5: 0.0051682, 0.0064756, 0.0052344, 0.0065769, -0.0009807, 0.0006771
6: -0.0017946, 0.0033931, -0.0015317, 0.0037947, -0.0038910, 0.0026864
7: -0.0071778, -0.0001126, -0.0077248, -0.0004707, -0.0036586, 0.0052992
8: 0.9841577, 0.9891346, 0.9837723, 0.9888823, -0.0025772, 0.0037329
9: -0.0060243, -0.0015067, -0.0057953, -0.0011569, -0.0033884, 0.0023394

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_A2_A1_B1

### Relational analysis result of IS_A2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016897, upper bound: 0.0017396
time: 0.83 seconds

## Relational analysis of IS_A2_A2_A2_A1_B2

### Relational analysis result of IS_A2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016897, upper bound: 0.0017396
time: 0.81 seconds

## BFS IS instance: IS_A2_A2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0026370, 0.0049806, 0.0025407, 0.0049085, -0.0011847, 0.0017373
1: 0.0017033, 0.0020419, 0.0016894, 0.0020314, -0.0001712, 0.0002510
2: 0.0116062, 0.0129019, 0.0116461, 0.0129552, -0.0009605, 0.0006550
3: -0.0026768, -0.0013367, -0.0026356, -0.0012816, -0.0009934, 0.0006774
4: -0.0025899, -0.0011392, -0.0026495, -0.0011838, -0.0007334, 0.0010754
5: 0.0051914, 0.0065643, 0.0052336, 0.0066207, -0.0010177, 0.0006940
6: -0.0017024, 0.0037448, -0.0015348, 0.0039687, -0.0040380, 0.0027536
7: -0.0076568, -0.0002382, -0.0079617, -0.0004665, -0.0037502, 0.0054994
8: 0.9838202, 0.9890460, 0.9836054, 0.9888853, -0.0026417, 0.0038739
9: -0.0059440, -0.0012004, -0.0057981, -0.0010054, -0.0035165, 0.0023980

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_A2_A2_B1

### Relational analysis result of IS_A2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017854, upper bound: 0.0017597
time: 1.01 seconds

## Relational analysis of IS_A2_A2_A2_A2_B2

### Relational analysis result of IS_A2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017854, upper bound: 0.0017597
time: 0.80 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.29 seconds
IS_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 8, lower bound: -0.0016944, upper bound: 0.0017207
IS_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 8, lower bound: -0.0016944, upper bound: 0.0017427
IS_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 8, lower bound: -0.0017617, upper bound: 0.0017386
IS_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 8, lower bound: -0.0017617, upper bound: 0.0017603
IS_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 8, lower bound: -0.0016944, upper bound: 0.0017207
IS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 8, lower bound: -0.0016944, upper bound: 0.0017474
IS_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 8, lower bound: -0.0017617, upper bound: 0.0017399
IS_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 8, lower bound: -0.0017617, upper bound: 0.0017669
IS_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 8, lower bound: -0.0016944, upper bound: 0.0017370
IS_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 8, lower bound: -0.0016944, upper bound: 0.0017645
IS_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 8, lower bound: -0.0017623, upper bound: 0.0017549
IS_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 8, lower bound: -0.0017623, upper bound: 0.0017826
IS_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 8, lower bound: -0.0016944, upper bound: 0.0017396
IS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 8, lower bound: -0.0016944, upper bound: 0.0017657
IS_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 8, lower bound: -0.0017623, upper bound: 0.0017597
IS_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 8, lower bound: -0.0017623, upper bound: 0.0017853
IS_A2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 8, lower bound: -0.0016778, upper bound: 0.0017367
IS_A2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 8, lower bound: -0.0016778, upper bound: 0.0017367
IS_A2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 8, lower bound: -0.0017669, upper bound: 0.0017541
IS_A2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 8, lower bound: -0.0017669, upper bound: 0.0017541
IS_A2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 8, lower bound: -0.0016778, upper bound: 0.0017395
IS_A2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 8, lower bound: -0.0016778, upper bound: 0.0017395
IS_A2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 8, lower bound: -0.0017669, upper bound: 0.0017590
IS_A2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 8, lower bound: -0.0017669, upper bound: 0.0017590
IS_A2_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 8, lower bound: -0.0016897, upper bound: 0.0017370
IS_A2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 8, lower bound: -0.0016897, upper bound: 0.0017370
IS_A2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 8, lower bound: -0.0017853, upper bound: 0.0017549
IS_A2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 8, lower bound: -0.0017853, upper bound: 0.0017549
IS_A2_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 8, lower bound: -0.0016897, upper bound: 0.0017396
IS_A2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 8, lower bound: -0.0016897, upper bound: 0.0017396
IS_A2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 8, lower bound: -0.0017854, upper bound: 0.0017597
IS_A2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 8, lower bound: -0.0017854, upper bound: 0.0017597

## BFS IS instance: IS_A1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0027686, 0.0049375, 0.0027298, 0.0049144, -0.0011116, 0.0011963
1: 0.0017223, 0.0020356, 0.0017167, 0.0020323, -0.0001606, 0.0001728
2: 0.0116300, 0.0128292, 0.0116428, 0.0128506, -0.0006614, 0.0006146
3: -0.0026521, -0.0014119, -0.0026389, -0.0013897, -0.0006841, 0.0006356
4: -0.0025085, -0.0011659, -0.0025325, -0.0011802, -0.0006881, 0.0007405
5: 0.0052167, 0.0064872, 0.0052302, 0.0065099, -0.0007008, 0.0006512
6: -0.0016021, 0.0034391, -0.0015485, 0.0035292, -0.0027806, 0.0025836
7: -0.0072405, -0.0003748, -0.0073632, -0.0004478, -0.0035187, 0.0037869
8: 0.9841136, 0.9889498, 0.9840271, 0.9888984, -0.0024786, 0.0026676
9: -0.0058567, -0.0014666, -0.0058100, -0.0013881, -0.0024214, 0.0022499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016828, upper bound: 0.0017207
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016828, upper bound: 0.0017207
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0027686, 0.0049375, 0.0027599, 0.0050186, -0.0013432, 0.0012871
1: 0.0017223, 0.0020356, 0.0017210, 0.0020473, -0.0001941, 0.0001859
2: 0.0116300, 0.0128292, 0.0115852, 0.0128340, -0.0007116, 0.0007426
3: -0.0026521, -0.0014119, -0.0026985, -0.0014069, -0.0007359, 0.0007681
4: -0.0025085, -0.0011659, -0.0025139, -0.0011157, -0.0008315, 0.0007967
5: 0.0052167, 0.0064872, 0.0051691, 0.0064923, -0.0007540, 0.0007869
6: -0.0016021, 0.0034391, -0.0017907, 0.0034592, -0.0029915, 0.0031221
7: -0.0072405, -0.0003748, -0.0072679, -0.0001179, -0.0042520, 0.0040741
8: 0.9841136, 0.9889498, 0.9840942, 0.9891307, -0.0029952, 0.0028699
9: -0.0058567, -0.0014666, -0.0060210, -0.0014490, -0.0026051, 0.0027188

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016828, upper bound: 0.0017427
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016828, upper bound: 0.0017427
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0026286, 0.0048868, 0.0026593, 0.0049158, -0.0011190, 0.0012796
1: 0.0017021, 0.0020283, 0.0017065, 0.0020325, -0.0001617, 0.0001849
2: 0.0116581, 0.0129066, 0.0116420, 0.0128896, -0.0007075, 0.0006187
3: -0.0026232, -0.0013319, -0.0026397, -0.0013494, -0.0007317, 0.0006399
4: -0.0025952, -0.0011972, -0.0025761, -0.0011793, -0.0006927, 0.0007921
5: 0.0052463, 0.0065692, 0.0052294, 0.0065512, -0.0007496, 0.0006555
6: -0.0014844, 0.0037645, -0.0015517, 0.0036930, -0.0029742, 0.0026010
7: -0.0076836, -0.0005351, -0.0075863, -0.0004434, -0.0035423, 0.0040506
8: 0.9838014, 0.9888369, 0.9838700, 0.9889016, -0.0024953, 0.0028533
9: -0.0057542, -0.0011832, -0.0058128, -0.0012455, -0.0025901, 0.0022650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017425, upper bound: 0.0016663
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017425, upper bound: 0.0016900
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0026286, 0.0048868, 0.0026843, 0.0050199, -0.0013793, 0.0013623
1: 0.0017021, 0.0020283, 0.0017101, 0.0020475, -0.0001993, 0.0001968
2: 0.0116581, 0.0129066, 0.0115845, 0.0128758, -0.0007532, 0.0007626
3: -0.0026232, -0.0013319, -0.0026992, -0.0013637, -0.0007790, 0.0007887
4: -0.0025952, -0.0011972, -0.0025606, -0.0011149, -0.0008538, 0.0008433
5: 0.0052463, 0.0065692, 0.0051684, 0.0065366, -0.0007980, 0.0008080
6: -0.0014844, 0.0037645, -0.0017936, 0.0036348, -0.0031663, 0.0032058
7: -0.0076836, -0.0005351, -0.0075071, -0.0001140, -0.0043660, 0.0043123
8: 0.9838014, 0.9888369, 0.9839258, 0.9891335, -0.0030755, 0.0030377
9: -0.0057542, -0.0011832, -0.0060234, -0.0012961, -0.0027574, 0.0027917

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017425, upper bound: 0.0016729
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017425, upper bound: 0.0016975
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0027233, 0.0049202, 0.0027315, 0.0048920, -0.0012107, 0.0012031
1: 0.0017157, 0.0020331, 0.0017169, 0.0020291, -0.0001749, 0.0001738
2: 0.0116396, 0.0128542, 0.0116552, 0.0128497, -0.0006652, 0.0006694
3: -0.0026422, -0.0013860, -0.0026261, -0.0013907, -0.0006880, 0.0006923
4: -0.0025365, -0.0011766, -0.0025315, -0.0011940, -0.0007495, 0.0007448
5: 0.0052268, 0.0065137, 0.0052433, 0.0065090, -0.0007048, 0.0007092
6: -0.0015619, 0.0035443, -0.0014964, 0.0035253, -0.0027964, 0.0028141
7: -0.0073837, -0.0004296, -0.0073579, -0.0005187, -0.0038325, 0.0038085
8: 0.9840126, 0.9889113, 0.9840308, 0.9888485, -0.0026997, 0.0026828
9: -0.0058217, -0.0013750, -0.0057647, -0.0013915, -0.0024352, 0.0024506

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016828, upper bound: 0.0017207
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016828, upper bound: 0.0017207
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0027233, 0.0049202, 0.0027616, 0.0049991, -0.0014330, 0.0012939
1: 0.0017157, 0.0020331, 0.0017213, 0.0020445, -0.0002070, 0.0001869
2: 0.0116396, 0.0128542, 0.0115960, 0.0128331, -0.0007154, 0.0007923
3: -0.0026422, -0.0013860, -0.0026873, -0.0014079, -0.0007399, 0.0008194
4: -0.0025365, -0.0011766, -0.0025128, -0.0011278, -0.0008870, 0.0008010
5: 0.0052268, 0.0065137, 0.0051806, 0.0064913, -0.0007580, 0.0008394
6: -0.0015619, 0.0035443, -0.0017453, 0.0034553, -0.0030074, 0.0033306
7: -0.0073837, -0.0004296, -0.0072626, -0.0001798, -0.0045360, 0.0040959
8: 0.9840126, 0.9889113, 0.9840979, 0.9890872, -0.0031953, 0.0028852
9: -0.0058217, -0.0013750, -0.0059814, -0.0014525, -0.0026190, 0.0029005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016828, upper bound: 0.0017474
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016828, upper bound: 0.0017474
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0025833, 0.0048678, 0.0026610, 0.0048934, -0.0012353, 0.0012855
1: 0.0016955, 0.0020256, 0.0017067, 0.0020293, -0.0001785, 0.0001857
2: 0.0116686, 0.0129316, 0.0116544, 0.0128887, -0.0007107, 0.0006829
3: -0.0026123, -0.0013060, -0.0026269, -0.0013504, -0.0007351, 0.0007063
4: -0.0026232, -0.0012090, -0.0025751, -0.0011932, -0.0007646, 0.0007957
5: 0.0052575, 0.0065958, 0.0052425, 0.0065503, -0.0007530, 0.0007236
6: -0.0014402, 0.0038698, -0.0014996, 0.0036892, -0.0029879, 0.0028711
7: -0.0078270, -0.0005953, -0.0075811, -0.0005144, -0.0039102, 0.0040692
8: 0.9837004, 0.9887945, 0.9838736, 0.9888515, -0.0027544, 0.0028664
9: -0.0057157, -0.0010916, -0.0057674, -0.0012488, -0.0026020, 0.0025003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017425, upper bound: 0.0016717
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A2_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017425, upper bound: 0.0016968
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0025833, 0.0048678, 0.0026860, 0.0050003, -0.0014795, 0.0013682
1: 0.0016955, 0.0020256, 0.0017103, 0.0020447, -0.0002137, 0.0001977
2: 0.0116686, 0.0129316, 0.0115953, 0.0128749, -0.0007564, 0.0008180
3: -0.0026123, -0.0013060, -0.0026880, -0.0013647, -0.0007823, 0.0008460
4: -0.0026232, -0.0012090, -0.0025596, -0.0011270, -0.0009158, 0.0008469
5: 0.0052575, 0.0065958, 0.0051799, 0.0065356, -0.0008015, 0.0008667
6: -0.0014402, 0.0038698, -0.0017481, 0.0036310, -0.0031800, 0.0034388
7: -0.0078270, -0.0005953, -0.0075018, -0.0001760, -0.0046833, 0.0043309
8: 0.9837004, 0.9887945, 0.9839294, 0.9890900, -0.0032990, 0.0030508
9: -0.0057157, -0.0010916, -0.0059838, -0.0012995, -0.0027693, 0.0029946

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017425, upper bound: 0.0016778
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017425, upper bound: 0.0017047
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0027566, 0.0049378, 0.0027024, 0.0049240, -0.0012501, 0.0011126
1: 0.0017206, 0.0020357, 0.0017127, 0.0020337, -0.0001806, 0.0001607
2: 0.0116299, 0.0128358, 0.0116375, 0.0128658, -0.0006151, 0.0006911
3: -0.0026523, -0.0014051, -0.0026444, -0.0013740, -0.0006362, 0.0007148
4: -0.0025159, -0.0011657, -0.0025495, -0.0011743, -0.0007738, 0.0006887
5: 0.0052165, 0.0064942, 0.0052246, 0.0065260, -0.0006518, 0.0007323
6: -0.0016028, 0.0034668, -0.0015706, 0.0035930, -0.0025860, 0.0029056
7: -0.0072782, -0.0003738, -0.0074500, -0.0004176, -0.0039571, 0.0035220
8: 0.9840869, 0.9889506, 0.9839659, 0.9889197, -0.0027875, 0.0024809
9: -0.0058573, -0.0014425, -0.0058293, -0.0013326, -0.0022520, 0.0025303

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016828, upper bound: 0.0017367
time: 1.05 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016828, upper bound: 0.0017215
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0027566, 0.0049378, 0.0027259, 0.0050380, -0.0014757, 0.0012039
1: 0.0017206, 0.0020357, 0.0017161, 0.0020501, -0.0002132, 0.0001739
2: 0.0116299, 0.0128358, 0.0115745, 0.0128528, -0.0006656, 0.0008158
3: -0.0026523, -0.0014051, -0.0027096, -0.0013875, -0.0006884, 0.0008438
4: -0.0025159, -0.0011657, -0.0025349, -0.0011037, -0.0009135, 0.0007452
5: 0.0052165, 0.0064942, 0.0051578, 0.0065122, -0.0007052, 0.0008644
6: -0.0016028, 0.0034668, -0.0018356, 0.0035383, -0.0027982, 0.0034298
7: -0.0072782, -0.0003738, -0.0073756, -0.0000568, -0.0046711, 0.0038109
8: 0.9840869, 0.9889506, 0.9840184, 0.9891739, -0.0032904, 0.0026845
9: -0.0058573, -0.0014425, -0.0060601, -0.0013802, -0.0024368, 0.0029868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016828, upper bound: 0.0017642
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016828, upper bound: 0.0017436
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0026132, 0.0048872, 0.0026285, 0.0049253, -0.0012686, 0.0012115
1: 0.0016998, 0.0020284, 0.0017020, 0.0020339, -0.0001833, 0.0001750
2: 0.0116579, 0.0129151, 0.0116368, 0.0129066, -0.0006698, 0.0007014
3: -0.0026234, -0.0013231, -0.0026452, -0.0013318, -0.0006928, 0.0007254
4: -0.0026046, -0.0011970, -0.0025952, -0.0011734, -0.0007853, 0.0007499
5: 0.0052461, 0.0065782, 0.0052238, 0.0065693, -0.0007097, 0.0007431
6: -0.0014852, 0.0038001, -0.0015738, 0.0037646, -0.0028159, 0.0029485
7: -0.0077322, -0.0005340, -0.0076837, -0.0004133, -0.0040156, 0.0038350
8: 0.9837672, 0.9888377, 0.9838014, 0.9889227, -0.0028287, 0.0027014
9: -0.0057549, -0.0011522, -0.0058320, -0.0011832, -0.0024522, 0.0025677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1_A2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017427, upper bound: 0.0016775
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017427, upper bound: 0.0017024
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0026132, 0.0048872, 0.0026499, 0.0050392, -0.0015084, 0.0012952
1: 0.0016998, 0.0020284, 0.0017051, 0.0020503, -0.0002179, 0.0001871
2: 0.0116579, 0.0129151, 0.0115738, 0.0128948, -0.0007161, 0.0008339
3: -0.0026234, -0.0013231, -0.0027103, -0.0013440, -0.0007406, 0.0008625
4: -0.0026046, -0.0011970, -0.0025819, -0.0011030, -0.0009337, 0.0008018
5: 0.0052461, 0.0065782, 0.0051571, 0.0065567, -0.0007587, 0.0008836
6: -0.0014852, 0.0038001, -0.0018384, 0.0037149, -0.0030105, 0.0035059
7: -0.0077322, -0.0005340, -0.0076161, -0.0000529, -0.0047747, 0.0041000
8: 0.9837672, 0.9888377, 0.9838489, 0.9891766, -0.0033634, 0.0028881
9: -0.0057549, -0.0011522, -0.0060625, -0.0012264, -0.0026217, 0.0030531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1_A2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017427, upper bound: 0.0016852
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017427, upper bound: 0.0017113
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0027113, 0.0049205, 0.0027040, 0.0049029, -0.0013460, 0.0011193
1: 0.0017140, 0.0020332, 0.0017130, 0.0020306, -0.0001945, 0.0001617
2: 0.0116394, 0.0128609, 0.0116492, 0.0128649, -0.0006189, 0.0007442
3: -0.0026424, -0.0013792, -0.0026324, -0.0013750, -0.0006400, 0.0007696
4: -0.0025439, -0.0011764, -0.0025485, -0.0011873, -0.0008332, 0.0006929
5: 0.0052266, 0.0065208, 0.0052369, 0.0065250, -0.0006557, 0.0007885
6: -0.0015627, 0.0035722, -0.0015218, 0.0035891, -0.0026016, 0.0031284
7: -0.0074217, -0.0004285, -0.0074448, -0.0004842, -0.0042607, 0.0035432
8: 0.9839858, 0.9889120, 0.9839696, 0.9888728, -0.0030013, 0.0024959
9: -0.0058224, -0.0013507, -0.0057868, -0.0013359, -0.0022656, 0.0027244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016828, upper bound: 0.0017395
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016828, upper bound: 0.0017213
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0027113, 0.0049205, 0.0027276, 0.0050171, -0.0015593, 0.0012106
1: 0.0017140, 0.0020332, 0.0017164, 0.0020471, -0.0002253, 0.0001749
2: 0.0116394, 0.0128609, 0.0115860, 0.0128518, -0.0006693, 0.0008621
3: -0.0026424, -0.0013792, -0.0026977, -0.0013885, -0.0006922, 0.0008916
4: -0.0025439, -0.0011764, -0.0025338, -0.0011166, -0.0009652, 0.0007494
5: 0.0052266, 0.0065208, 0.0051700, 0.0065112, -0.0007092, 0.0009134
6: -0.0015627, 0.0035722, -0.0017872, 0.0035343, -0.0028137, 0.0036243
7: -0.0074217, -0.0004285, -0.0073701, -0.0001227, -0.0049360, 0.0038321
8: 0.9839858, 0.9889120, 0.9840223, 0.9891275, -0.0034770, 0.0026994
9: -0.0058224, -0.0013507, -0.0060179, -0.0013837, -0.0024503, 0.0031562

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016828, upper bound: 0.0017653
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016828, upper bound: 0.0017485
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0025681, 0.0048682, 0.0026301, 0.0049043, -0.0013704, 0.0012101
1: 0.0016933, 0.0020256, 0.0017023, 0.0020308, -0.0001980, 0.0001748
2: 0.0116684, 0.0129400, 0.0116484, 0.0129057, -0.0006690, 0.0007577
3: -0.0026125, -0.0012973, -0.0026331, -0.0013328, -0.0006920, 0.0007836
4: -0.0026326, -0.0012088, -0.0025942, -0.0011864, -0.0008483, 0.0007491
5: 0.0052573, 0.0066047, 0.0052361, 0.0065683, -0.0007089, 0.0008028
6: -0.0014410, 0.0039050, -0.0015249, 0.0037608, -0.0028127, 0.0031853
7: -0.0078750, -0.0005942, -0.0076786, -0.0004799, -0.0043381, 0.0038306
8: 0.9836666, 0.9887953, 0.9838049, 0.9888757, -0.0030558, 0.0026984
9: -0.0057164, -0.0010609, -0.0057895, -0.0011864, -0.0024494, 0.0027739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017427, upper bound: 0.0016840
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017427, upper bound: 0.0017113
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0025681, 0.0048682, 0.0026516, 0.0050183, -0.0016005, 0.0012937
1: 0.0016933, 0.0020256, 0.0017054, 0.0020473, -0.0002312, 0.0001869
2: 0.0116684, 0.0129400, 0.0115854, 0.0128938, -0.0007153, 0.0008849
3: -0.0026125, -0.0012973, -0.0026983, -0.0013450, -0.0007398, 0.0009152
4: -0.0026326, -0.0012088, -0.0025809, -0.0011159, -0.0009908, 0.0008008
5: 0.0052573, 0.0066047, 0.0051693, 0.0065557, -0.0007579, 0.0009376
6: -0.0014410, 0.0039050, -0.0017900, 0.0037109, -0.0030070, 0.0037201
7: -0.0078750, -0.0005942, -0.0076106, -0.0001189, -0.0050665, 0.0040953
8: 0.9836666, 0.9887953, 0.9838527, 0.9891301, -0.0035689, 0.0028848
9: -0.0057164, -0.0010609, -0.0060203, -0.0012299, -0.0026186, 0.0032396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017427, upper bound: 0.0016897
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017427, upper bound: 0.0017183
time: 1.04 seconds

## BFS IS instance: IS_A2_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0028669, 0.0050237, 0.0026661, 0.0049251, -0.0012116, 0.0014410
1: 0.0017365, 0.0020481, 0.0017075, 0.0020338, -0.0001750, 0.0002082
2: 0.0115824, 0.0127748, 0.0116369, 0.0128859, -0.0007967, 0.0006698
3: -0.0027014, -0.0014681, -0.0026450, -0.0013533, -0.0008240, 0.0006928
4: -0.0024476, -0.0011125, -0.0025719, -0.0011735, -0.0007500, 0.0008920
5: 0.0051662, 0.0064296, 0.0052239, 0.0065473, -0.0008441, 0.0007097
6: -0.0018025, 0.0032105, -0.0015734, 0.0036774, -0.0033492, 0.0028160
7: -0.0069291, -0.0001019, -0.0075650, -0.0004139, -0.0038351, 0.0045613
8: 0.9843329, 0.9891422, 0.9838849, 0.9889223, -0.0027015, 0.0032131
9: -0.0060312, -0.0016657, -0.0058317, -0.0012591, -0.0029166, 0.0024523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A1_A1_A1_B1_B1

### Relational analysis result of IS_A2_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016778, upper bound: 0.0017207
time: 0.85 seconds

## Relational analysis of IS_A2_A1_A1_A1_B1_B2

### Relational analysis result of IS_A2_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016778, upper bound: 0.0017367
time: 0.86 seconds

## BFS IS instance: IS_A2_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0028669, 0.0050237, 0.0026925, 0.0050390, -0.0011566, 0.0012163
1: 0.0017365, 0.0020481, 0.0017113, 0.0020503, -0.0001671, 0.0001757
2: 0.0115824, 0.0127748, 0.0115739, 0.0128712, -0.0006724, 0.0006395
3: -0.0027014, -0.0014681, -0.0027102, -0.0013684, -0.0006955, 0.0006614
4: -0.0024476, -0.0011125, -0.0025556, -0.0011030, -0.0007160, 0.0007529
5: 0.0051662, 0.0064296, 0.0051572, 0.0065318, -0.0007125, 0.0006776
6: -0.0018025, 0.0032105, -0.0018381, 0.0036158, -0.0028270, 0.0026883
7: -0.0069291, -0.0001019, -0.0074811, -0.0000534, -0.0036613, 0.0038501
8: 0.9843329, 0.9891422, 0.9839440, 0.9891762, -0.0025791, 0.0027121
9: -0.0060312, -0.0016657, -0.0060622, -0.0013127, -0.0024618, 0.0023411

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A1_A1_A1_B2_B1

### Relational analysis result of IS_A2_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016778, upper bound: 0.0017207
time: 0.84 seconds

## Relational analysis of IS_A2_A1_A1_A1_B2_B2

### Relational analysis result of IS_A2_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016778, upper bound: 0.0017367
time: 0.83 seconds

## BFS IS instance: IS_A2_A1_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0027163, 0.0049792, 0.0025936, 0.0049265, -0.0012316, 0.0015067
1: 0.0017147, 0.0020416, 0.0016970, 0.0020340, -0.0001779, 0.0002177
2: 0.0116070, 0.0128581, 0.0116361, 0.0129259, -0.0008330, 0.0006809
3: -0.0026759, -0.0013820, -0.0026458, -0.0013119, -0.0008616, 0.0007042
4: -0.0025408, -0.0011401, -0.0026168, -0.0011727, -0.0007624, 0.0009327
5: 0.0051923, 0.0065178, 0.0052231, 0.0065897, -0.0008826, 0.0007215
6: -0.0016989, 0.0035605, -0.0015765, 0.0038456, -0.0035021, 0.0028625
7: -0.0074058, -0.0002429, -0.0077941, -0.0004096, -0.0038985, 0.0047695
8: 0.9839971, 0.9890428, 0.9837235, 0.9889254, -0.0027462, 0.0033597
9: -0.0059410, -0.0013609, -0.0058344, -0.0011126, -0.0030497, 0.0024928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A2_A1_A1_A2_B1_B1

### Relational analysis result of IS_A2_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0016774
time: 0.83 seconds

## Relational analysis of IS_A2_A1_A1_A2_B1_B2

### Relational analysis result of IS_A2_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0017018
time: 1.01 seconds

## BFS IS instance: IS_A2_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0027163, 0.0049792, 0.0026179, 0.0050403, -0.0011792, 0.0013169
1: 0.0017147, 0.0020416, 0.0017005, 0.0020505, -0.0001704, 0.0001903
2: 0.0116070, 0.0128581, 0.0115732, 0.0129125, -0.0007281, 0.0006520
3: -0.0026759, -0.0013820, -0.0027109, -0.0013257, -0.0007530, 0.0006743
4: -0.0025408, -0.0011401, -0.0026018, -0.0011023, -0.0007300, 0.0008152
5: 0.0051923, 0.0065178, 0.0051565, 0.0065755, -0.0007715, 0.0006908
6: -0.0016989, 0.0035605, -0.0018410, 0.0037894, -0.0030610, 0.0027409
7: -0.0074058, -0.0002429, -0.0077175, -0.0000495, -0.0037329, 0.0041688
8: 0.9839971, 0.9890428, 0.9837775, 0.9891790, -0.0026295, 0.0029366
9: -0.0059410, -0.0013609, -0.0060647, -0.0011616, -0.0026656, 0.0023869

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A2_A1_A1_A2_B2_B1

### Relational analysis result of IS_A2_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0016774
time: 0.82 seconds

## Relational analysis of IS_A2_A1_A1_A2_B2_B2

### Relational analysis result of IS_A2_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0017018
time: 0.98 seconds

## BFS IS instance: IS_A2_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0028237, 0.0050046, 0.0026677, 0.0049041, -0.0012827, 0.0014420
1: 0.0017302, 0.0020453, 0.0017077, 0.0020308, -0.0001853, 0.0002083
2: 0.0115930, 0.0127987, 0.0116485, 0.0128849, -0.0007972, 0.0007092
3: -0.0026905, -0.0014434, -0.0026330, -0.0013542, -0.0008245, 0.0007335
4: -0.0024744, -0.0011244, -0.0025709, -0.0011865, -0.0007940, 0.0008926
5: 0.0051774, 0.0064549, 0.0052362, 0.0065463, -0.0008447, 0.0007514
6: -0.0017580, 0.0033110, -0.0015245, 0.0036734, -0.0033515, 0.0029814
7: -0.0070660, -0.0001624, -0.0075596, -0.0004804, -0.0040604, 0.0045645
8: 0.9842364, 0.9890994, 0.9838887, 0.9888754, -0.0028602, 0.0032153
9: -0.0059925, -0.0015781, -0.0057892, -0.0012625, -0.0029187, 0.0025963

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 159

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A1_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016778, upper bound: 0.0017207
time: 1.02 seconds

## Relational analysis of IS_A2_A1_A2_A1_B1_B2

### Relational analysis result of IS_A2_A1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016778, upper bound: 0.0017395
time: 1.03 seconds

## BFS IS instance: IS_A2_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0028237, 0.0050046, 0.0026942, 0.0050182, -0.0012576, 0.0012237
1: 0.0017302, 0.0020453, 0.0017115, 0.0020473, -0.0001817, 0.0001768
2: 0.0115930, 0.0127987, 0.0115854, 0.0128703, -0.0006766, 0.0006953
3: -0.0026905, -0.0014434, -0.0026983, -0.0013694, -0.0006998, 0.0007191
4: -0.0024744, -0.0011244, -0.0025545, -0.0011159, -0.0007785, 0.0007575
5: 0.0051774, 0.0064549, 0.0051694, 0.0065308, -0.0007169, 0.0007367
6: -0.0017580, 0.0033110, -0.0017897, 0.0036119, -0.0028443, 0.0029231
7: -0.0070660, -0.0001624, -0.0074758, -0.0001193, -0.0039810, 0.0038737
8: 0.9842364, 0.9890994, 0.9839477, 0.9891298, -0.0028043, 0.0027287
9: -0.0059925, -0.0015781, -0.0060200, -0.0013161, -0.0024770, 0.0025456

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A1_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016778, upper bound: 0.0017207
time: 1.03 seconds

## Relational analysis of IS_A2_A1_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016778, upper bound: 0.0017395
time: 0.95 seconds

## BFS IS instance: IS_A2_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0026698, 0.0049633, 0.0025953, 0.0049055, -0.0013085, 0.0015016
1: 0.0017080, 0.0020394, 0.0016972, 0.0020310, -0.0001890, 0.0002169
2: 0.0116158, 0.0128838, 0.0116477, 0.0129250, -0.0008302, 0.0007234
3: -0.0026669, -0.0013554, -0.0026338, -0.0013128, -0.0008586, 0.0007482
4: -0.0025696, -0.0011499, -0.0026157, -0.0011857, -0.0008100, 0.0009295
5: 0.0052015, 0.0065451, 0.0052354, 0.0065887, -0.0008797, 0.0007665
6: -0.0016621, 0.0036687, -0.0015277, 0.0038418, -0.0034902, 0.0030413
7: -0.0075532, -0.0002930, -0.0077889, -0.0004761, -0.0041420, 0.0047534
8: 0.9838932, 0.9890075, 0.9837272, 0.9888784, -0.0029177, 0.0033484
9: -0.0059090, -0.0012666, -0.0057919, -0.0011159, -0.0030394, 0.0026485

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A2_A1_A2_A2_B1_B1

### Relational analysis result of IS_A2_A1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0016839
time: 1.03 seconds

## Relational analysis of IS_A2_A1_A2_A2_B1_B2

### Relational analysis result of IS_A2_A1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0017107
time: 0.96 seconds

## BFS IS instance: IS_A2_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0026698, 0.0049633, 0.0026195, 0.0050194, -0.0012852, 0.0013142
1: 0.0017080, 0.0020394, 0.0017007, 0.0020475, -0.0001857, 0.0001899
2: 0.0116158, 0.0128838, 0.0115848, 0.0129116, -0.0007266, 0.0007106
3: -0.0026669, -0.0013554, -0.0026990, -0.0013267, -0.0007515, 0.0007349
4: -0.0025696, -0.0011499, -0.0026007, -0.0011152, -0.0007956, 0.0008135
5: 0.0052015, 0.0065451, 0.0051687, 0.0065745, -0.0007699, 0.0007529
6: -0.0016621, 0.0036687, -0.0017925, 0.0037855, -0.0030547, 0.0029873
7: -0.0075532, -0.0002930, -0.0077122, -0.0001155, -0.0040684, 0.0041602
8: 0.9838932, 0.9890075, 0.9837812, 0.9891326, -0.0028659, 0.0029305
9: -0.0059090, -0.0012666, -0.0060225, -0.0011649, -0.0026601, 0.0026014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A2_A1_A2_A2_B2_B1

### Relational analysis result of IS_A2_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0016840
time: 1.09 seconds

## Relational analysis of IS_A2_A1_A2_A2_B2_B2

### Relational analysis result of IS_A2_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0017107
time: 1.06 seconds

## BFS IS instance: IS_A2_A2_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0028349, 0.0050386, 0.0026527, 0.0049255, -0.0011392, 0.0015675
1: 0.0017319, 0.0020502, 0.0017055, 0.0020339, -0.0001646, 0.0002265
2: 0.0115741, 0.0127925, 0.0116367, 0.0128932, -0.0008666, 0.0006298
3: -0.0027099, -0.0014498, -0.0026452, -0.0013457, -0.0008963, 0.0006514
4: -0.0024674, -0.0011033, -0.0025802, -0.0011733, -0.0007052, 0.0009703
5: 0.0051574, 0.0064484, 0.0052237, 0.0065551, -0.0009182, 0.0006673
6: -0.0018371, 0.0032849, -0.0015742, 0.0037083, -0.0036432, 0.0026477
7: -0.0070305, -0.0000547, -0.0076071, -0.0004128, -0.0036060, 0.0049617
8: 0.9842615, 0.9891754, 0.9838552, 0.9889230, -0.0025401, 0.0034951
9: -0.0060614, -0.0016009, -0.0058324, -0.0012321, -0.0031727, 0.0023058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A2_A1_A1_B1_B1

### Relational analysis result of IS_A2_A2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016897, upper bound: 0.0017207
time: 0.85 seconds

## Relational analysis of IS_A2_A2_A1_A1_B1_B2

### Relational analysis result of IS_A2_A2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016897, upper bound: 0.0017215
time: 1.04 seconds

## BFS IS instance: IS_A2_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0028349, 0.0050386, 0.0026773, 0.0050395, -0.0010906, 0.0013649
1: 0.0017319, 0.0020502, 0.0017091, 0.0020504, -0.0001576, 0.0001972
2: 0.0115741, 0.0127925, 0.0115736, 0.0128796, -0.0007546, 0.0006030
3: -0.0027099, -0.0014498, -0.0027105, -0.0013597, -0.0007805, 0.0006236
4: -0.0024674, -0.0011033, -0.0025650, -0.0011027, -0.0006751, 0.0008449
5: 0.0051574, 0.0064484, 0.0051569, 0.0065407, -0.0007996, 0.0006389
6: -0.0018371, 0.0032849, -0.0018393, 0.0036511, -0.0031725, 0.0025350
7: -0.0070305, -0.0000547, -0.0075292, -0.0000517, -0.0034524, 0.0043207
8: 0.9842615, 0.9891754, 0.9839101, 0.9891774, -0.0024319, 0.0030436
9: -0.0060614, -0.0016009, -0.0060633, -0.0012819, -0.0027628, 0.0022076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A2_A1_A1_B2_B1

### Relational analysis result of IS_A2_A2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016897, upper bound: 0.0017207
time: 0.85 seconds

## Relational analysis of IS_A2_A2_A1_A1_B2_B2

### Relational analysis result of IS_A2_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016897, upper bound: 0.0017215
time: 1.00 seconds

## BFS IS instance: IS_A2_A2_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0026820, 0.0049985, 0.0025798, 0.0049269, -0.0011500, 0.0016350
1: 0.0017098, 0.0020444, 0.0016950, 0.0020341, -0.0001661, 0.0002362
2: 0.0115963, 0.0128771, 0.0116359, 0.0129336, -0.0009039, 0.0006358
3: -0.0026870, -0.0013624, -0.0026460, -0.0013040, -0.0009349, 0.0006576
4: -0.0025621, -0.0011281, -0.0026253, -0.0011725, -0.0007119, 0.0010121
5: 0.0051809, 0.0065379, 0.0052229, 0.0065978, -0.0009578, 0.0006737
6: -0.0017439, 0.0036403, -0.0015774, 0.0038778, -0.0038001, 0.0026729
7: -0.0075145, -0.0001817, -0.0078380, -0.0004085, -0.0036402, 0.0051754
8: 0.9839206, 0.9890859, 0.9836926, 0.9889262, -0.0025642, 0.0036457
9: -0.0059802, -0.0012914, -0.0058352, -0.0010845, -0.0033093, 0.0023276

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A2_A2_A1_A2_B1_B1

### Relational analysis result of IS_A2_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017657, upper bound: 0.0016775
time: 0.82 seconds

## Relational analysis of IS_A2_A2_A1_A2_B1_B2

### Relational analysis result of IS_A2_A2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017657, upper bound: 0.0017024
time: 1.05 seconds

## BFS IS instance: IS_A2_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0026820, 0.0049985, 0.0026022, 0.0050408, -0.0011042, 0.0014489
1: 0.0017098, 0.0020444, 0.0016982, 0.0020505, -0.0001595, 0.0002093
2: 0.0115963, 0.0128771, 0.0115729, 0.0129211, -0.0008011, 0.0006105
3: -0.0026870, -0.0013624, -0.0027112, -0.0013168, -0.0008285, 0.0006314
4: -0.0025621, -0.0011281, -0.0026114, -0.0011020, -0.0006835, 0.0008969
5: 0.0051809, 0.0065379, 0.0051562, 0.0065847, -0.0008488, 0.0006468
6: -0.0017439, 0.0036403, -0.0018421, 0.0038257, -0.0033678, 0.0025665
7: -0.0075145, -0.0001817, -0.0077669, -0.0000479, -0.0034953, 0.0045866
8: 0.9839206, 0.9890859, 0.9837427, 0.9891801, -0.0024622, 0.0032309
9: -0.0059802, -0.0012914, -0.0060657, -0.0011300, -0.0029328, 0.0022350

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A2_A2_A1_A2_B2_B1

### Relational analysis result of IS_A2_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017657, upper bound: 0.0016775
time: 0.79 seconds

## Relational analysis of IS_A2_A2_A1_A2_B2_B2

### Relational analysis result of IS_A2_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017657, upper bound: 0.0017024
time: 1.00 seconds

## BFS IS instance: IS_A2_A2_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0027883, 0.0050203, 0.0026544, 0.0049045, -0.0012094, 0.0015668
1: 0.0017251, 0.0020476, 0.0017058, 0.0020309, -0.0001747, 0.0002264
2: 0.0115843, 0.0128183, 0.0116483, 0.0128923, -0.0008662, 0.0006686
3: -0.0026995, -0.0014232, -0.0026332, -0.0013466, -0.0008959, 0.0006915
4: -0.0024962, -0.0011146, -0.0025792, -0.0011863, -0.0007486, 0.0009699
5: 0.0051682, 0.0064756, 0.0052360, 0.0065541, -0.0009178, 0.0007085
6: -0.0017946, 0.0033931, -0.0015254, 0.0037045, -0.0036417, 0.0028110
7: -0.0071778, -0.0001126, -0.0076019, -0.0004793, -0.0038283, 0.0049597
8: 0.9841577, 0.9891346, 0.9838589, 0.9888762, -0.0026967, 0.0034937
9: -0.0060243, -0.0015067, -0.0057899, -0.0012355, -0.0031714, 0.0024479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 159

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A2_A2_A1_B1_B1

### Relational analysis result of IS_A2_A2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016897, upper bound: 0.0017207
time: 0.86 seconds

## Relational analysis of IS_A2_A2_A2_A1_B1_B2

### Relational analysis result of IS_A2_A2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016897, upper bound: 0.0017213
time: 1.05 seconds

## BFS IS instance: IS_A2_A2_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0027883, 0.0050203, 0.0026790, 0.0050187, -0.0011860, 0.0013709
1: 0.0017251, 0.0020476, 0.0017093, 0.0020474, -0.0001713, 0.0001981
2: 0.0115843, 0.0128183, 0.0115851, 0.0128787, -0.0007580, 0.0006557
3: -0.0026995, -0.0014232, -0.0026986, -0.0013607, -0.0007839, 0.0006782
4: -0.0024962, -0.0011146, -0.0025639, -0.0011156, -0.0007341, 0.0008486
5: 0.0051682, 0.0064756, 0.0051691, 0.0065397, -0.0008031, 0.0006947
6: -0.0017946, 0.0033931, -0.0017909, 0.0036472, -0.0031864, 0.0027565
7: -0.0071778, -0.0001126, -0.0075238, -0.0001177, -0.0037542, 0.0043397
8: 0.9841577, 0.9891346, 0.9839139, 0.9891310, -0.0026445, 0.0030569
9: -0.0060243, -0.0015067, -0.0060211, -0.0012854, -0.0027749, 0.0024005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 159

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A2_A2_A1_B2_B1

### Relational analysis result of IS_A2_A2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016897, upper bound: 0.0017207
time: 0.81 seconds

## Relational analysis of IS_A2_A2_A2_A1_B2_B2

### Relational analysis result of IS_A2_A2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016897, upper bound: 0.0017213
time: 1.08 seconds

## BFS IS instance: IS_A2_A2_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0026370, 0.0049806, 0.0025814, 0.0049059, -0.0012352, 0.0016308
1: 0.0017033, 0.0020419, 0.0016952, 0.0020311, -0.0001784, 0.0002356
2: 0.0116062, 0.0129019, 0.0116475, 0.0129327, -0.0009016, 0.0006829
3: -0.0026768, -0.0013367, -0.0026340, -0.0013049, -0.0009325, 0.0007063
4: -0.0025899, -0.0011392, -0.0026243, -0.0011855, -0.0007646, 0.0010095
5: 0.0051914, 0.0065643, 0.0052352, 0.0065969, -0.0009553, 0.0007236
6: -0.0017024, 0.0037448, -0.0015286, 0.0038741, -0.0037905, 0.0028709
7: -0.0076568, -0.0002382, -0.0078329, -0.0004750, -0.0039099, 0.0051624
8: 0.9838202, 0.9890460, 0.9836962, 0.9888793, -0.0027542, 0.0036365
9: -0.0059440, -0.0012004, -0.0057927, -0.0010878, -0.0033010, 0.0025001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A2_A2_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017657, upper bound: 0.0016840
time: 0.87 seconds

## Relational analysis of IS_A2_A2_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017657, upper bound: 0.0017113
time: 0.98 seconds

## BFS IS instance: IS_A2_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0026370, 0.0049806, 0.0026039, 0.0050199, -0.0012166, 0.0014531
1: 0.0017033, 0.0020419, 0.0016985, 0.0020475, -0.0001758, 0.0002099
2: 0.0116062, 0.0129019, 0.0115845, 0.0129202, -0.0008034, 0.0006726
3: -0.0026768, -0.0013367, -0.0026992, -0.0013178, -0.0008309, 0.0006957
4: -0.0025899, -0.0011392, -0.0026104, -0.0011149, -0.0007531, 0.0008995
5: 0.0051914, 0.0065643, 0.0051684, 0.0065837, -0.0008512, 0.0007127
6: -0.0017024, 0.0037448, -0.0017937, 0.0038218, -0.0033774, 0.0028278
7: -0.0076568, -0.0002382, -0.0077616, -0.0001139, -0.0038512, 0.0045997
8: 0.9838202, 0.9890460, 0.9837465, 0.9891336, -0.0027129, 0.0032401
9: -0.0059440, -0.0012004, -0.0060235, -0.0011334, -0.0029412, 0.0024626

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A2_A2_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017657, upper bound: 0.0016840
time: 0.82 seconds

## Relational analysis of IS_A2_A2_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017657, upper bound: 0.0017113
time: 0.96 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.31 seconds
IS_A1_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0016828, upper bound: 0.0017207
IS_A1_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0016828, upper bound: 0.0017207
IS_A1_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0016828, upper bound: 0.0017427
IS_A1_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0016828, upper bound: 0.0017427
IS_A1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0017425, upper bound: 0.0016663
IS_A1_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0017425, upper bound: 0.0016900
IS_A1_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0017425, upper bound: 0.0016729
IS_A1_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0017425, upper bound: 0.0016975
IS_A1_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0016828, upper bound: 0.0017207
IS_A1_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0016828, upper bound: 0.0017207
IS_A1_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0016828, upper bound: 0.0017474
IS_A1_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0016828, upper bound: 0.0017474
IS_A1_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0017425, upper bound: 0.0016717
IS_A1_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0017425, upper bound: 0.0016968
IS_A1_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0017425, upper bound: 0.0016778
IS_A1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0017425, upper bound: 0.0017047
IS_A1_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0016828, upper bound: 0.0017367
IS_A1_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0016828, upper bound: 0.0017215
IS_A1_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0016828, upper bound: 0.0017642
IS_A1_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0016828, upper bound: 0.0017436
IS_A1_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0017427, upper bound: 0.0016775
IS_A1_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0017427, upper bound: 0.0017024
IS_A1_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0017427, upper bound: 0.0016852
IS_A1_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0017427, upper bound: 0.0017113
IS_A1_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0016828, upper bound: 0.0017395
IS_A1_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0016828, upper bound: 0.0017213
IS_A1_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0016828, upper bound: 0.0017653
IS_A1_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0016828, upper bound: 0.0017485
IS_A1_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0017427, upper bound: 0.0016840
IS_A1_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0017427, upper bound: 0.0017113
IS_A1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0017427, upper bound: 0.0016897
IS_A1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0017427, upper bound: 0.0017183
IS_A2_A1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0016778, upper bound: 0.0017207
IS_A2_A1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0016778, upper bound: 0.0017367
IS_A2_A1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0016778, upper bound: 0.0017207
IS_A2_A1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0016778, upper bound: 0.0017367
IS_A2_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0016774
IS_A2_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0017018
IS_A2_A1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0016774
IS_A2_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0017018
IS_A2_A1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0016778, upper bound: 0.0017207
IS_A2_A1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0016778, upper bound: 0.0017395
IS_A2_A1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0016778, upper bound: 0.0017207
IS_A2_A1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0016778, upper bound: 0.0017395
IS_A2_A1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0016839
IS_A2_A1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0017107
IS_A2_A1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0016840
IS_A2_A1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0017107
IS_A2_A2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0016897, upper bound: 0.0017207
IS_A2_A2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0016897, upper bound: 0.0017215
IS_A2_A2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0016897, upper bound: 0.0017207
IS_A2_A2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0016897, upper bound: 0.0017215
IS_A2_A2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0017657, upper bound: 0.0016775
IS_A2_A2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0017657, upper bound: 0.0017024
IS_A2_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0017657, upper bound: 0.0016775
IS_A2_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0017657, upper bound: 0.0017024
IS_A2_A2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0016897, upper bound: 0.0017207
IS_A2_A2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0016897, upper bound: 0.0017213
IS_A2_A2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0016897, upper bound: 0.0017207
IS_A2_A2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0016897, upper bound: 0.0017213
IS_A2_A2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0017657, upper bound: 0.0016840
IS_A2_A2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0017657, upper bound: 0.0017113
IS_A2_A2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0017657, upper bound: 0.0016840
IS_A2_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 8, lower bound: -0.0017657, upper bound: 0.0017113

## BFS IS instance: IS_A1_B1_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0028311, 0.0049298, 0.0027298, 0.0049144, -0.0010304, 0.0010974
1: 0.0017313, 0.0020345, 0.0017167, 0.0020323, -0.0001489, 0.0001585
2: 0.0116343, 0.0127946, 0.0116428, 0.0128506, -0.0006067, 0.0005697
3: -0.0026477, -0.0014477, -0.0026389, -0.0013897, -0.0006275, 0.0005892
4: -0.0024698, -0.0011706, -0.0025325, -0.0011802, -0.0006378, 0.0006793
5: 0.0052212, 0.0064506, 0.0052302, 0.0065099, -0.0006428, 0.0006036
6: -0.0015843, 0.0032937, -0.0015485, 0.0035292, -0.0025506, 0.0023950
7: -0.0070425, -0.0003991, -0.0073632, -0.0004478, -0.0032617, 0.0034737
8: 0.9842530, 0.9889327, 0.9840271, 0.9888984, -0.0022976, 0.0024470
9: -0.0058412, -0.0015932, -0.0058100, -0.0013881, -0.0022212, 0.0020856

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015181, upper bound: 0.0015589
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014522, upper bound: 0.0015757
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0028054, 0.0049364, 0.0027298, 0.0049144, -0.0011426, 0.0011956
1: 0.0017276, 0.0020355, 0.0017167, 0.0020323, -0.0001651, 0.0001727
2: 0.0116307, 0.0128089, 0.0116428, 0.0128506, -0.0006610, 0.0006317
3: -0.0026515, -0.0014329, -0.0026389, -0.0013897, -0.0006837, 0.0006533
4: -0.0024857, -0.0011666, -0.0025325, -0.0011802, -0.0007073, 0.0007401
5: 0.0052173, 0.0064657, 0.0052302, 0.0065099, -0.0007004, 0.0006693
6: -0.0015995, 0.0033536, -0.0015485, 0.0035292, -0.0027789, 0.0026556
7: -0.0071240, -0.0003783, -0.0073632, -0.0004478, -0.0036167, 0.0037846
8: 0.9841956, 0.9889474, 0.9840271, 0.9888984, -0.0025477, 0.0026660
9: -0.0058545, -0.0015411, -0.0058100, -0.0013881, -0.0024200, 0.0023126

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014581, upper bound: 0.0016134
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014522, upper bound: 0.0015757
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0028311, 0.0049298, 0.0027599, 0.0050186, -0.0012621, 0.0011881
1: 0.0017313, 0.0020345, 0.0017210, 0.0020473, -0.0001823, 0.0001716
2: 0.0116343, 0.0127946, 0.0115852, 0.0128340, -0.0006569, 0.0006978
3: -0.0026477, -0.0014477, -0.0026985, -0.0014069, -0.0006794, 0.0007217
4: -0.0024698, -0.0011706, -0.0025139, -0.0011157, -0.0007812, 0.0007355
5: 0.0052212, 0.0064506, 0.0051691, 0.0064923, -0.0006960, 0.0007393
6: -0.0015843, 0.0032937, -0.0017907, 0.0034592, -0.0027615, 0.0029334
7: -0.0070425, -0.0003991, -0.0072679, -0.0001179, -0.0039951, 0.0037609
8: 0.9842530, 0.9889327, 0.9840942, 0.9891307, -0.0028142, 0.0026493
9: -0.0058412, -0.0015932, -0.0060210, -0.0014490, -0.0024048, 0.0025545

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014877, upper bound: 0.0015285
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014150, upper bound: 0.0015351
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0028054, 0.0049364, 0.0027599, 0.0050186, -0.0013742, 0.0012863
1: 0.0017276, 0.0020355, 0.0017210, 0.0020473, -0.0001985, 0.0001858
2: 0.0116307, 0.0128089, 0.0115852, 0.0128340, -0.0007112, 0.0007598
3: -0.0026515, -0.0014329, -0.0026985, -0.0014069, -0.0007355, 0.0007858
4: -0.0024857, -0.0011666, -0.0025139, -0.0011157, -0.0008507, 0.0007963
5: 0.0052173, 0.0064657, 0.0051691, 0.0064923, -0.0007535, 0.0008050
6: -0.0015995, 0.0033536, -0.0017907, 0.0034592, -0.0029898, 0.0031941
7: -0.0071240, -0.0003783, -0.0072679, -0.0001179, -0.0043501, 0.0040718
8: 0.9841956, 0.9889474, 0.9840942, 0.9891307, -0.0030643, 0.0028683
9: -0.0058545, -0.0015411, -0.0060210, -0.0014490, -0.0026036, 0.0027816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014877, upper bound: 0.0015285
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014150, upper bound: 0.0015351
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0026286, 0.0048868, 0.0028264, 0.0049708, -0.0013043, 0.0010793
1: 0.0017021, 0.0020283, 0.0017306, 0.0020404, -0.0001884, 0.0001559
2: 0.0116581, 0.0129066, 0.0116116, 0.0127972, -0.0005967, 0.0007211
3: -0.0026232, -0.0013319, -0.0026712, -0.0014449, -0.0006172, 0.0007458
4: -0.0025952, -0.0011972, -0.0024727, -0.0011453, -0.0008074, 0.0006681
5: 0.0052463, 0.0065692, 0.0051971, 0.0064534, -0.0006323, 0.0007641
6: -0.0014844, 0.0037645, -0.0016796, 0.0033048, -0.0025086, 0.0030316
7: -0.0076836, -0.0005351, -0.0070575, -0.0002693, -0.0041287, 0.0034165
8: 0.9838014, 0.9888369, 0.9842424, 0.9890242, -0.0029084, 0.0024067
9: -0.0057542, -0.0011832, -0.0059242, -0.0015836, -0.0021846, 0.0026400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017278, upper bound: 0.0016840
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017278, upper bound: 0.0016840
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0026286, 0.0048868, 0.0026899, 0.0049152, -0.0011182, 0.0010925
1: 0.0017021, 0.0020283, 0.0017109, 0.0020324, -0.0001615, 0.0001578
2: 0.0116581, 0.0129066, 0.0116424, 0.0128727, -0.0006040, 0.0006182
3: -0.0026232, -0.0013319, -0.0026394, -0.0013669, -0.0006247, 0.0006394
4: -0.0025952, -0.0011972, -0.0025572, -0.0011797, -0.0006922, 0.0006763
5: 0.0052463, 0.0065692, 0.0052297, 0.0065333, -0.0006400, 0.0006550
6: -0.0014844, 0.0037645, -0.0015503, 0.0036219, -0.0025393, 0.0025990
7: -0.0076836, -0.0005351, -0.0074895, -0.0004454, -0.0035396, 0.0034584
8: 0.9838014, 0.9888369, 0.9839381, 0.9889001, -0.0024934, 0.0024362
9: -0.0057542, -0.0011832, -0.0058116, -0.0013074, -0.0022114, 0.0022633

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017278, upper bound: 0.0017048
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017278, upper bound: 0.0017048
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0026286, 0.0048868, 0.0028623, 0.0050638, -0.0015644, 0.0011713
1: 0.0017021, 0.0020283, 0.0017358, 0.0020539, -0.0002260, 0.0001692
2: 0.0116581, 0.0129066, 0.0115602, 0.0127774, -0.0006476, 0.0008649
3: -0.0026232, -0.0013319, -0.0027243, -0.0014655, -0.0006698, 0.0008945
4: -0.0025952, -0.0011972, -0.0024505, -0.0010877, -0.0009684, 0.0007251
5: 0.0052463, 0.0065692, 0.0051427, 0.0064323, -0.0006862, 0.0009164
6: -0.0014844, 0.0037645, -0.0018957, 0.0032212, -0.0027225, 0.0036361
7: -0.0076836, -0.0005351, -0.0069437, 0.0000251, -0.0049520, 0.0037078
8: 0.9838014, 0.9888369, 0.9843227, 0.9892315, -0.0034883, 0.0026118
9: -0.0057542, -0.0011832, -0.0061124, -0.0016564, -0.0023708, 0.0031664

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_A1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017238, upper bound: 0.0016729
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_A2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017238, upper bound: 0.0016729
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0026286, 0.0048868, 0.0027117, 0.0050193, -0.0013786, 0.0011869
1: 0.0017021, 0.0020283, 0.0017141, 0.0020474, -0.0001992, 0.0001715
2: 0.0116581, 0.0129066, 0.0115848, 0.0128607, -0.0006562, 0.0007622
3: -0.0026232, -0.0013319, -0.0026989, -0.0013794, -0.0006787, 0.0007883
4: -0.0025952, -0.0011972, -0.0025437, -0.0011153, -0.0008534, 0.0007347
5: 0.0052463, 0.0065692, 0.0051688, 0.0065206, -0.0006953, 0.0008076
6: -0.0014844, 0.0037645, -0.0017922, 0.0035714, -0.0027586, 0.0032042
7: -0.0076836, -0.0005351, -0.0074206, -0.0001159, -0.0043638, 0.0037570
8: 0.9838014, 0.9888369, 0.9839866, 0.9891323, -0.0030740, 0.0026465
9: -0.0057542, -0.0011832, -0.0060223, -0.0013514, -0.0024023, 0.0027903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017238, upper bound: 0.0016975
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_A2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017238, upper bound: 0.0016975
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0027859, 0.0049128, 0.0027315, 0.0048920, -0.0011306, 0.0011048
1: 0.0017248, 0.0020321, 0.0017169, 0.0020291, -0.0001633, 0.0001596
2: 0.0116437, 0.0128196, 0.0116552, 0.0128497, -0.0006108, 0.0006251
3: -0.0026380, -0.0014218, -0.0026261, -0.0013907, -0.0006317, 0.0006465
4: -0.0024977, -0.0011812, -0.0025315, -0.0011940, -0.0006998, 0.0006839
5: 0.0052311, 0.0064771, 0.0052433, 0.0065090, -0.0006472, 0.0006623
6: -0.0015447, 0.0033987, -0.0014964, 0.0035253, -0.0025679, 0.0026278
7: -0.0071855, -0.0004529, -0.0073579, -0.0005187, -0.0035788, 0.0034973
8: 0.9841522, 0.9888949, 0.9840308, 0.9888485, -0.0025210, 0.0024635
9: -0.0058068, -0.0015017, -0.0057647, -0.0013915, -0.0022362, 0.0022884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015140, upper bound: 0.0015458
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014456, upper bound: 0.0015633
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0027600, 0.0049193, 0.0027315, 0.0048920, -0.0012362, 0.0012025
1: 0.0017210, 0.0020330, 0.0017169, 0.0020291, -0.0001786, 0.0001737
2: 0.0116401, 0.0128339, 0.0116552, 0.0128497, -0.0006648, 0.0006835
3: -0.0026417, -0.0014070, -0.0026261, -0.0013907, -0.0006876, 0.0007069
4: -0.0025138, -0.0011772, -0.0025315, -0.0011940, -0.0007653, 0.0007444
5: 0.0052273, 0.0064922, 0.0052433, 0.0065090, -0.0007044, 0.0007242
6: -0.0015598, 0.0034589, -0.0014964, 0.0035253, -0.0027950, 0.0028734
7: -0.0072675, -0.0004324, -0.0073579, -0.0005187, -0.0039133, 0.0038065
8: 0.9840945, 0.9889092, 0.9840308, 0.9888485, -0.0027566, 0.0026814
9: -0.0058198, -0.0014493, -0.0057647, -0.0013915, -0.0024340, 0.0025023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015140, upper bound: 0.0015458
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014456, upper bound: 0.0015633
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0027859, 0.0049128, 0.0027616, 0.0049991, -0.0013528, 0.0011956
1: 0.0017248, 0.0020321, 0.0017213, 0.0020445, -0.0001954, 0.0001727
2: 0.0116437, 0.0128196, 0.0115960, 0.0128331, -0.0006610, 0.0007479
3: -0.0026380, -0.0014218, -0.0026873, -0.0014079, -0.0006837, 0.0007735
4: -0.0024977, -0.0011812, -0.0025128, -0.0011278, -0.0008374, 0.0007401
5: 0.0052311, 0.0064771, 0.0051806, 0.0064913, -0.0007004, 0.0007925
6: -0.0015447, 0.0033987, -0.0017453, 0.0034553, -0.0027789, 0.0031443
7: -0.0071855, -0.0004529, -0.0072626, -0.0001798, -0.0042823, 0.0037847
8: 0.9841522, 0.9888949, 0.9840979, 0.9890872, -0.0030165, 0.0026660
9: -0.0058068, -0.0015017, -0.0059814, -0.0014525, -0.0024200, 0.0027382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014811, upper bound: 0.0015127
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014098, upper bound: 0.0015225
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0027600, 0.0049193, 0.0027616, 0.0049991, -0.0014585, 0.0012933
1: 0.0017210, 0.0020330, 0.0017213, 0.0020445, -0.0002107, 0.0001868
2: 0.0116401, 0.0128339, 0.0115960, 0.0128331, -0.0007150, 0.0008064
3: -0.0026417, -0.0014070, -0.0026873, -0.0014079, -0.0007395, 0.0008340
4: -0.0025138, -0.0011772, -0.0025128, -0.0011278, -0.0009028, 0.0008006
5: 0.0052273, 0.0064922, 0.0051806, 0.0064913, -0.0007576, 0.0008544
6: -0.0015598, 0.0034589, -0.0017453, 0.0034553, -0.0030060, 0.0033899
7: -0.0072675, -0.0004324, -0.0072626, -0.0001798, -0.0046168, 0.0040940
8: 0.9840945, 0.9889092, 0.9840979, 0.9890872, -0.0032521, 0.0028839
9: -0.0058198, -0.0014493, -0.0059814, -0.0014525, -0.0026178, 0.0029521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014811, upper bound: 0.0015127
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014098, upper bound: 0.0015225
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0025833, 0.0048678, 0.0028279, 0.0049515, -0.0013970, 0.0010852
1: 0.0016955, 0.0020256, 0.0017308, 0.0020376, -0.0002018, 0.0001568
2: 0.0116686, 0.0129316, 0.0116223, 0.0127964, -0.0006000, 0.0007724
3: -0.0026123, -0.0013060, -0.0026601, -0.0014458, -0.0006206, 0.0007988
4: -0.0026232, -0.0012090, -0.0024718, -0.0011572, -0.0008648, 0.0006718
5: 0.0052575, 0.0065958, 0.0052085, 0.0064525, -0.0006357, 0.0008184
6: -0.0014402, 0.0038698, -0.0016346, 0.0033012, -0.0025224, 0.0032470
7: -0.0078270, -0.0005953, -0.0070527, -0.0003305, -0.0044221, 0.0034353
8: 0.9837004, 0.9887945, 0.9842458, 0.9889810, -0.0031151, 0.0024199
9: -0.0057157, -0.0010916, -0.0058850, -0.0015866, -0.0021966, 0.0028276

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017278, upper bound: 0.0016890
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017278, upper bound: 0.0016890
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0025833, 0.0048678, 0.0026915, 0.0048928, -0.0012344, 0.0010989
1: 0.0016955, 0.0020256, 0.0017112, 0.0020292, -0.0001783, 0.0001588
2: 0.0116686, 0.0129316, 0.0116548, 0.0128718, -0.0006075, 0.0006825
3: -0.0026123, -0.0013060, -0.0026266, -0.0013679, -0.0006283, 0.0007058
4: -0.0026232, -0.0012090, -0.0025562, -0.0011936, -0.0007641, 0.0006802
5: 0.0052575, 0.0065958, 0.0052429, 0.0065323, -0.0006437, 0.0007231
6: -0.0014402, 0.0038698, -0.0014982, 0.0036181, -0.0025541, 0.0028691
7: -0.0078270, -0.0005953, -0.0074843, -0.0005163, -0.0039074, 0.0034784
8: 0.9837004, 0.9887945, 0.9839419, 0.9888502, -0.0027525, 0.0024503
9: -0.0057157, -0.0010916, -0.0057662, -0.0013107, -0.0022242, 0.0024985

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017278, upper bound: 0.0017115
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017278, upper bound: 0.0017115
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0025833, 0.0048678, 0.0028639, 0.0050437, -0.0016414, 0.0011772
1: 0.0016955, 0.0020256, 0.0017360, 0.0020510, -0.0002371, 0.0001701
2: 0.0116686, 0.0129316, 0.0115713, 0.0127765, -0.0006509, 0.0009075
3: -0.0026123, -0.0013060, -0.0027129, -0.0014664, -0.0006732, 0.0009386
4: -0.0026232, -0.0012090, -0.0024495, -0.0011001, -0.0010160, 0.0007287
5: 0.0052575, 0.0065958, 0.0051544, 0.0064314, -0.0006896, 0.0009615
6: -0.0014402, 0.0038698, -0.0018490, 0.0032176, -0.0027362, 0.0038150
7: -0.0078270, -0.0005953, -0.0069388, -0.0000385, -0.0051957, 0.0037265
8: 0.9837004, 0.9887945, 0.9843261, 0.9891867, -0.0036600, 0.0026250
9: -0.0057157, -0.0010916, -0.0060717, -0.0016595, -0.0023828, 0.0033223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_A1

### Relational analysis result of IS_A1_B1_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017238, upper bound: 0.0016778
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_A2

### Relational analysis result of IS_A1_B1_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017238, upper bound: 0.0016778
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0025833, 0.0048678, 0.0027133, 0.0049997, -0.0014788, 0.0011933
1: 0.0016955, 0.0020256, 0.0017143, 0.0020446, -0.0002136, 0.0001724
2: 0.0116686, 0.0129316, 0.0115956, 0.0128598, -0.0006597, 0.0008176
3: -0.0026123, -0.0013060, -0.0026877, -0.0013803, -0.0006823, 0.0008456
4: -0.0026232, -0.0012090, -0.0025427, -0.0011274, -0.0009154, 0.0007386
5: 0.0052575, 0.0065958, 0.0051802, 0.0065196, -0.0006990, 0.0008663
6: -0.0014402, 0.0038698, -0.0017468, 0.0035676, -0.0027735, 0.0034371
7: -0.0078270, -0.0005953, -0.0074154, -0.0001778, -0.0046811, 0.0037772
8: 0.9837004, 0.9887945, 0.9839904, 0.9890886, -0.0032975, 0.0026608
9: -0.0057157, -0.0010916, -0.0059827, -0.0013547, -0.0024153, 0.0029932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_A1

### Relational analysis result of IS_A1_B1_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017238, upper bound: 0.0017047
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_A2

### Relational analysis result of IS_A1_B1_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017238, upper bound: 0.0017047
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0028311, 0.0049298, 0.0027024, 0.0049240, -0.0011185, 0.0012044
1: 0.0017313, 0.0020345, 0.0017127, 0.0020337, -0.0001616, 0.0001740
2: 0.0116343, 0.0127946, 0.0116375, 0.0128658, -0.0006659, 0.0006184
3: -0.0026477, -0.0014477, -0.0026444, -0.0013740, -0.0006887, 0.0006396
4: -0.0024698, -0.0011706, -0.0025495, -0.0011743, -0.0006924, 0.0007455
5: 0.0052212, 0.0064506, 0.0052246, 0.0065260, -0.0007055, 0.0006552
6: -0.0015843, 0.0032937, -0.0015706, 0.0035930, -0.0027992, 0.0025997
7: -0.0070425, -0.0003991, -0.0074500, -0.0004176, -0.0035406, 0.0038123
8: 0.9842530, 0.9889327, 0.9839659, 0.9889197, -0.0024941, 0.0026855
9: -0.0058412, -0.0015932, -0.0058293, -0.0013326, -0.0024377, 0.0022640

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015181, upper bound: 0.0015710
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014522, upper bound: 0.0015934
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0028054, 0.0049364, 0.0027024, 0.0049240, -0.0010464, 0.0011119
1: 0.0017276, 0.0020355, 0.0017127, 0.0020337, -0.0001512, 0.0001606
2: 0.0116307, 0.0128089, 0.0116375, 0.0128658, -0.0006147, 0.0005785
3: -0.0026515, -0.0014329, -0.0026444, -0.0013740, -0.0006358, 0.0005984
4: -0.0024857, -0.0011666, -0.0025495, -0.0011743, -0.0006478, 0.0006883
5: 0.0052173, 0.0064657, 0.0052246, 0.0065260, -0.0006513, 0.0006130
6: -0.0015995, 0.0033536, -0.0015706, 0.0035930, -0.0025844, 0.0024322
7: -0.0071240, -0.0003783, -0.0074500, -0.0004176, -0.0033124, 0.0035197
8: 0.9841956, 0.9889474, 0.9839659, 0.9889197, -0.0023334, 0.0024793
9: -0.0058545, -0.0015411, -0.0058293, -0.0013326, -0.0022506, 0.0021181

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015181, upper bound: 0.0015583
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014522, upper bound: 0.0015757
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0028311, 0.0049298, 0.0027259, 0.0050380, -0.0013441, 0.0012832
1: 0.0017313, 0.0020345, 0.0017161, 0.0020501, -0.0001942, 0.0001854
2: 0.0116343, 0.0127946, 0.0115745, 0.0128528, -0.0007095, 0.0007431
3: -0.0026477, -0.0014477, -0.0027096, -0.0013875, -0.0007337, 0.0007686
4: -0.0024698, -0.0011706, -0.0025349, -0.0011037, -0.0008320, 0.0007943
5: 0.0052212, 0.0064506, 0.0051578, 0.0065122, -0.0007517, 0.0007874
6: -0.0015843, 0.0032937, -0.0018356, 0.0035383, -0.0029825, 0.0031240
7: -0.0070425, -0.0003991, -0.0073756, -0.0000568, -0.0042546, 0.0040619
8: 0.9842530, 0.9889327, 0.9840184, 0.9891739, -0.0029970, 0.0028613
9: -0.0058412, -0.0015932, -0.0060601, -0.0013802, -0.0025973, 0.0027205

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014877, upper bound: 0.0015475
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014150, upper bound: 0.0015562
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0028054, 0.0049364, 0.0027259, 0.0050380, -0.0012808, 0.0012032
1: 0.0017276, 0.0020355, 0.0017161, 0.0020501, -0.0001850, 0.0001738
2: 0.0116307, 0.0128089, 0.0115745, 0.0128528, -0.0006652, 0.0007081
3: -0.0026515, -0.0014329, -0.0027096, -0.0013875, -0.0006880, 0.0007324
4: -0.0024857, -0.0011666, -0.0025349, -0.0011037, -0.0007929, 0.0007448
5: 0.0052173, 0.0064657, 0.0051578, 0.0065122, -0.0007048, 0.0007503
6: -0.0015995, 0.0033536, -0.0018356, 0.0035383, -0.0027965, 0.0029770
7: -0.0071240, -0.0003783, -0.0073756, -0.0000568, -0.0040544, 0.0038086
8: 0.9841956, 0.9889474, 0.9840184, 0.9891739, -0.0028560, 0.0026829
9: -0.0058545, -0.0015411, -0.0060601, -0.0013802, -0.0024353, 0.0025925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014877, upper bound: 0.0015320
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014150, upper bound: 0.0015384
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0026132, 0.0048872, 0.0028005, 0.0049763, -0.0014477, 0.0010121
1: 0.0016998, 0.0020284, 0.0017269, 0.0020412, -0.0002092, 0.0001462
2: 0.0116579, 0.0129151, 0.0116086, 0.0128115, -0.0005596, 0.0008004
3: -0.0026234, -0.0013231, -0.0026743, -0.0014302, -0.0005787, 0.0008278
4: -0.0026046, -0.0011970, -0.0024887, -0.0011419, -0.0008962, 0.0006265
5: 0.0052461, 0.0065782, 0.0051940, 0.0064685, -0.0005929, 0.0008481
6: -0.0014852, 0.0038001, -0.0016922, 0.0033648, -0.0023524, 0.0033649
7: -0.0077322, -0.0005340, -0.0071393, -0.0002521, -0.0045826, 0.0032037
8: 0.9837672, 0.9888377, 0.9841847, 0.9890364, -0.0032281, 0.0022568
9: -0.0057549, -0.0011522, -0.0059352, -0.0015313, -0.0020485, 0.0029303

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_A2_B1_B1_A1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017278, upper bound: 0.0016908
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B1_A2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017278, upper bound: 0.0016821
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0026132, 0.0048872, 0.0026573, 0.0049247, -0.0012678, 0.0010105
1: 0.0016998, 0.0020284, 0.0017062, 0.0020338, -0.0001832, 0.0001460
2: 0.0116579, 0.0129151, 0.0116371, 0.0128907, -0.0005587, 0.0007009
3: -0.0026234, -0.0013231, -0.0026448, -0.0013483, -0.0005778, 0.0007250
4: -0.0026046, -0.0011970, -0.0025774, -0.0011738, -0.0007848, 0.0006255
5: 0.0052461, 0.0065782, 0.0052242, 0.0065524, -0.0005919, 0.0007427
6: -0.0014852, 0.0038001, -0.0015724, 0.0036978, -0.0023486, 0.0029468
7: -0.0077322, -0.0005340, -0.0075928, -0.0004153, -0.0040133, 0.0031986
8: 0.9837672, 0.9888377, 0.9838653, 0.9889213, -0.0028270, 0.0022531
9: -0.0057549, -0.0011522, -0.0058308, -0.0012413, -0.0020453, 0.0025662

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017278, upper bound: 0.0017123
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B2_A2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017278, upper bound: 0.0017042
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0026132, 0.0048872, 0.0028302, 0.0050791, -0.0016873, 0.0011036
1: 0.0016998, 0.0020284, 0.0017312, 0.0020561, -0.0002438, 0.0001594
2: 0.0116579, 0.0129151, 0.0115518, 0.0127951, -0.0006101, 0.0009328
3: -0.0026234, -0.0013231, -0.0027331, -0.0014472, -0.0006310, 0.0009648
4: -0.0026046, -0.0011970, -0.0024703, -0.0010782, -0.0010444, 0.0006831
5: 0.0052461, 0.0065782, 0.0051337, 0.0064511, -0.0006465, 0.0009884
6: -0.0014852, 0.0038001, -0.0019312, 0.0032957, -0.0025650, 0.0039216
7: -0.0077322, -0.0005340, -0.0070452, 0.0000735, -0.0053409, 0.0034933
8: 0.9837672, 0.9888377, 0.9842511, 0.9892656, -0.0037623, 0.0024607
9: -0.0057549, -0.0011522, -0.0061433, -0.0015915, -0.0022337, 0.0034151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017238, upper bound: 0.0016850
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017238, upper bound: 0.0016728
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0026132, 0.0048872, 0.0026773, 0.0050386, -0.0015077, 0.0011043
1: 0.0016998, 0.0020284, 0.0017091, 0.0020502, -0.0002178, 0.0001595
2: 0.0116579, 0.0129151, 0.0115742, 0.0128796, -0.0006106, 0.0008336
3: -0.0026234, -0.0013231, -0.0027099, -0.0013597, -0.0006315, 0.0008621
4: -0.0026046, -0.0011970, -0.0025650, -0.0011033, -0.0009333, 0.0006836
5: 0.0052461, 0.0065782, 0.0051574, 0.0065407, -0.0006469, 0.0008832
6: -0.0014852, 0.0038001, -0.0018371, 0.0036512, -0.0025668, 0.0035043
7: -0.0077322, -0.0005340, -0.0075293, -0.0000548, -0.0047725, 0.0034958
8: 0.9837672, 0.9888377, 0.9839100, 0.9891753, -0.0033618, 0.0024625
9: -0.0057549, -0.0011522, -0.0060613, -0.0012819, -0.0022353, 0.0030517

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017238, upper bound: 0.0017105
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_A2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017238, upper bound: 0.0016981
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0027859, 0.0049128, 0.0027040, 0.0049029, -0.0012193, 0.0012118
1: 0.0017248, 0.0020321, 0.0017130, 0.0020306, -0.0001761, 0.0001751
2: 0.0116437, 0.0128196, 0.0116492, 0.0128649, -0.0006700, 0.0006741
3: -0.0026380, -0.0014218, -0.0026324, -0.0013750, -0.0006929, 0.0006972
4: -0.0024977, -0.0011812, -0.0025485, -0.0011873, -0.0007547, 0.0007501
5: 0.0052311, 0.0064771, 0.0052369, 0.0065250, -0.0007099, 0.0007142
6: -0.0015447, 0.0033987, -0.0015218, 0.0035891, -0.0028165, 0.0028339
7: -0.0071855, -0.0004529, -0.0074448, -0.0004842, -0.0038595, 0.0038358
8: 0.9841522, 0.9888949, 0.9839696, 0.9888728, -0.0027187, 0.0027020
9: -0.0058068, -0.0015017, -0.0057868, -0.0013359, -0.0024527, 0.0024679

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015140, upper bound: 0.0015596
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014456, upper bound: 0.0015793
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0027600, 0.0049193, 0.0027040, 0.0049029, -0.0011461, 0.0011188
1: 0.0017210, 0.0020330, 0.0017130, 0.0020306, -0.0001656, 0.0001616
2: 0.0116401, 0.0128339, 0.0116492, 0.0128649, -0.0006185, 0.0006336
3: -0.0026417, -0.0014070, -0.0026324, -0.0013750, -0.0006397, 0.0006553
4: -0.0025138, -0.0011772, -0.0025485, -0.0011873, -0.0007094, 0.0006925
5: 0.0052273, 0.0064922, 0.0052369, 0.0065250, -0.0006554, 0.0006714
6: -0.0015598, 0.0034589, -0.0015218, 0.0035891, -0.0026003, 0.0026638
7: -0.0072675, -0.0004324, -0.0074448, -0.0004842, -0.0036278, 0.0035414
8: 0.9840945, 0.9889092, 0.9839696, 0.9888728, -0.0025555, 0.0024946
9: -0.0058198, -0.0014493, -0.0057868, -0.0013359, -0.0022645, 0.0023197

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015140, upper bound: 0.0015458
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014456, upper bound: 0.0015633
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0027859, 0.0049128, 0.0027276, 0.0050171, -0.0014326, 0.0012906
1: 0.0017248, 0.0020321, 0.0017164, 0.0020471, -0.0002070, 0.0001865
2: 0.0116437, 0.0128196, 0.0115860, 0.0128518, -0.0007136, 0.0007920
3: -0.0026380, -0.0014218, -0.0026977, -0.0013885, -0.0007380, 0.0008192
4: -0.0024977, -0.0011812, -0.0025338, -0.0011166, -0.0008868, 0.0007989
5: 0.0052311, 0.0064771, 0.0051700, 0.0065112, -0.0007561, 0.0008392
6: -0.0015447, 0.0033987, -0.0017872, 0.0035343, -0.0029998, 0.0033297
7: -0.0071855, -0.0004529, -0.0073701, -0.0001227, -0.0045348, 0.0040855
8: 0.9841522, 0.9888949, 0.9840223, 0.9891275, -0.0031944, 0.0028779
9: -0.0058068, -0.0015017, -0.0060179, -0.0013837, -0.0026124, 0.0028997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014811, upper bound: 0.0015343
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014098, upper bound: 0.0015426
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0027600, 0.0049193, 0.0027276, 0.0050171, -0.0013705, 0.0012100
1: 0.0017210, 0.0020330, 0.0017164, 0.0020471, -0.0001980, 0.0001748
2: 0.0116401, 0.0128339, 0.0115860, 0.0128518, -0.0006690, 0.0007577
3: -0.0026417, -0.0014070, -0.0026977, -0.0013885, -0.0006919, 0.0007837
4: -0.0025138, -0.0011772, -0.0025338, -0.0011166, -0.0008484, 0.0007490
5: 0.0052273, 0.0064922, 0.0051700, 0.0065112, -0.0007088, 0.0008029
6: -0.0015598, 0.0034589, -0.0017872, 0.0035343, -0.0028124, 0.0031855
7: -0.0072675, -0.0004324, -0.0073701, -0.0001227, -0.0043383, 0.0038303
8: 0.9840945, 0.9889092, 0.9840223, 0.9891275, -0.0030560, 0.0026981
9: -0.0058198, -0.0014493, -0.0060179, -0.0013837, -0.0024492, 0.0027741

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014811, upper bound: 0.0015147
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014098, upper bound: 0.0015244
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0025681, 0.0048682, 0.0028021, 0.0049567, -0.0015307, 0.0010107
1: 0.0016933, 0.0020256, 0.0017271, 0.0020384, -0.0002211, 0.0001460
2: 0.0116684, 0.0129400, 0.0116194, 0.0128106, -0.0005588, 0.0008463
3: -0.0026125, -0.0012973, -0.0026631, -0.0014311, -0.0005779, 0.0008753
4: -0.0026326, -0.0012088, -0.0024877, -0.0011540, -0.0009475, 0.0006256
5: 0.0052573, 0.0066047, 0.0052054, 0.0064676, -0.0005920, 0.0008967
6: -0.0014410, 0.0039050, -0.0016467, 0.0033611, -0.0023491, 0.0035577
7: -0.0078750, -0.0005942, -0.0071342, -0.0003140, -0.0048453, 0.0031992
8: 0.9836666, 0.9887953, 0.9841884, 0.9889927, -0.0034132, 0.0022536
9: -0.0057164, -0.0010609, -0.0058955, -0.0015345, -0.0020457, 0.0030982

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B2_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017278, upper bound: 0.0016982
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B2_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017278, upper bound: 0.0016867
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0025681, 0.0048682, 0.0026589, 0.0049037, -0.0013697, 0.0010159
1: 0.0016933, 0.0020256, 0.0017064, 0.0020307, -0.0001979, 0.0001468
2: 0.0116684, 0.0129400, 0.0116488, 0.0128898, -0.0005617, 0.0007573
3: -0.0026125, -0.0012973, -0.0026328, -0.0013492, -0.0005809, 0.0007832
4: -0.0026326, -0.0012088, -0.0025764, -0.0011868, -0.0008478, 0.0006289
5: 0.0052573, 0.0066047, 0.0052365, 0.0065515, -0.0005951, 0.0008024
6: -0.0014410, 0.0039050, -0.0015235, 0.0036941, -0.0023613, 0.0031835
7: -0.0078750, -0.0005942, -0.0075877, -0.0004819, -0.0043356, 0.0032159
8: 0.9836666, 0.9887953, 0.9838690, 0.9888744, -0.0030541, 0.0022654
9: -0.0057164, -0.0010609, -0.0057882, -0.0012446, -0.0020563, 0.0027723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B2_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017278, upper bound: 0.0017222
time: 1.05 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B2_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017278, upper bound: 0.0017108
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0025681, 0.0048682, 0.0028318, 0.0050594, -0.0017609, 0.0011022
1: 0.0016933, 0.0020256, 0.0017314, 0.0020532, -0.0002544, 0.0001592
2: 0.0116684, 0.0129400, 0.0115626, 0.0127942, -0.0006094, 0.0009735
3: -0.0026125, -0.0012973, -0.0027218, -0.0014481, -0.0006302, 0.0010069
4: -0.0026326, -0.0012088, -0.0024694, -0.0010904, -0.0010900, 0.0006823
5: 0.0052573, 0.0066047, 0.0051452, 0.0064502, -0.0006457, 0.0010315
6: -0.0014410, 0.0039050, -0.0018855, 0.0032921, -0.0025618, 0.0040927
7: -0.0078750, -0.0005942, -0.0070403, 0.0000112, -0.0055739, 0.0034889
8: 0.9836666, 0.9887953, 0.9842545, 0.9892218, -0.0039264, 0.0024577
9: -0.0057164, -0.0010609, -0.0061035, -0.0015946, -0.0022309, 0.0035641

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_A2_B2_B1_A1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017238, upper bound: 0.0016897
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B1_A2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017238, upper bound: 0.0016768
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0025681, 0.0048682, 0.0026790, 0.0050178, -0.0015999, 0.0011098
1: 0.0016933, 0.0020256, 0.0017093, 0.0020472, -0.0002311, 0.0001603
2: 0.0116684, 0.0129400, 0.0115857, 0.0128787, -0.0006136, 0.0008845
3: -0.0026125, -0.0012973, -0.0026980, -0.0013607, -0.0006346, 0.0009148
4: -0.0026326, -0.0012088, -0.0025639, -0.0011162, -0.0009903, 0.0006870
5: 0.0052573, 0.0066047, 0.0051697, 0.0065397, -0.0006501, 0.0009372
6: -0.0014410, 0.0039050, -0.0017886, 0.0036472, -0.0025795, 0.0037185
7: -0.0078750, -0.0005942, -0.0075239, -0.0001207, -0.0050643, 0.0035130
8: 0.9836666, 0.9887953, 0.9839139, 0.9891288, -0.0035674, 0.0024746
9: -0.0057164, -0.0010609, -0.0060191, -0.0012854, -0.0022463, 0.0032382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017238, upper bound: 0.0017174
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017238, upper bound: 0.0017049
time: 1.07 seconds

## BFS IS instance: IS_A2_A1_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0028669, 0.0050237, 0.0027298, 0.0049144, -0.0011226, 0.0013606
1: 0.0017365, 0.0020481, 0.0017167, 0.0020323, -0.0001622, 0.0001966
2: 0.0115824, 0.0127748, 0.0116428, 0.0128506, -0.0007522, 0.0006206
3: -0.0027014, -0.0014681, -0.0026389, -0.0013897, -0.0007780, 0.0006419
4: -0.0024476, -0.0011125, -0.0025325, -0.0011802, -0.0006949, 0.0008422
5: 0.0051662, 0.0064296, 0.0052302, 0.0065099, -0.0007970, 0.0006576
6: -0.0018025, 0.0032105, -0.0015485, 0.0035292, -0.0031624, 0.0026092
7: -0.0069291, -0.0001019, -0.0073632, -0.0004478, -0.0035534, 0.0043070
8: 0.9843329, 0.9891422, 0.9840271, 0.9888984, -0.0025031, 0.0030339
9: -0.0060312, -0.0016657, -0.0058100, -0.0013881, -0.0027540, 0.0022722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_A1_A1_A1_B1_B1_A1

### Relational analysis result of IS_A2_A1_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013992, upper bound: 0.0016005
time: 0.90 seconds

## Relational analysis of IS_A2_A1_A1_A1_B1_B1_A2

### Relational analysis result of IS_A2_A1_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013866, upper bound: 0.0015357
time: 0.94 seconds

## BFS IS instance: IS_A2_A1_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0028669, 0.0050237, 0.0027024, 0.0049240, -0.0012107, 0.0014676
1: 0.0017365, 0.0020481, 0.0017127, 0.0020337, -0.0001749, 0.0002120
2: 0.0115824, 0.0127748, 0.0116375, 0.0128658, -0.0008114, 0.0006694
3: -0.0027014, -0.0014681, -0.0026444, -0.0013740, -0.0008392, 0.0006923
4: -0.0024476, -0.0011125, -0.0025495, -0.0011743, -0.0007494, 0.0009085
5: 0.0051662, 0.0064296, 0.0052246, 0.0065260, -0.0008597, 0.0007092
6: -0.0018025, 0.0032105, -0.0015706, 0.0035930, -0.0034111, 0.0028140
7: -0.0069291, -0.0001019, -0.0074500, -0.0004176, -0.0038324, 0.0046456
8: 0.9843329, 0.9891422, 0.9839659, 0.9889197, -0.0026996, 0.0032724
9: -0.0060312, -0.0016657, -0.0058293, -0.0013326, -0.0029705, 0.0024505

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_A1_A1_A1_B1_B2_A1

### Relational analysis result of IS_A2_A1_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013992, upper bound: 0.0016224
time: 0.83 seconds

## Relational analysis of IS_A2_A1_A1_A1_B1_B2_A2

### Relational analysis result of IS_A2_A1_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013866, upper bound: 0.0015500
time: 0.96 seconds

## BFS IS instance: IS_A2_A1_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0028669, 0.0050237, 0.0027599, 0.0050186, -0.0010692, 0.0011352
1: 0.0017365, 0.0020481, 0.0017210, 0.0020473, -0.0001545, 0.0001640
2: 0.0115824, 0.0127748, 0.0115852, 0.0128340, -0.0006276, 0.0005911
3: -0.0027014, -0.0014681, -0.0026985, -0.0014069, -0.0006491, 0.0006114
4: -0.0024476, -0.0011125, -0.0025139, -0.0011157, -0.0006618, 0.0007027
5: 0.0051662, 0.0064296, 0.0051691, 0.0064923, -0.0006650, 0.0006263
6: -0.0018025, 0.0032105, -0.0017907, 0.0034592, -0.0026385, 0.0024851
7: -0.0069291, -0.0001019, -0.0072679, -0.0001179, -0.0033844, 0.0035935
8: 0.9843329, 0.9891422, 0.9840942, 0.9891307, -0.0023841, 0.0025313
9: -0.0060312, -0.0016657, -0.0060210, -0.0014490, -0.0022978, 0.0021641

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_A1_A1_A1_B2_B1_A1

### Relational analysis result of IS_A2_A1_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013992, upper bound: 0.0015969
time: 0.88 seconds

## Relational analysis of IS_A2_A1_A1_A1_B2_B1_A2

### Relational analysis result of IS_A2_A1_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013866, upper bound: 0.0015273
time: 1.03 seconds

## BFS IS instance: IS_A2_A1_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0028669, 0.0050237, 0.0027259, 0.0050380, -0.0011558, 0.0012438
1: 0.0017365, 0.0020481, 0.0017161, 0.0020501, -0.0001670, 0.0001797
2: 0.0115824, 0.0127748, 0.0115745, 0.0128528, -0.0006877, 0.0006390
3: -0.0027014, -0.0014681, -0.0027096, -0.0013875, -0.0007112, 0.0006609
4: -0.0024476, -0.0011125, -0.0025349, -0.0011037, -0.0007155, 0.0007700
5: 0.0051662, 0.0064296, 0.0051578, 0.0065122, -0.0007286, 0.0006771
6: -0.0018025, 0.0032105, -0.0018356, 0.0035383, -0.0028910, 0.0026864
7: -0.0069291, -0.0001019, -0.0073756, -0.0000568, -0.0036587, 0.0039373
8: 0.9843329, 0.9891422, 0.9840184, 0.9891739, -0.0025772, 0.0027735
9: -0.0060312, -0.0016657, -0.0060601, -0.0013802, -0.0025176, 0.0023395

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_A1_A1_A1_B2_B2_A1

### Relational analysis result of IS_A2_A1_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013992, upper bound: 0.0016183
time: 0.94 seconds

## Relational analysis of IS_A2_A1_A1_A1_B2_B2_A2

### Relational analysis result of IS_A2_A1_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013866, upper bound: 0.0015452
time: 0.96 seconds

## BFS IS instance: IS_A2_A1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0027163, 0.0049792, 0.0027638, 0.0049774, -0.0014008, 0.0013078
1: 0.0017147, 0.0020416, 0.0017216, 0.0020414, -0.0002024, 0.0001889
2: 0.0116070, 0.0128581, 0.0116080, 0.0128318, -0.0007231, 0.0007745
3: -0.0026759, -0.0013820, -0.0026749, -0.0014092, -0.0007478, 0.0008010
4: -0.0025408, -0.0011401, -0.0025114, -0.0011412, -0.0008671, 0.0008096
5: 0.0051923, 0.0065178, 0.0051933, 0.0064900, -0.0007661, 0.0008206
6: -0.0016989, 0.0035605, -0.0016948, 0.0034501, -0.0030398, 0.0032559
7: -0.0074058, -0.0002429, -0.0072555, -0.0002486, -0.0044343, 0.0041399
8: 0.9839971, 0.9890428, 0.9841030, 0.9890388, -0.0031236, 0.0029163
9: -0.0059410, -0.0013609, -0.0059374, -0.0014570, -0.0026472, 0.0028354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A1_A1_A2_B1_B1_B1

### Relational analysis result of IS_A2_A1_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0016779
time: 0.84 seconds

## Relational analysis of IS_A2_A1_A1_A2_B1_B1_B2

### Relational analysis result of IS_A2_A1_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0016866
time: 0.81 seconds

## BFS IS instance: IS_A2_A1_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0027163, 0.0049792, 0.0026238, 0.0049259, -0.0012308, 0.0013377
1: 0.0017147, 0.0020416, 0.0017014, 0.0020339, -0.0001778, 0.0001933
2: 0.0116070, 0.0128581, 0.0116365, 0.0129092, -0.0007396, 0.0006805
3: -0.0026759, -0.0013820, -0.0026455, -0.0013291, -0.0007649, 0.0007038
4: -0.0025408, -0.0011401, -0.0025981, -0.0011731, -0.0007619, 0.0008280
5: 0.0051923, 0.0065178, 0.0052235, 0.0065720, -0.0007836, 0.0007210
6: -0.0016989, 0.0035605, -0.0015751, 0.0037756, -0.0031091, 0.0028608
7: -0.0074058, -0.0002429, -0.0076988, -0.0004115, -0.0038961, 0.0042343
8: 0.9839971, 0.9890428, 0.9837906, 0.9889239, -0.0027445, 0.0029827
9: -0.0059410, -0.0013609, -0.0058332, -0.0011736, -0.0027075, 0.0024913

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A1_A1_A2_B1_B2_B1

### Relational analysis result of IS_A2_A1_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0017010
time: 1.00 seconds

## Relational analysis of IS_A2_A1_A1_A2_B1_B2_B2

### Relational analysis result of IS_A2_A1_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0017100
time: 1.06 seconds

## BFS IS instance: IS_A2_A1_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0027163, 0.0049792, 0.0027954, 0.0050800, -0.0013628, 0.0011143
1: 0.0017147, 0.0020416, 0.0017262, 0.0020562, -0.0001969, 0.0001610
2: 0.0116070, 0.0128581, 0.0115512, 0.0128143, -0.0006161, 0.0007534
3: -0.0026759, -0.0013820, -0.0027336, -0.0014273, -0.0006372, 0.0007793
4: -0.0025408, -0.0011401, -0.0024919, -0.0010777, -0.0008436, 0.0006898
5: 0.0051923, 0.0065178, 0.0051332, 0.0064715, -0.0006528, 0.0007983
6: -0.0016989, 0.0035605, -0.0019334, 0.0033766, -0.0025900, 0.0031675
7: -0.0074058, -0.0002429, -0.0071554, 0.0000764, -0.0043138, 0.0035273
8: 0.9839971, 0.9890428, 0.9841735, 0.9892677, -0.0030388, 0.0024847
9: -0.0059410, -0.0013609, -0.0061452, -0.0015210, -0.0022555, 0.0027584

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A1_A1_A2_B2_B1_B1

### Relational analysis result of IS_A2_A1_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0016655
time: 0.86 seconds

## Relational analysis of IS_A2_A1_A1_A2_B2_B1_B2

### Relational analysis result of IS_A2_A1_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0016774
time: 0.81 seconds

## BFS IS instance: IS_A2_A1_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0027163, 0.0049792, 0.0026453, 0.0050397, -0.0011785, 0.0011159
1: 0.0017147, 0.0020416, 0.0017045, 0.0020504, -0.0001703, 0.0001612
2: 0.0116070, 0.0128581, 0.0115736, 0.0128974, -0.0006170, 0.0006516
3: -0.0026759, -0.0013820, -0.0027105, -0.0013414, -0.0006381, 0.0006739
4: -0.0025408, -0.0011401, -0.0025848, -0.0011026, -0.0007295, 0.0006908
5: 0.0051923, 0.0065178, 0.0051568, 0.0065594, -0.0006537, 0.0006904
6: -0.0016989, 0.0035605, -0.0018396, 0.0037256, -0.0025937, 0.0027392
7: -0.0074058, -0.0002429, -0.0076307, -0.0000513, -0.0037306, 0.0035324
8: 0.9839971, 0.9890428, 0.9838386, 0.9891777, -0.0026279, 0.0024883
9: -0.0059410, -0.0013609, -0.0060635, -0.0012171, -0.0022587, 0.0023854

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A1_A1_A2_B2_B2_B1

### Relational analysis result of IS_A2_A1_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0016893
time: 1.08 seconds

## Relational analysis of IS_A2_A1_A1_A2_B2_B2_B2

### Relational analysis result of IS_A2_A1_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0017018
time: 1.03 seconds

## BFS IS instance: IS_A2_A1_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0028237, 0.0050046, 0.0027315, 0.0048920, -0.0011932, 0.0013616
1: 0.0017302, 0.0020453, 0.0017169, 0.0020291, -0.0001724, 0.0001967
2: 0.0115930, 0.0127987, 0.0116552, 0.0128497, -0.0007528, 0.0006597
3: -0.0026905, -0.0014434, -0.0026261, -0.0013907, -0.0007786, 0.0006823
4: -0.0024744, -0.0011244, -0.0025315, -0.0011940, -0.0007386, 0.0008429
5: 0.0051774, 0.0064549, 0.0052433, 0.0065090, -0.0007976, 0.0006989
6: -0.0017580, 0.0033110, -0.0014964, 0.0035253, -0.0031648, 0.0027732
7: -0.0070660, -0.0001624, -0.0073579, -0.0005187, -0.0037769, 0.0043102
8: 0.9842364, 0.9890994, 0.9840308, 0.9888485, -0.0026605, 0.0030362
9: -0.0059925, -0.0015781, -0.0057647, -0.0013915, -0.0027560, 0.0024150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_A1_A2_A1_B1_B1_B1

### Relational analysis result of IS_A2_A1_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014593, upper bound: 0.0015138
time: 0.99 seconds

## Relational analysis of IS_A2_A1_A2_A1_B1_B1_B2

### Relational analysis result of IS_A2_A1_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013798, upper bound: 0.0015214
time: 0.96 seconds

## BFS IS instance: IS_A2_A1_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0028237, 0.0050046, 0.0027040, 0.0049029, -0.0012818, 0.0014686
1: 0.0017302, 0.0020453, 0.0017130, 0.0020306, -0.0001852, 0.0002122
2: 0.0115930, 0.0127987, 0.0116492, 0.0128649, -0.0008119, 0.0007087
3: -0.0026905, -0.0014434, -0.0026324, -0.0013750, -0.0008397, 0.0007330
4: -0.0024744, -0.0011244, -0.0025485, -0.0011873, -0.0007935, 0.0009091
5: 0.0051774, 0.0064549, 0.0052369, 0.0065250, -0.0008603, 0.0007509
6: -0.0017580, 0.0033110, -0.0015218, 0.0035891, -0.0034134, 0.0029793
7: -0.0070660, -0.0001624, -0.0074448, -0.0004842, -0.0040576, 0.0046487
8: 0.9842364, 0.9890994, 0.9839696, 0.9888728, -0.0028583, 0.0032747
9: -0.0059925, -0.0015781, -0.0057868, -0.0013359, -0.0029725, 0.0025945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_A1_A2_A1_B1_B2_B1

### Relational analysis result of IS_A2_A1_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014593, upper bound: 0.0015272
time: 1.00 seconds

## Relational analysis of IS_A2_A1_A2_A1_B1_B2_B2

### Relational analysis result of IS_A2_A1_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013798, upper bound: 0.0015370
time: 0.94 seconds

## BFS IS instance: IS_A2_A1_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0028237, 0.0050046, 0.0027616, 0.0049991, -0.0011650, 0.0011427
1: 0.0017302, 0.0020453, 0.0017213, 0.0020445, -0.0001683, 0.0001651
2: 0.0115930, 0.0127987, 0.0115960, 0.0128331, -0.0006318, 0.0006441
3: -0.0026905, -0.0014434, -0.0026873, -0.0014079, -0.0006534, 0.0006661
4: -0.0024744, -0.0011244, -0.0025128, -0.0011278, -0.0007211, 0.0007074
5: 0.0051774, 0.0064549, 0.0051806, 0.0064913, -0.0006694, 0.0006824
6: -0.0017580, 0.0033110, -0.0017453, 0.0034553, -0.0026560, 0.0027077
7: -0.0070660, -0.0001624, -0.0072626, -0.0001798, -0.0036877, 0.0036172
8: 0.9842364, 0.9890994, 0.9840979, 0.9890872, -0.0025977, 0.0025480
9: -0.0059925, -0.0015781, -0.0059814, -0.0014525, -0.0023129, 0.0023580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_A1_A2_A1_B2_B1_B1

### Relational analysis result of IS_A2_A1_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014593, upper bound: 0.0015051
time: 1.04 seconds

## Relational analysis of IS_A2_A1_A2_A1_B2_B1_B2

### Relational analysis result of IS_A2_A1_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013798, upper bound: 0.0015140
time: 0.95 seconds

## BFS IS instance: IS_A2_A1_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0028237, 0.0050046, 0.0027276, 0.0050171, -0.0012568, 0.0012512
1: 0.0017302, 0.0020453, 0.0017164, 0.0020471, -0.0001816, 0.0001808
2: 0.0115930, 0.0127987, 0.0115860, 0.0128518, -0.0006918, 0.0006949
3: -0.0026905, -0.0014434, -0.0026977, -0.0013885, -0.0007154, 0.0007187
4: -0.0024744, -0.0011244, -0.0025338, -0.0011166, -0.0007780, 0.0007745
5: 0.0051774, 0.0064549, 0.0051700, 0.0065112, -0.0007329, 0.0007362
6: -0.0017580, 0.0033110, -0.0017872, 0.0035343, -0.0029081, 0.0029212
7: -0.0070660, -0.0001624, -0.0073701, -0.0001227, -0.0039784, 0.0039606
8: 0.9842364, 0.9890994, 0.9840223, 0.9891275, -0.0028025, 0.0027899
9: -0.0059925, -0.0015781, -0.0060179, -0.0013837, -0.0025325, 0.0025439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_A1_A2_A1_B2_B2_B1

### Relational analysis result of IS_A2_A1_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014593, upper bound: 0.0015215
time: 1.03 seconds

## Relational analysis of IS_A2_A1_A2_A1_B2_B2_B2

### Relational analysis result of IS_A2_A1_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013798, upper bound: 0.0015315
time: 0.91 seconds

## BFS IS instance: IS_A2_A1_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0026698, 0.0049633, 0.0027654, 0.0049578, -0.0014633, 0.0013027
1: 0.0017080, 0.0020394, 0.0017218, 0.0020386, -0.0002114, 0.0001882
2: 0.0116158, 0.0128838, 0.0116188, 0.0128310, -0.0007202, 0.0008090
3: -0.0026669, -0.0013554, -0.0026637, -0.0014101, -0.0007449, 0.0008367
4: -0.0025696, -0.0011499, -0.0025105, -0.0011533, -0.0009058, 0.0008064
5: 0.0052015, 0.0065451, 0.0052048, 0.0064891, -0.0007631, 0.0008572
6: -0.0016621, 0.0036687, -0.0016492, 0.0034465, -0.0030279, 0.0034011
7: -0.0075532, -0.0002930, -0.0072505, -0.0003106, -0.0046320, 0.0041238
8: 0.9838932, 0.9890075, 0.9841065, 0.9889951, -0.0032629, 0.0029049
9: -0.0059090, -0.0012666, -0.0058977, -0.0014601, -0.0026368, 0.0029618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A1_A2_A2_B1_B1_B1

### Relational analysis result of IS_A2_A1_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0016828
time: 0.95 seconds

## Relational analysis of IS_A2_A1_A2_A2_B1_B1_B2

### Relational analysis result of IS_A2_A1_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0016944
time: 0.79 seconds

## BFS IS instance: IS_A2_A1_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0026698, 0.0049633, 0.0026254, 0.0049049, -0.0013077, 0.0013377
1: 0.0017080, 0.0020394, 0.0017016, 0.0020309, -0.0001889, 0.0001933
2: 0.0116158, 0.0128838, 0.0116481, 0.0129083, -0.0007396, 0.0007230
3: -0.0026669, -0.0013554, -0.0026335, -0.0013301, -0.0007649, 0.0007478
4: -0.0025696, -0.0011499, -0.0025971, -0.0011861, -0.0008095, 0.0008281
5: 0.0052015, 0.0065451, 0.0052358, 0.0065711, -0.0007836, 0.0007661
6: -0.0016621, 0.0036687, -0.0015263, 0.0037717, -0.0031092, 0.0030395
7: -0.0075532, -0.0002930, -0.0076935, -0.0004781, -0.0041395, 0.0042344
8: 0.9838932, 0.9890075, 0.9837945, 0.9888771, -0.0029160, 0.0029828
9: -0.0059090, -0.0012666, -0.0057907, -0.0011769, -0.0027076, 0.0026469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A1_A2_A2_B1_B2_B1

### Relational analysis result of IS_A2_A1_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0017073
time: 0.95 seconds

## Relational analysis of IS_A2_A1_A2_A2_B1_B2_B2

### Relational analysis result of IS_A2_A1_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0017199
time: 1.05 seconds

## BFS IS instance: IS_A2_A1_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0026698, 0.0049633, 0.0027970, 0.0050604, -0.0014431, 0.0011116
1: 0.0017080, 0.0020394, 0.0017264, 0.0020534, -0.0002085, 0.0001606
2: 0.0116158, 0.0128838, 0.0115621, 0.0128135, -0.0006146, 0.0007978
3: -0.0026669, -0.0013554, -0.0027224, -0.0014281, -0.0006356, 0.0008252
4: -0.0025696, -0.0011499, -0.0024909, -0.0010898, -0.0008933, 0.0006881
5: 0.0052015, 0.0065451, 0.0051447, 0.0064706, -0.0006512, 0.0008454
6: -0.0016621, 0.0036687, -0.0018878, 0.0033731, -0.0025837, 0.0033541
7: -0.0075532, -0.0002930, -0.0071505, 0.0000143, -0.0045680, 0.0035188
8: 0.9838932, 0.9890075, 0.9841769, 0.9892239, -0.0032178, 0.0024787
9: -0.0059090, -0.0012666, -0.0061055, -0.0015241, -0.0022500, 0.0029209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A1_A2_A2_B2_B1_B1

### Relational analysis result of IS_A2_A1_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0016703
time: 1.05 seconds

## Relational analysis of IS_A2_A1_A2_A2_B2_B1_B2

### Relational analysis result of IS_A2_A1_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0016840
time: 1.04 seconds

## BFS IS instance: IS_A2_A1_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0026698, 0.0049633, 0.0026469, 0.0050188, -0.0012845, 0.0011238
1: 0.0017080, 0.0020394, 0.0017047, 0.0020474, -0.0001856, 0.0001624
2: 0.0116158, 0.0128838, 0.0115851, 0.0128964, -0.0006213, 0.0007102
3: -0.0026669, -0.0013554, -0.0026986, -0.0013424, -0.0006426, 0.0007345
4: -0.0025696, -0.0011499, -0.0025838, -0.0011155, -0.0007951, 0.0006956
5: 0.0052015, 0.0065451, 0.0051690, 0.0065585, -0.0006583, 0.0007525
6: -0.0016621, 0.0036687, -0.0017912, 0.0037218, -0.0026120, 0.0029856
7: -0.0075532, -0.0002930, -0.0076255, -0.0001173, -0.0040661, 0.0035573
8: 0.9838932, 0.9890075, 0.9838423, 0.9891313, -0.0028642, 0.0025059
9: -0.0059090, -0.0012666, -0.0060213, -0.0012204, -0.0022746, 0.0026000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A1_A2_A2_B2_B2_B1

### Relational analysis result of IS_A2_A1_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0016959
time: 1.22 seconds

## Relational analysis of IS_A2_A1_A2_A2_B2_B2_B2

### Relational analysis result of IS_A2_A1_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0017107
time: 1.15 seconds

## BFS IS instance: IS_A2_A2_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0028349, 0.0050386, 0.0027298, 0.0049144, -0.0012128, 0.0014385
1: 0.0017319, 0.0020502, 0.0017167, 0.0020323, -0.0001752, 0.0002078
2: 0.0115741, 0.0127925, 0.0116428, 0.0128506, -0.0007953, 0.0006705
3: -0.0027099, -0.0014498, -0.0026389, -0.0013897, -0.0008226, 0.0006935
4: -0.0024674, -0.0011033, -0.0025325, -0.0011802, -0.0007507, 0.0008905
5: 0.0051574, 0.0064484, 0.0052302, 0.0065099, -0.0008427, 0.0007104
6: -0.0018371, 0.0032849, -0.0015485, 0.0035292, -0.0033435, 0.0028188
7: -0.0070305, -0.0000547, -0.0073632, -0.0004478, -0.0038389, 0.0045536
8: 0.9842615, 0.9891754, 0.9840271, 0.9888984, -0.0027042, 0.0032077
9: -0.0060614, -0.0016009, -0.0058100, -0.0013881, -0.0029117, 0.0024547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_A2_A1_A1_B1_B1_A1

### Relational analysis result of IS_A2_A2_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014082, upper bound: 0.0016003
time: 0.97 seconds

## Relational analysis of IS_A2_A2_A1_A1_B1_B1_A2

### Relational analysis result of IS_A2_A2_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013972, upper bound: 0.0015357
time: 0.99 seconds

## BFS IS instance: IS_A2_A2_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0028349, 0.0050386, 0.0027024, 0.0049240, -0.0011381, 0.0013787
1: 0.0017319, 0.0020502, 0.0017127, 0.0020337, -0.0001644, 0.0001992
2: 0.0115741, 0.0127925, 0.0116375, 0.0128658, -0.0007622, 0.0006292
3: -0.0027099, -0.0014498, -0.0026444, -0.0013740, -0.0007883, 0.0006508
4: -0.0024674, -0.0011033, -0.0025495, -0.0011743, -0.0007045, 0.0008534
5: 0.0051574, 0.0064484, 0.0052246, 0.0065260, -0.0008076, 0.0006667
6: -0.0018371, 0.0032849, -0.0015706, 0.0035930, -0.0032044, 0.0026452
7: -0.0070305, -0.0000547, -0.0074500, -0.0004176, -0.0036025, 0.0043641
8: 0.9842615, 0.9891754, 0.9839659, 0.9889197, -0.0025377, 0.0030741
9: -0.0060614, -0.0016009, -0.0058293, -0.0013326, -0.0027905, 0.0023035

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_A2_A1_A1_B1_B2_A1

### Relational analysis result of IS_A2_A2_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014082, upper bound: 0.0016007
time: 0.95 seconds

## Relational analysis of IS_A2_A2_A1_A1_B1_B2_A2

### Relational analysis result of IS_A2_A2_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013972, upper bound: 0.0015370
time: 1.00 seconds

## BFS IS instance: IS_A2_A2_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0028349, 0.0050386, 0.0027599, 0.0050186, -0.0011810, 0.0012344
1: 0.0017319, 0.0020502, 0.0017210, 0.0020473, -0.0001706, 0.0001783
2: 0.0115741, 0.0127925, 0.0115852, 0.0128340, -0.0006825, 0.0006530
3: -0.0027099, -0.0014498, -0.0026985, -0.0014069, -0.0007059, 0.0006753
4: -0.0024674, -0.0011033, -0.0025139, -0.0011157, -0.0007311, 0.0007641
5: 0.0051574, 0.0064484, 0.0051691, 0.0064923, -0.0007231, 0.0006919
6: -0.0018371, 0.0032849, -0.0017907, 0.0034592, -0.0028691, 0.0027451
7: -0.0070305, -0.0000547, -0.0072679, -0.0001179, -0.0037385, 0.0039075
8: 0.9842615, 0.9891754, 0.9840942, 0.9891307, -0.0026335, 0.0027525
9: -0.0060614, -0.0016009, -0.0060210, -0.0014490, -0.0024986, 0.0023905

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_A2_A1_A1_B2_B1_A1

### Relational analysis result of IS_A2_A2_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014082, upper bound: 0.0015967
time: 0.99 seconds

## Relational analysis of IS_A2_A2_A1_A1_B2_B1_A2

### Relational analysis result of IS_A2_A2_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013972, upper bound: 0.0015273
time: 1.00 seconds

## BFS IS instance: IS_A2_A2_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0028349, 0.0050386, 0.0027259, 0.0050380, -0.0010895, 0.0011547
1: 0.0017319, 0.0020502, 0.0017161, 0.0020501, -0.0001574, 0.0001668
2: 0.0115741, 0.0127925, 0.0115745, 0.0128528, -0.0006384, 0.0006024
3: -0.0027099, -0.0014498, -0.0027096, -0.0013875, -0.0006603, 0.0006230
4: -0.0024674, -0.0011033, -0.0025349, -0.0011037, -0.0006744, 0.0007148
5: 0.0051574, 0.0064484, 0.0051578, 0.0065122, -0.0006764, 0.0006382
6: -0.0018371, 0.0032849, -0.0018356, 0.0035383, -0.0026839, 0.0025323
7: -0.0070305, -0.0000547, -0.0073756, -0.0000568, -0.0034488, 0.0036552
8: 0.9842615, 0.9891754, 0.9840184, 0.9891739, -0.0024294, 0.0025748
9: -0.0060614, -0.0016009, -0.0060601, -0.0013802, -0.0023373, 0.0022053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_A2_A1_A1_B2_B2_A1

### Relational analysis result of IS_A2_A2_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014082, upper bound: 0.0015976
time: 1.02 seconds

## Relational analysis of IS_A2_A2_A1_A1_B2_B2_A2

### Relational analysis result of IS_A2_A2_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013972, upper bound: 0.0015294
time: 1.00 seconds

## BFS IS instance: IS_A2_A2_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0026820, 0.0049985, 0.0027519, 0.0049777, -0.0013184, 0.0014386
1: 0.0017098, 0.0020444, 0.0017199, 0.0020414, -0.0001905, 0.0002078
2: 0.0115963, 0.0128771, 0.0116078, 0.0128384, -0.0007954, 0.0007289
3: -0.0026870, -0.0013624, -0.0026751, -0.0014024, -0.0008226, 0.0007538
4: -0.0025621, -0.0011281, -0.0025188, -0.0011410, -0.0008161, 0.0008905
5: 0.0051809, 0.0065379, 0.0051931, 0.0064970, -0.0008427, 0.0007723
6: -0.0017439, 0.0036403, -0.0016955, 0.0034779, -0.0033437, 0.0030642
7: -0.0075145, -0.0001817, -0.0072933, -0.0002476, -0.0041732, 0.0045539
8: 0.9839206, 0.9890859, 0.9840764, 0.9890394, -0.0029397, 0.0032078
9: -0.0059802, -0.0012914, -0.0059380, -0.0014328, -0.0029119, 0.0026685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A2_A1_A2_B1_B1_B1

### Relational analysis result of IS_A2_A2_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017653, upper bound: 0.0016779
time: 0.94 seconds

## Relational analysis of IS_A2_A2_A1_A2_B1_B1_B2

### Relational analysis result of IS_A2_A2_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017653, upper bound: 0.0016781
time: 1.00 seconds

## BFS IS instance: IS_A2_A2_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0026820, 0.0049985, 0.0026084, 0.0049262, -0.0011491, 0.0014676
1: 0.0017098, 0.0020444, 0.0016991, 0.0020340, -0.0001660, 0.0002120
2: 0.0115963, 0.0128771, 0.0116363, 0.0129177, -0.0008114, 0.0006353
3: -0.0026870, -0.0013624, -0.0026457, -0.0013203, -0.0008392, 0.0006571
4: -0.0025621, -0.0011281, -0.0026076, -0.0011729, -0.0007113, 0.0009085
5: 0.0051809, 0.0065379, 0.0052233, 0.0065811, -0.0008597, 0.0006732
6: -0.0017439, 0.0036403, -0.0015759, 0.0038114, -0.0034112, 0.0026709
7: -0.0075145, -0.0001817, -0.0077474, -0.0004104, -0.0036375, 0.0046457
8: 0.9839206, 0.9890859, 0.9837565, 0.9889247, -0.0025623, 0.0032725
9: -0.0059802, -0.0012914, -0.0058339, -0.0011424, -0.0029706, 0.0023259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A2_A1_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017653, upper bound: 0.0017010
time: 1.01 seconds

## Relational analysis of IS_A2_A2_A1_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017653, upper bound: 0.0017019
time: 1.01 seconds

## BFS IS instance: IS_A2_A2_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0026820, 0.0049985, 0.0027809, 0.0050805, -0.0012845, 0.0012488
1: 0.0017098, 0.0020444, 0.0017241, 0.0020563, -0.0001856, 0.0001804
2: 0.0115963, 0.0128771, 0.0115510, 0.0128224, -0.0006904, 0.0007102
3: -0.0026870, -0.0013624, -0.0027339, -0.0014189, -0.0007141, 0.0007345
4: -0.0025621, -0.0011281, -0.0025009, -0.0010773, -0.0007951, 0.0007730
5: 0.0051809, 0.0065379, 0.0051329, 0.0064800, -0.0007316, 0.0007525
6: -0.0017439, 0.0036403, -0.0019346, 0.0034105, -0.0029026, 0.0029856
7: -0.0075145, -0.0001817, -0.0072016, 0.0000780, -0.0040661, 0.0039531
8: 0.9839206, 0.9890859, 0.9841409, 0.9892688, -0.0028642, 0.0027847
9: -0.0059802, -0.0012914, -0.0061462, -0.0014915, -0.0025277, 0.0026000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A2_A1_A2_B2_B1_B1

### Relational analysis result of IS_A2_A2_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017653, upper bound: 0.0016655
time: 0.90 seconds

## Relational analysis of IS_A2_A2_A1_A2_B2_B1_B2

### Relational analysis result of IS_A2_A2_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017653, upper bound: 0.0016659
time: 1.02 seconds

## BFS IS instance: IS_A2_A2_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0026820, 0.0049985, 0.0026300, 0.0050402, -0.0011034, 0.0012654
1: 0.0017098, 0.0020444, 0.0017023, 0.0020505, -0.0001594, 0.0001828
2: 0.0115963, 0.0128771, 0.0115733, 0.0129058, -0.0006996, 0.0006100
3: -0.0026870, -0.0013624, -0.0027108, -0.0013327, -0.0007235, 0.0006309
4: -0.0025621, -0.0011281, -0.0025943, -0.0011023, -0.0006830, 0.0007833
5: 0.0051809, 0.0065379, 0.0051565, 0.0065684, -0.0007412, 0.0006464
6: -0.0017439, 0.0036403, -0.0018408, 0.0037611, -0.0029411, 0.0025646
7: -0.0075145, -0.0001817, -0.0076790, -0.0000497, -0.0034927, 0.0040055
8: 0.9839206, 0.9890859, 0.9838046, 0.9891789, -0.0024604, 0.0028215
9: -0.0059802, -0.0012914, -0.0060646, -0.0011862, -0.0025612, 0.0022334

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A2_A1_A2_B2_B2_B1

### Relational analysis result of IS_A2_A2_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017653, upper bound: 0.0016893
time: 0.91 seconds

## Relational analysis of IS_A2_A2_A1_A2_B2_B2_B2

### Relational analysis result of IS_A2_A2_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017653, upper bound: 0.0016905
time: 1.02 seconds

## BFS IS instance: IS_A2_A2_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0027883, 0.0050203, 0.0027315, 0.0048920, -0.0012816, 0.0014379
1: 0.0017251, 0.0020476, 0.0017169, 0.0020291, -0.0001852, 0.0002077
2: 0.0115843, 0.0128183, 0.0116552, 0.0128497, -0.0007950, 0.0007086
3: -0.0026995, -0.0014232, -0.0026261, -0.0013907, -0.0008222, 0.0007328
4: -0.0024962, -0.0011146, -0.0025315, -0.0011940, -0.0007933, 0.0008901
5: 0.0051682, 0.0064756, 0.0052433, 0.0065090, -0.0008423, 0.0007508
6: -0.0017946, 0.0033931, -0.0014964, 0.0035253, -0.0033420, 0.0029788
7: -0.0071778, -0.0001126, -0.0073579, -0.0005187, -0.0040568, 0.0045516
8: 0.9841577, 0.9891346, 0.9840308, 0.9888485, -0.0028577, 0.0032062
9: -0.0060243, -0.0015067, -0.0057647, -0.0013915, -0.0029104, 0.0025940

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_A2_A2_A1_B1_B1_B1

### Relational analysis result of IS_A2_A2_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014775, upper bound: 0.0015138
time: 0.96 seconds

## Relational analysis of IS_A2_A2_A2_A1_B1_B1_B2

### Relational analysis result of IS_A2_A2_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013915, upper bound: 0.0015214
time: 1.01 seconds

## BFS IS instance: IS_A2_A2_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0027883, 0.0050203, 0.0027040, 0.0049029, -0.0012083, 0.0013791
1: 0.0017251, 0.0020476, 0.0017130, 0.0020306, -0.0001746, 0.0001992
2: 0.0115843, 0.0128183, 0.0116492, 0.0128649, -0.0007625, 0.0006680
3: -0.0026995, -0.0014232, -0.0026324, -0.0013750, -0.0007886, 0.0006909
4: -0.0024962, -0.0011146, -0.0025485, -0.0011873, -0.0007479, 0.0008537
5: 0.0051682, 0.0064756, 0.0052369, 0.0065250, -0.0008079, 0.0007078
6: -0.0017946, 0.0033931, -0.0015218, 0.0035891, -0.0032054, 0.0028083
7: -0.0071778, -0.0001126, -0.0074448, -0.0004842, -0.0038247, 0.0043655
8: 0.9841577, 0.9891346, 0.9839696, 0.9888728, -0.0026942, 0.0030751
9: -0.0060243, -0.0015067, -0.0057868, -0.0013359, -0.0027914, 0.0024456

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_A2_A2_A1_B1_B2_B1

### Relational analysis result of IS_A2_A2_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014775, upper bound: 0.0015140
time: 0.98 seconds

## Relational analysis of IS_A2_A2_A2_A1_B1_B2_B2

### Relational analysis result of IS_A2_A2_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013915, upper bound: 0.0015219
time: 0.99 seconds

## BFS IS instance: IS_A2_A2_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0027883, 0.0050203, 0.0027616, 0.0049991, -0.0012692, 0.0012405
1: 0.0017251, 0.0020476, 0.0017213, 0.0020445, -0.0001834, 0.0001792
2: 0.0115843, 0.0128183, 0.0115960, 0.0128331, -0.0006858, 0.0007017
3: -0.0026995, -0.0014232, -0.0026873, -0.0014079, -0.0007093, 0.0007257
4: -0.0024962, -0.0011146, -0.0025128, -0.0011278, -0.0007856, 0.0007679
5: 0.0051682, 0.0064756, 0.0051806, 0.0064913, -0.0007267, 0.0007435
6: -0.0017946, 0.0033931, -0.0017453, 0.0034553, -0.0028832, 0.0029499
7: -0.0071778, -0.0001126, -0.0072626, -0.0001798, -0.0040175, 0.0039266
8: 0.9841577, 0.9891346, 0.9840979, 0.9890872, -0.0028300, 0.0027660
9: -0.0060243, -0.0015067, -0.0059814, -0.0014525, -0.0025108, 0.0025689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_A2_A2_A1_B2_B1_B1

### Relational analysis result of IS_A2_A2_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014775, upper bound: 0.0015051
time: 0.96 seconds

## Relational analysis of IS_A2_A2_A2_A1_B2_B1_B2

### Relational analysis result of IS_A2_A2_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013915, upper bound: 0.0015139
time: 0.97 seconds

## BFS IS instance: IS_A2_A2_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0027883, 0.0050203, 0.0027276, 0.0050171, -0.0011848, 0.0011616
1: 0.0017251, 0.0020476, 0.0017164, 0.0020471, -0.0001712, 0.0001678
2: 0.0115843, 0.0128183, 0.0115860, 0.0128518, -0.0006422, 0.0006550
3: -0.0026995, -0.0014232, -0.0026977, -0.0013885, -0.0006642, 0.0006775
4: -0.0024962, -0.0011146, -0.0025338, -0.0011166, -0.0007334, 0.0007191
5: 0.0051682, 0.0064756, 0.0051700, 0.0065112, -0.0006805, 0.0006940
6: -0.0017946, 0.0033931, -0.0017872, 0.0035343, -0.0026999, 0.0027537
7: -0.0071778, -0.0001126, -0.0073701, -0.0001227, -0.0037503, 0.0036770
8: 0.9841577, 0.9891346, 0.9840223, 0.9891275, -0.0026418, 0.0025902
9: -0.0060243, -0.0015067, -0.0060179, -0.0013837, -0.0023512, 0.0023981

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_A2_A2_A1_B2_B2_B1

### Relational analysis result of IS_A2_A2_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014775, upper bound: 0.0015059
time: 0.88 seconds

## Relational analysis of IS_A2_A2_A2_A1_B2_B2_B2

### Relational analysis result of IS_A2_A2_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013915, upper bound: 0.0015151
time: 0.98 seconds

## BFS IS instance: IS_A2_A2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0026370, 0.0049806, 0.0027535, 0.0049581, -0.0013881, 0.0014344
1: 0.0017033, 0.0020419, 0.0017201, 0.0020386, -0.0002005, 0.0002072
2: 0.0116062, 0.0129019, 0.0116187, 0.0128375, -0.0007930, 0.0007674
3: -0.0026768, -0.0013367, -0.0026639, -0.0014033, -0.0008202, 0.0007937
4: -0.0025899, -0.0011392, -0.0025178, -0.0011532, -0.0008592, 0.0008879
5: 0.0051914, 0.0065643, 0.0052046, 0.0064961, -0.0008403, 0.0008131
6: -0.0017024, 0.0037448, -0.0016499, 0.0034742, -0.0033340, 0.0032262
7: -0.0076568, -0.0002382, -0.0072883, -0.0003097, -0.0043938, 0.0045406
8: 0.9838202, 0.9890460, 0.9840799, 0.9889957, -0.0030951, 0.0031985
9: -0.0059440, -0.0012004, -0.0058983, -0.0014360, -0.0029034, 0.0028095

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A2_A2_A2_B1_B1_B1

### Relational analysis result of IS_A2_A2_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017653, upper bound: 0.0016828
time: 1.07 seconds

## Relational analysis of IS_A2_A2_A2_A2_B1_B1_B2

### Relational analysis result of IS_A2_A2_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017653, upper bound: 0.0016830
time: 1.08 seconds

## BFS IS instance: IS_A2_A2_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0026370, 0.0049806, 0.0026100, 0.0049052, -0.0012343, 0.0014661
1: 0.0017033, 0.0020419, 0.0016994, 0.0020310, -0.0001783, 0.0002118
2: 0.0116062, 0.0129019, 0.0116479, 0.0129169, -0.0008106, 0.0006824
3: -0.0026768, -0.0013367, -0.0026337, -0.0013212, -0.0008383, 0.0007058
4: -0.0025899, -0.0011392, -0.0026066, -0.0011859, -0.0007640, 0.0009075
5: 0.0051914, 0.0065643, 0.0052356, 0.0065801, -0.0008588, 0.0007230
6: -0.0017024, 0.0037448, -0.0015271, 0.0038076, -0.0034076, 0.0028688
7: -0.0076568, -0.0002382, -0.0077424, -0.0004769, -0.0039071, 0.0046409
8: 0.9838202, 0.9890460, 0.9837599, 0.9888779, -0.0027523, 0.0032691
9: -0.0059440, -0.0012004, -0.0057914, -0.0011457, -0.0029675, 0.0024983

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A2_A2_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017653, upper bound: 0.0017073
time: 1.02 seconds

## Relational analysis of IS_A2_A2_A2_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017653, upper bound: 0.0017086
time: 1.03 seconds

## BFS IS instance: IS_A2_A2_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0026370, 0.0049806, 0.0027824, 0.0050609, -0.0013710, 0.0012530
1: 0.0017033, 0.0020419, 0.0017243, 0.0020535, -0.0001981, 0.0001810
2: 0.0116062, 0.0129019, 0.0115618, 0.0128216, -0.0006927, 0.0007580
3: -0.0026768, -0.0013367, -0.0027227, -0.0014198, -0.0007165, 0.0007839
4: -0.0025899, -0.0011392, -0.0025000, -0.0010895, -0.0008487, 0.0007756
5: 0.0051914, 0.0065643, 0.0051444, 0.0064791, -0.0007340, 0.0008031
6: -0.0017024, 0.0037448, -0.0018890, 0.0034070, -0.0029123, 0.0031866
7: -0.0076568, -0.0002382, -0.0071968, 0.0000159, -0.0043398, 0.0039663
8: 0.9838202, 0.9890460, 0.9841443, 0.9892251, -0.0030571, 0.0027939
9: -0.0059440, -0.0012004, -0.0061065, -0.0014945, -0.0025361, 0.0027750

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A2_A2_A2_B2_B1_B1

### Relational analysis result of IS_A2_A2_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017653, upper bound: 0.0016703
time: 1.02 seconds

## Relational analysis of IS_A2_A2_A2_A2_B2_B1_B2

### Relational analysis result of IS_A2_A2_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017653, upper bound: 0.0016704
time: 1.11 seconds

## BFS IS instance: IS_A2_A2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0026370, 0.0049806, 0.0026317, 0.0050193, -0.0012158, 0.0012729
1: 0.0017033, 0.0020419, 0.0017025, 0.0020474, -0.0001756, 0.0001839
2: 0.0116062, 0.0129019, 0.0115848, 0.0129049, -0.0007037, 0.0006722
3: -0.0026768, -0.0013367, -0.0026989, -0.0013336, -0.0007278, 0.0006952
4: -0.0025899, -0.0011392, -0.0025932, -0.0011152, -0.0007526, 0.0007879
5: 0.0051914, 0.0065643, 0.0051687, 0.0065674, -0.0007456, 0.0007122
6: -0.0017024, 0.0037448, -0.0017923, 0.0037572, -0.0029585, 0.0028258
7: -0.0076568, -0.0002382, -0.0076737, -0.0001157, -0.0038486, 0.0040292
8: 0.9838202, 0.9890460, 0.9838084, 0.9891323, -0.0027110, 0.0028383
9: -0.0059440, -0.0012004, -0.0060224, -0.0011896, -0.0025764, 0.0024609

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A2_A2_A2_B2_B2_B1

### Relational analysis result of IS_A2_A2_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017653, upper bound: 0.0016959
time: 0.85 seconds

## Relational analysis of IS_A2_A2_A2_A2_B2_B2_B2

### Relational analysis result of IS_A2_A2_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017653, upper bound: 0.0016972
time: 1.05 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.51 seconds
IS_A1_B1_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0015181, upper bound: 0.0015589
IS_A1_B1_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014522, upper bound: 0.0015757
IS_A1_B1_A1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014581, upper bound: 0.0016134
IS_A1_B1_A1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014522, upper bound: 0.0015757
IS_A1_B1_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014877, upper bound: 0.0015285
IS_A1_B1_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014150, upper bound: 0.0015351
IS_A1_B1_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014877, upper bound: 0.0015285
IS_A1_B1_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014150, upper bound: 0.0015351
IS_A1_B1_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017278, upper bound: 0.0016840
IS_A1_B1_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017278, upper bound: 0.0016840
IS_A1_B1_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017278, upper bound: 0.0017048
IS_A1_B1_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017278, upper bound: 0.0017048
IS_A1_B1_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017238, upper bound: 0.0016729
IS_A1_B1_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017238, upper bound: 0.0016729
IS_A1_B1_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017238, upper bound: 0.0016975
IS_A1_B1_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017238, upper bound: 0.0016975
IS_A1_B1_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0015140, upper bound: 0.0015458
IS_A1_B1_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014456, upper bound: 0.0015633
IS_A1_B1_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0015140, upper bound: 0.0015458
IS_A1_B1_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014456, upper bound: 0.0015633
IS_A1_B1_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014811, upper bound: 0.0015127
IS_A1_B1_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014098, upper bound: 0.0015225
IS_A1_B1_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014811, upper bound: 0.0015127
IS_A1_B1_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014098, upper bound: 0.0015225
IS_A1_B1_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017278, upper bound: 0.0016890
IS_A1_B1_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017278, upper bound: 0.0016890
IS_A1_B1_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017278, upper bound: 0.0017115
IS_A1_B1_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017278, upper bound: 0.0017115
IS_A1_B1_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017238, upper bound: 0.0016778
IS_A1_B1_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017238, upper bound: 0.0016778
IS_A1_B1_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017238, upper bound: 0.0017047
IS_A1_B1_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017238, upper bound: 0.0017047
IS_A1_B2_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0015181, upper bound: 0.0015710
IS_A1_B2_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014522, upper bound: 0.0015934
IS_A1_B2_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0015181, upper bound: 0.0015583
IS_A1_B2_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014522, upper bound: 0.0015757
IS_A1_B2_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014877, upper bound: 0.0015475
IS_A1_B2_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014150, upper bound: 0.0015562
IS_A1_B2_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014877, upper bound: 0.0015320
IS_A1_B2_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014150, upper bound: 0.0015384
IS_A1_B2_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017278, upper bound: 0.0016908
IS_A1_B2_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017278, upper bound: 0.0016821
IS_A1_B2_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017278, upper bound: 0.0017123
IS_A1_B2_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017278, upper bound: 0.0017042
IS_A1_B2_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017238, upper bound: 0.0016850
IS_A1_B2_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017238, upper bound: 0.0016728
IS_A1_B2_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017238, upper bound: 0.0017105
IS_A1_B2_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017238, upper bound: 0.0016981
IS_A1_B2_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0015140, upper bound: 0.0015596
IS_A1_B2_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014456, upper bound: 0.0015793
IS_A1_B2_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0015140, upper bound: 0.0015458
IS_A1_B2_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014456, upper bound: 0.0015633
IS_A1_B2_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014811, upper bound: 0.0015343
IS_A1_B2_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014098, upper bound: 0.0015426
IS_A1_B2_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014811, upper bound: 0.0015147
IS_A1_B2_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014098, upper bound: 0.0015244
IS_A1_B2_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017278, upper bound: 0.0016982
IS_A1_B2_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017278, upper bound: 0.0016867
IS_A1_B2_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017278, upper bound: 0.0017222
IS_A1_B2_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017278, upper bound: 0.0017108
IS_A1_B2_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017238, upper bound: 0.0016897
IS_A1_B2_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017238, upper bound: 0.0016768
IS_A1_B2_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017238, upper bound: 0.0017174
IS_A1_B2_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017238, upper bound: 0.0017049
IS_A2_A1_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0013992, upper bound: 0.0016005
IS_A2_A1_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0013866, upper bound: 0.0015357
IS_A2_A1_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0013992, upper bound: 0.0016224
IS_A2_A1_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0013866, upper bound: 0.0015500
IS_A2_A1_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0013992, upper bound: 0.0015969
IS_A2_A1_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0013866, upper bound: 0.0015273
IS_A2_A1_A1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0013992, upper bound: 0.0016183
IS_A2_A1_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0013866, upper bound: 0.0015452
IS_A2_A1_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0016779
IS_A2_A1_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0016866
IS_A2_A1_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0017010
IS_A2_A1_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0017100
IS_A2_A1_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0016655
IS_A2_A1_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0016774
IS_A2_A1_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0016893
IS_A2_A1_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0017018
IS_A2_A1_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014593, upper bound: 0.0015138
IS_A2_A1_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0013798, upper bound: 0.0015214
IS_A2_A1_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014593, upper bound: 0.0015272
IS_A2_A1_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0013798, upper bound: 0.0015370
IS_A2_A1_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014593, upper bound: 0.0015051
IS_A2_A1_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0013798, upper bound: 0.0015140
IS_A2_A1_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014593, upper bound: 0.0015215
IS_A2_A1_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0013798, upper bound: 0.0015315
IS_A2_A1_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0016828
IS_A2_A1_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0016944
IS_A2_A1_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0017073
IS_A2_A1_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0017199
IS_A2_A1_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0016703
IS_A2_A1_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0016840
IS_A2_A1_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0016959
IS_A2_A1_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017474, upper bound: 0.0017107
IS_A2_A2_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014082, upper bound: 0.0016003
IS_A2_A2_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0013972, upper bound: 0.0015357
IS_A2_A2_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014082, upper bound: 0.0016007
IS_A2_A2_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0013972, upper bound: 0.0015370
IS_A2_A2_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014082, upper bound: 0.0015967
IS_A2_A2_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0013972, upper bound: 0.0015273
IS_A2_A2_A1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014082, upper bound: 0.0015976
IS_A2_A2_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0013972, upper bound: 0.0015294
IS_A2_A2_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017653, upper bound: 0.0016779
IS_A2_A2_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017653, upper bound: 0.0016781
IS_A2_A2_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017653, upper bound: 0.0017010
IS_A2_A2_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017653, upper bound: 0.0017019
IS_A2_A2_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017653, upper bound: 0.0016655
IS_A2_A2_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017653, upper bound: 0.0016659
IS_A2_A2_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017653, upper bound: 0.0016893
IS_A2_A2_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017653, upper bound: 0.0016905
IS_A2_A2_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014775, upper bound: 0.0015138
IS_A2_A2_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0013915, upper bound: 0.0015214
IS_A2_A2_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014775, upper bound: 0.0015140
IS_A2_A2_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0013915, upper bound: 0.0015219
IS_A2_A2_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014775, upper bound: 0.0015051
IS_A2_A2_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0013915, upper bound: 0.0015139
IS_A2_A2_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0014775, upper bound: 0.0015059
IS_A2_A2_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0013915, upper bound: 0.0015151
IS_A2_A2_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017653, upper bound: 0.0016828
IS_A2_A2_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017653, upper bound: 0.0016830
IS_A2_A2_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017653, upper bound: 0.0017073
IS_A2_A2_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017653, upper bound: 0.0017086
IS_A2_A2_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017653, upper bound: 0.0016703
IS_A2_A2_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017653, upper bound: 0.0016704
IS_A2_A2_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017653, upper bound: 0.0016959
IS_A2_A2_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 8, lower bound: -0.0017653, upper bound: 0.0016972

## BFS IS instance: IS_A1_B1_A1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0028322, 0.0049297, 0.0027717, 0.0049103, -0.0010250, 0.0010536
1: 0.0017315, 0.0020345, 0.0017227, 0.0020317, -0.0001481, 0.0001522
2: 0.0116343, 0.0127940, 0.0116451, 0.0128274, -0.0005825, 0.0005667
3: -0.0026477, -0.0014483, -0.0026366, -0.0014137, -0.0006025, 0.0005861
4: -0.0024691, -0.0011707, -0.0025065, -0.0011827, -0.0006345, 0.0006522
5: 0.0052212, 0.0064499, 0.0052326, 0.0064854, -0.0006172, 0.0006005
6: -0.0015840, 0.0032911, -0.0015389, 0.0034317, -0.0024490, 0.0023825
7: -0.0070390, -0.0003994, -0.0072304, -0.0004608, -0.0032447, 0.0033353
8: 0.9842555, 0.9889325, 0.9841206, 0.9888893, -0.0022857, 0.0023494
9: -0.0058410, -0.0015954, -0.0058017, -0.0014730, -0.0021327, 0.0020748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015218, upper bound: 0.0015589
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015218, upper bound: 0.0015589
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0028674, 0.0049256, 0.0028145, 0.0050245, -0.0011237, 0.0010440
1: 0.0017366, 0.0020339, 0.0017289, 0.0020482, -0.0001623, 0.0001508
2: 0.0116366, 0.0127746, 0.0115820, 0.0128038, -0.0005772, 0.0006213
3: -0.0026453, -0.0014684, -0.0027018, -0.0014382, -0.0005970, 0.0006425
4: -0.0024473, -0.0011733, -0.0024800, -0.0011121, -0.0006956, 0.0006462
5: 0.0052237, 0.0064293, 0.0051657, 0.0064603, -0.0006116, 0.0006583
6: -0.0015744, 0.0032094, -0.0018042, 0.0033323, -0.0024265, 0.0026118
7: -0.0069276, -0.0004125, -0.0070950, -0.0000995, -0.0035570, 0.0033047
8: 0.9843339, 0.9889233, 0.9842160, 0.9891437, -0.0025056, 0.0023279
9: -0.0058326, -0.0016666, -0.0060327, -0.0015596, -0.0021131, 0.0022744

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014553, upper bound: 0.0015757
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014553, upper bound: 0.0015757
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0028457, 0.0049326, 0.0027309, 0.0049143, -0.0010980, 0.0011908
1: 0.0017334, 0.0020349, 0.0017168, 0.0020323, -0.0001586, 0.0001720
2: 0.0116327, 0.0127866, 0.0116429, 0.0128500, -0.0006584, 0.0006071
3: -0.0026493, -0.0014560, -0.0026389, -0.0013904, -0.0006809, 0.0006278
4: -0.0024608, -0.0011689, -0.0025318, -0.0011802, -0.0006797, 0.0007371
5: 0.0052195, 0.0064421, 0.0052302, 0.0065093, -0.0006976, 0.0006432
6: -0.0015908, 0.0032599, -0.0015482, 0.0035266, -0.0027678, 0.0025521
7: -0.0069964, -0.0003902, -0.0073596, -0.0004482, -0.0034757, 0.0037695
8: 0.9842854, 0.9889390, 0.9840297, 0.9888982, -0.0024484, 0.0026553
9: -0.0058469, -0.0016227, -0.0058098, -0.0013904, -0.0024103, 0.0022225

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014656, upper bound: 0.0016134
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014656, upper bound: 0.0016134
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0028921, 0.0050445, 0.0027652, 0.0049097, -0.0010876, 0.0012869
1: 0.0017401, 0.0020511, 0.0017218, 0.0020316, -0.0001571, 0.0001859
2: 0.0115709, 0.0127609, 0.0116454, 0.0128311, -0.0007115, 0.0006013
3: -0.0027133, -0.0014826, -0.0026362, -0.0014100, -0.0007359, 0.0006219
4: -0.0024320, -0.0010996, -0.0025106, -0.0011831, -0.0006732, 0.0007966
5: 0.0051540, 0.0064148, 0.0052329, 0.0064892, -0.0007539, 0.0006371
6: -0.0018509, 0.0031519, -0.0015375, 0.0034470, -0.0029911, 0.0025278
7: -0.0068493, -0.0000360, -0.0072513, -0.0004627, -0.0034427, 0.0040736
8: 0.9843891, 0.9891886, 0.9841059, 0.9888879, -0.0024251, 0.0028695
9: -0.0060734, -0.0017167, -0.0058005, -0.0014597, -0.0026048, 0.0022013

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014608, upper bound: 0.0015757
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A1_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014608, upper bound: 0.0015757
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0028322, 0.0049297, 0.0028024, 0.0050146, -0.0012565, 0.0011468
1: 0.0017315, 0.0020345, 0.0017272, 0.0020468, -0.0001815, 0.0001657
2: 0.0116343, 0.0127940, 0.0115874, 0.0128105, -0.0006340, 0.0006947
3: -0.0026477, -0.0014483, -0.0026962, -0.0014312, -0.0006557, 0.0007185
4: -0.0024691, -0.0011707, -0.0024876, -0.0011182, -0.0007778, 0.0007099
5: 0.0052212, 0.0064499, 0.0051715, 0.0064674, -0.0006718, 0.0007361
6: -0.0015840, 0.0032911, -0.0017813, 0.0033605, -0.0026654, 0.0029204
7: -0.0070390, -0.0003994, -0.0071335, -0.0001307, -0.0039774, 0.0036300
8: 0.9842555, 0.9889325, 0.9841889, 0.9891218, -0.0028018, 0.0025571
9: -0.0058410, -0.0015954, -0.0060128, -0.0015350, -0.0023211, 0.0025433

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014891, upper bound: 0.0015285
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014891, upper bound: 0.0015285
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0028674, 0.0049256, 0.0028389, 0.0051374, -0.0013521, 0.0011334
1: 0.0017366, 0.0020339, 0.0017324, 0.0020645, -0.0001953, 0.0001637
2: 0.0116366, 0.0127746, 0.0115195, 0.0127903, -0.0006266, 0.0007475
3: -0.0026453, -0.0014684, -0.0027664, -0.0014521, -0.0006481, 0.0007731
4: -0.0024473, -0.0011733, -0.0024650, -0.0010421, -0.0008370, 0.0007016
5: 0.0052237, 0.0064293, 0.0050995, 0.0064460, -0.0006640, 0.0007921
6: -0.0015744, 0.0032094, -0.0020668, 0.0032757, -0.0026344, 0.0031427
7: -0.0069276, -0.0004125, -0.0070179, 0.0002581, -0.0042800, 0.0035878
8: 0.9843339, 0.9889233, 0.9842703, 0.9893957, -0.0030149, 0.0025273
9: -0.0058326, -0.0016666, -0.0062614, -0.0016089, -0.0022941, 0.0027368

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014184, upper bound: 0.0015351
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014184, upper bound: 0.0015351
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0028064, 0.0049363, 0.0028024, 0.0050146, -0.0013687, 0.0012450
1: 0.0017278, 0.0020355, 0.0017272, 0.0020468, -0.0001977, 0.0001799
2: 0.0116307, 0.0128083, 0.0115874, 0.0128105, -0.0006883, 0.0007567
3: -0.0026514, -0.0014336, -0.0026962, -0.0014312, -0.0007119, 0.0007826
4: -0.0024850, -0.0011666, -0.0024876, -0.0011182, -0.0008472, 0.0007707
5: 0.0052174, 0.0064650, 0.0051715, 0.0064674, -0.0007293, 0.0008018
6: -0.0015993, 0.0033510, -0.0017813, 0.0033605, -0.0028937, 0.0031811
7: -0.0071205, -0.0003786, -0.0071335, -0.0001307, -0.0043324, 0.0039409
8: 0.9841980, 0.9889472, 0.9841889, 0.9891218, -0.0030519, 0.0027761
9: -0.0058543, -0.0015433, -0.0060128, -0.0015350, -0.0025199, 0.0027703

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014996, upper bound: 0.0015285
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014996, upper bound: 0.0015285
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0028428, 0.0049323, 0.0028389, 0.0051374, -0.0014635, 0.0012312
1: 0.0017330, 0.0020349, 0.0017324, 0.0020645, -0.0002114, 0.0001779
2: 0.0116329, 0.0127882, 0.0115195, 0.0127903, -0.0006807, 0.0008091
3: -0.0026491, -0.0014543, -0.0027664, -0.0014521, -0.0007040, 0.0008368
4: -0.0024626, -0.0011691, -0.0024650, -0.0010421, -0.0009059, 0.0007621
5: 0.0052197, 0.0064438, 0.0050995, 0.0064460, -0.0007212, 0.0008573
6: -0.0015900, 0.0032666, -0.0020668, 0.0032757, -0.0028616, 0.0034016
7: -0.0070056, -0.0003913, -0.0070179, 0.0002581, -0.0046327, 0.0038972
8: 0.9842790, 0.9889383, 0.9842703, 0.9893957, -0.0032633, 0.0027453
9: -0.0058462, -0.0016168, -0.0062614, -0.0016089, -0.0024920, 0.0029623

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014220, upper bound: 0.0015351
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014220, upper bound: 0.0015351
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0026946, 0.0048765, 0.0028264, 0.0049708, -0.0012235, 0.0009950
1: 0.0017116, 0.0020268, 0.0017306, 0.0020404, -0.0001768, 0.0001438
2: 0.0116638, 0.0128701, 0.0116116, 0.0127972, -0.0005501, 0.0006765
3: -0.0026172, -0.0013696, -0.0026712, -0.0014449, -0.0005690, 0.0006996
4: -0.0025543, -0.0012036, -0.0024727, -0.0011453, -0.0007574, 0.0006159
5: 0.0052524, 0.0065305, 0.0051971, 0.0064534, -0.0005829, 0.0007167
6: -0.0014604, 0.0036109, -0.0016796, 0.0033048, -0.0023127, 0.0028438
7: -0.0074745, -0.0005678, -0.0070575, -0.0002693, -0.0038731, 0.0031497
8: 0.9839487, 0.9888139, 0.9842424, 0.9890242, -0.0027283, 0.0022187
9: -0.0057333, -0.0013170, -0.0059242, -0.0015836, -0.0020140, 0.0024765

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016101, upper bound: 0.0014682
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015643, upper bound: 0.0014604
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0026621, 0.0048856, 0.0028264, 0.0049708, -0.0013252, 0.0010784
1: 0.0017069, 0.0020281, 0.0017306, 0.0020404, -0.0001915, 0.0001558
2: 0.0116587, 0.0128881, 0.0116116, 0.0127972, -0.0005962, 0.0007327
3: -0.0026225, -0.0013510, -0.0026712, -0.0014449, -0.0006166, 0.0007578
4: -0.0025744, -0.0011980, -0.0024727, -0.0011453, -0.0008203, 0.0006675
5: 0.0052470, 0.0065496, 0.0051971, 0.0064534, -0.0006317, 0.0007763
6: -0.0014816, 0.0036866, -0.0016796, 0.0033048, -0.0025065, 0.0030801
7: -0.0075775, -0.0005390, -0.0070575, -0.0002693, -0.0041948, 0.0034136
8: 0.9838761, 0.9888342, 0.9842424, 0.9890242, -0.0029549, 0.0024046
9: -0.0057517, -0.0012511, -0.0059242, -0.0015836, -0.0021828, 0.0026823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016101, upper bound: 0.0014682
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015643, upper bound: 0.0014604
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0026946, 0.0048765, 0.0026899, 0.0049152, -0.0010371, 0.0009927
1: 0.0017116, 0.0020268, 0.0017109, 0.0020324, -0.0001498, 0.0001434
2: 0.0116638, 0.0128701, 0.0116424, 0.0128727, -0.0005488, 0.0005734
3: -0.0026172, -0.0013696, -0.0026394, -0.0013669, -0.0005676, 0.0005930
4: -0.0025543, -0.0012036, -0.0025572, -0.0011797, -0.0006420, 0.0006145
5: 0.0052524, 0.0065305, 0.0052297, 0.0065333, -0.0005815, 0.0006075
6: -0.0014604, 0.0036109, -0.0015503, 0.0036219, -0.0023072, 0.0024104
7: -0.0074745, -0.0005678, -0.0074895, -0.0004454, -0.0032828, 0.0031422
8: 0.9839487, 0.9888139, 0.9839381, 0.9889001, -0.0023125, 0.0022135
9: -0.0057333, -0.0013170, -0.0058116, -0.0013074, -0.0020092, 0.0020991

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016876, upper bound: 0.0016081
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016876, upper bound: 0.0016430
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0026621, 0.0048856, 0.0026899, 0.0049152, -0.0011500, 0.0010918
1: 0.0017069, 0.0020281, 0.0017109, 0.0020324, -0.0001661, 0.0001577
2: 0.0116587, 0.0128881, 0.0116424, 0.0128727, -0.0006036, 0.0006358
3: -0.0026225, -0.0013510, -0.0026394, -0.0013669, -0.0006243, 0.0006576
4: -0.0025744, -0.0011980, -0.0025572, -0.0011797, -0.0007118, 0.0006759
5: 0.0052470, 0.0065496, 0.0052297, 0.0065333, -0.0006396, 0.0006736
6: -0.0014816, 0.0036866, -0.0015503, 0.0036219, -0.0025377, 0.0026728
7: -0.0075775, -0.0005390, -0.0074895, -0.0004454, -0.0036401, 0.0034561
8: 0.9838761, 0.9888342, 0.9839381, 0.9889001, -0.0025642, 0.0024346
9: -0.0057517, -0.0012511, -0.0058116, -0.0013074, -0.0022099, 0.0023276

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016876, upper bound: 0.0016081
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016876, upper bound: 0.0016430
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0026946, 0.0048765, 0.0028623, 0.0050638, -0.0014836, 0.0010870
1: 0.0017116, 0.0020268, 0.0017358, 0.0020539, -0.0002143, 0.0001570
2: 0.0116638, 0.0128701, 0.0115602, 0.0127774, -0.0006010, 0.0008203
3: -0.0026172, -0.0013696, -0.0027243, -0.0014655, -0.0006216, 0.0008483
4: -0.0025543, -0.0012036, -0.0024505, -0.0010877, -0.0009184, 0.0006729
5: 0.0052524, 0.0065305, 0.0051427, 0.0064323, -0.0006368, 0.0008691
6: -0.0014604, 0.0036109, -0.0018957, 0.0032212, -0.0025265, 0.0034483
7: -0.0074745, -0.0005678, -0.0069437, 0.0000251, -0.0046963, 0.0034409
8: 0.9839487, 0.9888139, 0.9843227, 0.9892315, -0.0033082, 0.0024239
9: -0.0057333, -0.0013170, -0.0061124, -0.0016564, -0.0022002, 0.0030030

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015964, upper bound: 0.0014057
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015219, upper bound: 0.0013899
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0026621, 0.0048856, 0.0028623, 0.0050638, -0.0015853, 0.0011704
1: 0.0017069, 0.0020281, 0.0017358, 0.0020539, -0.0002290, 0.0001691
2: 0.0116587, 0.0128881, 0.0115602, 0.0127774, -0.0006471, 0.0008764
3: -0.0026225, -0.0013510, -0.0027243, -0.0014655, -0.0006692, 0.0009065
4: -0.0025744, -0.0011980, -0.0024505, -0.0010877, -0.0009813, 0.0007245
5: 0.0052470, 0.0065496, 0.0051427, 0.0064323, -0.0006856, 0.0009286
6: -0.0014816, 0.0036866, -0.0018957, 0.0032212, -0.0027203, 0.0036846
7: -0.0075775, -0.0005390, -0.0069437, 0.0000251, -0.0050181, 0.0037048
8: 0.9838761, 0.9888342, 0.9843227, 0.9892315, -0.0035348, 0.0026098
9: -0.0057517, -0.0012511, -0.0061124, -0.0016564, -0.0023690, 0.0032087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015964, upper bound: 0.0014057
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015219, upper bound: 0.0013899
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0026946, 0.0048765, 0.0027117, 0.0050193, -0.0012974, 0.0010870
1: 0.0017116, 0.0020268, 0.0017141, 0.0020474, -0.0001874, 0.0001570
2: 0.0116638, 0.0128701, 0.0115848, 0.0128607, -0.0006010, 0.0007173
3: -0.0026172, -0.0013696, -0.0026989, -0.0013794, -0.0006216, 0.0007419
4: -0.0025543, -0.0012036, -0.0025437, -0.0011153, -0.0008031, 0.0006729
5: 0.0052524, 0.0065305, 0.0051688, 0.0065206, -0.0006368, 0.0007600
6: -0.0014604, 0.0036109, -0.0017922, 0.0035714, -0.0025265, 0.0030156
7: -0.0074745, -0.0005678, -0.0074206, -0.0001159, -0.0041070, 0.0034408
8: 0.9839487, 0.9888139, 0.9839866, 0.9891323, -0.0028930, 0.0024238
9: -0.0057333, -0.0013170, -0.0060223, -0.0013514, -0.0022002, 0.0026261

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016851, upper bound: 0.0016028
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016851, upper bound: 0.0016362
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0026621, 0.0048856, 0.0027117, 0.0050193, -0.0014103, 0.0011862
1: 0.0017069, 0.0020281, 0.0017141, 0.0020474, -0.0002038, 0.0001714
2: 0.0116587, 0.0128881, 0.0115848, 0.0128607, -0.0006558, 0.0007797
3: -0.0026225, -0.0013510, -0.0026989, -0.0013794, -0.0006783, 0.0008064
4: -0.0025744, -0.0011980, -0.0025437, -0.0011153, -0.0008730, 0.0007342
5: 0.0052470, 0.0065496, 0.0051688, 0.0065206, -0.0006948, 0.0008262
6: -0.0014816, 0.0036866, -0.0017922, 0.0035714, -0.0027569, 0.0032780
7: -0.0075775, -0.0005390, -0.0074206, -0.0001159, -0.0044643, 0.0037547
8: 0.9838761, 0.9888342, 0.9839866, 0.9891323, -0.0031448, 0.0026449
9: -0.0057517, -0.0012511, -0.0060223, -0.0013514, -0.0024009, 0.0028546

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016851, upper bound: 0.0016028
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016851, upper bound: 0.0016362
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0027870, 0.0049127, 0.0027734, 0.0048878, -0.0011252, 0.0010611
1: 0.0017249, 0.0020320, 0.0017230, 0.0020284, -0.0001626, 0.0001533
2: 0.0116437, 0.0128190, 0.0116575, 0.0128265, -0.0005867, 0.0006221
3: -0.0026380, -0.0014225, -0.0026237, -0.0014147, -0.0006068, 0.0006434
4: -0.0024971, -0.0011812, -0.0025055, -0.0011967, -0.0006965, 0.0006569
5: 0.0052312, 0.0064764, 0.0052458, 0.0064844, -0.0006216, 0.0006592
6: -0.0015445, 0.0033962, -0.0014865, 0.0034279, -0.0024664, 0.0026154
7: -0.0071820, -0.0004532, -0.0072252, -0.0005322, -0.0035619, 0.0033590
8: 0.9841547, 0.9888946, 0.9841244, 0.9888390, -0.0025091, 0.0023662
9: -0.0058066, -0.0015040, -0.0057560, -0.0014764, -0.0021479, 0.0022776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015174, upper bound: 0.0014477
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015174, upper bound: 0.0015464
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0028216, 0.0049089, 0.0028163, 0.0050025, -0.0012241, 0.0010515
1: 0.0017299, 0.0020315, 0.0017292, 0.0020450, -0.0001768, 0.0001519
2: 0.0116459, 0.0127999, 0.0115941, 0.0128028, -0.0005814, 0.0006768
3: -0.0026357, -0.0014422, -0.0026893, -0.0014392, -0.0006013, 0.0006999
4: -0.0024757, -0.0011836, -0.0024790, -0.0011257, -0.0007577, 0.0006509
5: 0.0052335, 0.0064562, 0.0051786, 0.0064593, -0.0006160, 0.0007171
6: -0.0015355, 0.0033158, -0.0017531, 0.0033282, -0.0024440, 0.0028451
7: -0.0070726, -0.0004655, -0.0070895, -0.0001692, -0.0038748, 0.0033286
8: 0.9842318, 0.9888860, 0.9842200, 0.9890947, -0.0027295, 0.0023447
9: -0.0057987, -0.0015740, -0.0059882, -0.0015632, -0.0021284, 0.0024776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_A2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014329, upper bound: 0.0014329
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A2_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014329, upper bound: 0.0015644
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0027611, 0.0049192, 0.0027734, 0.0048878, -0.0012309, 0.0011588
1: 0.0017212, 0.0020330, 0.0017230, 0.0020284, -0.0001778, 0.0001674
2: 0.0116402, 0.0128333, 0.0116575, 0.0128265, -0.0006407, 0.0006806
3: -0.0026417, -0.0014077, -0.0026237, -0.0014147, -0.0006626, 0.0007039
4: -0.0025131, -0.0011772, -0.0025055, -0.0011967, -0.0007620, 0.0007173
5: 0.0052274, 0.0064916, 0.0052458, 0.0064844, -0.0006788, 0.0007211
6: -0.0015596, 0.0034563, -0.0014865, 0.0034279, -0.0026935, 0.0028610
7: -0.0072639, -0.0004327, -0.0072252, -0.0005322, -0.0038965, 0.0036682
8: 0.9840970, 0.9889091, 0.9841244, 0.9888390, -0.0027448, 0.0025840
9: -0.0058197, -0.0014516, -0.0057560, -0.0014764, -0.0023456, 0.0024915

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015249, upper bound: 0.0014476
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015249, upper bound: 0.0015458
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0027958, 0.0049155, 0.0028163, 0.0050025, -0.0013271, 0.0011486
1: 0.0017262, 0.0020325, 0.0017292, 0.0020450, -0.0001917, 0.0001659
2: 0.0116422, 0.0128141, 0.0115941, 0.0128028, -0.0006351, 0.0007337
3: -0.0026396, -0.0014275, -0.0026893, -0.0014392, -0.0006568, 0.0007588
4: -0.0024916, -0.0011795, -0.0024790, -0.0011257, -0.0008215, 0.0007110
5: 0.0052295, 0.0064713, 0.0051786, 0.0064593, -0.0006729, 0.0007774
6: -0.0015511, 0.0033757, -0.0017531, 0.0033282, -0.0026698, 0.0030844
7: -0.0071541, -0.0004443, -0.0070895, -0.0001692, -0.0042007, 0.0036360
8: 0.9841744, 0.9889009, 0.9842200, 0.9890947, -0.0029591, 0.0025613
9: -0.0058122, -0.0015218, -0.0059882, -0.0015632, -0.0023249, 0.0026861

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_A2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014356, upper bound: 0.0014327
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014356, upper bound: 0.0014327
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0027870, 0.0049127, 0.0028041, 0.0049948, -0.0013471, 0.0011543
1: 0.0017249, 0.0020320, 0.0017274, 0.0020439, -0.0001946, 0.0001668
2: 0.0116437, 0.0128190, 0.0115984, 0.0128096, -0.0006382, 0.0007448
3: -0.0026380, -0.0014225, -0.0026849, -0.0014322, -0.0006600, 0.0007703
4: -0.0024971, -0.0011812, -0.0024865, -0.0011304, -0.0008339, 0.0007145
5: 0.0052312, 0.0064764, 0.0051831, 0.0064664, -0.0006762, 0.0007891
6: -0.0015445, 0.0033962, -0.0017353, 0.0033566, -0.0026829, 0.0031309
7: -0.0071820, -0.0004532, -0.0071281, -0.0001933, -0.0042641, 0.0036538
8: 0.9841547, 0.9888946, 0.9841927, 0.9890777, -0.0030037, 0.0025738
9: -0.0058066, -0.0015040, -0.0059727, -0.0015384, -0.0023364, 0.0027266

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014793, upper bound: 0.0013759
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014793, upper bound: 0.0015127
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0028216, 0.0049089, 0.0028406, 0.0051186, -0.0014427, 0.0011410
1: 0.0017299, 0.0020315, 0.0017327, 0.0020618, -0.0002084, 0.0001648
2: 0.0116459, 0.0127999, 0.0115299, 0.0127894, -0.0006308, 0.0007976
3: -0.0026357, -0.0014422, -0.0027557, -0.0014531, -0.0006524, 0.0008249
4: -0.0024757, -0.0011836, -0.0024639, -0.0010538, -0.0008930, 0.0007063
5: 0.0052335, 0.0064562, 0.0051106, 0.0064450, -0.0006684, 0.0008451
6: -0.0015355, 0.0033158, -0.0020230, 0.0032716, -0.0026519, 0.0033532
7: -0.0070726, -0.0004655, -0.0070123, 0.0001985, -0.0045667, 0.0036117
8: 0.9842318, 0.9888860, 0.9842743, 0.9893537, -0.0032169, 0.0025442
9: -0.0057987, -0.0015740, -0.0062233, -0.0016125, -0.0023094, 0.0029201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_A2_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0011224, upper bound: 0.0012648
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0011177, upper bound: 0.0012508
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0027611, 0.0049192, 0.0028041, 0.0049948, -0.0014528, 0.0012520
1: 0.0017212, 0.0020330, 0.0017274, 0.0020439, -0.0002099, 0.0001809
2: 0.0116402, 0.0128333, 0.0115984, 0.0128096, -0.0006922, 0.0008032
3: -0.0026417, -0.0014077, -0.0026849, -0.0014322, -0.0007159, 0.0008307
4: -0.0025131, -0.0011772, -0.0024865, -0.0011304, -0.0008993, 0.0007750
5: 0.0052274, 0.0064916, 0.0051831, 0.0064664, -0.0007334, 0.0008510
6: -0.0015596, 0.0034563, -0.0017353, 0.0033566, -0.0029099, 0.0033766
7: -0.0072639, -0.0004327, -0.0071281, -0.0001933, -0.0045987, 0.0039631
8: 0.9840970, 0.9889091, 0.9841927, 0.9890777, -0.0032394, 0.0027917
9: -0.0058197, -0.0014516, -0.0059727, -0.0015384, -0.0025341, 0.0029405

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014932, upper bound: 0.0013759
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014932, upper bound: 0.0015127
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0027958, 0.0049155, 0.0028406, 0.0051186, -0.0015457, 0.0012381
1: 0.0017262, 0.0020325, 0.0017327, 0.0020618, -0.0002233, 0.0001789
2: 0.0116422, 0.0128141, 0.0115299, 0.0127894, -0.0006845, 0.0008546
3: -0.0026396, -0.0014275, -0.0027557, -0.0014531, -0.0007080, 0.0008838
4: -0.0024916, -0.0011795, -0.0024639, -0.0010538, -0.0009568, 0.0007664
5: 0.0052295, 0.0064713, 0.0051106, 0.0064450, -0.0007253, 0.0009054
6: -0.0015511, 0.0033757, -0.0020230, 0.0032716, -0.0028777, 0.0035925
7: -0.0071541, -0.0004443, -0.0070123, 0.0001985, -0.0048927, 0.0039191
8: 0.9841744, 0.9889009, 0.9842743, 0.9893537, -0.0034465, 0.0027607
9: -0.0058122, -0.0015218, -0.0062233, -0.0016125, -0.0025060, 0.0031285

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_A2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0011377, upper bound: 0.0012648
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0011298, upper bound: 0.0012508
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0026498, 0.0048553, 0.0028279, 0.0049515, -0.0013170, 0.0009940
1: 0.0017051, 0.0020238, 0.0017308, 0.0020376, -0.0001903, 0.0001436
2: 0.0116755, 0.0128948, 0.0116223, 0.0127964, -0.0005496, 0.0007281
3: -0.0026051, -0.0013440, -0.0026601, -0.0014458, -0.0005684, 0.0007531
4: -0.0025820, -0.0012168, -0.0024718, -0.0011572, -0.0008153, 0.0006153
5: 0.0052648, 0.0065568, 0.0052085, 0.0064525, -0.0005823, 0.0007715
6: -0.0014111, 0.0037151, -0.0016346, 0.0033012, -0.0023103, 0.0030611
7: -0.0076163, -0.0006349, -0.0070527, -0.0003305, -0.0041690, 0.0031465
8: 0.9838487, 0.9887666, 0.9842458, 0.9889810, -0.0029367, 0.0022164
9: -0.0056903, -0.0012263, -0.0058850, -0.0015866, -0.0020119, 0.0026658

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A2_B1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016101, upper bound: 0.0014582
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_A2_B1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015642, upper bound: 0.0014490
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0026157, 0.0048668, 0.0028279, 0.0049515, -0.0014195, 0.0010844
1: 0.0017002, 0.0020254, 0.0017308, 0.0020376, -0.0002051, 0.0001567
2: 0.0116692, 0.0129137, 0.0116223, 0.0127964, -0.0005996, 0.0007848
3: -0.0026117, -0.0013245, -0.0026601, -0.0014458, -0.0006201, 0.0008117
4: -0.0026031, -0.0012097, -0.0024718, -0.0011572, -0.0008787, 0.0006713
5: 0.0052581, 0.0065768, 0.0052085, 0.0064525, -0.0006353, 0.0008316
6: -0.0014377, 0.0037943, -0.0016346, 0.0033012, -0.0025205, 0.0032994
7: -0.0077242, -0.0005987, -0.0070527, -0.0003305, -0.0044934, 0.0034327
8: 0.9837727, 0.9887921, 0.9842458, 0.9889810, -0.0031653, 0.0024181
9: -0.0057135, -0.0011573, -0.0058850, -0.0015866, -0.0021950, 0.0028732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A2_B1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016101, upper bound: 0.0014582
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A2_A2_B1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015642, upper bound: 0.0014490
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0026498, 0.0048553, 0.0026915, 0.0048928, -0.0011542, 0.0009991
1: 0.0017051, 0.0020238, 0.0017112, 0.0020292, -0.0001668, 0.0001443
2: 0.0116755, 0.0128948, 0.0116548, 0.0128718, -0.0005524, 0.0006381
3: -0.0026051, -0.0013440, -0.0026266, -0.0013679, -0.0005713, 0.0006600
4: -0.0025820, -0.0012168, -0.0025562, -0.0011936, -0.0007145, 0.0006185
5: 0.0052648, 0.0065568, 0.0052429, 0.0065323, -0.0005853, 0.0006761
6: -0.0014111, 0.0037151, -0.0014982, 0.0036181, -0.0023222, 0.0026827
7: -0.0076163, -0.0006349, -0.0074843, -0.0005163, -0.0036537, 0.0031626
8: 0.9838487, 0.9887666, 0.9839419, 0.9888502, -0.0025737, 0.0022278
9: -0.0056903, -0.0012263, -0.0057662, -0.0013107, -0.0020223, 0.0023363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A2_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016876, upper bound: 0.0016110
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A2_A2_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016876, upper bound: 0.0016503
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0026157, 0.0048668, 0.0026915, 0.0048928, -0.0012595, 0.0010982
1: 0.0017002, 0.0020254, 0.0017112, 0.0020292, -0.0001820, 0.0001587
2: 0.0116692, 0.0129137, 0.0116548, 0.0128718, -0.0006072, 0.0006963
3: -0.0026117, -0.0013245, -0.0026266, -0.0013679, -0.0006280, 0.0007202
4: -0.0026031, -0.0012097, -0.0025562, -0.0011936, -0.0007796, 0.0006798
5: 0.0052581, 0.0065768, 0.0052429, 0.0065323, -0.0006433, 0.0007378
6: -0.0014377, 0.0037943, -0.0014982, 0.0036181, -0.0025526, 0.0029274
7: -0.0077242, -0.0005987, -0.0074843, -0.0005163, -0.0039869, 0.0034764
8: 0.9837727, 0.9887921, 0.9839419, 0.9888502, -0.0028084, 0.0024489
9: -0.0057135, -0.0011573, -0.0057662, -0.0013107, -0.0022229, 0.0025493

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A2_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016876, upper bound: 0.0016110
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A2_A2_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016876, upper bound: 0.0016503
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0026498, 0.0048553, 0.0028639, 0.0050437, -0.0015614, 0.0010860
1: 0.0017051, 0.0020238, 0.0017360, 0.0020510, -0.0002256, 0.0001569
2: 0.0116755, 0.0128948, 0.0115713, 0.0127765, -0.0006004, 0.0008633
3: -0.0026051, -0.0013440, -0.0027129, -0.0014664, -0.0006210, 0.0008928
4: -0.0025820, -0.0012168, -0.0024495, -0.0011001, -0.0009665, 0.0006722
5: 0.0052648, 0.0065568, 0.0051544, 0.0064314, -0.0006362, 0.0009147
6: -0.0014111, 0.0037151, -0.0018490, 0.0032176, -0.0025241, 0.0036291
7: -0.0076163, -0.0006349, -0.0069388, -0.0000385, -0.0049425, 0.0034376
8: 0.9838487, 0.9887666, 0.9843261, 0.9891867, -0.0034816, 0.0024216
9: -0.0056903, -0.0012263, -0.0060717, -0.0016595, -0.0021981, 0.0031604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015964, upper bound: 0.0013971
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015214, upper bound: 0.0013798
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0026157, 0.0048668, 0.0028639, 0.0050437, -0.0016639, 0.0011764
1: 0.0017002, 0.0020254, 0.0017360, 0.0020510, -0.0002404, 0.0001700
2: 0.0116692, 0.0129137, 0.0115713, 0.0127765, -0.0006504, 0.0009199
3: -0.0026117, -0.0013245, -0.0027129, -0.0014664, -0.0006727, 0.0009514
4: -0.0026031, -0.0012097, -0.0024495, -0.0011001, -0.0010300, 0.0007282
5: 0.0052581, 0.0065768, 0.0051544, 0.0064314, -0.0006891, 0.0009747
6: -0.0014377, 0.0037943, -0.0018490, 0.0032176, -0.0027343, 0.0038674
7: -0.0077242, -0.0005987, -0.0069388, -0.0000385, -0.0052670, 0.0037239
8: 0.9837727, 0.9887921, 0.9843261, 0.9891867, -0.0037102, 0.0026232
9: -0.0057135, -0.0011573, -0.0060717, -0.0016595, -0.0023812, 0.0033679

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015964, upper bound: 0.0013971
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015214, upper bound: 0.0013798
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0026498, 0.0048553, 0.0027133, 0.0049997, -0.0013986, 0.0010935
1: 0.0017051, 0.0020238, 0.0017143, 0.0020446, -0.0002021, 0.0001580
2: 0.0116755, 0.0128948, 0.0115956, 0.0128598, -0.0006046, 0.0007733
3: -0.0026051, -0.0013440, -0.0026877, -0.0013803, -0.0006253, 0.0007998
4: -0.0025820, -0.0012168, -0.0025427, -0.0011274, -0.0008658, 0.0006769
5: 0.0052648, 0.0065568, 0.0051802, 0.0065196, -0.0006406, 0.0008193
6: -0.0014111, 0.0037151, -0.0017468, 0.0035676, -0.0025416, 0.0032508
7: -0.0076163, -0.0006349, -0.0074154, -0.0001778, -0.0044273, 0.0034614
8: 0.9838487, 0.9887666, 0.9839904, 0.9890886, -0.0031187, 0.0024383
9: -0.0056903, -0.0012263, -0.0059827, -0.0013547, -0.0022133, 0.0028309

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016851, upper bound: 0.0016055
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016851, upper bound: 0.0016423
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0026157, 0.0048668, 0.0027133, 0.0049997, -0.0015039, 0.0011926
1: 0.0017002, 0.0020254, 0.0017143, 0.0020446, -0.0002173, 0.0001723
2: 0.0116692, 0.0129137, 0.0115956, 0.0128598, -0.0006594, 0.0008315
3: -0.0026117, -0.0013245, -0.0026877, -0.0013803, -0.0006820, 0.0008599
4: -0.0026031, -0.0012097, -0.0025427, -0.0011274, -0.0009309, 0.0007383
5: 0.0052581, 0.0065768, 0.0051802, 0.0065196, -0.0006986, 0.0008810
6: -0.0014377, 0.0037943, -0.0017468, 0.0035676, -0.0027720, 0.0034955
7: -0.0077242, -0.0005987, -0.0074154, -0.0001778, -0.0047605, 0.0037752
8: 0.9837727, 0.9887921, 0.9839904, 0.9890886, -0.0033534, 0.0026593
9: -0.0057135, -0.0011573, -0.0059827, -0.0013547, -0.0024140, 0.0030440

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016851, upper bound: 0.0016055
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016851, upper bound: 0.0016423
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0028322, 0.0049297, 0.0027441, 0.0049198, -0.0011132, 0.0011612
1: 0.0017315, 0.0020345, 0.0017187, 0.0020331, -0.0001608, 0.0001678
2: 0.0116343, 0.0127940, 0.0116398, 0.0128427, -0.0006420, 0.0006155
3: -0.0026477, -0.0014483, -0.0026420, -0.0013979, -0.0006640, 0.0006365
4: -0.0024691, -0.0011707, -0.0025237, -0.0011768, -0.0006891, 0.0007188
5: 0.0052212, 0.0064499, 0.0052270, 0.0065016, -0.0006802, 0.0006521
6: -0.0015840, 0.0032911, -0.0015611, 0.0034961, -0.0026990, 0.0025874
7: -0.0070390, -0.0003994, -0.0073180, -0.0004307, -0.0035238, 0.0036758
8: 0.9842555, 0.9889325, 0.9840589, 0.9889106, -0.0024822, 0.0025893
9: -0.0058410, -0.0015954, -0.0058210, -0.0014170, -0.0023504, 0.0022532

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015216, upper bound: 0.0015710
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015216, upper bound: 0.0015710
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0028674, 0.0049256, 0.0027861, 0.0050390, -0.0012047, 0.0011516
1: 0.0017366, 0.0020339, 0.0017248, 0.0020503, -0.0001740, 0.0001664
2: 0.0116366, 0.0127746, 0.0115739, 0.0128195, -0.0006367, 0.0006660
3: -0.0026453, -0.0014684, -0.0027102, -0.0014219, -0.0006585, 0.0006889
4: -0.0024473, -0.0011733, -0.0024976, -0.0011030, -0.0007457, 0.0007128
5: 0.0052237, 0.0064293, 0.0051572, 0.0064770, -0.0006746, 0.0007057
6: -0.0015744, 0.0032094, -0.0018381, 0.0033983, -0.0026766, 0.0028000
7: -0.0069276, -0.0004125, -0.0071850, -0.0000533, -0.0038134, 0.0036452
8: 0.9843339, 0.9889233, 0.9841526, 0.9891763, -0.0026862, 0.0025678
9: -0.0058326, -0.0016666, -0.0060622, -0.0015021, -0.0023309, 0.0024384

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014550, upper bound: 0.0015934
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014550, upper bound: 0.0015934
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0028064, 0.0049363, 0.0027441, 0.0049198, -0.0010411, 0.0010685
1: 0.0017278, 0.0020355, 0.0017187, 0.0020331, -0.0001504, 0.0001544
2: 0.0116307, 0.0128083, 0.0116398, 0.0128427, -0.0005907, 0.0005756
3: -0.0026514, -0.0014336, -0.0026420, -0.0013979, -0.0006110, 0.0005953
4: -0.0024850, -0.0011666, -0.0025237, -0.0011768, -0.0006444, 0.0006614
5: 0.0052174, 0.0064650, 0.0052270, 0.0065016, -0.0006259, 0.0006099
6: -0.0015993, 0.0033510, -0.0015611, 0.0034961, -0.0024835, 0.0024198
7: -0.0071205, -0.0003786, -0.0073180, -0.0004307, -0.0032955, 0.0033823
8: 0.9841980, 0.9889472, 0.9840589, 0.9889106, -0.0023214, 0.0023826
9: -0.0058543, -0.0015433, -0.0058210, -0.0014170, -0.0021627, 0.0021072

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015286, upper bound: 0.0015583
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015286, upper bound: 0.0015583
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0028428, 0.0049323, 0.0027861, 0.0050390, -0.0011405, 0.0010589
1: 0.0017330, 0.0020349, 0.0017248, 0.0020503, -0.0001648, 0.0001530
2: 0.0116329, 0.0127882, 0.0115739, 0.0128195, -0.0005855, 0.0006305
3: -0.0026491, -0.0014543, -0.0027102, -0.0014219, -0.0006055, 0.0006521
4: -0.0024626, -0.0011691, -0.0024976, -0.0011030, -0.0007060, 0.0006555
5: 0.0052197, 0.0064438, 0.0051572, 0.0064770, -0.0006203, 0.0006681
6: -0.0015900, 0.0032666, -0.0018381, 0.0033983, -0.0024613, 0.0026507
7: -0.0070056, -0.0003913, -0.0071850, -0.0000533, -0.0036101, 0.0033520
8: 0.9842790, 0.9889383, 0.9841526, 0.9891763, -0.0025430, 0.0023613
9: -0.0058462, -0.0016168, -0.0060622, -0.0015021, -0.0021434, 0.0023084

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014608, upper bound: 0.0015757
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B2_A1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014608, upper bound: 0.0015757
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0028322, 0.0049297, 0.0027691, 0.0050339, -0.0013387, 0.0012439
1: 0.0017315, 0.0020345, 0.0017223, 0.0020495, -0.0001934, 0.0001797
2: 0.0116343, 0.0127940, 0.0115768, 0.0128289, -0.0006877, 0.0007401
3: -0.0026477, -0.0014483, -0.0027072, -0.0014122, -0.0007113, 0.0007655
4: -0.0024691, -0.0011707, -0.0025082, -0.0011062, -0.0008287, 0.0007700
5: 0.0052212, 0.0064499, 0.0051602, 0.0064869, -0.0007287, 0.0007842
6: -0.0015840, 0.0032911, -0.0018261, 0.0034379, -0.0028911, 0.0031115
7: -0.0070390, -0.0003994, -0.0072389, -0.0000698, -0.0042376, 0.0039375
8: 0.9842555, 0.9889325, 0.9841146, 0.9891647, -0.0029851, 0.0027736
9: -0.0058410, -0.0015954, -0.0060517, -0.0014676, -0.0025177, 0.0027097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014891, upper bound: 0.0015475
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014891, upper bound: 0.0015475
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0028674, 0.0049256, 0.0028082, 0.0051544, -0.0014255, 0.0012279
1: 0.0017366, 0.0020339, 0.0017280, 0.0020670, -0.0002059, 0.0001774
2: 0.0116366, 0.0127746, 0.0115101, 0.0128073, -0.0006789, 0.0007881
3: -0.0026453, -0.0014684, -0.0027761, -0.0014346, -0.0007021, 0.0008151
4: -0.0024473, -0.0011733, -0.0024839, -0.0010316, -0.0008824, 0.0007601
5: 0.0052237, 0.0064293, 0.0050896, 0.0064640, -0.0007193, 0.0008351
6: -0.0015744, 0.0032094, -0.0021062, 0.0033469, -0.0028540, 0.0033133
7: -0.0069276, -0.0004125, -0.0071149, 0.0003118, -0.0045125, 0.0038868
8: 0.9843339, 0.9889233, 0.9842020, 0.9894335, -0.0031787, 0.0027380
9: -0.0058326, -0.0016666, -0.0062957, -0.0015469, -0.0024854, 0.0028854

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014184, upper bound: 0.0015562
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014184, upper bound: 0.0015562
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0028064, 0.0049363, 0.0027691, 0.0050339, -0.0012752, 0.0011620
1: 0.0017278, 0.0020355, 0.0017223, 0.0020495, -0.0001842, 0.0001679
2: 0.0116307, 0.0128083, 0.0115768, 0.0128289, -0.0006425, 0.0007050
3: -0.0026514, -0.0014336, -0.0027072, -0.0014122, -0.0006645, 0.0007292
4: -0.0024850, -0.0011666, -0.0025082, -0.0011062, -0.0007894, 0.0007193
5: 0.0052174, 0.0064650, 0.0051602, 0.0064869, -0.0006807, 0.0007470
6: -0.0015993, 0.0033510, -0.0018261, 0.0034379, -0.0027009, 0.0029640
7: -0.0071205, -0.0003786, -0.0072389, -0.0000698, -0.0040367, 0.0036784
8: 0.9841980, 0.9889472, 0.9841146, 0.9891647, -0.0028435, 0.0025911
9: -0.0058543, -0.0015433, -0.0060517, -0.0014676, -0.0023521, 0.0025812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014996, upper bound: 0.0015320
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014996, upper bound: 0.0015320
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0028428, 0.0049323, 0.0028082, 0.0051544, -0.0013711, 0.0011487
1: 0.0017330, 0.0020349, 0.0017280, 0.0020670, -0.0001981, 0.0001659
2: 0.0116329, 0.0127882, 0.0115101, 0.0128073, -0.0006351, 0.0007581
3: -0.0026491, -0.0014543, -0.0027761, -0.0014346, -0.0006568, 0.0007840
4: -0.0024626, -0.0011691, -0.0024839, -0.0010316, -0.0008488, 0.0007110
5: 0.0052197, 0.0064438, 0.0050896, 0.0064640, -0.0006729, 0.0008032
6: -0.0015900, 0.0032666, -0.0021062, 0.0033469, -0.0026698, 0.0031869
7: -0.0070056, -0.0003913, -0.0071149, 0.0003118, -0.0043403, 0.0036361
8: 0.9842790, 0.9889383, 0.9842020, 0.9894335, -0.0030574, 0.0025613
9: -0.0058462, -0.0016168, -0.0062957, -0.0015469, -0.0023250, 0.0027753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014220, upper bound: 0.0015384
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014220, upper bound: 0.0015384
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0026946, 0.0048765, 0.0028005, 0.0049763, -0.0013212, 0.0011073
1: 0.0017116, 0.0020268, 0.0017269, 0.0020412, -0.0001909, 0.0001600
2: 0.0116638, 0.0128701, 0.0116086, 0.0128115, -0.0006122, 0.0007305
3: -0.0026172, -0.0013696, -0.0026743, -0.0014302, -0.0006332, 0.0007555
4: -0.0025543, -0.0012036, -0.0024887, -0.0011419, -0.0008178, 0.0006854
5: 0.0052524, 0.0065305, 0.0051940, 0.0064685, -0.0006486, 0.0007740
6: -0.0014604, 0.0036109, -0.0016922, 0.0033648, -0.0025736, 0.0030708
7: -0.0074745, -0.0005678, -0.0071393, -0.0002521, -0.0041822, 0.0035051
8: 0.9839487, 0.9888139, 0.9841847, 0.9890364, -0.0029460, 0.0024691
9: -0.0057333, -0.0013170, -0.0059352, -0.0015313, -0.0022412, 0.0026742

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A2_B1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016101, upper bound: 0.0014712
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015637, upper bound: 0.0014630
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0026621, 0.0048856, 0.0028005, 0.0049763, -0.0012385, 0.0010109
1: 0.0017069, 0.0020281, 0.0017269, 0.0020412, -0.0001789, 0.0001460
2: 0.0116587, 0.0128881, 0.0116086, 0.0128115, -0.0005589, 0.0006848
3: -0.0026225, -0.0013510, -0.0026743, -0.0014302, -0.0005780, 0.0007082
4: -0.0025744, -0.0011980, -0.0024887, -0.0011419, -0.0007667, 0.0006258
5: 0.0052470, 0.0065496, 0.0051940, 0.0064685, -0.0005922, 0.0007255
6: -0.0014816, 0.0036866, -0.0016922, 0.0033648, -0.0023496, 0.0028787
7: -0.0075775, -0.0005390, -0.0071393, -0.0002521, -0.0039206, 0.0032000
8: 0.9838761, 0.9888342, 0.9841847, 0.9890364, -0.0027617, 0.0022541
9: -0.0057517, -0.0012511, -0.0059352, -0.0015313, -0.0020462, 0.0025069

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A2_B1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016101, upper bound: 0.0014645
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015637, upper bound: 0.0014573
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0026946, 0.0048765, 0.0026573, 0.0049247, -0.0011357, 0.0011057
1: 0.0017116, 0.0020268, 0.0017062, 0.0020338, -0.0001641, 0.0001597
2: 0.0116638, 0.0128701, 0.0116371, 0.0128907, -0.0006113, 0.0006279
3: -0.0026172, -0.0013696, -0.0026448, -0.0013483, -0.0006322, 0.0006494
4: -0.0025543, -0.0012036, -0.0025774, -0.0011738, -0.0007030, 0.0006844
5: 0.0052524, 0.0065305, 0.0052242, 0.0065524, -0.0006477, 0.0006653
6: -0.0014604, 0.0036109, -0.0015724, 0.0036978, -0.0025699, 0.0026396
7: -0.0074745, -0.0005678, -0.0075928, -0.0004153, -0.0035949, 0.0035000
8: 0.9839487, 0.9888139, 0.9838653, 0.9889213, -0.0025323, 0.0024655
9: -0.0057333, -0.0013170, -0.0058308, -0.0012413, -0.0022380, 0.0022987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A2_B1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016876, upper bound: 0.0016194
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016876, upper bound: 0.0016517
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0026621, 0.0048856, 0.0026573, 0.0049247, -0.0010544, 0.0010097
1: 0.0017069, 0.0020281, 0.0017062, 0.0020338, -0.0001523, 0.0001459
2: 0.0116587, 0.0128881, 0.0116371, 0.0128907, -0.0005583, 0.0005829
3: -0.0026225, -0.0013510, -0.0026448, -0.0013483, -0.0005774, 0.0006029
4: -0.0025744, -0.0011980, -0.0025774, -0.0011738, -0.0006527, 0.0006250
5: 0.0052470, 0.0065496, 0.0052242, 0.0065524, -0.0005915, 0.0006177
6: -0.0014816, 0.0036866, -0.0015724, 0.0036978, -0.0023469, 0.0024507
7: -0.0075775, -0.0005390, -0.0075928, -0.0004153, -0.0033376, 0.0031963
8: 0.9838761, 0.9888342, 0.9838653, 0.9889213, -0.0023511, 0.0022515
9: -0.0057517, -0.0012511, -0.0058308, -0.0012413, -0.0020438, 0.0021342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A2_B1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016876, upper bound: 0.0016099
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016876, upper bound: 0.0016444
time: 1.18 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0026946, 0.0048765, 0.0028302, 0.0050791, -0.0015608, 0.0011773
1: 0.0017116, 0.0020268, 0.0017312, 0.0020561, -0.0002255, 0.0001701
2: 0.0116638, 0.0128701, 0.0115518, 0.0127951, -0.0006509, 0.0008629
3: -0.0026172, -0.0013696, -0.0027331, -0.0014472, -0.0006732, 0.0008925
4: -0.0025543, -0.0012036, -0.0024703, -0.0010782, -0.0009661, 0.0007287
5: 0.0052524, 0.0065305, 0.0051337, 0.0064511, -0.0006896, 0.0009143
6: -0.0014604, 0.0036109, -0.0019312, 0.0032957, -0.0027363, 0.0036276
7: -0.0074745, -0.0005678, -0.0070452, 0.0000735, -0.0049405, 0.0037266
8: 0.9839487, 0.9888139, 0.9842511, 0.9892656, -0.0034802, 0.0026251
9: -0.0057333, -0.0013170, -0.0061433, -0.0015915, -0.0023829, 0.0031591

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015963, upper bound: 0.0014131
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015219, upper bound: 0.0014003
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0026621, 0.0048856, 0.0028302, 0.0050791, -0.0015016, 0.0011024
1: 0.0017069, 0.0020281, 0.0017312, 0.0020561, -0.0002169, 0.0001593
2: 0.0116587, 0.0128881, 0.0115518, 0.0127951, -0.0006095, 0.0008302
3: -0.0026225, -0.0013510, -0.0027331, -0.0014472, -0.0006303, 0.0008587
4: -0.0025744, -0.0011980, -0.0024703, -0.0010782, -0.0009295, 0.0006824
5: 0.0052470, 0.0065496, 0.0051337, 0.0064511, -0.0006458, 0.0008797
6: -0.0014816, 0.0036866, -0.0019312, 0.0032957, -0.0025622, 0.0034902
7: -0.0075775, -0.0005390, -0.0070452, 0.0000735, -0.0047534, 0.0034895
8: 0.9838761, 0.9888342, 0.9842511, 0.9892656, -0.0033484, 0.0024581
9: -0.0057517, -0.0012511, -0.0061433, -0.0015915, -0.0022313, 0.0030394

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015963, upper bound: 0.0014042
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015219, upper bound: 0.0013912
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0026946, 0.0048765, 0.0026773, 0.0050386, -0.0013755, 0.0011771
1: 0.0017116, 0.0020268, 0.0017091, 0.0020502, -0.0001987, 0.0001701
2: 0.0116638, 0.0128701, 0.0115742, 0.0128796, -0.0006508, 0.0007605
3: -0.0026172, -0.0013696, -0.0027099, -0.0013597, -0.0006731, 0.0007865
4: -0.0025543, -0.0012036, -0.0025650, -0.0011033, -0.0008515, 0.0007287
5: 0.0052524, 0.0065305, 0.0051574, 0.0065407, -0.0006895, 0.0008058
6: -0.0014604, 0.0036109, -0.0018371, 0.0036512, -0.0027359, 0.0031971
7: -0.0074745, -0.0005678, -0.0075293, -0.0000548, -0.0043541, 0.0037261
8: 0.9839487, 0.9888139, 0.9839100, 0.9891753, -0.0030671, 0.0026247
9: -0.0057333, -0.0013170, -0.0060613, -0.0012819, -0.0023826, 0.0027841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016851, upper bound: 0.0016173
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016851, upper bound: 0.0016497
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0026621, 0.0048856, 0.0026773, 0.0050386, -0.0013179, 0.0011036
1: 0.0017069, 0.0020281, 0.0017091, 0.0020502, -0.0001904, 0.0001594
2: 0.0116587, 0.0128881, 0.0115742, 0.0128796, -0.0006102, 0.0007286
3: -0.0026225, -0.0013510, -0.0027099, -0.0013597, -0.0006311, 0.0007536
4: -0.0025744, -0.0011980, -0.0025650, -0.0011033, -0.0008158, 0.0006832
5: 0.0052470, 0.0065496, 0.0051574, 0.0065407, -0.0006465, 0.0007720
6: -0.0014816, 0.0036866, -0.0018371, 0.0036512, -0.0025651, 0.0030632
7: -0.0075775, -0.0005390, -0.0075293, -0.0000548, -0.0041718, 0.0034935
8: 0.9838761, 0.9888342, 0.9839100, 0.9891753, -0.0029387, 0.0024609
9: -0.0057517, -0.0012511, -0.0060613, -0.0012819, -0.0022338, 0.0026675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 2.97 + 598.32 = 601.28 seconds
