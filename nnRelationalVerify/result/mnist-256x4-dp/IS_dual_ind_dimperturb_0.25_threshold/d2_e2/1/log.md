## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.10584259


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0402193, 0.0621039, -0.0402193, 0.0621039, -0.1022427, 0.1022427)
1: (-0.0128951, 0.0132891, -0.0128951, 0.0132891, -0.0261842, 0.0261842)
2: (-0.0010430, 0.0380089, -0.0010430, 0.0380089, -0.0361754, 0.0361754)
3: (-0.0014132, 0.0692004, -0.0014132, 0.0692004, -0.0549832, 0.0549832)
4: (-0.0237420, -0.0027564, -0.0237420, -0.0027564, -0.0209856, 0.0209856)
5: (0.0076022, 0.0490565, 0.0076022, 0.0490565, -0.0414542, 0.0414542)
6: (-0.0318482, 0.0527335, -0.0318482, 0.0527335, -0.0845817, 0.0845817)
7: (-0.0145990, 0.0107700, -0.0145990, 0.0107700, -0.0253690, 0.0253690)
8: (0.6929997, 0.9393872, 0.6929997, 0.9393872, -0.1973720, 0.1973720)
9: (0.0520203, 0.0932842, 0.0520203, 0.0932842, -0.0412639, 0.0412639)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.24 + 1.37 = 2.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.1214126, upper bound: 0.1214126

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1097196, upper bound: 0.1188688
time: 0.57 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1201256, upper bound: 0.1201256
time: 0.49 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.18 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.18
Output dim: 8, lower bound: -0.1097196, upper bound: 0.1188688
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.18
Output dim: 8, lower bound: -0.1201256, upper bound: 0.1201256

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0442874, 0.0571432, -0.0401832, 0.0613591, -0.1055586, 0.0972385
1: -0.0152352, 0.0143338, -0.0128743, 0.0132797, -0.0285149, 0.0272081
2: -0.0034271, 0.0363141, -0.0010219, 0.0377545, -0.0385648, 0.0373360
3: -0.0049560, 0.0659597, -0.0013819, 0.0687138, -0.0579583, 0.0539058
4: -0.0250527, -0.0023502, -0.0237303, -0.0028743, -0.0221784, 0.0213801
5: 0.0049732, 0.0473181, 0.0076257, 0.0487955, -0.0438223, 0.0396924
6: -0.0349179, 0.0478597, -0.0318209, 0.0520015, -0.0869194, 0.0796806
7: -0.0160732, 0.0126215, -0.0145858, 0.0107534, -0.0268266, 0.0272073
8: 0.7069579, 0.9457194, 0.6950959, 0.9393302, -0.1796720, 0.1985388
9: 0.0490810, 0.0916184, 0.0520465, 0.0930341, -0.0439531, 0.0395719

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1093933, upper bound: 0.1093933
time: 0.56 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1093933, upper bound: 0.1188688
time: 0.46 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0401978, 0.0617608, -0.0402193, 0.0621039, -0.1022161, 0.1018975
1: -0.0128830, 0.0132836, -0.0128951, 0.0132891, -0.0261721, 0.0261787
2: -0.0010306, 0.0378918, -0.0010430, 0.0380089, -0.0363154, 0.0359923
3: -0.0013948, 0.0689764, -0.0014132, 0.0692004, -0.0552354, 0.0540608
4: -0.0237351, -0.0028107, -0.0237420, -0.0027564, -0.0209786, 0.0209313
5: 0.0076159, 0.0489362, 0.0076022, 0.0490565, -0.0414405, 0.0413340
6: -0.0318324, 0.0523963, -0.0318482, 0.0527335, -0.0845659, 0.0842445
7: -0.0145914, 0.0107602, -0.0145990, 0.0107700, -0.0253614, 0.0253592
8: 0.6939644, 0.9393535, 0.6929997, 0.9393872, -0.1950138, 0.1942306
9: 0.0520358, 0.0931691, 0.0520203, 0.0932842, -0.0412484, 0.0411487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1188688, upper bound: 0.1097196
time: 0.54 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1188688, upper bound: 0.1201256
time: 0.54 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.33 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.33
Output dim: 8, lower bound: -0.1093933, upper bound: 0.1093933
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.33
Output dim: 8, lower bound: -0.1093933, upper bound: 0.1188688
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.33
Output dim: 8, lower bound: -0.1188688, upper bound: 0.1097196
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.33
Output dim: 8, lower bound: -0.1188688, upper bound: 0.1201256

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0442874, 0.0571432, -0.0442874, 0.0571432, -0.1013423, 0.1013422
1: -0.0152352, 0.0143338, -0.0152352, 0.0143338, -0.0295689, 0.0295689
2: -0.0034271, 0.0363141, -0.0034271, 0.0363141, -0.0397349, 0.0397348
3: -0.0049560, 0.0659597, -0.0049560, 0.0659597, -0.0571244, 0.0571244
4: -0.0250527, -0.0023502, -0.0250527, -0.0023502, -0.0227024, 0.0227024
5: 0.0049732, 0.0473181, 0.0049732, 0.0473181, -0.0423449, 0.0423449
6: -0.0349179, 0.0478597, -0.0349179, 0.0478597, -0.0827776, 0.0827776
7: -0.0160732, 0.0126215, -0.0160732, 0.0126215, -0.0286947, 0.0286947
8: 0.7069579, 0.9457194, 0.7069579, 0.9457194, -0.1860874, 0.1860874
9: 0.0490810, 0.0916184, 0.0490810, 0.0916184, -0.0425373, 0.0425373

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1023777, upper bound: 0.0997138
time: 0.49 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0997138, upper bound: 0.0997138
time: 0.49 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0442874, 0.0571432, -0.0401978, 0.0617608, -0.1059613, 0.0972538
1: -0.0152352, 0.0143338, -0.0128830, 0.0132836, -0.0285187, 0.0272167
2: -0.0034271, 0.0363141, -0.0010306, 0.0378918, -0.0386057, 0.0373446
3: -0.0049560, 0.0659597, -0.0013948, 0.0689764, -0.0582418, 0.0539215
4: -0.0250527, -0.0023502, -0.0237351, -0.0028107, -0.0222420, 0.0213848
5: 0.0049732, 0.0473181, 0.0076159, 0.0489362, -0.0439630, 0.0397021
6: -0.0349179, 0.0478597, -0.0318324, 0.0523963, -0.0873142, 0.0796921
7: -0.0160732, 0.0126215, -0.0145914, 0.0107602, -0.0268334, 0.0272128
8: 0.7069579, 0.9457194, 0.6939644, 0.9393535, -0.1796930, 0.1996529
9: 0.0490810, 0.0916184, 0.0520358, 0.0931691, -0.0440880, 0.0395826

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1023777, upper bound: 0.1109350
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0997138, upper bound: 0.1109350
time: 0.58 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0401978, 0.0617608, -0.0442874, 0.0571432, -0.0972537, 0.1059613
1: -0.0128830, 0.0132836, -0.0152352, 0.0143338, -0.0272167, 0.0285187
2: -0.0010306, 0.0378918, -0.0034271, 0.0363141, -0.0373446, 0.0386057
3: -0.0013948, 0.0689764, -0.0049560, 0.0659597, -0.0539216, 0.0582418
4: -0.0237351, -0.0028107, -0.0250527, -0.0023502, -0.0213848, 0.0222420
5: 0.0076159, 0.0489362, 0.0049732, 0.0473181, -0.0397021, 0.0439630
6: -0.0318324, 0.0523963, -0.0349179, 0.0478597, -0.0796921, 0.0873142
7: -0.0145914, 0.0107602, -0.0160732, 0.0126215, -0.0272128, 0.0268334
8: 0.6939644, 0.9393535, 0.7069579, 0.9457194, -0.1996529, 0.1796930
9: 0.0520358, 0.0931691, 0.0490810, 0.0916184, -0.0395826, 0.0440880

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
time: 0.52 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0401978, 0.0617608, -0.0401978, 0.0617608, -0.1018703, 0.1018703
1: -0.0128830, 0.0132836, -0.0128830, 0.0132836, -0.0261666, 0.0261666
2: -0.0010306, 0.0378918, -0.0010306, 0.0378918, -0.0361574, 0.0361574
3: -0.0013948, 0.0689764, -0.0013948, 0.0689764, -0.0541713, 0.0541713
4: -0.0237351, -0.0028107, -0.0237351, -0.0028107, -0.0209244, 0.0209244
5: 0.0076159, 0.0489362, 0.0076159, 0.0489362, -0.0413203, 0.0413203
6: -0.0318324, 0.0523963, -0.0318324, 0.0523963, -0.0842287, 0.0842287
7: -0.0145914, 0.0107602, -0.0145914, 0.0107602, -0.0253515, 0.0253515
8: 0.6939644, 0.9393535, 0.6939644, 0.9393535, -0.1915889, 0.1915889
9: 0.0520358, 0.0931691, 0.0520358, 0.0931691, -0.0411332, 0.0411332

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1108777, upper bound: 0.1068057
time: 0.49 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1109350, upper bound: 0.1068673
time: 0.53 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.35 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.35
Output dim: 8, lower bound: -0.1023777, upper bound: 0.0997138
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.35
Output dim: 8, lower bound: -0.0997138, upper bound: 0.0997138
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.35
Output dim: 8, lower bound: -0.1023777, upper bound: 0.1109350
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.35
Output dim: 8, lower bound: -0.0997138, upper bound: 0.1109350
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.35
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.35
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.35
Output dim: 8, lower bound: -0.1108777, upper bound: 0.1068057
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.35
Output dim: 8, lower bound: -0.1109350, upper bound: 0.1068673

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0401978, 0.0617608, -0.1059196, 0.0949187
1: -0.0152119, 0.0143234, -0.0128830, 0.0132836, -0.0284955, 0.0272064
2: -0.0034036, 0.0355157, -0.0010306, 0.0378918, -0.0384062, 0.0365462
3: -0.0049208, 0.0644339, -0.0013948, 0.0689764, -0.0561618, 0.0518001
4: -0.0250397, -0.0023599, -0.0237351, -0.0028107, -0.0222290, 0.0213751
5: 0.0049993, 0.0464991, 0.0076159, 0.0489362, -0.0439370, 0.0388832
6: -0.0348876, 0.0455635, -0.0318324, 0.0523963, -0.0872839, 0.0773959
7: -0.0160587, 0.0126032, -0.0145914, 0.0107602, -0.0268188, 0.0271946
8: 0.7135336, 0.9456578, 0.6939644, 0.9393535, -0.1726243, 0.1973648
9: 0.0491103, 0.0908419, 0.0520358, 0.0931691, -0.0440588, 0.0388061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.49 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0401981, 0.0614806, -0.1056821, 0.0936019
1: -0.0152351, 0.0143338, -0.0128829, 0.0132836, -0.0285187, 0.0272167
2: -0.0034273, 0.0350586, -0.0010304, 0.0377960, -0.0385649, 0.0360889
3: -0.0049559, 0.0635683, -0.0013949, 0.0687934, -0.0581020, 0.0493299
4: -0.0250527, -0.0023502, -0.0237350, -0.0028550, -0.0221977, 0.0213848
5: 0.0049731, 0.0460303, 0.0076160, 0.0488381, -0.0438650, 0.0384143
6: -0.0349180, 0.0442495, -0.0318326, 0.0521212, -0.0870392, 0.0760820
7: -0.0160732, 0.0126215, -0.0145913, 0.0107602, -0.0268334, 0.0272128
8: 0.7172979, 0.9457193, 0.6947533, 0.9393542, -0.1680386, 0.1989450
9: 0.0490810, 0.0904766, 0.0520357, 0.0930750, -0.0439940, 0.0384409

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.50 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442871, 0.0570295, -0.0982908, 0.1026354
1: -0.0135457, 0.0135796, -0.0152352, 0.0143338, -0.0278795, 0.0288148
2: -0.0017059, 0.0367558, -0.0034273, 0.0362753, -0.0378196, 0.0383452
3: -0.0023984, 0.0668044, -0.0049560, 0.0658857, -0.0525380, 0.0557213
4: -0.0241064, -0.0030614, -0.0250527, -0.0023502, -0.0217561, 0.0219912
5: 0.0068713, 0.0477711, 0.0049731, 0.0472783, -0.0404070, 0.0427980
6: -0.0327018, 0.0491298, -0.0349182, 0.0477482, -0.0804500, 0.0840480
7: -0.0150089, 0.0112847, -0.0160732, 0.0126215, -0.0276304, 0.0273579
8: 0.7033206, 0.9411473, 0.7072781, 0.9457201, -0.1894240, 0.1790278
9: 0.0512032, 0.0920525, 0.0490810, 0.0915802, -0.0403771, 0.0429715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.50 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442874, 0.0571432, -0.0972537, 0.1029809
1: -0.0128829, 0.0132836, -0.0152352, 0.0143338, -0.0272167, 0.0285187
2: -0.0010305, 0.0368735, -0.0034271, 0.0363141, -0.0373446, 0.0377327
3: -0.0013947, 0.0670294, -0.0049560, 0.0659597, -0.0538889, 0.0538707
4: -0.0237351, -0.0032822, -0.0250527, -0.0023502, -0.0213848, 0.0217704
5: 0.0076161, 0.0478919, 0.0049732, 0.0473181, -0.0397019, 0.0429187
6: -0.0318324, 0.0494685, -0.0349179, 0.0478597, -0.0796921, 0.0843864
7: -0.0145913, 0.0107602, -0.0160732, 0.0126215, -0.0272128, 0.0268334
8: 0.7023503, 0.9393543, 0.7069579, 0.9457194, -0.1894739, 0.1796851
9: 0.0520359, 0.0921682, 0.0490810, 0.0916184, -0.0395825, 0.0430872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
time: 0.51 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
time: 0.51 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0401979, 0.0612278, -0.1024873, 0.0985439
1: -0.0135457, 0.0135796, -0.0128830, 0.0132836, -0.0268293, 0.0264626
2: -0.0017059, 0.0367558, -0.0010306, 0.0377096, -0.0365139, 0.0359827
3: -0.0023984, 0.0668044, -0.0013949, 0.0686281, -0.0524577, 0.0517634
4: -0.0241064, -0.0030614, -0.0237351, -0.0028950, -0.0212113, 0.0206736
5: 0.0068713, 0.0477711, 0.0076162, 0.0487495, -0.0418782, 0.0401549
6: -0.0327018, 0.0491298, -0.0318325, 0.0518725, -0.0845743, 0.0809623
7: -0.0150089, 0.0112847, -0.0145914, 0.0107601, -0.0257690, 0.0258761
8: 0.7033206, 0.9411473, 0.6954654, 0.9393534, -0.1813266, 0.1897099
9: 0.0512032, 0.0920525, 0.0520358, 0.0929900, -0.0417868, 0.0400167

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0963391, upper bound: 0.1023510
time: 0.50 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1027889, upper bound: 0.0967478
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0401981, 0.0614809, -0.1015906, 0.0988877
1: -0.0128829, 0.0132836, -0.0128830, 0.0132836, -0.0261665, 0.0261665
2: -0.0010305, 0.0368735, -0.0010304, 0.0377961, -0.0361170, 0.0353521
3: -0.0013947, 0.0670294, -0.0013948, 0.0687936, -0.0540461, 0.0492774
4: -0.0237351, -0.0032822, -0.0237351, -0.0028550, -0.0208801, 0.0204529
5: 0.0076161, 0.0478919, 0.0076160, 0.0488381, -0.0412220, 0.0402759
6: -0.0318324, 0.0494685, -0.0318323, 0.0521212, -0.0839536, 0.0813008
7: -0.0145913, 0.0107602, -0.0145914, 0.0107602, -0.0253515, 0.0253516
8: 0.7023503, 0.9393543, 0.6947527, 0.9393533, -0.1807969, 0.1908872
9: 0.0520359, 0.0921682, 0.0520358, 0.0930750, -0.0410391, 0.0401325

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1121336, upper bound: 0.1068673
time: 0.50 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1121336, upper bound: 0.1068673
time: 0.50 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.32 seconds
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.32
Output dim: 8, lower bound: -0.0963391, upper bound: 0.1023510
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.32
Output dim: 8, lower bound: -0.1027889, upper bound: 0.0967478
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 8, lower bound: -0.1121336, upper bound: 0.1068673
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 8, lower bound: -0.1121336, upper bound: 0.1068673

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0413505, 0.0584358, -0.1025935, 0.0960693
1: -0.0152119, 0.0143234, -0.0135457, 0.0135796, -0.0287915, 0.0278691
2: -0.0034036, 0.0355157, -0.0017059, 0.0367558, -0.0381460, 0.0372216
3: -0.0049208, 0.0644339, -0.0023984, 0.0668044, -0.0536429, 0.0505283
4: -0.0250397, -0.0023599, -0.0241064, -0.0030614, -0.0219782, 0.0217464
5: 0.0049993, 0.0464991, 0.0068713, 0.0477711, -0.0427719, 0.0396278
6: -0.0348876, 0.0455635, -0.0327018, 0.0491298, -0.0840174, 0.0782653
7: -0.0160587, 0.0126032, -0.0150089, 0.0112847, -0.0273434, 0.0276121
8: 0.7135336, 0.9456578, 0.7033206, 0.9411473, -0.1723211, 0.1871364
9: 0.0491103, 0.0908419, 0.0512032, 0.0920525, -0.0429423, 0.0396387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.49 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.50 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0401982, 0.0587806, -0.1029403, 0.0949187
1: -0.0152119, 0.0143234, -0.0128829, 0.0132836, -0.0284955, 0.0272064
2: -0.0034036, 0.0355157, -0.0010305, 0.0368735, -0.0380357, 0.0365461
3: -0.0049208, 0.0644339, -0.0013947, 0.0670294, -0.0553089, 0.0517675
4: -0.0250397, -0.0023599, -0.0237351, -0.0032822, -0.0217575, 0.0213751
5: 0.0049993, 0.0464991, 0.0076161, 0.0478919, -0.0428927, 0.0388830
6: -0.0348876, 0.0455635, -0.0318324, 0.0494685, -0.0843561, 0.0773959
7: -0.0160587, 0.0126032, -0.0145913, 0.0107602, -0.0268189, 0.0271946
8: 0.7135336, 0.9456578, 0.7023503, 0.9393543, -0.1726165, 0.1907670
9: 0.0491103, 0.0908419, 0.0520359, 0.0921682, -0.0430580, 0.0388060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1109350
time: 0.49 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.50 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0413505, 0.0584358, -0.1026357, 0.0947535
1: -0.0152351, 0.0143338, -0.0135457, 0.0135796, -0.0288147, 0.0278795
2: -0.0034273, 0.0350586, -0.0017059, 0.0367558, -0.0383397, 0.0367645
3: -0.0049559, 0.0635683, -0.0023984, 0.0668044, -0.0556710, 0.0515055
4: -0.0250527, -0.0023502, -0.0241064, -0.0030614, -0.0219912, 0.0217561
5: 0.0049731, 0.0460303, 0.0068713, 0.0477711, -0.0427980, 0.0391590
6: -0.0349180, 0.0442495, -0.0327018, 0.0491298, -0.0840478, 0.0769513
7: -0.0160732, 0.0126215, -0.0150089, 0.0112847, -0.0273579, 0.0276304
8: 0.7172979, 0.9457193, 0.7033206, 0.9411473, -0.1712241, 0.1894178
9: 0.0490810, 0.0904766, 0.0512032, 0.0920525, -0.0429715, 0.0392734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0401982, 0.0587806, -0.1029809, 0.0936015
1: -0.0152351, 0.0143338, -0.0128829, 0.0132836, -0.0285187, 0.0272167
2: -0.0034273, 0.0350586, -0.0010305, 0.0368735, -0.0377323, 0.0360891
3: -0.0049559, 0.0635683, -0.0013947, 0.0670294, -0.0538581, 0.0493166
4: -0.0250527, -0.0023502, -0.0237351, -0.0032822, -0.0217705, 0.0213849
5: 0.0049731, 0.0460303, 0.0076161, 0.0478919, -0.0429188, 0.0384142
6: -0.0349180, 0.0442495, -0.0318324, 0.0494685, -0.0843865, 0.0760819
7: -0.0160732, 0.0126215, -0.0145913, 0.0107602, -0.0268334, 0.0272128
8: 0.7172979, 0.9457193, 0.7023503, 0.9393543, -0.1680372, 0.1894736
9: 0.0490810, 0.0904766, 0.0520359, 0.0921682, -0.0430872, 0.0384407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.50 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442470, 0.0548080, -0.0960693, 0.1025934
1: -0.0135457, 0.0135796, -0.0152119, 0.0143234, -0.0278691, 0.0287915
2: -0.0017059, 0.0367558, -0.0034036, 0.0355157, -0.0372216, 0.0381460
3: -0.0023984, 0.0668044, -0.0049208, 0.0644339, -0.0505283, 0.0536429
4: -0.0241064, -0.0030614, -0.0250397, -0.0023599, -0.0217464, 0.0219782
5: 0.0068713, 0.0477711, 0.0049993, 0.0464991, -0.0396278, 0.0427719
6: -0.0327018, 0.0491298, -0.0348876, 0.0455635, -0.0782653, 0.0840174
7: -0.0150089, 0.0112847, -0.0160587, 0.0126032, -0.0276121, 0.0273434
8: 0.7033206, 0.9411473, 0.7135336, 0.9456578, -0.1871364, 0.1723211
9: 0.0512032, 0.0920525, 0.0491103, 0.0908419, -0.0396387, 0.0429423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442871, 0.0534906, -0.0947534, 0.1026359
1: -0.0135457, 0.0135796, -0.0152351, 0.0143338, -0.0278795, 0.0288147
2: -0.0017059, 0.0367558, -0.0034273, 0.0350586, -0.0367645, 0.0383396
3: -0.0023984, 0.0668044, -0.0049559, 0.0635683, -0.0515055, 0.0556709
4: -0.0241064, -0.0030614, -0.0250527, -0.0023502, -0.0217561, 0.0219912
5: 0.0068713, 0.0477711, 0.0049731, 0.0460303, -0.0391590, 0.0427980
6: -0.0327018, 0.0491298, -0.0349180, 0.0442495, -0.0769513, 0.0840478
7: -0.0150089, 0.0112847, -0.0160732, 0.0126215, -0.0276304, 0.0273579
8: 0.7033206, 0.9411473, 0.7172979, 0.9457193, -0.1894178, 0.1712241
9: 0.0512032, 0.0920525, 0.0490810, 0.0904766, -0.0392734, 0.0429715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442470, 0.0548080, -0.0949187, 0.1029401
1: -0.0128829, 0.0132836, -0.0152119, 0.0143234, -0.0272064, 0.0284955
2: -0.0010305, 0.0368735, -0.0034036, 0.0355157, -0.0365461, 0.0380358
3: -0.0013947, 0.0670294, -0.0049208, 0.0644339, -0.0517675, 0.0553089
4: -0.0237351, -0.0032822, -0.0250397, -0.0023599, -0.0213751, 0.0217575
5: 0.0076161, 0.0478919, 0.0049993, 0.0464991, -0.0388830, 0.0428927
6: -0.0318324, 0.0494685, -0.0348876, 0.0455635, -0.0773959, 0.0843561
7: -0.0145913, 0.0107602, -0.0160587, 0.0126032, -0.0271946, 0.0268189
8: 0.7023503, 0.9393543, 0.7135336, 0.9456578, -0.1907670, 0.1726165
9: 0.0520359, 0.0921682, 0.0491103, 0.0908419, -0.0388060, 0.0430580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442871, 0.0534906, -0.0936016, 0.1029809
1: -0.0128829, 0.0132836, -0.0152351, 0.0143338, -0.0272167, 0.0285187
2: -0.0010305, 0.0368735, -0.0034273, 0.0350586, -0.0360891, 0.0377323
3: -0.0013947, 0.0670294, -0.0049559, 0.0635683, -0.0493166, 0.0538580
4: -0.0237351, -0.0032822, -0.0250527, -0.0023502, -0.0213849, 0.0217705
5: 0.0076161, 0.0478919, 0.0049731, 0.0460303, -0.0384142, 0.0429188
6: -0.0318324, 0.0494685, -0.0349180, 0.0442495, -0.0760819, 0.0843865
7: -0.0145913, 0.0107602, -0.0160732, 0.0126215, -0.0272128, 0.0268334
8: 0.7023503, 0.9393543, 0.7172979, 0.9457193, -0.1894736, 0.1680372
9: 0.0520359, 0.0921682, 0.0490810, 0.0904766, -0.0384407, 0.0430872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0413505, 0.0584358, -0.0985444, 0.1000414
1: -0.0128829, 0.0132836, -0.0135457, 0.0135796, -0.0264625, 0.0268293
2: -0.0010305, 0.0368735, -0.0017059, 0.0367558, -0.0359803, 0.0362400
3: -0.0013947, 0.0670294, -0.0023984, 0.0668044, -0.0517358, 0.0521227
4: -0.0237351, -0.0032822, -0.0241064, -0.0030614, -0.0206736, 0.0208242
5: 0.0076161, 0.0478919, 0.0068713, 0.0477711, -0.0401550, 0.0410206
6: -0.0318324, 0.0494685, -0.0327018, 0.0491298, -0.0809622, 0.0821703
7: -0.0145913, 0.0107602, -0.0150089, 0.0112847, -0.0258760, 0.0257691
8: 0.7023503, 0.9393543, 0.7033206, 0.9411473, -0.1846697, 0.1813202
9: 0.0520359, 0.0921682, 0.0512032, 0.0920525, -0.0400166, 0.0409651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1074684, upper bound: 0.0902061
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1024660, upper bound: 0.0969314
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0401982, 0.0587806, -0.0988878, 0.0988877
1: -0.0128829, 0.0132836, -0.0128829, 0.0132836, -0.0261665, 0.0261665
2: -0.0010305, 0.0368735, -0.0010305, 0.0368735, -0.0353519, 0.0353519
3: -0.0013947, 0.0670294, -0.0013947, 0.0670294, -0.0492703, 0.0492703
4: -0.0237351, -0.0032822, -0.0237351, -0.0032822, -0.0204529, 0.0204529
5: 0.0076161, 0.0478919, 0.0076161, 0.0478919, -0.0402758, 0.0402758
6: -0.0318324, 0.0494685, -0.0318324, 0.0494685, -0.0813009, 0.0813009
7: -0.0145913, 0.0107602, -0.0145913, 0.0107602, -0.0253515, 0.0253515
8: 0.7023503, 0.9393543, 0.7023503, 0.9393543, -0.1807964, 0.1807964
9: 0.0520359, 0.0921682, 0.0520359, 0.0921682, -0.0401324, 0.0401324

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1118365, upper bound: 0.1068057
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1121336, upper bound: 0.1068673
time: 0.52 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.43 seconds
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1109350
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 8, lower bound: -0.1074684, upper bound: 0.0902061
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.43
Output dim: 8, lower bound: -0.1024660, upper bound: 0.0969314
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 8, lower bound: -0.1118365, upper bound: 0.1068057
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 8, lower bound: -0.1121336, upper bound: 0.1068673

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0413505, 0.0584358, -0.1025935, 0.0960693
1: -0.0152119, 0.0143234, -0.0135457, 0.0135796, -0.0287915, 0.0278691
2: -0.0034036, 0.0355157, -0.0017059, 0.0367558, -0.0381460, 0.0372216
3: -0.0049208, 0.0644339, -0.0023984, 0.0668044, -0.0536429, 0.0505283
4: -0.0250397, -0.0023599, -0.0241064, -0.0030614, -0.0219782, 0.0217464
5: 0.0049993, 0.0464991, 0.0068713, 0.0477711, -0.0427719, 0.0396278
6: -0.0348876, 0.0455635, -0.0327018, 0.0491298, -0.0840174, 0.0782653
7: -0.0160587, 0.0126032, -0.0150089, 0.0112847, -0.0273434, 0.0276121
8: 0.7135336, 0.9456578, 0.7033206, 0.9411473, -0.1723211, 0.1871364
9: 0.0491103, 0.0908419, 0.0512032, 0.0920525, -0.0429423, 0.0396387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0413505, 0.0584358, -0.1026357, 0.0947535
1: -0.0152351, 0.0143338, -0.0135457, 0.0135796, -0.0288147, 0.0278795
2: -0.0034273, 0.0350586, -0.0017059, 0.0367558, -0.0383397, 0.0367645
3: -0.0049559, 0.0635683, -0.0023984, 0.0668044, -0.0556710, 0.0515055
4: -0.0250527, -0.0023502, -0.0241064, -0.0030614, -0.0219912, 0.0217561
5: 0.0049731, 0.0460303, 0.0068713, 0.0477711, -0.0427980, 0.0391590
6: -0.0349180, 0.0442495, -0.0327018, 0.0491298, -0.0840478, 0.0769513
7: -0.0160732, 0.0126215, -0.0150089, 0.0112847, -0.0273579, 0.0276304
8: 0.7172979, 0.9457193, 0.7033206, 0.9411473, -0.1712241, 0.1894178
9: 0.0490810, 0.0904766, 0.0512032, 0.0920525, -0.0429715, 0.0392734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.50 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0401982, 0.0587806, -0.1029403, 0.0949187
1: -0.0152119, 0.0143234, -0.0128829, 0.0132836, -0.0284955, 0.0272064
2: -0.0034036, 0.0355157, -0.0010305, 0.0368735, -0.0380357, 0.0365461
3: -0.0049208, 0.0644339, -0.0013947, 0.0670294, -0.0553089, 0.0517675
4: -0.0250397, -0.0023599, -0.0237351, -0.0032822, -0.0217575, 0.0213751
5: 0.0049993, 0.0464991, 0.0076161, 0.0478919, -0.0428927, 0.0388830
6: -0.0348876, 0.0455635, -0.0318324, 0.0494685, -0.0843561, 0.0773959
7: -0.0160587, 0.0126032, -0.0145913, 0.0107602, -0.0268189, 0.0271946
8: 0.7135336, 0.9456578, 0.7023503, 0.9393543, -0.1726165, 0.1907670
9: 0.0491103, 0.0908419, 0.0520359, 0.0921682, -0.0430580, 0.0388060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0401982, 0.0587806, -0.1029809, 0.0936015
1: -0.0152351, 0.0143338, -0.0128829, 0.0132836, -0.0285187, 0.0272167
2: -0.0034273, 0.0350586, -0.0010305, 0.0368735, -0.0377323, 0.0360891
3: -0.0049559, 0.0635683, -0.0013947, 0.0670294, -0.0538581, 0.0493166
4: -0.0250527, -0.0023502, -0.0237351, -0.0032822, -0.0217705, 0.0213849
5: 0.0049731, 0.0460303, 0.0076161, 0.0478919, -0.0429188, 0.0384142
6: -0.0349180, 0.0442495, -0.0318324, 0.0494685, -0.0843865, 0.0760819
7: -0.0160732, 0.0126215, -0.0145913, 0.0107602, -0.0268334, 0.0272128
8: 0.7172979, 0.9457193, 0.7023503, 0.9393543, -0.1680372, 0.1894736
9: 0.0490810, 0.0904766, 0.0520359, 0.0921682, -0.0430872, 0.0384407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0413505, 0.0584358, -0.1025935, 0.0960693
1: -0.0152119, 0.0143234, -0.0135457, 0.0135796, -0.0287915, 0.0278691
2: -0.0034036, 0.0355157, -0.0017059, 0.0367558, -0.0381460, 0.0372216
3: -0.0049208, 0.0644339, -0.0023984, 0.0668044, -0.0536429, 0.0505283
4: -0.0250397, -0.0023599, -0.0241064, -0.0030614, -0.0219782, 0.0217464
5: 0.0049993, 0.0464991, 0.0068713, 0.0477711, -0.0427719, 0.0396278
6: -0.0348876, 0.0455635, -0.0327018, 0.0491298, -0.0840174, 0.0782653
7: -0.0160587, 0.0126032, -0.0150089, 0.0112847, -0.0273434, 0.0276121
8: 0.7135336, 0.9456578, 0.7033206, 0.9411473, -0.1723211, 0.1871364
9: 0.0491103, 0.0908419, 0.0512032, 0.0920525, -0.0429423, 0.0396387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0413505, 0.0584358, -0.1026357, 0.0947535
1: -0.0152351, 0.0143338, -0.0135457, 0.0135796, -0.0288147, 0.0278795
2: -0.0034273, 0.0350586, -0.0017059, 0.0367558, -0.0383397, 0.0367645
3: -0.0049559, 0.0635683, -0.0023984, 0.0668044, -0.0556710, 0.0515055
4: -0.0250527, -0.0023502, -0.0241064, -0.0030614, -0.0219912, 0.0217561
5: 0.0049731, 0.0460303, 0.0068713, 0.0477711, -0.0427980, 0.0391590
6: -0.0349180, 0.0442495, -0.0327018, 0.0491298, -0.0840478, 0.0769513
7: -0.0160732, 0.0126215, -0.0150089, 0.0112847, -0.0273579, 0.0276304
8: 0.7172979, 0.9457193, 0.7033206, 0.9411473, -0.1712241, 0.1894178
9: 0.0490810, 0.0904766, 0.0512032, 0.0920525, -0.0429715, 0.0392734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0401982, 0.0587806, -0.1029403, 0.0949187
1: -0.0152119, 0.0143234, -0.0128829, 0.0132836, -0.0284955, 0.0272064
2: -0.0034036, 0.0355157, -0.0010305, 0.0368735, -0.0380357, 0.0365461
3: -0.0049208, 0.0644339, -0.0013947, 0.0670294, -0.0553089, 0.0517675
4: -0.0250397, -0.0023599, -0.0237351, -0.0032822, -0.0217575, 0.0213751
5: 0.0049993, 0.0464991, 0.0076161, 0.0478919, -0.0428927, 0.0388830
6: -0.0348876, 0.0455635, -0.0318324, 0.0494685, -0.0843561, 0.0773959
7: -0.0160587, 0.0126032, -0.0145913, 0.0107602, -0.0268189, 0.0271946
8: 0.7135336, 0.9456578, 0.7023503, 0.9393543, -0.1726165, 0.1907670
9: 0.0491103, 0.0908419, 0.0520359, 0.0921682, -0.0430580, 0.0388060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0401982, 0.0587806, -0.1029809, 0.0936015
1: -0.0152351, 0.0143338, -0.0128829, 0.0132836, -0.0285187, 0.0272167
2: -0.0034273, 0.0350586, -0.0010305, 0.0368735, -0.0377323, 0.0360891
3: -0.0049559, 0.0635683, -0.0013947, 0.0670294, -0.0538581, 0.0493166
4: -0.0250527, -0.0023502, -0.0237351, -0.0032822, -0.0217705, 0.0213849
5: 0.0049731, 0.0460303, 0.0076161, 0.0478919, -0.0429188, 0.0384142
6: -0.0349180, 0.0442495, -0.0318324, 0.0494685, -0.0843865, 0.0760819
7: -0.0160732, 0.0126215, -0.0145913, 0.0107602, -0.0268334, 0.0272128
8: 0.7172979, 0.9457193, 0.7023503, 0.9393543, -0.1680372, 0.1894736
9: 0.0490810, 0.0904766, 0.0520359, 0.0921682, -0.0430872, 0.0384407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442470, 0.0548080, -0.0960693, 0.1025934
1: -0.0135457, 0.0135796, -0.0152119, 0.0143234, -0.0278691, 0.0287915
2: -0.0017059, 0.0367558, -0.0034036, 0.0355157, -0.0372216, 0.0381460
3: -0.0023984, 0.0668044, -0.0049208, 0.0644339, -0.0505283, 0.0536429
4: -0.0241064, -0.0030614, -0.0250397, -0.0023599, -0.0217464, 0.0219782
5: 0.0068713, 0.0477711, 0.0049993, 0.0464991, -0.0396278, 0.0427719
6: -0.0327018, 0.0491298, -0.0348876, 0.0455635, -0.0782653, 0.0840174
7: -0.0150089, 0.0112847, -0.0160587, 0.0126032, -0.0276121, 0.0273434
8: 0.7033206, 0.9411473, 0.7135336, 0.9456578, -0.1871364, 0.1723211
9: 0.0512032, 0.0920525, 0.0491103, 0.0908419, -0.0396387, 0.0429423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442470, 0.0548080, -0.0949187, 0.1029401
1: -0.0128829, 0.0132836, -0.0152119, 0.0143234, -0.0272064, 0.0284955
2: -0.0010305, 0.0368735, -0.0034036, 0.0355157, -0.0365461, 0.0380358
3: -0.0013947, 0.0670294, -0.0049208, 0.0644339, -0.0517675, 0.0553089
4: -0.0237351, -0.0032822, -0.0250397, -0.0023599, -0.0213751, 0.0217575
5: 0.0076161, 0.0478919, 0.0049993, 0.0464991, -0.0388830, 0.0428927
6: -0.0318324, 0.0494685, -0.0348876, 0.0455635, -0.0773959, 0.0843561
7: -0.0145913, 0.0107602, -0.0160587, 0.0126032, -0.0271946, 0.0268189
8: 0.7023503, 0.9393543, 0.7135336, 0.9456578, -0.1907670, 0.1726165
9: 0.0520359, 0.0921682, 0.0491103, 0.0908419, -0.0388060, 0.0430580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442871, 0.0534906, -0.0947534, 0.1026359
1: -0.0135457, 0.0135796, -0.0152351, 0.0143338, -0.0278795, 0.0288147
2: -0.0017059, 0.0367558, -0.0034273, 0.0350586, -0.0367645, 0.0383396
3: -0.0023984, 0.0668044, -0.0049559, 0.0635683, -0.0515055, 0.0556709
4: -0.0241064, -0.0030614, -0.0250527, -0.0023502, -0.0217561, 0.0219912
5: 0.0068713, 0.0477711, 0.0049731, 0.0460303, -0.0391590, 0.0427980
6: -0.0327018, 0.0491298, -0.0349180, 0.0442495, -0.0769513, 0.0840478
7: -0.0150089, 0.0112847, -0.0160732, 0.0126215, -0.0276304, 0.0273579
8: 0.7033206, 0.9411473, 0.7172979, 0.9457193, -0.1894178, 0.1712241
9: 0.0512032, 0.0920525, 0.0490810, 0.0904766, -0.0392734, 0.0429715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442871, 0.0534906, -0.0936016, 0.1029809
1: -0.0128829, 0.0132836, -0.0152351, 0.0143338, -0.0272167, 0.0285187
2: -0.0010305, 0.0368735, -0.0034273, 0.0350586, -0.0360891, 0.0377323
3: -0.0013947, 0.0670294, -0.0049559, 0.0635683, -0.0493166, 0.0538580
4: -0.0237351, -0.0032822, -0.0250527, -0.0023502, -0.0213849, 0.0217705
5: 0.0076161, 0.0478919, 0.0049731, 0.0460303, -0.0384142, 0.0429188
6: -0.0318324, 0.0494685, -0.0349180, 0.0442495, -0.0760819, 0.0843865
7: -0.0145913, 0.0107602, -0.0160732, 0.0126215, -0.0272128, 0.0268334
8: 0.7023503, 0.9393543, 0.7172979, 0.9457193, -0.1894736, 0.1680372
9: 0.0520359, 0.0921682, 0.0490810, 0.0904766, -0.0384407, 0.0430872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442470, 0.0548080, -0.0960693, 0.1025934
1: -0.0135457, 0.0135796, -0.0152119, 0.0143234, -0.0278691, 0.0287915
2: -0.0017059, 0.0367558, -0.0034036, 0.0355157, -0.0372216, 0.0381460
3: -0.0023984, 0.0668044, -0.0049208, 0.0644339, -0.0505283, 0.0536429
4: -0.0241064, -0.0030614, -0.0250397, -0.0023599, -0.0217464, 0.0219782
5: 0.0068713, 0.0477711, 0.0049993, 0.0464991, -0.0396278, 0.0427719
6: -0.0327018, 0.0491298, -0.0348876, 0.0455635, -0.0782653, 0.0840174
7: -0.0150089, 0.0112847, -0.0160587, 0.0126032, -0.0276121, 0.0273434
8: 0.7033206, 0.9411473, 0.7135336, 0.9456578, -0.1871364, 0.1723211
9: 0.0512032, 0.0920525, 0.0491103, 0.0908419, -0.0396387, 0.0429423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442470, 0.0548080, -0.0949187, 0.1029401
1: -0.0128829, 0.0132836, -0.0152119, 0.0143234, -0.0272064, 0.0284955
2: -0.0010305, 0.0368735, -0.0034036, 0.0355157, -0.0365461, 0.0380358
3: -0.0013947, 0.0670294, -0.0049208, 0.0644339, -0.0517675, 0.0553089
4: -0.0237351, -0.0032822, -0.0250397, -0.0023599, -0.0213751, 0.0217575
5: 0.0076161, 0.0478919, 0.0049993, 0.0464991, -0.0388830, 0.0428927
6: -0.0318324, 0.0494685, -0.0348876, 0.0455635, -0.0773959, 0.0843561
7: -0.0145913, 0.0107602, -0.0160587, 0.0126032, -0.0271946, 0.0268189
8: 0.7023503, 0.9393543, 0.7135336, 0.9456578, -0.1907670, 0.1726165
9: 0.0520359, 0.0921682, 0.0491103, 0.0908419, -0.0388060, 0.0430580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1109350, upper bound: 0.1028533
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442871, 0.0534906, -0.0947534, 0.1026359
1: -0.0135457, 0.0135796, -0.0152351, 0.0143338, -0.0278795, 0.0288147
2: -0.0017059, 0.0367558, -0.0034273, 0.0350586, -0.0367645, 0.0383396
3: -0.0023984, 0.0668044, -0.0049559, 0.0635683, -0.0515055, 0.0556709
4: -0.0241064, -0.0030614, -0.0250527, -0.0023502, -0.0217561, 0.0219912
5: 0.0068713, 0.0477711, 0.0049731, 0.0460303, -0.0391590, 0.0427980
6: -0.0327018, 0.0491298, -0.0349180, 0.0442495, -0.0769513, 0.0840478
7: -0.0150089, 0.0112847, -0.0160732, 0.0126215, -0.0276304, 0.0273579
8: 0.7033206, 0.9411473, 0.7172979, 0.9457193, -0.1894178, 0.1712241
9: 0.0512032, 0.0920525, 0.0490810, 0.0904766, -0.0392734, 0.0429715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442871, 0.0534906, -0.0936016, 0.1029809
1: -0.0128829, 0.0132836, -0.0152351, 0.0143338, -0.0272167, 0.0285187
2: -0.0010305, 0.0368735, -0.0034273, 0.0350586, -0.0360891, 0.0377323
3: -0.0013947, 0.0670294, -0.0049559, 0.0635683, -0.0493166, 0.0538580
4: -0.0237351, -0.0032822, -0.0250527, -0.0023502, -0.0213849, 0.0217705
5: 0.0076161, 0.0478919, 0.0049731, 0.0460303, -0.0384142, 0.0429188
6: -0.0318324, 0.0494685, -0.0349180, 0.0442495, -0.0760819, 0.0843865
7: -0.0145913, 0.0107602, -0.0160732, 0.0126215, -0.0272128, 0.0268334
8: 0.7023503, 0.9393543, 0.7172979, 0.9457193, -0.1894736, 0.1680372
9: 0.0520359, 0.0921682, 0.0490810, 0.0904766, -0.0384407, 0.0430872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0401980, 0.0572768, -0.0413503, 0.0583259, -0.0984344, 0.0985379
1: -0.0128829, 0.0132835, -0.0135457, 0.0135796, -0.0264625, 0.0268293
2: -0.0010304, 0.0363597, -0.0017060, 0.0367182, -0.0359369, 0.0355918
3: -0.0013946, 0.0660469, -0.0023983, 0.0667324, -0.0515402, 0.0500631
4: -0.0237351, -0.0033405, -0.0241063, -0.0030615, -0.0206736, 0.0207658
5: 0.0076161, 0.0473649, 0.0068712, 0.0477326, -0.0401165, 0.0404937
6: -0.0318319, 0.0479909, -0.0327020, 0.0490217, -0.0808536, 0.0806929
7: -0.0145914, 0.0107601, -0.0150089, 0.0112847, -0.0258760, 0.0257690
8: 0.7065819, 0.9393533, 0.7036301, 0.9411469, -0.1803710, 0.1809983
9: 0.0520357, 0.0916632, 0.0512030, 0.0920156, -0.0399799, 0.0404602

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1074684, upper bound: 0.0915589
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1074684, upper bound: 0.0902061
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0401982, 0.0587806, -0.1000412, 0.0985441
1: -0.0135457, 0.0135796, -0.0128829, 0.0132836, -0.0268293, 0.0264625
2: -0.0017059, 0.0367558, -0.0010305, 0.0368735, -0.0362400, 0.0359803
3: -0.0023984, 0.0668044, -0.0013947, 0.0670294, -0.0521227, 0.0517359
4: -0.0241064, -0.0030614, -0.0237351, -0.0032822, -0.0208242, 0.0206736
5: 0.0068713, 0.0477711, 0.0076161, 0.0478919, -0.0410206, 0.0401550
6: -0.0327018, 0.0491298, -0.0318324, 0.0494685, -0.0821703, 0.0809622
7: -0.0150089, 0.0112847, -0.0145913, 0.0107602, -0.0257691, 0.0258760
8: 0.7033206, 0.9411473, 0.7023503, 0.9393543, -0.1813202, 0.1846695
9: 0.0512032, 0.0920525, 0.0520359, 0.0921682, -0.0409651, 0.0400166

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0957134, upper bound: 0.1023510
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1022408, upper bound: 0.0967478
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0401982, 0.0587806, -0.0988878, 0.0988877
1: -0.0128829, 0.0132836, -0.0128829, 0.0132836, -0.0261665, 0.0261665
2: -0.0010305, 0.0368735, -0.0010305, 0.0368735, -0.0353519, 0.0353519
3: -0.0013947, 0.0670294, -0.0013947, 0.0670294, -0.0492703, 0.0492703
4: -0.0237351, -0.0032822, -0.0237351, -0.0032822, -0.0204529, 0.0204529
5: 0.0076161, 0.0478919, 0.0076161, 0.0478919, -0.0402758, 0.0402758
6: -0.0318324, 0.0494685, -0.0318324, 0.0494685, -0.0813009, 0.0813009
7: -0.0145913, 0.0107602, -0.0145913, 0.0107602, -0.0253515, 0.0253515
8: 0.7023503, 0.9393543, 0.7023503, 0.9393543, -0.1807964, 0.1807964
9: 0.0520359, 0.0921682, 0.0520359, 0.0921682, -0.0401324, 0.0401324

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1121336, upper bound: 0.1068673
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1121336, upper bound: 0.1068673
time: 0.55 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.45 seconds
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.1109350, upper bound: 0.1028533
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.1074684, upper bound: 0.0915589
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.1074684, upper bound: 0.0902061
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.0957134, upper bound: 0.1023510
IS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.1022408, upper bound: 0.0967478
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.1121336, upper bound: 0.1068673
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 8, lower bound: -0.1121336, upper bound: 0.1068673

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0413505, 0.0584358, -0.1025935, 0.0960693
1: -0.0152119, 0.0143234, -0.0135457, 0.0135796, -0.0287915, 0.0278691
2: -0.0034036, 0.0355157, -0.0017059, 0.0367558, -0.0381460, 0.0372216
3: -0.0049208, 0.0644339, -0.0023984, 0.0668044, -0.0536429, 0.0505283
4: -0.0250397, -0.0023599, -0.0241064, -0.0030614, -0.0219782, 0.0217464
5: 0.0049993, 0.0464991, 0.0068713, 0.0477711, -0.0427719, 0.0396278
6: -0.0348876, 0.0455635, -0.0327018, 0.0491298, -0.0840174, 0.0782653
7: -0.0160587, 0.0126032, -0.0150089, 0.0112847, -0.0273434, 0.0276121
8: 0.7135336, 0.9456578, 0.7033206, 0.9411473, -0.1723211, 0.1871364
9: 0.0491103, 0.0908419, 0.0512032, 0.0920525, -0.0429423, 0.0396387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0401982, 0.0587806, -0.1029403, 0.0949187
1: -0.0152119, 0.0143234, -0.0128829, 0.0132836, -0.0284955, 0.0272064
2: -0.0034036, 0.0355157, -0.0010305, 0.0368735, -0.0380357, 0.0365461
3: -0.0049208, 0.0644339, -0.0013947, 0.0670294, -0.0553089, 0.0517675
4: -0.0250397, -0.0023599, -0.0237351, -0.0032822, -0.0217575, 0.0213751
5: 0.0049993, 0.0464991, 0.0076161, 0.0478919, -0.0428927, 0.0388830
6: -0.0348876, 0.0455635, -0.0318324, 0.0494685, -0.0843561, 0.0773959
7: -0.0160587, 0.0126032, -0.0145913, 0.0107602, -0.0268189, 0.0271946
8: 0.7135336, 0.9456578, 0.7023503, 0.9393543, -0.1726165, 0.1907670
9: 0.0491103, 0.0908419, 0.0520359, 0.0921682, -0.0430580, 0.0388060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0413505, 0.0584358, -0.1026357, 0.0947535
1: -0.0152351, 0.0143338, -0.0135457, 0.0135796, -0.0288147, 0.0278795
2: -0.0034273, 0.0350586, -0.0017059, 0.0367558, -0.0383397, 0.0367645
3: -0.0049559, 0.0635683, -0.0023984, 0.0668044, -0.0556710, 0.0515055
4: -0.0250527, -0.0023502, -0.0241064, -0.0030614, -0.0219912, 0.0217561
5: 0.0049731, 0.0460303, 0.0068713, 0.0477711, -0.0427980, 0.0391590
6: -0.0349180, 0.0442495, -0.0327018, 0.0491298, -0.0840478, 0.0769513
7: -0.0160732, 0.0126215, -0.0150089, 0.0112847, -0.0273579, 0.0276304
8: 0.7172979, 0.9457193, 0.7033206, 0.9411473, -0.1712241, 0.1894178
9: 0.0490810, 0.0904766, 0.0512032, 0.0920525, -0.0429715, 0.0392734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0401982, 0.0587806, -0.1029809, 0.0936015
1: -0.0152351, 0.0143338, -0.0128829, 0.0132836, -0.0285187, 0.0272167
2: -0.0034273, 0.0350586, -0.0010305, 0.0368735, -0.0377323, 0.0360891
3: -0.0049559, 0.0635683, -0.0013947, 0.0670294, -0.0538581, 0.0493166
4: -0.0250527, -0.0023502, -0.0237351, -0.0032822, -0.0217705, 0.0213849
5: 0.0049731, 0.0460303, 0.0076161, 0.0478919, -0.0429188, 0.0384142
6: -0.0349180, 0.0442495, -0.0318324, 0.0494685, -0.0843865, 0.0760819
7: -0.0160732, 0.0126215, -0.0145913, 0.0107602, -0.0268334, 0.0272128
8: 0.7172979, 0.9457193, 0.7023503, 0.9393543, -0.1680372, 0.1894736
9: 0.0490810, 0.0904766, 0.0520359, 0.0921682, -0.0430872, 0.0384407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0413505, 0.0584358, -0.1025935, 0.0960693
1: -0.0152119, 0.0143234, -0.0135457, 0.0135796, -0.0287915, 0.0278691
2: -0.0034036, 0.0355157, -0.0017059, 0.0367558, -0.0381460, 0.0372216
3: -0.0049208, 0.0644339, -0.0023984, 0.0668044, -0.0536429, 0.0505283
4: -0.0250397, -0.0023599, -0.0241064, -0.0030614, -0.0219782, 0.0217464
5: 0.0049993, 0.0464991, 0.0068713, 0.0477711, -0.0427719, 0.0396278
6: -0.0348876, 0.0455635, -0.0327018, 0.0491298, -0.0840174, 0.0782653
7: -0.0160587, 0.0126032, -0.0150089, 0.0112847, -0.0273434, 0.0276121
8: 0.7135336, 0.9456578, 0.7033206, 0.9411473, -0.1723211, 0.1871364
9: 0.0491103, 0.0908419, 0.0512032, 0.0920525, -0.0429423, 0.0396387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0401982, 0.0587806, -0.1029403, 0.0949187
1: -0.0152119, 0.0143234, -0.0128829, 0.0132836, -0.0284955, 0.0272064
2: -0.0034036, 0.0355157, -0.0010305, 0.0368735, -0.0380357, 0.0365461
3: -0.0049208, 0.0644339, -0.0013947, 0.0670294, -0.0553089, 0.0517675
4: -0.0250397, -0.0023599, -0.0237351, -0.0032822, -0.0217575, 0.0213751
5: 0.0049993, 0.0464991, 0.0076161, 0.0478919, -0.0428927, 0.0388830
6: -0.0348876, 0.0455635, -0.0318324, 0.0494685, -0.0843561, 0.0773959
7: -0.0160587, 0.0126032, -0.0145913, 0.0107602, -0.0268189, 0.0271946
8: 0.7135336, 0.9456578, 0.7023503, 0.9393543, -0.1726165, 0.1907670
9: 0.0491103, 0.0908419, 0.0520359, 0.0921682, -0.0430580, 0.0388060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1109350
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0413505, 0.0584358, -0.1026357, 0.0947535
1: -0.0152351, 0.0143338, -0.0135457, 0.0135796, -0.0288147, 0.0278795
2: -0.0034273, 0.0350586, -0.0017059, 0.0367558, -0.0383397, 0.0367645
3: -0.0049559, 0.0635683, -0.0023984, 0.0668044, -0.0556710, 0.0515055
4: -0.0250527, -0.0023502, -0.0241064, -0.0030614, -0.0219912, 0.0217561
5: 0.0049731, 0.0460303, 0.0068713, 0.0477711, -0.0427980, 0.0391590
6: -0.0349180, 0.0442495, -0.0327018, 0.0491298, -0.0840478, 0.0769513
7: -0.0160732, 0.0126215, -0.0150089, 0.0112847, -0.0273579, 0.0276304
8: 0.7172979, 0.9457193, 0.7033206, 0.9411473, -0.1712241, 0.1894178
9: 0.0490810, 0.0904766, 0.0512032, 0.0920525, -0.0429715, 0.0392734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0401982, 0.0587806, -0.1029809, 0.0936015
1: -0.0152351, 0.0143338, -0.0128829, 0.0132836, -0.0285187, 0.0272167
2: -0.0034273, 0.0350586, -0.0010305, 0.0368735, -0.0377323, 0.0360891
3: -0.0049559, 0.0635683, -0.0013947, 0.0670294, -0.0538581, 0.0493166
4: -0.0250527, -0.0023502, -0.0237351, -0.0032822, -0.0217705, 0.0213849
5: 0.0049731, 0.0460303, 0.0076161, 0.0478919, -0.0429188, 0.0384142
6: -0.0349180, 0.0442495, -0.0318324, 0.0494685, -0.0843865, 0.0760819
7: -0.0160732, 0.0126215, -0.0145913, 0.0107602, -0.0268334, 0.0272128
8: 0.7172979, 0.9457193, 0.7023503, 0.9393543, -0.1680372, 0.1894736
9: 0.0490810, 0.0904766, 0.0520359, 0.0921682, -0.0430872, 0.0384407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0413505, 0.0584358, -0.1025935, 0.0960693
1: -0.0152119, 0.0143234, -0.0135457, 0.0135796, -0.0287915, 0.0278691
2: -0.0034036, 0.0355157, -0.0017059, 0.0367558, -0.0381460, 0.0372216
3: -0.0049208, 0.0644339, -0.0023984, 0.0668044, -0.0536429, 0.0505283
4: -0.0250397, -0.0023599, -0.0241064, -0.0030614, -0.0219782, 0.0217464
5: 0.0049993, 0.0464991, 0.0068713, 0.0477711, -0.0427719, 0.0396278
6: -0.0348876, 0.0455635, -0.0327018, 0.0491298, -0.0840174, 0.0782653
7: -0.0160587, 0.0126032, -0.0150089, 0.0112847, -0.0273434, 0.0276121
8: 0.7135336, 0.9456578, 0.7033206, 0.9411473, -0.1723211, 0.1871364
9: 0.0491103, 0.0908419, 0.0512032, 0.0920525, -0.0429423, 0.0396387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0401982, 0.0587806, -0.1029403, 0.0949187
1: -0.0152119, 0.0143234, -0.0128829, 0.0132836, -0.0284955, 0.0272064
2: -0.0034036, 0.0355157, -0.0010305, 0.0368735, -0.0380357, 0.0365461
3: -0.0049208, 0.0644339, -0.0013947, 0.0670294, -0.0553089, 0.0517675
4: -0.0250397, -0.0023599, -0.0237351, -0.0032822, -0.0217575, 0.0213751
5: 0.0049993, 0.0464991, 0.0076161, 0.0478919, -0.0428927, 0.0388830
6: -0.0348876, 0.0455635, -0.0318324, 0.0494685, -0.0843561, 0.0773959
7: -0.0160587, 0.0126032, -0.0145913, 0.0107602, -0.0268189, 0.0271946
8: 0.7135336, 0.9456578, 0.7023503, 0.9393543, -0.1726165, 0.1907670
9: 0.0491103, 0.0908419, 0.0520359, 0.0921682, -0.0430580, 0.0388060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0413505, 0.0584358, -0.1026357, 0.0947535
1: -0.0152351, 0.0143338, -0.0135457, 0.0135796, -0.0288147, 0.0278795
2: -0.0034273, 0.0350586, -0.0017059, 0.0367558, -0.0383397, 0.0367645
3: -0.0049559, 0.0635683, -0.0023984, 0.0668044, -0.0556710, 0.0515055
4: -0.0250527, -0.0023502, -0.0241064, -0.0030614, -0.0219912, 0.0217561
5: 0.0049731, 0.0460303, 0.0068713, 0.0477711, -0.0427980, 0.0391590
6: -0.0349180, 0.0442495, -0.0327018, 0.0491298, -0.0840478, 0.0769513
7: -0.0160732, 0.0126215, -0.0150089, 0.0112847, -0.0273579, 0.0276304
8: 0.7172979, 0.9457193, 0.7033206, 0.9411473, -0.1712241, 0.1894178
9: 0.0490810, 0.0904766, 0.0512032, 0.0920525, -0.0429715, 0.0392734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0401982, 0.0587806, -0.1029809, 0.0936015
1: -0.0152351, 0.0143338, -0.0128829, 0.0132836, -0.0285187, 0.0272167
2: -0.0034273, 0.0350586, -0.0010305, 0.0368735, -0.0377323, 0.0360891
3: -0.0049559, 0.0635683, -0.0013947, 0.0670294, -0.0538581, 0.0493166
4: -0.0250527, -0.0023502, -0.0237351, -0.0032822, -0.0217705, 0.0213849
5: 0.0049731, 0.0460303, 0.0076161, 0.0478919, -0.0429188, 0.0384142
6: -0.0349180, 0.0442495, -0.0318324, 0.0494685, -0.0843865, 0.0760819
7: -0.0160732, 0.0126215, -0.0145913, 0.0107602, -0.0268334, 0.0272128
8: 0.7172979, 0.9457193, 0.7023503, 0.9393543, -0.1680372, 0.1894736
9: 0.0490810, 0.0904766, 0.0520359, 0.0921682, -0.0430872, 0.0384407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0413505, 0.0584358, -0.1025935, 0.0960693
1: -0.0152119, 0.0143234, -0.0135457, 0.0135796, -0.0287915, 0.0278691
2: -0.0034036, 0.0355157, -0.0017059, 0.0367558, -0.0381460, 0.0372216
3: -0.0049208, 0.0644339, -0.0023984, 0.0668044, -0.0536429, 0.0505283
4: -0.0250397, -0.0023599, -0.0241064, -0.0030614, -0.0219782, 0.0217464
5: 0.0049993, 0.0464991, 0.0068713, 0.0477711, -0.0427719, 0.0396278
6: -0.0348876, 0.0455635, -0.0327018, 0.0491298, -0.0840174, 0.0782653
7: -0.0160587, 0.0126032, -0.0150089, 0.0112847, -0.0273434, 0.0276121
8: 0.7135336, 0.9456578, 0.7033206, 0.9411473, -0.1723211, 0.1871364
9: 0.0491103, 0.0908419, 0.0512032, 0.0920525, -0.0429423, 0.0396387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0401982, 0.0587806, -0.1029403, 0.0949187
1: -0.0152119, 0.0143234, -0.0128829, 0.0132836, -0.0284955, 0.0272064
2: -0.0034036, 0.0355157, -0.0010305, 0.0368735, -0.0380357, 0.0365461
3: -0.0049208, 0.0644339, -0.0013947, 0.0670294, -0.0553089, 0.0517675
4: -0.0250397, -0.0023599, -0.0237351, -0.0032822, -0.0217575, 0.0213751
5: 0.0049993, 0.0464991, 0.0076161, 0.0478919, -0.0428927, 0.0388830
6: -0.0348876, 0.0455635, -0.0318324, 0.0494685, -0.0843561, 0.0773959
7: -0.0160587, 0.0126032, -0.0145913, 0.0107602, -0.0268189, 0.0271946
8: 0.7135336, 0.9456578, 0.7023503, 0.9393543, -0.1726165, 0.1907670
9: 0.0491103, 0.0908419, 0.0520359, 0.0921682, -0.0430580, 0.0388060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1109350
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0413505, 0.0584358, -0.1026357, 0.0947535
1: -0.0152351, 0.0143338, -0.0135457, 0.0135796, -0.0288147, 0.0278795
2: -0.0034273, 0.0350586, -0.0017059, 0.0367558, -0.0383397, 0.0367645
3: -0.0049559, 0.0635683, -0.0023984, 0.0668044, -0.0556710, 0.0515055
4: -0.0250527, -0.0023502, -0.0241064, -0.0030614, -0.0219912, 0.0217561
5: 0.0049731, 0.0460303, 0.0068713, 0.0477711, -0.0427980, 0.0391590
6: -0.0349180, 0.0442495, -0.0327018, 0.0491298, -0.0840478, 0.0769513
7: -0.0160732, 0.0126215, -0.0150089, 0.0112847, -0.0273579, 0.0276304
8: 0.7172979, 0.9457193, 0.7033206, 0.9411473, -0.1712241, 0.1894178
9: 0.0490810, 0.0904766, 0.0512032, 0.0920525, -0.0429715, 0.0392734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0401982, 0.0587806, -0.1029809, 0.0936015
1: -0.0152351, 0.0143338, -0.0128829, 0.0132836, -0.0285187, 0.0272167
2: -0.0034273, 0.0350586, -0.0010305, 0.0368735, -0.0377323, 0.0360891
3: -0.0049559, 0.0635683, -0.0013947, 0.0670294, -0.0538581, 0.0493166
4: -0.0250527, -0.0023502, -0.0237351, -0.0032822, -0.0217705, 0.0213849
5: 0.0049731, 0.0460303, 0.0076161, 0.0478919, -0.0429188, 0.0384142
6: -0.0349180, 0.0442495, -0.0318324, 0.0494685, -0.0843865, 0.0760819
7: -0.0160732, 0.0126215, -0.0145913, 0.0107602, -0.0268334, 0.0272128
8: 0.7172979, 0.9457193, 0.7023503, 0.9393543, -0.1680372, 0.1894736
9: 0.0490810, 0.0904766, 0.0520359, 0.0921682, -0.0430872, 0.0384407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442470, 0.0548080, -0.0960693, 0.1025934
1: -0.0135457, 0.0135796, -0.0152119, 0.0143234, -0.0278691, 0.0287915
2: -0.0017059, 0.0367558, -0.0034036, 0.0355157, -0.0372216, 0.0381460
3: -0.0023984, 0.0668044, -0.0049208, 0.0644339, -0.0505283, 0.0536429
4: -0.0241064, -0.0030614, -0.0250397, -0.0023599, -0.0217464, 0.0219782
5: 0.0068713, 0.0477711, 0.0049993, 0.0464991, -0.0396278, 0.0427719
6: -0.0327018, 0.0491298, -0.0348876, 0.0455635, -0.0782653, 0.0840174
7: -0.0150089, 0.0112847, -0.0160587, 0.0126032, -0.0276121, 0.0273434
8: 0.7033206, 0.9411473, 0.7135336, 0.9456578, -0.1871364, 0.1723211
9: 0.0512032, 0.0920525, 0.0491103, 0.0908419, -0.0396387, 0.0429423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442871, 0.0534906, -0.0947534, 0.1026359
1: -0.0135457, 0.0135796, -0.0152351, 0.0143338, -0.0278795, 0.0288147
2: -0.0017059, 0.0367558, -0.0034273, 0.0350586, -0.0367645, 0.0383396
3: -0.0023984, 0.0668044, -0.0049559, 0.0635683, -0.0515055, 0.0556709
4: -0.0241064, -0.0030614, -0.0250527, -0.0023502, -0.0217561, 0.0219912
5: 0.0068713, 0.0477711, 0.0049731, 0.0460303, -0.0391590, 0.0427980
6: -0.0327018, 0.0491298, -0.0349180, 0.0442495, -0.0769513, 0.0840478
7: -0.0150089, 0.0112847, -0.0160732, 0.0126215, -0.0276304, 0.0273579
8: 0.7033206, 0.9411473, 0.7172979, 0.9457193, -0.1894178, 0.1712241
9: 0.0512032, 0.0920525, 0.0490810, 0.0904766, -0.0392734, 0.0429715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442470, 0.0548080, -0.0949187, 0.1029401
1: -0.0128829, 0.0132836, -0.0152119, 0.0143234, -0.0272064, 0.0284955
2: -0.0010305, 0.0368735, -0.0034036, 0.0355157, -0.0365461, 0.0380358
3: -0.0013947, 0.0670294, -0.0049208, 0.0644339, -0.0517675, 0.0553089
4: -0.0237351, -0.0032822, -0.0250397, -0.0023599, -0.0213751, 0.0217575
5: 0.0076161, 0.0478919, 0.0049993, 0.0464991, -0.0388830, 0.0428927
6: -0.0318324, 0.0494685, -0.0348876, 0.0455635, -0.0773959, 0.0843561
7: -0.0145913, 0.0107602, -0.0160587, 0.0126032, -0.0271946, 0.0268189
8: 0.7023503, 0.9393543, 0.7135336, 0.9456578, -0.1907670, 0.1726165
9: 0.0520359, 0.0921682, 0.0491103, 0.0908419, -0.0388060, 0.0430580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442871, 0.0534906, -0.0936016, 0.1029809
1: -0.0128829, 0.0132836, -0.0152351, 0.0143338, -0.0272167, 0.0285187
2: -0.0010305, 0.0368735, -0.0034273, 0.0350586, -0.0360891, 0.0377323
3: -0.0013947, 0.0670294, -0.0049559, 0.0635683, -0.0493166, 0.0538580
4: -0.0237351, -0.0032822, -0.0250527, -0.0023502, -0.0213849, 0.0217705
5: 0.0076161, 0.0478919, 0.0049731, 0.0460303, -0.0384142, 0.0429188
6: -0.0318324, 0.0494685, -0.0349180, 0.0442495, -0.0760819, 0.0843865
7: -0.0145913, 0.0107602, -0.0160732, 0.0126215, -0.0272128, 0.0268334
8: 0.7023503, 0.9393543, 0.7172979, 0.9457193, -0.1894736, 0.1680372
9: 0.0520359, 0.0921682, 0.0490810, 0.0904766, -0.0384407, 0.0430872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442470, 0.0548080, -0.0960693, 0.1025934
1: -0.0135457, 0.0135796, -0.0152119, 0.0143234, -0.0278691, 0.0287915
2: -0.0017059, 0.0367558, -0.0034036, 0.0355157, -0.0372216, 0.0381460
3: -0.0023984, 0.0668044, -0.0049208, 0.0644339, -0.0505283, 0.0536429
4: -0.0241064, -0.0030614, -0.0250397, -0.0023599, -0.0217464, 0.0219782
5: 0.0068713, 0.0477711, 0.0049993, 0.0464991, -0.0396278, 0.0427719
6: -0.0327018, 0.0491298, -0.0348876, 0.0455635, -0.0782653, 0.0840174
7: -0.0150089, 0.0112847, -0.0160587, 0.0126032, -0.0276121, 0.0273434
8: 0.7033206, 0.9411473, 0.7135336, 0.9456578, -0.1871364, 0.1723211
9: 0.0512032, 0.0920525, 0.0491103, 0.0908419, -0.0396387, 0.0429423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442871, 0.0534906, -0.0947534, 0.1026359
1: -0.0135457, 0.0135796, -0.0152351, 0.0143338, -0.0278795, 0.0288147
2: -0.0017059, 0.0367558, -0.0034273, 0.0350586, -0.0367645, 0.0383396
3: -0.0023984, 0.0668044, -0.0049559, 0.0635683, -0.0515055, 0.0556709
4: -0.0241064, -0.0030614, -0.0250527, -0.0023502, -0.0217561, 0.0219912
5: 0.0068713, 0.0477711, 0.0049731, 0.0460303, -0.0391590, 0.0427980
6: -0.0327018, 0.0491298, -0.0349180, 0.0442495, -0.0769513, 0.0840478
7: -0.0150089, 0.0112847, -0.0160732, 0.0126215, -0.0276304, 0.0273579
8: 0.7033206, 0.9411473, 0.7172979, 0.9457193, -0.1894178, 0.1712241
9: 0.0512032, 0.0920525, 0.0490810, 0.0904766, -0.0392734, 0.0429715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442470, 0.0548080, -0.0949187, 0.1029401
1: -0.0128829, 0.0132836, -0.0152119, 0.0143234, -0.0272064, 0.0284955
2: -0.0010305, 0.0368735, -0.0034036, 0.0355157, -0.0365461, 0.0380358
3: -0.0013947, 0.0670294, -0.0049208, 0.0644339, -0.0517675, 0.0553089
4: -0.0237351, -0.0032822, -0.0250397, -0.0023599, -0.0213751, 0.0217575
5: 0.0076161, 0.0478919, 0.0049993, 0.0464991, -0.0388830, 0.0428927
6: -0.0318324, 0.0494685, -0.0348876, 0.0455635, -0.0773959, 0.0843561
7: -0.0145913, 0.0107602, -0.0160587, 0.0126032, -0.0271946, 0.0268189
8: 0.7023503, 0.9393543, 0.7135336, 0.9456578, -0.1907670, 0.1726165
9: 0.0520359, 0.0921682, 0.0491103, 0.0908419, -0.0388060, 0.0430580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442871, 0.0534906, -0.0936016, 0.1029809
1: -0.0128829, 0.0132836, -0.0152351, 0.0143338, -0.0272167, 0.0285187
2: -0.0010305, 0.0368735, -0.0034273, 0.0350586, -0.0360891, 0.0377323
3: -0.0013947, 0.0670294, -0.0049559, 0.0635683, -0.0493166, 0.0538580
4: -0.0237351, -0.0032822, -0.0250527, -0.0023502, -0.0213849, 0.0217705
5: 0.0076161, 0.0478919, 0.0049731, 0.0460303, -0.0384142, 0.0429188
6: -0.0318324, 0.0494685, -0.0349180, 0.0442495, -0.0760819, 0.0843865
7: -0.0145913, 0.0107602, -0.0160732, 0.0126215, -0.0272128, 0.0268334
8: 0.7023503, 0.9393543, 0.7172979, 0.9457193, -0.1894736, 0.1680372
9: 0.0520359, 0.0921682, 0.0490810, 0.0904766, -0.0384407, 0.0430872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442470, 0.0548080, -0.0960693, 0.1025934
1: -0.0135457, 0.0135796, -0.0152119, 0.0143234, -0.0278691, 0.0287915
2: -0.0017059, 0.0367558, -0.0034036, 0.0355157, -0.0372216, 0.0381460
3: -0.0023984, 0.0668044, -0.0049208, 0.0644339, -0.0505283, 0.0536429
4: -0.0241064, -0.0030614, -0.0250397, -0.0023599, -0.0217464, 0.0219782
5: 0.0068713, 0.0477711, 0.0049993, 0.0464991, -0.0396278, 0.0427719
6: -0.0327018, 0.0491298, -0.0348876, 0.0455635, -0.0782653, 0.0840174
7: -0.0150089, 0.0112847, -0.0160587, 0.0126032, -0.0276121, 0.0273434
8: 0.7033206, 0.9411473, 0.7135336, 0.9456578, -0.1871364, 0.1723211
9: 0.0512032, 0.0920525, 0.0491103, 0.0908419, -0.0396387, 0.0429423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442871, 0.0534906, -0.0947534, 0.1026359
1: -0.0135457, 0.0135796, -0.0152351, 0.0143338, -0.0278795, 0.0288147
2: -0.0017059, 0.0367558, -0.0034273, 0.0350586, -0.0367645, 0.0383396
3: -0.0023984, 0.0668044, -0.0049559, 0.0635683, -0.0515055, 0.0556709
4: -0.0241064, -0.0030614, -0.0250527, -0.0023502, -0.0217561, 0.0219912
5: 0.0068713, 0.0477711, 0.0049731, 0.0460303, -0.0391590, 0.0427980
6: -0.0327018, 0.0491298, -0.0349180, 0.0442495, -0.0769513, 0.0840478
7: -0.0150089, 0.0112847, -0.0160732, 0.0126215, -0.0276304, 0.0273579
8: 0.7033206, 0.9411473, 0.7172979, 0.9457193, -0.1894178, 0.1712241
9: 0.0512032, 0.0920525, 0.0490810, 0.0904766, -0.0392734, 0.0429715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442470, 0.0548080, -0.0949187, 0.1029401
1: -0.0128829, 0.0132836, -0.0152119, 0.0143234, -0.0272064, 0.0284955
2: -0.0010305, 0.0368735, -0.0034036, 0.0355157, -0.0365461, 0.0380358
3: -0.0013947, 0.0670294, -0.0049208, 0.0644339, -0.0517675, 0.0553089
4: -0.0237351, -0.0032822, -0.0250397, -0.0023599, -0.0213751, 0.0217575
5: 0.0076161, 0.0478919, 0.0049993, 0.0464991, -0.0388830, 0.0428927
6: -0.0318324, 0.0494685, -0.0348876, 0.0455635, -0.0773959, 0.0843561
7: -0.0145913, 0.0107602, -0.0160587, 0.0126032, -0.0271946, 0.0268189
8: 0.7023503, 0.9393543, 0.7135336, 0.9456578, -0.1907670, 0.1726165
9: 0.0520359, 0.0921682, 0.0491103, 0.0908419, -0.0388060, 0.0430580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442871, 0.0534906, -0.0936016, 0.1029809
1: -0.0128829, 0.0132836, -0.0152351, 0.0143338, -0.0272167, 0.0285187
2: -0.0010305, 0.0368735, -0.0034273, 0.0350586, -0.0360891, 0.0377323
3: -0.0013947, 0.0670294, -0.0049559, 0.0635683, -0.0493166, 0.0538580
4: -0.0237351, -0.0032822, -0.0250527, -0.0023502, -0.0213849, 0.0217705
5: 0.0076161, 0.0478919, 0.0049731, 0.0460303, -0.0384142, 0.0429188
6: -0.0318324, 0.0494685, -0.0349180, 0.0442495, -0.0760819, 0.0843865
7: -0.0145913, 0.0107602, -0.0160732, 0.0126215, -0.0272128, 0.0268334
8: 0.7023503, 0.9393543, 0.7172979, 0.9457193, -0.1894736, 0.1680372
9: 0.0520359, 0.0921682, 0.0490810, 0.0904766, -0.0384407, 0.0430872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442470, 0.0548080, -0.0960693, 0.1025934
1: -0.0135457, 0.0135796, -0.0152119, 0.0143234, -0.0278691, 0.0287915
2: -0.0017059, 0.0367558, -0.0034036, 0.0355157, -0.0372216, 0.0381460
3: -0.0023984, 0.0668044, -0.0049208, 0.0644339, -0.0505283, 0.0536429
4: -0.0241064, -0.0030614, -0.0250397, -0.0023599, -0.0217464, 0.0219782
5: 0.0068713, 0.0477711, 0.0049993, 0.0464991, -0.0396278, 0.0427719
6: -0.0327018, 0.0491298, -0.0348876, 0.0455635, -0.0782653, 0.0840174
7: -0.0150089, 0.0112847, -0.0160587, 0.0126032, -0.0276121, 0.0273434
8: 0.7033206, 0.9411473, 0.7135336, 0.9456578, -0.1871364, 0.1723211
9: 0.0512032, 0.0920525, 0.0491103, 0.0908419, -0.0396387, 0.0429423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442871, 0.0534906, -0.0947534, 0.1026359
1: -0.0135457, 0.0135796, -0.0152351, 0.0143338, -0.0278795, 0.0288147
2: -0.0017059, 0.0367558, -0.0034273, 0.0350586, -0.0367645, 0.0383396
3: -0.0023984, 0.0668044, -0.0049559, 0.0635683, -0.0515055, 0.0556709
4: -0.0241064, -0.0030614, -0.0250527, -0.0023502, -0.0217561, 0.0219912
5: 0.0068713, 0.0477711, 0.0049731, 0.0460303, -0.0391590, 0.0427980
6: -0.0327018, 0.0491298, -0.0349180, 0.0442495, -0.0769513, 0.0840478
7: -0.0150089, 0.0112847, -0.0160732, 0.0126215, -0.0276304, 0.0273579
8: 0.7033206, 0.9411473, 0.7172979, 0.9457193, -0.1894178, 0.1712241
9: 0.0512032, 0.0920525, 0.0490810, 0.0904766, -0.0392734, 0.0429715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442470, 0.0548080, -0.0949187, 0.1029401
1: -0.0128829, 0.0132836, -0.0152119, 0.0143234, -0.0272064, 0.0284955
2: -0.0010305, 0.0368735, -0.0034036, 0.0355157, -0.0365461, 0.0380358
3: -0.0013947, 0.0670294, -0.0049208, 0.0644339, -0.0517675, 0.0553089
4: -0.0237351, -0.0032822, -0.0250397, -0.0023599, -0.0213751, 0.0217575
5: 0.0076161, 0.0478919, 0.0049993, 0.0464991, -0.0388830, 0.0428927
6: -0.0318324, 0.0494685, -0.0348876, 0.0455635, -0.0773959, 0.0843561
7: -0.0145913, 0.0107602, -0.0160587, 0.0126032, -0.0271946, 0.0268189
8: 0.7023503, 0.9393543, 0.7135336, 0.9456578, -0.1907670, 0.1726165
9: 0.0520359, 0.0921682, 0.0491103, 0.0908419, -0.0388060, 0.0430580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442871, 0.0534906, -0.0936016, 0.1029809
1: -0.0128829, 0.0132836, -0.0152351, 0.0143338, -0.0272167, 0.0285187
2: -0.0010305, 0.0368735, -0.0034273, 0.0350586, -0.0360891, 0.0377323
3: -0.0013947, 0.0670294, -0.0049559, 0.0635683, -0.0493166, 0.0538580
4: -0.0237351, -0.0032822, -0.0250527, -0.0023502, -0.0213849, 0.0217705
5: 0.0076161, 0.0478919, 0.0049731, 0.0460303, -0.0384142, 0.0429188
6: -0.0318324, 0.0494685, -0.0349180, 0.0442495, -0.0760819, 0.0843865
7: -0.0145913, 0.0107602, -0.0160732, 0.0126215, -0.0272128, 0.0268334
8: 0.7023503, 0.9393543, 0.7172979, 0.9457193, -0.1894736, 0.1680372
9: 0.0520359, 0.0921682, 0.0490810, 0.0904766, -0.0384407, 0.0430872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0401980, 0.0572768, -0.0413505, 0.0584358, -0.0985444, 0.0985377
1: -0.0128829, 0.0132835, -0.0135457, 0.0135796, -0.0264625, 0.0268292
2: -0.0010304, 0.0363597, -0.0017059, 0.0367558, -0.0359790, 0.0355918
3: -0.0013946, 0.0660469, -0.0023984, 0.0668044, -0.0517202, 0.0500633
4: -0.0237351, -0.0033405, -0.0241064, -0.0030614, -0.0206736, 0.0207659
5: 0.0076161, 0.0473649, 0.0068713, 0.0477711, -0.0401550, 0.0404936
6: -0.0318319, 0.0479909, -0.0327018, 0.0491298, -0.0809617, 0.0806927
7: -0.0145914, 0.0107601, -0.0150089, 0.0112847, -0.0258761, 0.0257690
8: 0.7065819, 0.9393533, 0.7033206, 0.9411473, -0.1803710, 0.1813164
9: 0.0520357, 0.0916632, 0.0512032, 0.0920525, -0.0400168, 0.0404600

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1074684, upper bound: 0.0902061
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1024660, upper bound: 0.0902061
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0401980, 0.0572768, -0.0401982, 0.0587806, -0.0988878, 0.0973837
1: -0.0128829, 0.0132835, -0.0128829, 0.0132836, -0.0261665, 0.0261665
2: -0.0010304, 0.0363597, -0.0010305, 0.0368735, -0.0353518, 0.0347543
3: -0.0013946, 0.0660469, -0.0013947, 0.0670294, -0.0492666, 0.0469156
4: -0.0237351, -0.0033405, -0.0237351, -0.0032822, -0.0204528, 0.0203946
5: 0.0076161, 0.0473649, 0.0076161, 0.0478919, -0.0402758, 0.0397487
6: -0.0318319, 0.0479909, -0.0318324, 0.0494685, -0.0813005, 0.0798233
7: -0.0145914, 0.0107601, -0.0145913, 0.0107602, -0.0253516, 0.0253515
8: 0.7065819, 0.9393533, 0.7023503, 0.9393543, -0.1764905, 0.1807961
9: 0.0520357, 0.0916632, 0.0520359, 0.0921682, -0.0401325, 0.0396273

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1071041, upper bound: 0.0901719
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1071041, upper bound: 0.0902061
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0413505, 0.0584358, -0.0985444, 0.1000414
1: -0.0128829, 0.0132836, -0.0135457, 0.0135796, -0.0264625, 0.0268293
2: -0.0010305, 0.0368735, -0.0017059, 0.0367558, -0.0359803, 0.0362400
3: -0.0013947, 0.0670294, -0.0023984, 0.0668044, -0.0517358, 0.0521227
4: -0.0237351, -0.0032822, -0.0241064, -0.0030614, -0.0206736, 0.0208242
5: 0.0076161, 0.0478919, 0.0068713, 0.0477711, -0.0401550, 0.0410206
6: -0.0318324, 0.0494685, -0.0327018, 0.0491298, -0.0809622, 0.0821703
7: -0.0145913, 0.0107602, -0.0150089, 0.0112847, -0.0258760, 0.0257691
8: 0.7023503, 0.9393543, 0.7033206, 0.9411473, -0.1846697, 0.1813202
9: 0.0520359, 0.0921682, 0.0512032, 0.0920525, -0.0400166, 0.0409651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1074684, upper bound: 0.0902061
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1024660, upper bound: 0.0969314
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0401982, 0.0587806, -0.0988878, 0.0988877
1: -0.0128829, 0.0132836, -0.0128829, 0.0132836, -0.0261665, 0.0261665
2: -0.0010305, 0.0368735, -0.0010305, 0.0368735, -0.0353519, 0.0353519
3: -0.0013947, 0.0670294, -0.0013947, 0.0670294, -0.0492703, 0.0492703
4: -0.0237351, -0.0032822, -0.0237351, -0.0032822, -0.0204529, 0.0204529
5: 0.0076161, 0.0478919, 0.0076161, 0.0478919, -0.0402758, 0.0402758
6: -0.0318324, 0.0494685, -0.0318324, 0.0494685, -0.0813009, 0.0813009
7: -0.0145913, 0.0107602, -0.0145913, 0.0107602, -0.0253515, 0.0253515
8: 0.7023503, 0.9393543, 0.7023503, 0.9393543, -0.1807964, 0.1807964
9: 0.0520359, 0.0921682, 0.0520359, 0.0921682, -0.0401324, 0.0401324

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1118365, upper bound: 0.1068057
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1121336, upper bound: 0.1068673
time: 0.57 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.70 seconds
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1109350
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1109350
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1074684, upper bound: 0.0902061
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1024660, upper bound: 0.0902061
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1071041, upper bound: 0.0901719
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1071041, upper bound: 0.0902061
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1074684, upper bound: 0.0902061
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1024660, upper bound: 0.0969314
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1118365, upper bound: 0.1068057
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 8, lower bound: -0.1121336, upper bound: 0.1068673

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0413505, 0.0584358, -0.1025935, 0.0960693
1: -0.0152119, 0.0143234, -0.0135457, 0.0135796, -0.0287915, 0.0278691
2: -0.0034036, 0.0355157, -0.0017059, 0.0367558, -0.0381460, 0.0372216
3: -0.0049208, 0.0644339, -0.0023984, 0.0668044, -0.0536429, 0.0505283
4: -0.0250397, -0.0023599, -0.0241064, -0.0030614, -0.0219782, 0.0217464
5: 0.0049993, 0.0464991, 0.0068713, 0.0477711, -0.0427719, 0.0396278
6: -0.0348876, 0.0455635, -0.0327018, 0.0491298, -0.0840174, 0.0782653
7: -0.0160587, 0.0126032, -0.0150089, 0.0112847, -0.0273434, 0.0276121
8: 0.7135336, 0.9456578, 0.7033206, 0.9411473, -0.1723211, 0.1871364
9: 0.0491103, 0.0908419, 0.0512032, 0.0920525, -0.0429423, 0.0396387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0413505, 0.0584358, -0.1026357, 0.0947535
1: -0.0152351, 0.0143338, -0.0135457, 0.0135796, -0.0288147, 0.0278795
2: -0.0034273, 0.0350586, -0.0017059, 0.0367558, -0.0383397, 0.0367645
3: -0.0049559, 0.0635683, -0.0023984, 0.0668044, -0.0556710, 0.0515055
4: -0.0250527, -0.0023502, -0.0241064, -0.0030614, -0.0219912, 0.0217561
5: 0.0049731, 0.0460303, 0.0068713, 0.0477711, -0.0427980, 0.0391590
6: -0.0349180, 0.0442495, -0.0327018, 0.0491298, -0.0840478, 0.0769513
7: -0.0160732, 0.0126215, -0.0150089, 0.0112847, -0.0273579, 0.0276304
8: 0.7172979, 0.9457193, 0.7033206, 0.9411473, -0.1712241, 0.1894178
9: 0.0490810, 0.0904766, 0.0512032, 0.0920525, -0.0429715, 0.0392734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0401982, 0.0587806, -0.1029403, 0.0949187
1: -0.0152119, 0.0143234, -0.0128829, 0.0132836, -0.0284955, 0.0272064
2: -0.0034036, 0.0355157, -0.0010305, 0.0368735, -0.0380357, 0.0365461
3: -0.0049208, 0.0644339, -0.0013947, 0.0670294, -0.0553089, 0.0517675
4: -0.0250397, -0.0023599, -0.0237351, -0.0032822, -0.0217575, 0.0213751
5: 0.0049993, 0.0464991, 0.0076161, 0.0478919, -0.0428927, 0.0388830
6: -0.0348876, 0.0455635, -0.0318324, 0.0494685, -0.0843561, 0.0773959
7: -0.0160587, 0.0126032, -0.0145913, 0.0107602, -0.0268189, 0.0271946
8: 0.7135336, 0.9456578, 0.7023503, 0.9393543, -0.1726165, 0.1907670
9: 0.0491103, 0.0908419, 0.0520359, 0.0921682, -0.0430580, 0.0388060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0401982, 0.0587806, -0.1029809, 0.0936015
1: -0.0152351, 0.0143338, -0.0128829, 0.0132836, -0.0285187, 0.0272167
2: -0.0034273, 0.0350586, -0.0010305, 0.0368735, -0.0377323, 0.0360891
3: -0.0049559, 0.0635683, -0.0013947, 0.0670294, -0.0538581, 0.0493166
4: -0.0250527, -0.0023502, -0.0237351, -0.0032822, -0.0217705, 0.0213849
5: 0.0049731, 0.0460303, 0.0076161, 0.0478919, -0.0429188, 0.0384142
6: -0.0349180, 0.0442495, -0.0318324, 0.0494685, -0.0843865, 0.0760819
7: -0.0160732, 0.0126215, -0.0145913, 0.0107602, -0.0268334, 0.0272128
8: 0.7172979, 0.9457193, 0.7023503, 0.9393543, -0.1680372, 0.1894736
9: 0.0490810, 0.0904766, 0.0520359, 0.0921682, -0.0430872, 0.0384407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0413505, 0.0584358, -0.1025935, 0.0960693
1: -0.0152119, 0.0143234, -0.0135457, 0.0135796, -0.0287915, 0.0278691
2: -0.0034036, 0.0355157, -0.0017059, 0.0367558, -0.0381460, 0.0372216
3: -0.0049208, 0.0644339, -0.0023984, 0.0668044, -0.0536429, 0.0505283
4: -0.0250397, -0.0023599, -0.0241064, -0.0030614, -0.0219782, 0.0217464
5: 0.0049993, 0.0464991, 0.0068713, 0.0477711, -0.0427719, 0.0396278
6: -0.0348876, 0.0455635, -0.0327018, 0.0491298, -0.0840174, 0.0782653
7: -0.0160587, 0.0126032, -0.0150089, 0.0112847, -0.0273434, 0.0276121
8: 0.7135336, 0.9456578, 0.7033206, 0.9411473, -0.1723211, 0.1871364
9: 0.0491103, 0.0908419, 0.0512032, 0.0920525, -0.0429423, 0.0396387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0413505, 0.0584358, -0.1026357, 0.0947535
1: -0.0152351, 0.0143338, -0.0135457, 0.0135796, -0.0288147, 0.0278795
2: -0.0034273, 0.0350586, -0.0017059, 0.0367558, -0.0383397, 0.0367645
3: -0.0049559, 0.0635683, -0.0023984, 0.0668044, -0.0556710, 0.0515055
4: -0.0250527, -0.0023502, -0.0241064, -0.0030614, -0.0219912, 0.0217561
5: 0.0049731, 0.0460303, 0.0068713, 0.0477711, -0.0427980, 0.0391590
6: -0.0349180, 0.0442495, -0.0327018, 0.0491298, -0.0840478, 0.0769513
7: -0.0160732, 0.0126215, -0.0150089, 0.0112847, -0.0273579, 0.0276304
8: 0.7172979, 0.9457193, 0.7033206, 0.9411473, -0.1712241, 0.1894178
9: 0.0490810, 0.0904766, 0.0512032, 0.0920525, -0.0429715, 0.0392734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0401982, 0.0587806, -0.1029403, 0.0949187
1: -0.0152119, 0.0143234, -0.0128829, 0.0132836, -0.0284955, 0.0272064
2: -0.0034036, 0.0355157, -0.0010305, 0.0368735, -0.0380357, 0.0365461
3: -0.0049208, 0.0644339, -0.0013947, 0.0670294, -0.0553089, 0.0517675
4: -0.0250397, -0.0023599, -0.0237351, -0.0032822, -0.0217575, 0.0213751
5: 0.0049993, 0.0464991, 0.0076161, 0.0478919, -0.0428927, 0.0388830
6: -0.0348876, 0.0455635, -0.0318324, 0.0494685, -0.0843561, 0.0773959
7: -0.0160587, 0.0126032, -0.0145913, 0.0107602, -0.0268189, 0.0271946
8: 0.7135336, 0.9456578, 0.7023503, 0.9393543, -0.1726165, 0.1907670
9: 0.0491103, 0.0908419, 0.0520359, 0.0921682, -0.0430580, 0.0388060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0401982, 0.0587806, -0.1029809, 0.0936015
1: -0.0152351, 0.0143338, -0.0128829, 0.0132836, -0.0285187, 0.0272167
2: -0.0034273, 0.0350586, -0.0010305, 0.0368735, -0.0377323, 0.0360891
3: -0.0049559, 0.0635683, -0.0013947, 0.0670294, -0.0538581, 0.0493166
4: -0.0250527, -0.0023502, -0.0237351, -0.0032822, -0.0217705, 0.0213849
5: 0.0049731, 0.0460303, 0.0076161, 0.0478919, -0.0429188, 0.0384142
6: -0.0349180, 0.0442495, -0.0318324, 0.0494685, -0.0843865, 0.0760819
7: -0.0160732, 0.0126215, -0.0145913, 0.0107602, -0.0268334, 0.0272128
8: 0.7172979, 0.9457193, 0.7023503, 0.9393543, -0.1680372, 0.1894736
9: 0.0490810, 0.0904766, 0.0520359, 0.0921682, -0.0430872, 0.0384407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0413505, 0.0584358, -0.1025935, 0.0960693
1: -0.0152119, 0.0143234, -0.0135457, 0.0135796, -0.0287915, 0.0278691
2: -0.0034036, 0.0355157, -0.0017059, 0.0367558, -0.0381460, 0.0372216
3: -0.0049208, 0.0644339, -0.0023984, 0.0668044, -0.0536429, 0.0505283
4: -0.0250397, -0.0023599, -0.0241064, -0.0030614, -0.0219782, 0.0217464
5: 0.0049993, 0.0464991, 0.0068713, 0.0477711, -0.0427719, 0.0396278
6: -0.0348876, 0.0455635, -0.0327018, 0.0491298, -0.0840174, 0.0782653
7: -0.0160587, 0.0126032, -0.0150089, 0.0112847, -0.0273434, 0.0276121
8: 0.7135336, 0.9456578, 0.7033206, 0.9411473, -0.1723211, 0.1871364
9: 0.0491103, 0.0908419, 0.0512032, 0.0920525, -0.0429423, 0.0396387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0413505, 0.0584358, -0.1026357, 0.0947535
1: -0.0152351, 0.0143338, -0.0135457, 0.0135796, -0.0288147, 0.0278795
2: -0.0034273, 0.0350586, -0.0017059, 0.0367558, -0.0383397, 0.0367645
3: -0.0049559, 0.0635683, -0.0023984, 0.0668044, -0.0556710, 0.0515055
4: -0.0250527, -0.0023502, -0.0241064, -0.0030614, -0.0219912, 0.0217561
5: 0.0049731, 0.0460303, 0.0068713, 0.0477711, -0.0427980, 0.0391590
6: -0.0349180, 0.0442495, -0.0327018, 0.0491298, -0.0840478, 0.0769513
7: -0.0160732, 0.0126215, -0.0150089, 0.0112847, -0.0273579, 0.0276304
8: 0.7172979, 0.9457193, 0.7033206, 0.9411473, -0.1712241, 0.1894178
9: 0.0490810, 0.0904766, 0.0512032, 0.0920525, -0.0429715, 0.0392734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0401982, 0.0587806, -0.1029403, 0.0949187
1: -0.0152119, 0.0143234, -0.0128829, 0.0132836, -0.0284955, 0.0272064
2: -0.0034036, 0.0355157, -0.0010305, 0.0368735, -0.0380357, 0.0365461
3: -0.0049208, 0.0644339, -0.0013947, 0.0670294, -0.0553089, 0.0517675
4: -0.0250397, -0.0023599, -0.0237351, -0.0032822, -0.0217575, 0.0213751
5: 0.0049993, 0.0464991, 0.0076161, 0.0478919, -0.0428927, 0.0388830
6: -0.0348876, 0.0455635, -0.0318324, 0.0494685, -0.0843561, 0.0773959
7: -0.0160587, 0.0126032, -0.0145913, 0.0107602, -0.0268189, 0.0271946
8: 0.7135336, 0.9456578, 0.7023503, 0.9393543, -0.1726165, 0.1907670
9: 0.0491103, 0.0908419, 0.0520359, 0.0921682, -0.0430580, 0.0388060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0401982, 0.0587806, -0.1029809, 0.0936015
1: -0.0152351, 0.0143338, -0.0128829, 0.0132836, -0.0285187, 0.0272167
2: -0.0034273, 0.0350586, -0.0010305, 0.0368735, -0.0377323, 0.0360891
3: -0.0049559, 0.0635683, -0.0013947, 0.0670294, -0.0538581, 0.0493166
4: -0.0250527, -0.0023502, -0.0237351, -0.0032822, -0.0217705, 0.0213849
5: 0.0049731, 0.0460303, 0.0076161, 0.0478919, -0.0429188, 0.0384142
6: -0.0349180, 0.0442495, -0.0318324, 0.0494685, -0.0843865, 0.0760819
7: -0.0160732, 0.0126215, -0.0145913, 0.0107602, -0.0268334, 0.0272128
8: 0.7172979, 0.9457193, 0.7023503, 0.9393543, -0.1680372, 0.1894736
9: 0.0490810, 0.0904766, 0.0520359, 0.0921682, -0.0430872, 0.0384407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0413505, 0.0584358, -0.1025935, 0.0960693
1: -0.0152119, 0.0143234, -0.0135457, 0.0135796, -0.0287915, 0.0278691
2: -0.0034036, 0.0355157, -0.0017059, 0.0367558, -0.0381460, 0.0372216
3: -0.0049208, 0.0644339, -0.0023984, 0.0668044, -0.0536429, 0.0505283
4: -0.0250397, -0.0023599, -0.0241064, -0.0030614, -0.0219782, 0.0217464
5: 0.0049993, 0.0464991, 0.0068713, 0.0477711, -0.0427719, 0.0396278
6: -0.0348876, 0.0455635, -0.0327018, 0.0491298, -0.0840174, 0.0782653
7: -0.0160587, 0.0126032, -0.0150089, 0.0112847, -0.0273434, 0.0276121
8: 0.7135336, 0.9456578, 0.7033206, 0.9411473, -0.1723211, 0.1871364
9: 0.0491103, 0.0908419, 0.0512032, 0.0920525, -0.0429423, 0.0396387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0413505, 0.0584358, -0.1026357, 0.0947535
1: -0.0152351, 0.0143338, -0.0135457, 0.0135796, -0.0288147, 0.0278795
2: -0.0034273, 0.0350586, -0.0017059, 0.0367558, -0.0383397, 0.0367645
3: -0.0049559, 0.0635683, -0.0023984, 0.0668044, -0.0556710, 0.0515055
4: -0.0250527, -0.0023502, -0.0241064, -0.0030614, -0.0219912, 0.0217561
5: 0.0049731, 0.0460303, 0.0068713, 0.0477711, -0.0427980, 0.0391590
6: -0.0349180, 0.0442495, -0.0327018, 0.0491298, -0.0840478, 0.0769513
7: -0.0160732, 0.0126215, -0.0150089, 0.0112847, -0.0273579, 0.0276304
8: 0.7172979, 0.9457193, 0.7033206, 0.9411473, -0.1712241, 0.1894178
9: 0.0490810, 0.0904766, 0.0512032, 0.0920525, -0.0429715, 0.0392734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0401982, 0.0587806, -0.1029403, 0.0949187
1: -0.0152119, 0.0143234, -0.0128829, 0.0132836, -0.0284955, 0.0272064
2: -0.0034036, 0.0355157, -0.0010305, 0.0368735, -0.0380357, 0.0365461
3: -0.0049208, 0.0644339, -0.0013947, 0.0670294, -0.0553089, 0.0517675
4: -0.0250397, -0.0023599, -0.0237351, -0.0032822, -0.0217575, 0.0213751
5: 0.0049993, 0.0464991, 0.0076161, 0.0478919, -0.0428927, 0.0388830
6: -0.0348876, 0.0455635, -0.0318324, 0.0494685, -0.0843561, 0.0773959
7: -0.0160587, 0.0126032, -0.0145913, 0.0107602, -0.0268189, 0.0271946
8: 0.7135336, 0.9456578, 0.7023503, 0.9393543, -0.1726165, 0.1907670
9: 0.0491103, 0.0908419, 0.0520359, 0.0921682, -0.0430580, 0.0388060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0401982, 0.0587806, -0.1029809, 0.0936015
1: -0.0152351, 0.0143338, -0.0128829, 0.0132836, -0.0285187, 0.0272167
2: -0.0034273, 0.0350586, -0.0010305, 0.0368735, -0.0377323, 0.0360891
3: -0.0049559, 0.0635683, -0.0013947, 0.0670294, -0.0538581, 0.0493166
4: -0.0250527, -0.0023502, -0.0237351, -0.0032822, -0.0217705, 0.0213849
5: 0.0049731, 0.0460303, 0.0076161, 0.0478919, -0.0429188, 0.0384142
6: -0.0349180, 0.0442495, -0.0318324, 0.0494685, -0.0843865, 0.0760819
7: -0.0160732, 0.0126215, -0.0145913, 0.0107602, -0.0268334, 0.0272128
8: 0.7172979, 0.9457193, 0.7023503, 0.9393543, -0.1680372, 0.1894736
9: 0.0490810, 0.0904766, 0.0520359, 0.0921682, -0.0430872, 0.0384407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0413505, 0.0584358, -0.1025935, 0.0960693
1: -0.0152119, 0.0143234, -0.0135457, 0.0135796, -0.0287915, 0.0278691
2: -0.0034036, 0.0355157, -0.0017059, 0.0367558, -0.0381460, 0.0372216
3: -0.0049208, 0.0644339, -0.0023984, 0.0668044, -0.0536429, 0.0505283
4: -0.0250397, -0.0023599, -0.0241064, -0.0030614, -0.0219782, 0.0217464
5: 0.0049993, 0.0464991, 0.0068713, 0.0477711, -0.0427719, 0.0396278
6: -0.0348876, 0.0455635, -0.0327018, 0.0491298, -0.0840174, 0.0782653
7: -0.0160587, 0.0126032, -0.0150089, 0.0112847, -0.0273434, 0.0276121
8: 0.7135336, 0.9456578, 0.7033206, 0.9411473, -0.1723211, 0.1871364
9: 0.0491103, 0.0908419, 0.0512032, 0.0920525, -0.0429423, 0.0396387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0413505, 0.0584358, -0.1026357, 0.0947535
1: -0.0152351, 0.0143338, -0.0135457, 0.0135796, -0.0288147, 0.0278795
2: -0.0034273, 0.0350586, -0.0017059, 0.0367558, -0.0383397, 0.0367645
3: -0.0049559, 0.0635683, -0.0023984, 0.0668044, -0.0556710, 0.0515055
4: -0.0250527, -0.0023502, -0.0241064, -0.0030614, -0.0219912, 0.0217561
5: 0.0049731, 0.0460303, 0.0068713, 0.0477711, -0.0427980, 0.0391590
6: -0.0349180, 0.0442495, -0.0327018, 0.0491298, -0.0840478, 0.0769513
7: -0.0160732, 0.0126215, -0.0150089, 0.0112847, -0.0273579, 0.0276304
8: 0.7172979, 0.9457193, 0.7033206, 0.9411473, -0.1712241, 0.1894178
9: 0.0490810, 0.0904766, 0.0512032, 0.0920525, -0.0429715, 0.0392734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0401982, 0.0587806, -0.1029403, 0.0949187
1: -0.0152119, 0.0143234, -0.0128829, 0.0132836, -0.0284955, 0.0272064
2: -0.0034036, 0.0355157, -0.0010305, 0.0368735, -0.0380357, 0.0365461
3: -0.0049208, 0.0644339, -0.0013947, 0.0670294, -0.0553089, 0.0517675
4: -0.0250397, -0.0023599, -0.0237351, -0.0032822, -0.0217575, 0.0213751
5: 0.0049993, 0.0464991, 0.0076161, 0.0478919, -0.0428927, 0.0388830
6: -0.0348876, 0.0455635, -0.0318324, 0.0494685, -0.0843561, 0.0773959
7: -0.0160587, 0.0126032, -0.0145913, 0.0107602, -0.0268189, 0.0271946
8: 0.7135336, 0.9456578, 0.7023503, 0.9393543, -0.1726165, 0.1907670
9: 0.0491103, 0.0908419, 0.0520359, 0.0921682, -0.0430580, 0.0388060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0401982, 0.0587806, -0.1029809, 0.0936015
1: -0.0152351, 0.0143338, -0.0128829, 0.0132836, -0.0285187, 0.0272167
2: -0.0034273, 0.0350586, -0.0010305, 0.0368735, -0.0377323, 0.0360891
3: -0.0049559, 0.0635683, -0.0013947, 0.0670294, -0.0538581, 0.0493166
4: -0.0250527, -0.0023502, -0.0237351, -0.0032822, -0.0217705, 0.0213849
5: 0.0049731, 0.0460303, 0.0076161, 0.0478919, -0.0429188, 0.0384142
6: -0.0349180, 0.0442495, -0.0318324, 0.0494685, -0.0843865, 0.0760819
7: -0.0160732, 0.0126215, -0.0145913, 0.0107602, -0.0268334, 0.0272128
8: 0.7172979, 0.9457193, 0.7023503, 0.9393543, -0.1680372, 0.1894736
9: 0.0490810, 0.0904766, 0.0520359, 0.0921682, -0.0430872, 0.0384407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0413505, 0.0584358, -0.1025935, 0.0960693
1: -0.0152119, 0.0143234, -0.0135457, 0.0135796, -0.0287915, 0.0278691
2: -0.0034036, 0.0355157, -0.0017059, 0.0367558, -0.0381460, 0.0372216
3: -0.0049208, 0.0644339, -0.0023984, 0.0668044, -0.0536429, 0.0505283
4: -0.0250397, -0.0023599, -0.0241064, -0.0030614, -0.0219782, 0.0217464
5: 0.0049993, 0.0464991, 0.0068713, 0.0477711, -0.0427719, 0.0396278
6: -0.0348876, 0.0455635, -0.0327018, 0.0491298, -0.0840174, 0.0782653
7: -0.0160587, 0.0126032, -0.0150089, 0.0112847, -0.0273434, 0.0276121
8: 0.7135336, 0.9456578, 0.7033206, 0.9411473, -0.1723211, 0.1871364
9: 0.0491103, 0.0908419, 0.0512032, 0.0920525, -0.0429423, 0.0396387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0413505, 0.0584358, -0.1026357, 0.0947535
1: -0.0152351, 0.0143338, -0.0135457, 0.0135796, -0.0288147, 0.0278795
2: -0.0034273, 0.0350586, -0.0017059, 0.0367558, -0.0383397, 0.0367645
3: -0.0049559, 0.0635683, -0.0023984, 0.0668044, -0.0556710, 0.0515055
4: -0.0250527, -0.0023502, -0.0241064, -0.0030614, -0.0219912, 0.0217561
5: 0.0049731, 0.0460303, 0.0068713, 0.0477711, -0.0427980, 0.0391590
6: -0.0349180, 0.0442495, -0.0327018, 0.0491298, -0.0840478, 0.0769513
7: -0.0160732, 0.0126215, -0.0150089, 0.0112847, -0.0273579, 0.0276304
8: 0.7172979, 0.9457193, 0.7033206, 0.9411473, -0.1712241, 0.1894178
9: 0.0490810, 0.0904766, 0.0512032, 0.0920525, -0.0429715, 0.0392734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0401982, 0.0587806, -0.1029403, 0.0949187
1: -0.0152119, 0.0143234, -0.0128829, 0.0132836, -0.0284955, 0.0272064
2: -0.0034036, 0.0355157, -0.0010305, 0.0368735, -0.0380357, 0.0365461
3: -0.0049208, 0.0644339, -0.0013947, 0.0670294, -0.0553089, 0.0517675
4: -0.0250397, -0.0023599, -0.0237351, -0.0032822, -0.0217575, 0.0213751
5: 0.0049993, 0.0464991, 0.0076161, 0.0478919, -0.0428927, 0.0388830
6: -0.0348876, 0.0455635, -0.0318324, 0.0494685, -0.0843561, 0.0773959
7: -0.0160587, 0.0126032, -0.0145913, 0.0107602, -0.0268189, 0.0271946
8: 0.7135336, 0.9456578, 0.7023503, 0.9393543, -0.1726165, 0.1907670
9: 0.0491103, 0.0908419, 0.0520359, 0.0921682, -0.0430580, 0.0388060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0401982, 0.0587806, -0.1029809, 0.0936015
1: -0.0152351, 0.0143338, -0.0128829, 0.0132836, -0.0285187, 0.0272167
2: -0.0034273, 0.0350586, -0.0010305, 0.0368735, -0.0377323, 0.0360891
3: -0.0049559, 0.0635683, -0.0013947, 0.0670294, -0.0538581, 0.0493166
4: -0.0250527, -0.0023502, -0.0237351, -0.0032822, -0.0217705, 0.0213849
5: 0.0049731, 0.0460303, 0.0076161, 0.0478919, -0.0429188, 0.0384142
6: -0.0349180, 0.0442495, -0.0318324, 0.0494685, -0.0843865, 0.0760819
7: -0.0160732, 0.0126215, -0.0145913, 0.0107602, -0.0268334, 0.0272128
8: 0.7172979, 0.9457193, 0.7023503, 0.9393543, -0.1680372, 0.1894736
9: 0.0490810, 0.0904766, 0.0520359, 0.0921682, -0.0430872, 0.0384407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0413505, 0.0584358, -0.1025935, 0.0960693
1: -0.0152119, 0.0143234, -0.0135457, 0.0135796, -0.0287915, 0.0278691
2: -0.0034036, 0.0355157, -0.0017059, 0.0367558, -0.0381460, 0.0372216
3: -0.0049208, 0.0644339, -0.0023984, 0.0668044, -0.0536429, 0.0505283
4: -0.0250397, -0.0023599, -0.0241064, -0.0030614, -0.0219782, 0.0217464
5: 0.0049993, 0.0464991, 0.0068713, 0.0477711, -0.0427719, 0.0396278
6: -0.0348876, 0.0455635, -0.0327018, 0.0491298, -0.0840174, 0.0782653
7: -0.0160587, 0.0126032, -0.0150089, 0.0112847, -0.0273434, 0.0276121
8: 0.7135336, 0.9456578, 0.7033206, 0.9411473, -0.1723211, 0.1871364
9: 0.0491103, 0.0908419, 0.0512032, 0.0920525, -0.0429423, 0.0396387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0413505, 0.0584358, -0.1026357, 0.0947535
1: -0.0152351, 0.0143338, -0.0135457, 0.0135796, -0.0288147, 0.0278795
2: -0.0034273, 0.0350586, -0.0017059, 0.0367558, -0.0383397, 0.0367645
3: -0.0049559, 0.0635683, -0.0023984, 0.0668044, -0.0556710, 0.0515055
4: -0.0250527, -0.0023502, -0.0241064, -0.0030614, -0.0219912, 0.0217561
5: 0.0049731, 0.0460303, 0.0068713, 0.0477711, -0.0427980, 0.0391590
6: -0.0349180, 0.0442495, -0.0327018, 0.0491298, -0.0840478, 0.0769513
7: -0.0160732, 0.0126215, -0.0150089, 0.0112847, -0.0273579, 0.0276304
8: 0.7172979, 0.9457193, 0.7033206, 0.9411473, -0.1712241, 0.1894178
9: 0.0490810, 0.0904766, 0.0512032, 0.0920525, -0.0429715, 0.0392734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0401982, 0.0587806, -0.1029403, 0.0949187
1: -0.0152119, 0.0143234, -0.0128829, 0.0132836, -0.0284955, 0.0272064
2: -0.0034036, 0.0355157, -0.0010305, 0.0368735, -0.0380357, 0.0365461
3: -0.0049208, 0.0644339, -0.0013947, 0.0670294, -0.0553089, 0.0517675
4: -0.0250397, -0.0023599, -0.0237351, -0.0032822, -0.0217575, 0.0213751
5: 0.0049993, 0.0464991, 0.0076161, 0.0478919, -0.0428927, 0.0388830
6: -0.0348876, 0.0455635, -0.0318324, 0.0494685, -0.0843561, 0.0773959
7: -0.0160587, 0.0126032, -0.0145913, 0.0107602, -0.0268189, 0.0271946
8: 0.7135336, 0.9456578, 0.7023503, 0.9393543, -0.1726165, 0.1907670
9: 0.0491103, 0.0908419, 0.0520359, 0.0921682, -0.0430580, 0.0388060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0401982, 0.0587806, -0.1029809, 0.0936015
1: -0.0152351, 0.0143338, -0.0128829, 0.0132836, -0.0285187, 0.0272167
2: -0.0034273, 0.0350586, -0.0010305, 0.0368735, -0.0377323, 0.0360891
3: -0.0049559, 0.0635683, -0.0013947, 0.0670294, -0.0538581, 0.0493166
4: -0.0250527, -0.0023502, -0.0237351, -0.0032822, -0.0217705, 0.0213849
5: 0.0049731, 0.0460303, 0.0076161, 0.0478919, -0.0429188, 0.0384142
6: -0.0349180, 0.0442495, -0.0318324, 0.0494685, -0.0843865, 0.0760819
7: -0.0160732, 0.0126215, -0.0145913, 0.0107602, -0.0268334, 0.0272128
8: 0.7172979, 0.9457193, 0.7023503, 0.9393543, -0.1680372, 0.1894736
9: 0.0490810, 0.0904766, 0.0520359, 0.0921682, -0.0430872, 0.0384407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0413505, 0.0584358, -0.1025935, 0.0960693
1: -0.0152119, 0.0143234, -0.0135457, 0.0135796, -0.0287915, 0.0278691
2: -0.0034036, 0.0355157, -0.0017059, 0.0367558, -0.0381460, 0.0372216
3: -0.0049208, 0.0644339, -0.0023984, 0.0668044, -0.0536429, 0.0505283
4: -0.0250397, -0.0023599, -0.0241064, -0.0030614, -0.0219782, 0.0217464
5: 0.0049993, 0.0464991, 0.0068713, 0.0477711, -0.0427719, 0.0396278
6: -0.0348876, 0.0455635, -0.0327018, 0.0491298, -0.0840174, 0.0782653
7: -0.0160587, 0.0126032, -0.0150089, 0.0112847, -0.0273434, 0.0276121
8: 0.7135336, 0.9456578, 0.7033206, 0.9411473, -0.1723211, 0.1871364
9: 0.0491103, 0.0908419, 0.0512032, 0.0920525, -0.0429423, 0.0396387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0413505, 0.0584358, -0.1026357, 0.0947535
1: -0.0152351, 0.0143338, -0.0135457, 0.0135796, -0.0288147, 0.0278795
2: -0.0034273, 0.0350586, -0.0017059, 0.0367558, -0.0383397, 0.0367645
3: -0.0049559, 0.0635683, -0.0023984, 0.0668044, -0.0556710, 0.0515055
4: -0.0250527, -0.0023502, -0.0241064, -0.0030614, -0.0219912, 0.0217561
5: 0.0049731, 0.0460303, 0.0068713, 0.0477711, -0.0427980, 0.0391590
6: -0.0349180, 0.0442495, -0.0327018, 0.0491298, -0.0840478, 0.0769513
7: -0.0160732, 0.0126215, -0.0150089, 0.0112847, -0.0273579, 0.0276304
8: 0.7172979, 0.9457193, 0.7033206, 0.9411473, -0.1712241, 0.1894178
9: 0.0490810, 0.0904766, 0.0512032, 0.0920525, -0.0429715, 0.0392734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0401982, 0.0587806, -0.1029403, 0.0949187
1: -0.0152119, 0.0143234, -0.0128829, 0.0132836, -0.0284955, 0.0272064
2: -0.0034036, 0.0355157, -0.0010305, 0.0368735, -0.0380357, 0.0365461
3: -0.0049208, 0.0644339, -0.0013947, 0.0670294, -0.0553089, 0.0517675
4: -0.0250397, -0.0023599, -0.0237351, -0.0032822, -0.0217575, 0.0213751
5: 0.0049993, 0.0464991, 0.0076161, 0.0478919, -0.0428927, 0.0388830
6: -0.0348876, 0.0455635, -0.0318324, 0.0494685, -0.0843561, 0.0773959
7: -0.0160587, 0.0126032, -0.0145913, 0.0107602, -0.0268189, 0.0271946
8: 0.7135336, 0.9456578, 0.7023503, 0.9393543, -0.1726165, 0.1907670
9: 0.0491103, 0.0908419, 0.0520359, 0.0921682, -0.0430580, 0.0388060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0401982, 0.0587806, -0.1029809, 0.0936015
1: -0.0152351, 0.0143338, -0.0128829, 0.0132836, -0.0285187, 0.0272167
2: -0.0034273, 0.0350586, -0.0010305, 0.0368735, -0.0377323, 0.0360891
3: -0.0049559, 0.0635683, -0.0013947, 0.0670294, -0.0538581, 0.0493166
4: -0.0250527, -0.0023502, -0.0237351, -0.0032822, -0.0217705, 0.0213849
5: 0.0049731, 0.0460303, 0.0076161, 0.0478919, -0.0429188, 0.0384142
6: -0.0349180, 0.0442495, -0.0318324, 0.0494685, -0.0843865, 0.0760819
7: -0.0160732, 0.0126215, -0.0145913, 0.0107602, -0.0268334, 0.0272128
8: 0.7172979, 0.9457193, 0.7023503, 0.9393543, -0.1680372, 0.1894736
9: 0.0490810, 0.0904766, 0.0520359, 0.0921682, -0.0430872, 0.0384407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442470, 0.0548080, -0.0960693, 0.1025934
1: -0.0135457, 0.0135796, -0.0152119, 0.0143234, -0.0278691, 0.0287915
2: -0.0017059, 0.0367558, -0.0034036, 0.0355157, -0.0372216, 0.0381460
3: -0.0023984, 0.0668044, -0.0049208, 0.0644339, -0.0505283, 0.0536429
4: -0.0241064, -0.0030614, -0.0250397, -0.0023599, -0.0217464, 0.0219782
5: 0.0068713, 0.0477711, 0.0049993, 0.0464991, -0.0396278, 0.0427719
6: -0.0327018, 0.0491298, -0.0348876, 0.0455635, -0.0782653, 0.0840174
7: -0.0150089, 0.0112847, -0.0160587, 0.0126032, -0.0276121, 0.0273434
8: 0.7033206, 0.9411473, 0.7135336, 0.9456578, -0.1871364, 0.1723211
9: 0.0512032, 0.0920525, 0.0491103, 0.0908419, -0.0396387, 0.0429423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442470, 0.0548080, -0.0949187, 0.1029401
1: -0.0128829, 0.0132836, -0.0152119, 0.0143234, -0.0272064, 0.0284955
2: -0.0010305, 0.0368735, -0.0034036, 0.0355157, -0.0365461, 0.0380358
3: -0.0013947, 0.0670294, -0.0049208, 0.0644339, -0.0517675, 0.0553089
4: -0.0237351, -0.0032822, -0.0250397, -0.0023599, -0.0213751, 0.0217575
5: 0.0076161, 0.0478919, 0.0049993, 0.0464991, -0.0388830, 0.0428927
6: -0.0318324, 0.0494685, -0.0348876, 0.0455635, -0.0773959, 0.0843561
7: -0.0145913, 0.0107602, -0.0160587, 0.0126032, -0.0271946, 0.0268189
8: 0.7023503, 0.9393543, 0.7135336, 0.9456578, -0.1907670, 0.1726165
9: 0.0520359, 0.0921682, 0.0491103, 0.0908419, -0.0388060, 0.0430580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442871, 0.0534906, -0.0947534, 0.1026359
1: -0.0135457, 0.0135796, -0.0152351, 0.0143338, -0.0278795, 0.0288147
2: -0.0017059, 0.0367558, -0.0034273, 0.0350586, -0.0367645, 0.0383396
3: -0.0023984, 0.0668044, -0.0049559, 0.0635683, -0.0515055, 0.0556709
4: -0.0241064, -0.0030614, -0.0250527, -0.0023502, -0.0217561, 0.0219912
5: 0.0068713, 0.0477711, 0.0049731, 0.0460303, -0.0391590, 0.0427980
6: -0.0327018, 0.0491298, -0.0349180, 0.0442495, -0.0769513, 0.0840478
7: -0.0150089, 0.0112847, -0.0160732, 0.0126215, -0.0276304, 0.0273579
8: 0.7033206, 0.9411473, 0.7172979, 0.9457193, -0.1894178, 0.1712241
9: 0.0512032, 0.0920525, 0.0490810, 0.0904766, -0.0392734, 0.0429715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442871, 0.0534906, -0.0936016, 0.1029809
1: -0.0128829, 0.0132836, -0.0152351, 0.0143338, -0.0272167, 0.0285187
2: -0.0010305, 0.0368735, -0.0034273, 0.0350586, -0.0360891, 0.0377323
3: -0.0013947, 0.0670294, -0.0049559, 0.0635683, -0.0493166, 0.0538580
4: -0.0237351, -0.0032822, -0.0250527, -0.0023502, -0.0213849, 0.0217705
5: 0.0076161, 0.0478919, 0.0049731, 0.0460303, -0.0384142, 0.0429188
6: -0.0318324, 0.0494685, -0.0349180, 0.0442495, -0.0760819, 0.0843865
7: -0.0145913, 0.0107602, -0.0160732, 0.0126215, -0.0272128, 0.0268334
8: 0.7023503, 0.9393543, 0.7172979, 0.9457193, -0.1894736, 0.1680372
9: 0.0520359, 0.0921682, 0.0490810, 0.0904766, -0.0384407, 0.0430872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442470, 0.0548080, -0.0960693, 0.1025934
1: -0.0135457, 0.0135796, -0.0152119, 0.0143234, -0.0278691, 0.0287915
2: -0.0017059, 0.0367558, -0.0034036, 0.0355157, -0.0372216, 0.0381460
3: -0.0023984, 0.0668044, -0.0049208, 0.0644339, -0.0505283, 0.0536429
4: -0.0241064, -0.0030614, -0.0250397, -0.0023599, -0.0217464, 0.0219782
5: 0.0068713, 0.0477711, 0.0049993, 0.0464991, -0.0396278, 0.0427719
6: -0.0327018, 0.0491298, -0.0348876, 0.0455635, -0.0782653, 0.0840174
7: -0.0150089, 0.0112847, -0.0160587, 0.0126032, -0.0276121, 0.0273434
8: 0.7033206, 0.9411473, 0.7135336, 0.9456578, -0.1871364, 0.1723211
9: 0.0512032, 0.0920525, 0.0491103, 0.0908419, -0.0396387, 0.0429423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442470, 0.0548080, -0.0949187, 0.1029401
1: -0.0128829, 0.0132836, -0.0152119, 0.0143234, -0.0272064, 0.0284955
2: -0.0010305, 0.0368735, -0.0034036, 0.0355157, -0.0365461, 0.0380358
3: -0.0013947, 0.0670294, -0.0049208, 0.0644339, -0.0517675, 0.0553089
4: -0.0237351, -0.0032822, -0.0250397, -0.0023599, -0.0213751, 0.0217575
5: 0.0076161, 0.0478919, 0.0049993, 0.0464991, -0.0388830, 0.0428927
6: -0.0318324, 0.0494685, -0.0348876, 0.0455635, -0.0773959, 0.0843561
7: -0.0145913, 0.0107602, -0.0160587, 0.0126032, -0.0271946, 0.0268189
8: 0.7023503, 0.9393543, 0.7135336, 0.9456578, -0.1907670, 0.1726165
9: 0.0520359, 0.0921682, 0.0491103, 0.0908419, -0.0388060, 0.0430580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442871, 0.0534906, -0.0947534, 0.1026359
1: -0.0135457, 0.0135796, -0.0152351, 0.0143338, -0.0278795, 0.0288147
2: -0.0017059, 0.0367558, -0.0034273, 0.0350586, -0.0367645, 0.0383396
3: -0.0023984, 0.0668044, -0.0049559, 0.0635683, -0.0515055, 0.0556709
4: -0.0241064, -0.0030614, -0.0250527, -0.0023502, -0.0217561, 0.0219912
5: 0.0068713, 0.0477711, 0.0049731, 0.0460303, -0.0391590, 0.0427980
6: -0.0327018, 0.0491298, -0.0349180, 0.0442495, -0.0769513, 0.0840478
7: -0.0150089, 0.0112847, -0.0160732, 0.0126215, -0.0276304, 0.0273579
8: 0.7033206, 0.9411473, 0.7172979, 0.9457193, -0.1894178, 0.1712241
9: 0.0512032, 0.0920525, 0.0490810, 0.0904766, -0.0392734, 0.0429715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442871, 0.0534906, -0.0936016, 0.1029809
1: -0.0128829, 0.0132836, -0.0152351, 0.0143338, -0.0272167, 0.0285187
2: -0.0010305, 0.0368735, -0.0034273, 0.0350586, -0.0360891, 0.0377323
3: -0.0013947, 0.0670294, -0.0049559, 0.0635683, -0.0493166, 0.0538580
4: -0.0237351, -0.0032822, -0.0250527, -0.0023502, -0.0213849, 0.0217705
5: 0.0076161, 0.0478919, 0.0049731, 0.0460303, -0.0384142, 0.0429188
6: -0.0318324, 0.0494685, -0.0349180, 0.0442495, -0.0760819, 0.0843865
7: -0.0145913, 0.0107602, -0.0160732, 0.0126215, -0.0272128, 0.0268334
8: 0.7023503, 0.9393543, 0.7172979, 0.9457193, -0.1894736, 0.1680372
9: 0.0520359, 0.0921682, 0.0490810, 0.0904766, -0.0384407, 0.0430872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442470, 0.0548080, -0.0960693, 0.1025934
1: -0.0135457, 0.0135796, -0.0152119, 0.0143234, -0.0278691, 0.0287915
2: -0.0017059, 0.0367558, -0.0034036, 0.0355157, -0.0372216, 0.0381460
3: -0.0023984, 0.0668044, -0.0049208, 0.0644339, -0.0505283, 0.0536429
4: -0.0241064, -0.0030614, -0.0250397, -0.0023599, -0.0217464, 0.0219782
5: 0.0068713, 0.0477711, 0.0049993, 0.0464991, -0.0396278, 0.0427719
6: -0.0327018, 0.0491298, -0.0348876, 0.0455635, -0.0782653, 0.0840174
7: -0.0150089, 0.0112847, -0.0160587, 0.0126032, -0.0276121, 0.0273434
8: 0.7033206, 0.9411473, 0.7135336, 0.9456578, -0.1871364, 0.1723211
9: 0.0512032, 0.0920525, 0.0491103, 0.0908419, -0.0396387, 0.0429423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442470, 0.0548080, -0.0949187, 0.1029401
1: -0.0128829, 0.0132836, -0.0152119, 0.0143234, -0.0272064, 0.0284955
2: -0.0010305, 0.0368735, -0.0034036, 0.0355157, -0.0365461, 0.0380358
3: -0.0013947, 0.0670294, -0.0049208, 0.0644339, -0.0517675, 0.0553089
4: -0.0237351, -0.0032822, -0.0250397, -0.0023599, -0.0213751, 0.0217575
5: 0.0076161, 0.0478919, 0.0049993, 0.0464991, -0.0388830, 0.0428927
6: -0.0318324, 0.0494685, -0.0348876, 0.0455635, -0.0773959, 0.0843561
7: -0.0145913, 0.0107602, -0.0160587, 0.0126032, -0.0271946, 0.0268189
8: 0.7023503, 0.9393543, 0.7135336, 0.9456578, -0.1907670, 0.1726165
9: 0.0520359, 0.0921682, 0.0491103, 0.0908419, -0.0388060, 0.0430580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442871, 0.0534906, -0.0947534, 0.1026359
1: -0.0135457, 0.0135796, -0.0152351, 0.0143338, -0.0278795, 0.0288147
2: -0.0017059, 0.0367558, -0.0034273, 0.0350586, -0.0367645, 0.0383396
3: -0.0023984, 0.0668044, -0.0049559, 0.0635683, -0.0515055, 0.0556709
4: -0.0241064, -0.0030614, -0.0250527, -0.0023502, -0.0217561, 0.0219912
5: 0.0068713, 0.0477711, 0.0049731, 0.0460303, -0.0391590, 0.0427980
6: -0.0327018, 0.0491298, -0.0349180, 0.0442495, -0.0769513, 0.0840478
7: -0.0150089, 0.0112847, -0.0160732, 0.0126215, -0.0276304, 0.0273579
8: 0.7033206, 0.9411473, 0.7172979, 0.9457193, -0.1894178, 0.1712241
9: 0.0512032, 0.0920525, 0.0490810, 0.0904766, -0.0392734, 0.0429715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442871, 0.0534906, -0.0936016, 0.1029809
1: -0.0128829, 0.0132836, -0.0152351, 0.0143338, -0.0272167, 0.0285187
2: -0.0010305, 0.0368735, -0.0034273, 0.0350586, -0.0360891, 0.0377323
3: -0.0013947, 0.0670294, -0.0049559, 0.0635683, -0.0493166, 0.0538580
4: -0.0237351, -0.0032822, -0.0250527, -0.0023502, -0.0213849, 0.0217705
5: 0.0076161, 0.0478919, 0.0049731, 0.0460303, -0.0384142, 0.0429188
6: -0.0318324, 0.0494685, -0.0349180, 0.0442495, -0.0760819, 0.0843865
7: -0.0145913, 0.0107602, -0.0160732, 0.0126215, -0.0272128, 0.0268334
8: 0.7023503, 0.9393543, 0.7172979, 0.9457193, -0.1894736, 0.1680372
9: 0.0520359, 0.0921682, 0.0490810, 0.0904766, -0.0384407, 0.0430872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442470, 0.0548080, -0.0960693, 0.1025934
1: -0.0135457, 0.0135796, -0.0152119, 0.0143234, -0.0278691, 0.0287915
2: -0.0017059, 0.0367558, -0.0034036, 0.0355157, -0.0372216, 0.0381460
3: -0.0023984, 0.0668044, -0.0049208, 0.0644339, -0.0505283, 0.0536429
4: -0.0241064, -0.0030614, -0.0250397, -0.0023599, -0.0217464, 0.0219782
5: 0.0068713, 0.0477711, 0.0049993, 0.0464991, -0.0396278, 0.0427719
6: -0.0327018, 0.0491298, -0.0348876, 0.0455635, -0.0782653, 0.0840174
7: -0.0150089, 0.0112847, -0.0160587, 0.0126032, -0.0276121, 0.0273434
8: 0.7033206, 0.9411473, 0.7135336, 0.9456578, -0.1871364, 0.1723211
9: 0.0512032, 0.0920525, 0.0491103, 0.0908419, -0.0396387, 0.0429423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442470, 0.0548080, -0.0949187, 0.1029401
1: -0.0128829, 0.0132836, -0.0152119, 0.0143234, -0.0272064, 0.0284955
2: -0.0010305, 0.0368735, -0.0034036, 0.0355157, -0.0365461, 0.0380358
3: -0.0013947, 0.0670294, -0.0049208, 0.0644339, -0.0517675, 0.0553089
4: -0.0237351, -0.0032822, -0.0250397, -0.0023599, -0.0213751, 0.0217575
5: 0.0076161, 0.0478919, 0.0049993, 0.0464991, -0.0388830, 0.0428927
6: -0.0318324, 0.0494685, -0.0348876, 0.0455635, -0.0773959, 0.0843561
7: -0.0145913, 0.0107602, -0.0160587, 0.0126032, -0.0271946, 0.0268189
8: 0.7023503, 0.9393543, 0.7135336, 0.9456578, -0.1907670, 0.1726165
9: 0.0520359, 0.0921682, 0.0491103, 0.0908419, -0.0388060, 0.0430580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442871, 0.0534906, -0.0947534, 0.1026359
1: -0.0135457, 0.0135796, -0.0152351, 0.0143338, -0.0278795, 0.0288147
2: -0.0017059, 0.0367558, -0.0034273, 0.0350586, -0.0367645, 0.0383396
3: -0.0023984, 0.0668044, -0.0049559, 0.0635683, -0.0515055, 0.0556709
4: -0.0241064, -0.0030614, -0.0250527, -0.0023502, -0.0217561, 0.0219912
5: 0.0068713, 0.0477711, 0.0049731, 0.0460303, -0.0391590, 0.0427980
6: -0.0327018, 0.0491298, -0.0349180, 0.0442495, -0.0769513, 0.0840478
7: -0.0150089, 0.0112847, -0.0160732, 0.0126215, -0.0276304, 0.0273579
8: 0.7033206, 0.9411473, 0.7172979, 0.9457193, -0.1894178, 0.1712241
9: 0.0512032, 0.0920525, 0.0490810, 0.0904766, -0.0392734, 0.0429715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442871, 0.0534906, -0.0936016, 0.1029809
1: -0.0128829, 0.0132836, -0.0152351, 0.0143338, -0.0272167, 0.0285187
2: -0.0010305, 0.0368735, -0.0034273, 0.0350586, -0.0360891, 0.0377323
3: -0.0013947, 0.0670294, -0.0049559, 0.0635683, -0.0493166, 0.0538580
4: -0.0237351, -0.0032822, -0.0250527, -0.0023502, -0.0213849, 0.0217705
5: 0.0076161, 0.0478919, 0.0049731, 0.0460303, -0.0384142, 0.0429188
6: -0.0318324, 0.0494685, -0.0349180, 0.0442495, -0.0760819, 0.0843865
7: -0.0145913, 0.0107602, -0.0160732, 0.0126215, -0.0272128, 0.0268334
8: 0.7023503, 0.9393543, 0.7172979, 0.9457193, -0.1894736, 0.1680372
9: 0.0520359, 0.0921682, 0.0490810, 0.0904766, -0.0384407, 0.0430872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442470, 0.0548080, -0.0960693, 0.1025934
1: -0.0135457, 0.0135796, -0.0152119, 0.0143234, -0.0278691, 0.0287915
2: -0.0017059, 0.0367558, -0.0034036, 0.0355157, -0.0372216, 0.0381460
3: -0.0023984, 0.0668044, -0.0049208, 0.0644339, -0.0505283, 0.0536429
4: -0.0241064, -0.0030614, -0.0250397, -0.0023599, -0.0217464, 0.0219782
5: 0.0068713, 0.0477711, 0.0049993, 0.0464991, -0.0396278, 0.0427719
6: -0.0327018, 0.0491298, -0.0348876, 0.0455635, -0.0782653, 0.0840174
7: -0.0150089, 0.0112847, -0.0160587, 0.0126032, -0.0276121, 0.0273434
8: 0.7033206, 0.9411473, 0.7135336, 0.9456578, -0.1871364, 0.1723211
9: 0.0512032, 0.0920525, 0.0491103, 0.0908419, -0.0396387, 0.0429423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442470, 0.0548080, -0.0949187, 0.1029401
1: -0.0128829, 0.0132836, -0.0152119, 0.0143234, -0.0272064, 0.0284955
2: -0.0010305, 0.0368735, -0.0034036, 0.0355157, -0.0365461, 0.0380358
3: -0.0013947, 0.0670294, -0.0049208, 0.0644339, -0.0517675, 0.0553089
4: -0.0237351, -0.0032822, -0.0250397, -0.0023599, -0.0213751, 0.0217575
5: 0.0076161, 0.0478919, 0.0049993, 0.0464991, -0.0388830, 0.0428927
6: -0.0318324, 0.0494685, -0.0348876, 0.0455635, -0.0773959, 0.0843561
7: -0.0145913, 0.0107602, -0.0160587, 0.0126032, -0.0271946, 0.0268189
8: 0.7023503, 0.9393543, 0.7135336, 0.9456578, -0.1907670, 0.1726165
9: 0.0520359, 0.0921682, 0.0491103, 0.0908419, -0.0388060, 0.0430580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442871, 0.0534906, -0.0947534, 0.1026359
1: -0.0135457, 0.0135796, -0.0152351, 0.0143338, -0.0278795, 0.0288147
2: -0.0017059, 0.0367558, -0.0034273, 0.0350586, -0.0367645, 0.0383396
3: -0.0023984, 0.0668044, -0.0049559, 0.0635683, -0.0515055, 0.0556709
4: -0.0241064, -0.0030614, -0.0250527, -0.0023502, -0.0217561, 0.0219912
5: 0.0068713, 0.0477711, 0.0049731, 0.0460303, -0.0391590, 0.0427980
6: -0.0327018, 0.0491298, -0.0349180, 0.0442495, -0.0769513, 0.0840478
7: -0.0150089, 0.0112847, -0.0160732, 0.0126215, -0.0276304, 0.0273579
8: 0.7033206, 0.9411473, 0.7172979, 0.9457193, -0.1894178, 0.1712241
9: 0.0512032, 0.0920525, 0.0490810, 0.0904766, -0.0392734, 0.0429715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442871, 0.0534906, -0.0936016, 0.1029809
1: -0.0128829, 0.0132836, -0.0152351, 0.0143338, -0.0272167, 0.0285187
2: -0.0010305, 0.0368735, -0.0034273, 0.0350586, -0.0360891, 0.0377323
3: -0.0013947, 0.0670294, -0.0049559, 0.0635683, -0.0493166, 0.0538580
4: -0.0237351, -0.0032822, -0.0250527, -0.0023502, -0.0213849, 0.0217705
5: 0.0076161, 0.0478919, 0.0049731, 0.0460303, -0.0384142, 0.0429188
6: -0.0318324, 0.0494685, -0.0349180, 0.0442495, -0.0760819, 0.0843865
7: -0.0145913, 0.0107602, -0.0160732, 0.0126215, -0.0272128, 0.0268334
8: 0.7023503, 0.9393543, 0.7172979, 0.9457193, -0.1894736, 0.1680372
9: 0.0520359, 0.0921682, 0.0490810, 0.0904766, -0.0384407, 0.0430872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442470, 0.0548080, -0.0960693, 0.1025934
1: -0.0135457, 0.0135796, -0.0152119, 0.0143234, -0.0278691, 0.0287915
2: -0.0017059, 0.0367558, -0.0034036, 0.0355157, -0.0372216, 0.0381460
3: -0.0023984, 0.0668044, -0.0049208, 0.0644339, -0.0505283, 0.0536429
4: -0.0241064, -0.0030614, -0.0250397, -0.0023599, -0.0217464, 0.0219782
5: 0.0068713, 0.0477711, 0.0049993, 0.0464991, -0.0396278, 0.0427719
6: -0.0327018, 0.0491298, -0.0348876, 0.0455635, -0.0782653, 0.0840174
7: -0.0150089, 0.0112847, -0.0160587, 0.0126032, -0.0276121, 0.0273434
8: 0.7033206, 0.9411473, 0.7135336, 0.9456578, -0.1871364, 0.1723211
9: 0.0512032, 0.0920525, 0.0491103, 0.0908419, -0.0396387, 0.0429423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442470, 0.0548080, -0.0949187, 0.1029401
1: -0.0128829, 0.0132836, -0.0152119, 0.0143234, -0.0272064, 0.0284955
2: -0.0010305, 0.0368735, -0.0034036, 0.0355157, -0.0365461, 0.0380358
3: -0.0013947, 0.0670294, -0.0049208, 0.0644339, -0.0517675, 0.0553089
4: -0.0237351, -0.0032822, -0.0250397, -0.0023599, -0.0213751, 0.0217575
5: 0.0076161, 0.0478919, 0.0049993, 0.0464991, -0.0388830, 0.0428927
6: -0.0318324, 0.0494685, -0.0348876, 0.0455635, -0.0773959, 0.0843561
7: -0.0145913, 0.0107602, -0.0160587, 0.0126032, -0.0271946, 0.0268189
8: 0.7023503, 0.9393543, 0.7135336, 0.9456578, -0.1907670, 0.1726165
9: 0.0520359, 0.0921682, 0.0491103, 0.0908419, -0.0388060, 0.0430580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1109350, upper bound: 0.1028533
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442871, 0.0534906, -0.0947534, 0.1026359
1: -0.0135457, 0.0135796, -0.0152351, 0.0143338, -0.0278795, 0.0288147
2: -0.0017059, 0.0367558, -0.0034273, 0.0350586, -0.0367645, 0.0383396
3: -0.0023984, 0.0668044, -0.0049559, 0.0635683, -0.0515055, 0.0556709
4: -0.0241064, -0.0030614, -0.0250527, -0.0023502, -0.0217561, 0.0219912
5: 0.0068713, 0.0477711, 0.0049731, 0.0460303, -0.0391590, 0.0427980
6: -0.0327018, 0.0491298, -0.0349180, 0.0442495, -0.0769513, 0.0840478
7: -0.0150089, 0.0112847, -0.0160732, 0.0126215, -0.0276304, 0.0273579
8: 0.7033206, 0.9411473, 0.7172979, 0.9457193, -0.1894178, 0.1712241
9: 0.0512032, 0.0920525, 0.0490810, 0.0904766, -0.0392734, 0.0429715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442871, 0.0534906, -0.0936016, 0.1029809
1: -0.0128829, 0.0132836, -0.0152351, 0.0143338, -0.0272167, 0.0285187
2: -0.0010305, 0.0368735, -0.0034273, 0.0350586, -0.0360891, 0.0377323
3: -0.0013947, 0.0670294, -0.0049559, 0.0635683, -0.0493166, 0.0538580
4: -0.0237351, -0.0032822, -0.0250527, -0.0023502, -0.0213849, 0.0217705
5: 0.0076161, 0.0478919, 0.0049731, 0.0460303, -0.0384142, 0.0429188
6: -0.0318324, 0.0494685, -0.0349180, 0.0442495, -0.0760819, 0.0843865
7: -0.0145913, 0.0107602, -0.0160732, 0.0126215, -0.0272128, 0.0268334
8: 0.7023503, 0.9393543, 0.7172979, 0.9457193, -0.1894736, 0.1680372
9: 0.0520359, 0.0921682, 0.0490810, 0.0904766, -0.0384407, 0.0430872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442470, 0.0548080, -0.0960693, 0.1025934
1: -0.0135457, 0.0135796, -0.0152119, 0.0143234, -0.0278691, 0.0287915
2: -0.0017059, 0.0367558, -0.0034036, 0.0355157, -0.0372216, 0.0381460
3: -0.0023984, 0.0668044, -0.0049208, 0.0644339, -0.0505283, 0.0536429
4: -0.0241064, -0.0030614, -0.0250397, -0.0023599, -0.0217464, 0.0219782
5: 0.0068713, 0.0477711, 0.0049993, 0.0464991, -0.0396278, 0.0427719
6: -0.0327018, 0.0491298, -0.0348876, 0.0455635, -0.0782653, 0.0840174
7: -0.0150089, 0.0112847, -0.0160587, 0.0126032, -0.0276121, 0.0273434
8: 0.7033206, 0.9411473, 0.7135336, 0.9456578, -0.1871364, 0.1723211
9: 0.0512032, 0.0920525, 0.0491103, 0.0908419, -0.0396387, 0.0429423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442470, 0.0548080, -0.0949187, 0.1029401
1: -0.0128829, 0.0132836, -0.0152119, 0.0143234, -0.0272064, 0.0284955
2: -0.0010305, 0.0368735, -0.0034036, 0.0355157, -0.0365461, 0.0380358
3: -0.0013947, 0.0670294, -0.0049208, 0.0644339, -0.0517675, 0.0553089
4: -0.0237351, -0.0032822, -0.0250397, -0.0023599, -0.0213751, 0.0217575
5: 0.0076161, 0.0478919, 0.0049993, 0.0464991, -0.0388830, 0.0428927
6: -0.0318324, 0.0494685, -0.0348876, 0.0455635, -0.0773959, 0.0843561
7: -0.0145913, 0.0107602, -0.0160587, 0.0126032, -0.0271946, 0.0268189
8: 0.7023503, 0.9393543, 0.7135336, 0.9456578, -0.1907670, 0.1726165
9: 0.0520359, 0.0921682, 0.0491103, 0.0908419, -0.0388060, 0.0430580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442871, 0.0534906, -0.0947534, 0.1026359
1: -0.0135457, 0.0135796, -0.0152351, 0.0143338, -0.0278795, 0.0288147
2: -0.0017059, 0.0367558, -0.0034273, 0.0350586, -0.0367645, 0.0383396
3: -0.0023984, 0.0668044, -0.0049559, 0.0635683, -0.0515055, 0.0556709
4: -0.0241064, -0.0030614, -0.0250527, -0.0023502, -0.0217561, 0.0219912
5: 0.0068713, 0.0477711, 0.0049731, 0.0460303, -0.0391590, 0.0427980
6: -0.0327018, 0.0491298, -0.0349180, 0.0442495, -0.0769513, 0.0840478
7: -0.0150089, 0.0112847, -0.0160732, 0.0126215, -0.0276304, 0.0273579
8: 0.7033206, 0.9411473, 0.7172979, 0.9457193, -0.1894178, 0.1712241
9: 0.0512032, 0.0920525, 0.0490810, 0.0904766, -0.0392734, 0.0429715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442871, 0.0534906, -0.0936016, 0.1029809
1: -0.0128829, 0.0132836, -0.0152351, 0.0143338, -0.0272167, 0.0285187
2: -0.0010305, 0.0368735, -0.0034273, 0.0350586, -0.0360891, 0.0377323
3: -0.0013947, 0.0670294, -0.0049559, 0.0635683, -0.0493166, 0.0538580
4: -0.0237351, -0.0032822, -0.0250527, -0.0023502, -0.0213849, 0.0217705
5: 0.0076161, 0.0478919, 0.0049731, 0.0460303, -0.0384142, 0.0429188
6: -0.0318324, 0.0494685, -0.0349180, 0.0442495, -0.0760819, 0.0843865
7: -0.0145913, 0.0107602, -0.0160732, 0.0126215, -0.0272128, 0.0268334
8: 0.7023503, 0.9393543, 0.7172979, 0.9457193, -0.1894736, 0.1680372
9: 0.0520359, 0.0921682, 0.0490810, 0.0904766, -0.0384407, 0.0430872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442470, 0.0548080, -0.0960693, 0.1025934
1: -0.0135457, 0.0135796, -0.0152119, 0.0143234, -0.0278691, 0.0287915
2: -0.0017059, 0.0367558, -0.0034036, 0.0355157, -0.0372216, 0.0381460
3: -0.0023984, 0.0668044, -0.0049208, 0.0644339, -0.0505283, 0.0536429
4: -0.0241064, -0.0030614, -0.0250397, -0.0023599, -0.0217464, 0.0219782
5: 0.0068713, 0.0477711, 0.0049993, 0.0464991, -0.0396278, 0.0427719
6: -0.0327018, 0.0491298, -0.0348876, 0.0455635, -0.0782653, 0.0840174
7: -0.0150089, 0.0112847, -0.0160587, 0.0126032, -0.0276121, 0.0273434
8: 0.7033206, 0.9411473, 0.7135336, 0.9456578, -0.1871364, 0.1723211
9: 0.0512032, 0.0920525, 0.0491103, 0.0908419, -0.0396387, 0.0429423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442470, 0.0548080, -0.0949187, 0.1029401
1: -0.0128829, 0.0132836, -0.0152119, 0.0143234, -0.0272064, 0.0284955
2: -0.0010305, 0.0368735, -0.0034036, 0.0355157, -0.0365461, 0.0380358
3: -0.0013947, 0.0670294, -0.0049208, 0.0644339, -0.0517675, 0.0553089
4: -0.0237351, -0.0032822, -0.0250397, -0.0023599, -0.0213751, 0.0217575
5: 0.0076161, 0.0478919, 0.0049993, 0.0464991, -0.0388830, 0.0428927
6: -0.0318324, 0.0494685, -0.0348876, 0.0455635, -0.0773959, 0.0843561
7: -0.0145913, 0.0107602, -0.0160587, 0.0126032, -0.0271946, 0.0268189
8: 0.7023503, 0.9393543, 0.7135336, 0.9456578, -0.1907670, 0.1726165
9: 0.0520359, 0.0921682, 0.0491103, 0.0908419, -0.0388060, 0.0430580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1109350, upper bound: 0.1028533
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442871, 0.0534906, -0.0947534, 0.1026359
1: -0.0135457, 0.0135796, -0.0152351, 0.0143338, -0.0278795, 0.0288147
2: -0.0017059, 0.0367558, -0.0034273, 0.0350586, -0.0367645, 0.0383396
3: -0.0023984, 0.0668044, -0.0049559, 0.0635683, -0.0515055, 0.0556709
4: -0.0241064, -0.0030614, -0.0250527, -0.0023502, -0.0217561, 0.0219912
5: 0.0068713, 0.0477711, 0.0049731, 0.0460303, -0.0391590, 0.0427980
6: -0.0327018, 0.0491298, -0.0349180, 0.0442495, -0.0769513, 0.0840478
7: -0.0150089, 0.0112847, -0.0160732, 0.0126215, -0.0276304, 0.0273579
8: 0.7033206, 0.9411473, 0.7172979, 0.9457193, -0.1894178, 0.1712241
9: 0.0512032, 0.0920525, 0.0490810, 0.0904766, -0.0392734, 0.0429715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442871, 0.0534906, -0.0936016, 0.1029809
1: -0.0128829, 0.0132836, -0.0152351, 0.0143338, -0.0272167, 0.0285187
2: -0.0010305, 0.0368735, -0.0034273, 0.0350586, -0.0360891, 0.0377323
3: -0.0013947, 0.0670294, -0.0049559, 0.0635683, -0.0493166, 0.0538580
4: -0.0237351, -0.0032822, -0.0250527, -0.0023502, -0.0213849, 0.0217705
5: 0.0076161, 0.0478919, 0.0049731, 0.0460303, -0.0384142, 0.0429188
6: -0.0318324, 0.0494685, -0.0349180, 0.0442495, -0.0760819, 0.0843865
7: -0.0145913, 0.0107602, -0.0160732, 0.0126215, -0.0272128, 0.0268334
8: 0.7023503, 0.9393543, 0.7172979, 0.9457193, -0.1894736, 0.1680372
9: 0.0520359, 0.0921682, 0.0490810, 0.0904766, -0.0384407, 0.0430872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0401980, 0.0572768, -0.0413503, 0.0583259, -0.0984344, 0.0985379
1: -0.0128829, 0.0132835, -0.0135457, 0.0135796, -0.0264625, 0.0268293
2: -0.0010304, 0.0363597, -0.0017060, 0.0367182, -0.0359369, 0.0355918
3: -0.0013946, 0.0660469, -0.0023983, 0.0667324, -0.0515402, 0.0500631
4: -0.0237351, -0.0033405, -0.0241063, -0.0030615, -0.0206736, 0.0207658
5: 0.0076161, 0.0473649, 0.0068712, 0.0477326, -0.0401165, 0.0404937
6: -0.0318319, 0.0479909, -0.0327020, 0.0490217, -0.0808536, 0.0806929
7: -0.0145914, 0.0107601, -0.0150089, 0.0112847, -0.0258760, 0.0257690
8: 0.7065819, 0.9393533, 0.7036301, 0.9411469, -0.1803710, 0.1809983
9: 0.0520357, 0.0916632, 0.0512030, 0.0920156, -0.0399799, 0.0404602

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1074684, upper bound: 0.0915589
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1074684, upper bound: 0.0902061
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0413507, 0.0568593, -0.0401982, 0.0587806, -0.1000412, 0.0969677
1: -0.0135457, 0.0135796, -0.0128829, 0.0132836, -0.0268293, 0.0264625
2: -0.0017059, 0.0362172, -0.0010305, 0.0368735, -0.0362400, 0.0353923
3: -0.0023985, 0.0657746, -0.0013947, 0.0670294, -0.0521207, 0.0493881
4: -0.0241064, -0.0030614, -0.0237351, -0.0032822, -0.0208242, 0.0206736
5: 0.0068713, 0.0472187, 0.0076161, 0.0478919, -0.0410207, 0.0396026
6: -0.0327018, 0.0475809, -0.0318324, 0.0494685, -0.0821703, 0.0794133
7: -0.0150089, 0.0112846, -0.0145913, 0.0107602, -0.0257691, 0.0258760
8: 0.7077572, 0.9411479, 0.7023503, 0.9393543, -0.1769090, 0.1846697
9: 0.0512030, 0.0915231, 0.0520359, 0.0921682, -0.0409652, 0.0394873

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0957134, upper bound: 0.0901718
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0957134, upper bound: 0.0901718
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0401980, 0.0572768, -0.0401982, 0.0587806, -0.0988878, 0.0973837
1: -0.0128829, 0.0132835, -0.0128829, 0.0132836, -0.0261665, 0.0261665
2: -0.0010304, 0.0363597, -0.0010305, 0.0368735, -0.0353518, 0.0347543
3: -0.0013946, 0.0660469, -0.0013947, 0.0670294, -0.0492666, 0.0469156
4: -0.0237351, -0.0033405, -0.0237351, -0.0032822, -0.0204528, 0.0203946
5: 0.0076161, 0.0473649, 0.0076161, 0.0478919, -0.0402758, 0.0397487
6: -0.0318319, 0.0479909, -0.0318324, 0.0494685, -0.0813005, 0.0798233
7: -0.0145914, 0.0107601, -0.0145913, 0.0107602, -0.0253516, 0.0253515
8: 0.7065819, 0.9393533, 0.7023503, 0.9393543, -0.1764905, 0.1807961
9: 0.0520357, 0.0916632, 0.0520359, 0.0921682, -0.0401325, 0.0396273

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1074684, upper bound: 0.0902061
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1074684, upper bound: 0.0902061
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0401980, 0.0572768, -0.0413503, 0.0583259, -0.0984344, 0.0985379
1: -0.0128829, 0.0132835, -0.0135457, 0.0135796, -0.0264625, 0.0268293
2: -0.0010304, 0.0363597, -0.0017060, 0.0367182, -0.0359369, 0.0355918
3: -0.0013946, 0.0660469, -0.0023983, 0.0667324, -0.0515402, 0.0500631
4: -0.0237351, -0.0033405, -0.0241063, -0.0030615, -0.0206736, 0.0207658
5: 0.0076161, 0.0473649, 0.0068712, 0.0477326, -0.0401165, 0.0404937
6: -0.0318319, 0.0479909, -0.0327020, 0.0490217, -0.0808536, 0.0806929
7: -0.0145914, 0.0107601, -0.0150089, 0.0112847, -0.0258760, 0.0257690
8: 0.7065819, 0.9393533, 0.7036301, 0.9411469, -0.1803710, 0.1809983
9: 0.0520357, 0.0916632, 0.0512030, 0.0920156, -0.0399799, 0.0404602

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1074684, upper bound: 0.0915589
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1074684, upper bound: 0.0902061
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0401982, 0.0587806, -0.1000412, 0.0985441
1: -0.0135457, 0.0135796, -0.0128829, 0.0132836, -0.0268293, 0.0264625
2: -0.0017059, 0.0367558, -0.0010305, 0.0368735, -0.0362400, 0.0359803
3: -0.0023984, 0.0668044, -0.0013947, 0.0670294, -0.0521227, 0.0517359
4: -0.0241064, -0.0030614, -0.0237351, -0.0032822, -0.0208242, 0.0206736
5: 0.0068713, 0.0477711, 0.0076161, 0.0478919, -0.0410206, 0.0401550
6: -0.0327018, 0.0491298, -0.0318324, 0.0494685, -0.0821703, 0.0809622
7: -0.0150089, 0.0112847, -0.0145913, 0.0107602, -0.0257691, 0.0258760
8: 0.7033206, 0.9411473, 0.7023503, 0.9393543, -0.1813202, 0.1846695
9: 0.0512032, 0.0920525, 0.0520359, 0.0921682, -0.0409651, 0.0400166

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0957134, upper bound: 0.1023510
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1022408, upper bound: 0.0967478
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0401982, 0.0587806, -0.0988878, 0.0988877
1: -0.0128829, 0.0132836, -0.0128829, 0.0132836, -0.0261665, 0.0261665
2: -0.0010305, 0.0368735, -0.0010305, 0.0368735, -0.0353519, 0.0353519
3: -0.0013947, 0.0670294, -0.0013947, 0.0670294, -0.0492703, 0.0492703
4: -0.0237351, -0.0032822, -0.0237351, -0.0032822, -0.0204529, 0.0204529
5: 0.0076161, 0.0478919, 0.0076161, 0.0478919, -0.0402758, 0.0402758
6: -0.0318324, 0.0494685, -0.0318324, 0.0494685, -0.0813009, 0.0813009
7: -0.0145913, 0.0107602, -0.0145913, 0.0107602, -0.0253515, 0.0253515
8: 0.7023503, 0.9393543, 0.7023503, 0.9393543, -0.1807964, 0.1807964
9: 0.0520359, 0.0921682, 0.0520359, 0.0921682, -0.0401324, 0.0401324

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1121336, upper bound: 0.1068673
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1121336, upper bound: 0.1068673
time: 0.57 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 2.62 seconds
IS_A1_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
IS_A1_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
IS_A1_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
IS_A1_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
IS_A1_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
IS_A1_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
IS_A1_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A1_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
IS_A1_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
IS_A1_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A1_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
IS_A1_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
IS_A1_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
IS_A1_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
IS_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
IS_A1_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
IS_A1_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A1_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A1_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
IS_A1_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1108777
IS_A1_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A2_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
IS_A2_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
IS_A2_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
IS_A2_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
IS_A2_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
IS_A2_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
IS_A2_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
IS_A2_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
IS_A2_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
IS_A2_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
IS_A2_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
IS_A2_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1109350, upper bound: 0.1028533
IS_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
IS_A2_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
IS_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
IS_A2_B1_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
IS_A2_B1_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
IS_A2_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
IS_A2_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1109350, upper bound: 0.1028533
IS_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
IS_A2_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
IS_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
IS_A2_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1074684, upper bound: 0.0915589
IS_A2_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1074684, upper bound: 0.0902061
IS_A2_B2_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0957134, upper bound: 0.0901718
IS_A2_B2_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0957134, upper bound: 0.0901718
IS_A2_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1074684, upper bound: 0.0902061
IS_A2_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1074684, upper bound: 0.0902061
IS_A2_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1074684, upper bound: 0.0915589
IS_A2_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1074684, upper bound: 0.0902061
IS_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.0957134, upper bound: 0.1023510
IS_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1022408, upper bound: 0.0967478
IS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1121336, upper bound: 0.1068673
IS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.62
Output dim: 8, lower bound: -0.1121336, upper bound: 0.1068673

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0413505, 0.0584358, -0.1025935, 0.0960693
1: -0.0152119, 0.0143234, -0.0135457, 0.0135796, -0.0287915, 0.0278691
2: -0.0034036, 0.0355157, -0.0017059, 0.0367558, -0.0381460, 0.0372216
3: -0.0049208, 0.0644339, -0.0023984, 0.0668044, -0.0536429, 0.0505283
4: -0.0250397, -0.0023599, -0.0241064, -0.0030614, -0.0219782, 0.0217464
5: 0.0049993, 0.0464991, 0.0068713, 0.0477711, -0.0427719, 0.0396278
6: -0.0348876, 0.0455635, -0.0327018, 0.0491298, -0.0840174, 0.0782653
7: -0.0160587, 0.0126032, -0.0150089, 0.0112847, -0.0273434, 0.0276121
8: 0.7135336, 0.9456578, 0.7033206, 0.9411473, -0.1723211, 0.1871364
9: 0.0491103, 0.0908419, 0.0512032, 0.0920525, -0.0429423, 0.0396387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0401982, 0.0587806, -0.1029403, 0.0949187
1: -0.0152119, 0.0143234, -0.0128829, 0.0132836, -0.0284955, 0.0272064
2: -0.0034036, 0.0355157, -0.0010305, 0.0368735, -0.0380357, 0.0365461
3: -0.0049208, 0.0644339, -0.0013947, 0.0670294, -0.0553089, 0.0517675
4: -0.0250397, -0.0023599, -0.0237351, -0.0032822, -0.0217575, 0.0213751
5: 0.0049993, 0.0464991, 0.0076161, 0.0478919, -0.0428927, 0.0388830
6: -0.0348876, 0.0455635, -0.0318324, 0.0494685, -0.0843561, 0.0773959
7: -0.0160587, 0.0126032, -0.0145913, 0.0107602, -0.0268189, 0.0271946
8: 0.7135336, 0.9456578, 0.7023503, 0.9393543, -0.1726165, 0.1907670
9: 0.0491103, 0.0908419, 0.0520359, 0.0921682, -0.0430580, 0.0388060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0413505, 0.0584358, -0.1026357, 0.0947535
1: -0.0152351, 0.0143338, -0.0135457, 0.0135796, -0.0288147, 0.0278795
2: -0.0034273, 0.0350586, -0.0017059, 0.0367558, -0.0383397, 0.0367645
3: -0.0049559, 0.0635683, -0.0023984, 0.0668044, -0.0556710, 0.0515055
4: -0.0250527, -0.0023502, -0.0241064, -0.0030614, -0.0219912, 0.0217561
5: 0.0049731, 0.0460303, 0.0068713, 0.0477711, -0.0427980, 0.0391590
6: -0.0349180, 0.0442495, -0.0327018, 0.0491298, -0.0840478, 0.0769513
7: -0.0160732, 0.0126215, -0.0150089, 0.0112847, -0.0273579, 0.0276304
8: 0.7172979, 0.9457193, 0.7033206, 0.9411473, -0.1712241, 0.1894178
9: 0.0490810, 0.0904766, 0.0512032, 0.0920525, -0.0429715, 0.0392734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0401982, 0.0587806, -0.1029809, 0.0936015
1: -0.0152351, 0.0143338, -0.0128829, 0.0132836, -0.0285187, 0.0272167
2: -0.0034273, 0.0350586, -0.0010305, 0.0368735, -0.0377323, 0.0360891
3: -0.0049559, 0.0635683, -0.0013947, 0.0670294, -0.0538581, 0.0493166
4: -0.0250527, -0.0023502, -0.0237351, -0.0032822, -0.0217705, 0.0213849
5: 0.0049731, 0.0460303, 0.0076161, 0.0478919, -0.0429188, 0.0384142
6: -0.0349180, 0.0442495, -0.0318324, 0.0494685, -0.0843865, 0.0760819
7: -0.0160732, 0.0126215, -0.0145913, 0.0107602, -0.0268334, 0.0272128
8: 0.7172979, 0.9457193, 0.7023503, 0.9393543, -0.1680372, 0.1894736
9: 0.0490810, 0.0904766, 0.0520359, 0.0921682, -0.0430872, 0.0384407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0413505, 0.0584358, -0.1025935, 0.0960693
1: -0.0152119, 0.0143234, -0.0135457, 0.0135796, -0.0287915, 0.0278691
2: -0.0034036, 0.0355157, -0.0017059, 0.0367558, -0.0381460, 0.0372216
3: -0.0049208, 0.0644339, -0.0023984, 0.0668044, -0.0536429, 0.0505283
4: -0.0250397, -0.0023599, -0.0241064, -0.0030614, -0.0219782, 0.0217464
5: 0.0049993, 0.0464991, 0.0068713, 0.0477711, -0.0427719, 0.0396278
6: -0.0348876, 0.0455635, -0.0327018, 0.0491298, -0.0840174, 0.0782653
7: -0.0160587, 0.0126032, -0.0150089, 0.0112847, -0.0273434, 0.0276121
8: 0.7135336, 0.9456578, 0.7033206, 0.9411473, -0.1723211, 0.1871364
9: 0.0491103, 0.0908419, 0.0512032, 0.0920525, -0.0429423, 0.0396387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0401982, 0.0587806, -0.1029403, 0.0949187
1: -0.0152119, 0.0143234, -0.0128829, 0.0132836, -0.0284955, 0.0272064
2: -0.0034036, 0.0355157, -0.0010305, 0.0368735, -0.0380357, 0.0365461
3: -0.0049208, 0.0644339, -0.0013947, 0.0670294, -0.0553089, 0.0517675
4: -0.0250397, -0.0023599, -0.0237351, -0.0032822, -0.0217575, 0.0213751
5: 0.0049993, 0.0464991, 0.0076161, 0.0478919, -0.0428927, 0.0388830
6: -0.0348876, 0.0455635, -0.0318324, 0.0494685, -0.0843561, 0.0773959
7: -0.0160587, 0.0126032, -0.0145913, 0.0107602, -0.0268189, 0.0271946
8: 0.7135336, 0.9456578, 0.7023503, 0.9393543, -0.1726165, 0.1907670
9: 0.0491103, 0.0908419, 0.0520359, 0.0921682, -0.0430580, 0.0388060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0413505, 0.0584358, -0.1026357, 0.0947535
1: -0.0152351, 0.0143338, -0.0135457, 0.0135796, -0.0288147, 0.0278795
2: -0.0034273, 0.0350586, -0.0017059, 0.0367558, -0.0383397, 0.0367645
3: -0.0049559, 0.0635683, -0.0023984, 0.0668044, -0.0556710, 0.0515055
4: -0.0250527, -0.0023502, -0.0241064, -0.0030614, -0.0219912, 0.0217561
5: 0.0049731, 0.0460303, 0.0068713, 0.0477711, -0.0427980, 0.0391590
6: -0.0349180, 0.0442495, -0.0327018, 0.0491298, -0.0840478, 0.0769513
7: -0.0160732, 0.0126215, -0.0150089, 0.0112847, -0.0273579, 0.0276304
8: 0.7172979, 0.9457193, 0.7033206, 0.9411473, -0.1712241, 0.1894178
9: 0.0490810, 0.0904766, 0.0512032, 0.0920525, -0.0429715, 0.0392734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0401982, 0.0587806, -0.1029809, 0.0936015
1: -0.0152351, 0.0143338, -0.0128829, 0.0132836, -0.0285187, 0.0272167
2: -0.0034273, 0.0350586, -0.0010305, 0.0368735, -0.0377323, 0.0360891
3: -0.0049559, 0.0635683, -0.0013947, 0.0670294, -0.0538581, 0.0493166
4: -0.0250527, -0.0023502, -0.0237351, -0.0032822, -0.0217705, 0.0213849
5: 0.0049731, 0.0460303, 0.0076161, 0.0478919, -0.0429188, 0.0384142
6: -0.0349180, 0.0442495, -0.0318324, 0.0494685, -0.0843865, 0.0760819
7: -0.0160732, 0.0126215, -0.0145913, 0.0107602, -0.0268334, 0.0272128
8: 0.7172979, 0.9457193, 0.7023503, 0.9393543, -0.1680372, 0.1894736
9: 0.0490810, 0.0904766, 0.0520359, 0.0921682, -0.0430872, 0.0384407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0413505, 0.0584358, -0.1025935, 0.0960693
1: -0.0152119, 0.0143234, -0.0135457, 0.0135796, -0.0287915, 0.0278691
2: -0.0034036, 0.0355157, -0.0017059, 0.0367558, -0.0381460, 0.0372216
3: -0.0049208, 0.0644339, -0.0023984, 0.0668044, -0.0536429, 0.0505283
4: -0.0250397, -0.0023599, -0.0241064, -0.0030614, -0.0219782, 0.0217464
5: 0.0049993, 0.0464991, 0.0068713, 0.0477711, -0.0427719, 0.0396278
6: -0.0348876, 0.0455635, -0.0327018, 0.0491298, -0.0840174, 0.0782653
7: -0.0160587, 0.0126032, -0.0150089, 0.0112847, -0.0273434, 0.0276121
8: 0.7135336, 0.9456578, 0.7033206, 0.9411473, -0.1723211, 0.1871364
9: 0.0491103, 0.0908419, 0.0512032, 0.0920525, -0.0429423, 0.0396387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0401982, 0.0587806, -0.1029403, 0.0949187
1: -0.0152119, 0.0143234, -0.0128829, 0.0132836, -0.0284955, 0.0272064
2: -0.0034036, 0.0355157, -0.0010305, 0.0368735, -0.0380357, 0.0365461
3: -0.0049208, 0.0644339, -0.0013947, 0.0670294, -0.0553089, 0.0517675
4: -0.0250397, -0.0023599, -0.0237351, -0.0032822, -0.0217575, 0.0213751
5: 0.0049993, 0.0464991, 0.0076161, 0.0478919, -0.0428927, 0.0388830
6: -0.0348876, 0.0455635, -0.0318324, 0.0494685, -0.0843561, 0.0773959
7: -0.0160587, 0.0126032, -0.0145913, 0.0107602, -0.0268189, 0.0271946
8: 0.7135336, 0.9456578, 0.7023503, 0.9393543, -0.1726165, 0.1907670
9: 0.0491103, 0.0908419, 0.0520359, 0.0921682, -0.0430580, 0.0388060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0413505, 0.0584358, -0.1026357, 0.0947535
1: -0.0152351, 0.0143338, -0.0135457, 0.0135796, -0.0288147, 0.0278795
2: -0.0034273, 0.0350586, -0.0017059, 0.0367558, -0.0383397, 0.0367645
3: -0.0049559, 0.0635683, -0.0023984, 0.0668044, -0.0556710, 0.0515055
4: -0.0250527, -0.0023502, -0.0241064, -0.0030614, -0.0219912, 0.0217561
5: 0.0049731, 0.0460303, 0.0068713, 0.0477711, -0.0427980, 0.0391590
6: -0.0349180, 0.0442495, -0.0327018, 0.0491298, -0.0840478, 0.0769513
7: -0.0160732, 0.0126215, -0.0150089, 0.0112847, -0.0273579, 0.0276304
8: 0.7172979, 0.9457193, 0.7033206, 0.9411473, -0.1712241, 0.1894178
9: 0.0490810, 0.0904766, 0.0512032, 0.0920525, -0.0429715, 0.0392734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0401982, 0.0587806, -0.1029809, 0.0936015
1: -0.0152351, 0.0143338, -0.0128829, 0.0132836, -0.0285187, 0.0272167
2: -0.0034273, 0.0350586, -0.0010305, 0.0368735, -0.0377323, 0.0360891
3: -0.0049559, 0.0635683, -0.0013947, 0.0670294, -0.0538581, 0.0493166
4: -0.0250527, -0.0023502, -0.0237351, -0.0032822, -0.0217705, 0.0213849
5: 0.0049731, 0.0460303, 0.0076161, 0.0478919, -0.0429188, 0.0384142
6: -0.0349180, 0.0442495, -0.0318324, 0.0494685, -0.0843865, 0.0760819
7: -0.0160732, 0.0126215, -0.0145913, 0.0107602, -0.0268334, 0.0272128
8: 0.7172979, 0.9457193, 0.7023503, 0.9393543, -0.1680372, 0.1894736
9: 0.0490810, 0.0904766, 0.0520359, 0.0921682, -0.0430872, 0.0384407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0413505, 0.0584358, -0.1025935, 0.0960693
1: -0.0152119, 0.0143234, -0.0135457, 0.0135796, -0.0287915, 0.0278691
2: -0.0034036, 0.0355157, -0.0017059, 0.0367558, -0.0381460, 0.0372216
3: -0.0049208, 0.0644339, -0.0023984, 0.0668044, -0.0536429, 0.0505283
4: -0.0250397, -0.0023599, -0.0241064, -0.0030614, -0.0219782, 0.0217464
5: 0.0049993, 0.0464991, 0.0068713, 0.0477711, -0.0427719, 0.0396278
6: -0.0348876, 0.0455635, -0.0327018, 0.0491298, -0.0840174, 0.0782653
7: -0.0160587, 0.0126032, -0.0150089, 0.0112847, -0.0273434, 0.0276121
8: 0.7135336, 0.9456578, 0.7033206, 0.9411473, -0.1723211, 0.1871364
9: 0.0491103, 0.0908419, 0.0512032, 0.0920525, -0.0429423, 0.0396387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0401982, 0.0587806, -0.1029403, 0.0949187
1: -0.0152119, 0.0143234, -0.0128829, 0.0132836, -0.0284955, 0.0272064
2: -0.0034036, 0.0355157, -0.0010305, 0.0368735, -0.0380357, 0.0365461
3: -0.0049208, 0.0644339, -0.0013947, 0.0670294, -0.0553089, 0.0517675
4: -0.0250397, -0.0023599, -0.0237351, -0.0032822, -0.0217575, 0.0213751
5: 0.0049993, 0.0464991, 0.0076161, 0.0478919, -0.0428927, 0.0388830
6: -0.0348876, 0.0455635, -0.0318324, 0.0494685, -0.0843561, 0.0773959
7: -0.0160587, 0.0126032, -0.0145913, 0.0107602, -0.0268189, 0.0271946
8: 0.7135336, 0.9456578, 0.7023503, 0.9393543, -0.1726165, 0.1907670
9: 0.0491103, 0.0908419, 0.0520359, 0.0921682, -0.0430580, 0.0388060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0413505, 0.0584358, -0.1026357, 0.0947535
1: -0.0152351, 0.0143338, -0.0135457, 0.0135796, -0.0288147, 0.0278795
2: -0.0034273, 0.0350586, -0.0017059, 0.0367558, -0.0383397, 0.0367645
3: -0.0049559, 0.0635683, -0.0023984, 0.0668044, -0.0556710, 0.0515055
4: -0.0250527, -0.0023502, -0.0241064, -0.0030614, -0.0219912, 0.0217561
5: 0.0049731, 0.0460303, 0.0068713, 0.0477711, -0.0427980, 0.0391590
6: -0.0349180, 0.0442495, -0.0327018, 0.0491298, -0.0840478, 0.0769513
7: -0.0160732, 0.0126215, -0.0150089, 0.0112847, -0.0273579, 0.0276304
8: 0.7172979, 0.9457193, 0.7033206, 0.9411473, -0.1712241, 0.1894178
9: 0.0490810, 0.0904766, 0.0512032, 0.0920525, -0.0429715, 0.0392734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0401982, 0.0587806, -0.1029809, 0.0936015
1: -0.0152351, 0.0143338, -0.0128829, 0.0132836, -0.0285187, 0.0272167
2: -0.0034273, 0.0350586, -0.0010305, 0.0368735, -0.0377323, 0.0360891
3: -0.0049559, 0.0635683, -0.0013947, 0.0670294, -0.0538581, 0.0493166
4: -0.0250527, -0.0023502, -0.0237351, -0.0032822, -0.0217705, 0.0213849
5: 0.0049731, 0.0460303, 0.0076161, 0.0478919, -0.0429188, 0.0384142
6: -0.0349180, 0.0442495, -0.0318324, 0.0494685, -0.0843865, 0.0760819
7: -0.0160732, 0.0126215, -0.0145913, 0.0107602, -0.0268334, 0.0272128
8: 0.7172979, 0.9457193, 0.7023503, 0.9393543, -0.1680372, 0.1894736
9: 0.0490810, 0.0904766, 0.0520359, 0.0921682, -0.0430872, 0.0384407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0413505, 0.0584358, -0.1025935, 0.0960693
1: -0.0152119, 0.0143234, -0.0135457, 0.0135796, -0.0287915, 0.0278691
2: -0.0034036, 0.0355157, -0.0017059, 0.0367558, -0.0381460, 0.0372216
3: -0.0049208, 0.0644339, -0.0023984, 0.0668044, -0.0536429, 0.0505283
4: -0.0250397, -0.0023599, -0.0241064, -0.0030614, -0.0219782, 0.0217464
5: 0.0049993, 0.0464991, 0.0068713, 0.0477711, -0.0427719, 0.0396278
6: -0.0348876, 0.0455635, -0.0327018, 0.0491298, -0.0840174, 0.0782653
7: -0.0160587, 0.0126032, -0.0150089, 0.0112847, -0.0273434, 0.0276121
8: 0.7135336, 0.9456578, 0.7033206, 0.9411473, -0.1723211, 0.1871364
9: 0.0491103, 0.0908419, 0.0512032, 0.0920525, -0.0429423, 0.0396387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0401982, 0.0587806, -0.1029403, 0.0949187
1: -0.0152119, 0.0143234, -0.0128829, 0.0132836, -0.0284955, 0.0272064
2: -0.0034036, 0.0355157, -0.0010305, 0.0368735, -0.0380357, 0.0365461
3: -0.0049208, 0.0644339, -0.0013947, 0.0670294, -0.0553089, 0.0517675
4: -0.0250397, -0.0023599, -0.0237351, -0.0032822, -0.0217575, 0.0213751
5: 0.0049993, 0.0464991, 0.0076161, 0.0478919, -0.0428927, 0.0388830
6: -0.0348876, 0.0455635, -0.0318324, 0.0494685, -0.0843561, 0.0773959
7: -0.0160587, 0.0126032, -0.0145913, 0.0107602, -0.0268189, 0.0271946
8: 0.7135336, 0.9456578, 0.7023503, 0.9393543, -0.1726165, 0.1907670
9: 0.0491103, 0.0908419, 0.0520359, 0.0921682, -0.0430580, 0.0388060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0413505, 0.0584358, -0.1026357, 0.0947535
1: -0.0152351, 0.0143338, -0.0135457, 0.0135796, -0.0288147, 0.0278795
2: -0.0034273, 0.0350586, -0.0017059, 0.0367558, -0.0383397, 0.0367645
3: -0.0049559, 0.0635683, -0.0023984, 0.0668044, -0.0556710, 0.0515055
4: -0.0250527, -0.0023502, -0.0241064, -0.0030614, -0.0219912, 0.0217561
5: 0.0049731, 0.0460303, 0.0068713, 0.0477711, -0.0427980, 0.0391590
6: -0.0349180, 0.0442495, -0.0327018, 0.0491298, -0.0840478, 0.0769513
7: -0.0160732, 0.0126215, -0.0150089, 0.0112847, -0.0273579, 0.0276304
8: 0.7172979, 0.9457193, 0.7033206, 0.9411473, -0.1712241, 0.1894178
9: 0.0490810, 0.0904766, 0.0512032, 0.0920525, -0.0429715, 0.0392734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0401982, 0.0587806, -0.1029809, 0.0936015
1: -0.0152351, 0.0143338, -0.0128829, 0.0132836, -0.0285187, 0.0272167
2: -0.0034273, 0.0350586, -0.0010305, 0.0368735, -0.0377323, 0.0360891
3: -0.0049559, 0.0635683, -0.0013947, 0.0670294, -0.0538581, 0.0493166
4: -0.0250527, -0.0023502, -0.0237351, -0.0032822, -0.0217705, 0.0213849
5: 0.0049731, 0.0460303, 0.0076161, 0.0478919, -0.0429188, 0.0384142
6: -0.0349180, 0.0442495, -0.0318324, 0.0494685, -0.0843865, 0.0760819
7: -0.0160732, 0.0126215, -0.0145913, 0.0107602, -0.0268334, 0.0272128
8: 0.7172979, 0.9457193, 0.7023503, 0.9393543, -0.1680372, 0.1894736
9: 0.0490810, 0.0904766, 0.0520359, 0.0921682, -0.0430872, 0.0384407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0413505, 0.0584358, -0.1025935, 0.0960693
1: -0.0152119, 0.0143234, -0.0135457, 0.0135796, -0.0287915, 0.0278691
2: -0.0034036, 0.0355157, -0.0017059, 0.0367558, -0.0381460, 0.0372216
3: -0.0049208, 0.0644339, -0.0023984, 0.0668044, -0.0536429, 0.0505283
4: -0.0250397, -0.0023599, -0.0241064, -0.0030614, -0.0219782, 0.0217464
5: 0.0049993, 0.0464991, 0.0068713, 0.0477711, -0.0427719, 0.0396278
6: -0.0348876, 0.0455635, -0.0327018, 0.0491298, -0.0840174, 0.0782653
7: -0.0160587, 0.0126032, -0.0150089, 0.0112847, -0.0273434, 0.0276121
8: 0.7135336, 0.9456578, 0.7033206, 0.9411473, -0.1723211, 0.1871364
9: 0.0491103, 0.0908419, 0.0512032, 0.0920525, -0.0429423, 0.0396387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0401982, 0.0587806, -0.1029403, 0.0949187
1: -0.0152119, 0.0143234, -0.0128829, 0.0132836, -0.0284955, 0.0272064
2: -0.0034036, 0.0355157, -0.0010305, 0.0368735, -0.0380357, 0.0365461
3: -0.0049208, 0.0644339, -0.0013947, 0.0670294, -0.0553089, 0.0517675
4: -0.0250397, -0.0023599, -0.0237351, -0.0032822, -0.0217575, 0.0213751
5: 0.0049993, 0.0464991, 0.0076161, 0.0478919, -0.0428927, 0.0388830
6: -0.0348876, 0.0455635, -0.0318324, 0.0494685, -0.0843561, 0.0773959
7: -0.0160587, 0.0126032, -0.0145913, 0.0107602, -0.0268189, 0.0271946
8: 0.7135336, 0.9456578, 0.7023503, 0.9393543, -0.1726165, 0.1907670
9: 0.0491103, 0.0908419, 0.0520359, 0.0921682, -0.0430580, 0.0388060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1109350
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0413505, 0.0584358, -0.1026357, 0.0947535
1: -0.0152351, 0.0143338, -0.0135457, 0.0135796, -0.0288147, 0.0278795
2: -0.0034273, 0.0350586, -0.0017059, 0.0367558, -0.0383397, 0.0367645
3: -0.0049559, 0.0635683, -0.0023984, 0.0668044, -0.0556710, 0.0515055
4: -0.0250527, -0.0023502, -0.0241064, -0.0030614, -0.0219912, 0.0217561
5: 0.0049731, 0.0460303, 0.0068713, 0.0477711, -0.0427980, 0.0391590
6: -0.0349180, 0.0442495, -0.0327018, 0.0491298, -0.0840478, 0.0769513
7: -0.0160732, 0.0126215, -0.0150089, 0.0112847, -0.0273579, 0.0276304
8: 0.7172979, 0.9457193, 0.7033206, 0.9411473, -0.1712241, 0.1894178
9: 0.0490810, 0.0904766, 0.0512032, 0.0920525, -0.0429715, 0.0392734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0401982, 0.0587806, -0.1029809, 0.0936015
1: -0.0152351, 0.0143338, -0.0128829, 0.0132836, -0.0285187, 0.0272167
2: -0.0034273, 0.0350586, -0.0010305, 0.0368735, -0.0377323, 0.0360891
3: -0.0049559, 0.0635683, -0.0013947, 0.0670294, -0.0538581, 0.0493166
4: -0.0250527, -0.0023502, -0.0237351, -0.0032822, -0.0217705, 0.0213849
5: 0.0049731, 0.0460303, 0.0076161, 0.0478919, -0.0429188, 0.0384142
6: -0.0349180, 0.0442495, -0.0318324, 0.0494685, -0.0843865, 0.0760819
7: -0.0160732, 0.0126215, -0.0145913, 0.0107602, -0.0268334, 0.0272128
8: 0.7172979, 0.9457193, 0.7023503, 0.9393543, -0.1680372, 0.1894736
9: 0.0490810, 0.0904766, 0.0520359, 0.0921682, -0.0430872, 0.0384407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0413505, 0.0584358, -0.1025935, 0.0960693
1: -0.0152119, 0.0143234, -0.0135457, 0.0135796, -0.0287915, 0.0278691
2: -0.0034036, 0.0355157, -0.0017059, 0.0367558, -0.0381460, 0.0372216
3: -0.0049208, 0.0644339, -0.0023984, 0.0668044, -0.0536429, 0.0505283
4: -0.0250397, -0.0023599, -0.0241064, -0.0030614, -0.0219782, 0.0217464
5: 0.0049993, 0.0464991, 0.0068713, 0.0477711, -0.0427719, 0.0396278
6: -0.0348876, 0.0455635, -0.0327018, 0.0491298, -0.0840174, 0.0782653
7: -0.0160587, 0.0126032, -0.0150089, 0.0112847, -0.0273434, 0.0276121
8: 0.7135336, 0.9456578, 0.7033206, 0.9411473, -0.1723211, 0.1871364
9: 0.0491103, 0.0908419, 0.0512032, 0.0920525, -0.0429423, 0.0396387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0401982, 0.0587806, -0.1029403, 0.0949187
1: -0.0152119, 0.0143234, -0.0128829, 0.0132836, -0.0284955, 0.0272064
2: -0.0034036, 0.0355157, -0.0010305, 0.0368735, -0.0380357, 0.0365461
3: -0.0049208, 0.0644339, -0.0013947, 0.0670294, -0.0553089, 0.0517675
4: -0.0250397, -0.0023599, -0.0237351, -0.0032822, -0.0217575, 0.0213751
5: 0.0049993, 0.0464991, 0.0076161, 0.0478919, -0.0428927, 0.0388830
6: -0.0348876, 0.0455635, -0.0318324, 0.0494685, -0.0843561, 0.0773959
7: -0.0160587, 0.0126032, -0.0145913, 0.0107602, -0.0268189, 0.0271946
8: 0.7135336, 0.9456578, 0.7023503, 0.9393543, -0.1726165, 0.1907670
9: 0.0491103, 0.0908419, 0.0520359, 0.0921682, -0.0430580, 0.0388060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0413505, 0.0584358, -0.1026357, 0.0947535
1: -0.0152351, 0.0143338, -0.0135457, 0.0135796, -0.0288147, 0.0278795
2: -0.0034273, 0.0350586, -0.0017059, 0.0367558, -0.0383397, 0.0367645
3: -0.0049559, 0.0635683, -0.0023984, 0.0668044, -0.0556710, 0.0515055
4: -0.0250527, -0.0023502, -0.0241064, -0.0030614, -0.0219912, 0.0217561
5: 0.0049731, 0.0460303, 0.0068713, 0.0477711, -0.0427980, 0.0391590
6: -0.0349180, 0.0442495, -0.0327018, 0.0491298, -0.0840478, 0.0769513
7: -0.0160732, 0.0126215, -0.0150089, 0.0112847, -0.0273579, 0.0276304
8: 0.7172979, 0.9457193, 0.7033206, 0.9411473, -0.1712241, 0.1894178
9: 0.0490810, 0.0904766, 0.0512032, 0.0920525, -0.0429715, 0.0392734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0401982, 0.0587806, -0.1029809, 0.0936015
1: -0.0152351, 0.0143338, -0.0128829, 0.0132836, -0.0285187, 0.0272167
2: -0.0034273, 0.0350586, -0.0010305, 0.0368735, -0.0377323, 0.0360891
3: -0.0049559, 0.0635683, -0.0013947, 0.0670294, -0.0538581, 0.0493166
4: -0.0250527, -0.0023502, -0.0237351, -0.0032822, -0.0217705, 0.0213849
5: 0.0049731, 0.0460303, 0.0076161, 0.0478919, -0.0429188, 0.0384142
6: -0.0349180, 0.0442495, -0.0318324, 0.0494685, -0.0843865, 0.0760819
7: -0.0160732, 0.0126215, -0.0145913, 0.0107602, -0.0268334, 0.0272128
8: 0.7172979, 0.9457193, 0.7023503, 0.9393543, -0.1680372, 0.1894736
9: 0.0490810, 0.0904766, 0.0520359, 0.0921682, -0.0430872, 0.0384407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0413505, 0.0584358, -0.1025935, 0.0960693
1: -0.0152119, 0.0143234, -0.0135457, 0.0135796, -0.0287915, 0.0278691
2: -0.0034036, 0.0355157, -0.0017059, 0.0367558, -0.0381460, 0.0372216
3: -0.0049208, 0.0644339, -0.0023984, 0.0668044, -0.0536429, 0.0505283
4: -0.0250397, -0.0023599, -0.0241064, -0.0030614, -0.0219782, 0.0217464
5: 0.0049993, 0.0464991, 0.0068713, 0.0477711, -0.0427719, 0.0396278
6: -0.0348876, 0.0455635, -0.0327018, 0.0491298, -0.0840174, 0.0782653
7: -0.0160587, 0.0126032, -0.0150089, 0.0112847, -0.0273434, 0.0276121
8: 0.7135336, 0.9456578, 0.7033206, 0.9411473, -0.1723211, 0.1871364
9: 0.0491103, 0.0908419, 0.0512032, 0.0920525, -0.0429423, 0.0396387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0401982, 0.0587806, -0.1029403, 0.0949187
1: -0.0152119, 0.0143234, -0.0128829, 0.0132836, -0.0284955, 0.0272064
2: -0.0034036, 0.0355157, -0.0010305, 0.0368735, -0.0380357, 0.0365461
3: -0.0049208, 0.0644339, -0.0013947, 0.0670294, -0.0553089, 0.0517675
4: -0.0250397, -0.0023599, -0.0237351, -0.0032822, -0.0217575, 0.0213751
5: 0.0049993, 0.0464991, 0.0076161, 0.0478919, -0.0428927, 0.0388830
6: -0.0348876, 0.0455635, -0.0318324, 0.0494685, -0.0843561, 0.0773959
7: -0.0160587, 0.0126032, -0.0145913, 0.0107602, -0.0268189, 0.0271946
8: 0.7135336, 0.9456578, 0.7023503, 0.9393543, -0.1726165, 0.1907670
9: 0.0491103, 0.0908419, 0.0520359, 0.0921682, -0.0430580, 0.0388060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1109350
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0413505, 0.0584358, -0.1026357, 0.0947535
1: -0.0152351, 0.0143338, -0.0135457, 0.0135796, -0.0288147, 0.0278795
2: -0.0034273, 0.0350586, -0.0017059, 0.0367558, -0.0383397, 0.0367645
3: -0.0049559, 0.0635683, -0.0023984, 0.0668044, -0.0556710, 0.0515055
4: -0.0250527, -0.0023502, -0.0241064, -0.0030614, -0.0219912, 0.0217561
5: 0.0049731, 0.0460303, 0.0068713, 0.0477711, -0.0427980, 0.0391590
6: -0.0349180, 0.0442495, -0.0327018, 0.0491298, -0.0840478, 0.0769513
7: -0.0160732, 0.0126215, -0.0150089, 0.0112847, -0.0273579, 0.0276304
8: 0.7172979, 0.9457193, 0.7033206, 0.9411473, -0.1712241, 0.1894178
9: 0.0490810, 0.0904766, 0.0512032, 0.0920525, -0.0429715, 0.0392734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0401982, 0.0587806, -0.1029809, 0.0936015
1: -0.0152351, 0.0143338, -0.0128829, 0.0132836, -0.0285187, 0.0272167
2: -0.0034273, 0.0350586, -0.0010305, 0.0368735, -0.0377323, 0.0360891
3: -0.0049559, 0.0635683, -0.0013947, 0.0670294, -0.0538581, 0.0493166
4: -0.0250527, -0.0023502, -0.0237351, -0.0032822, -0.0217705, 0.0213849
5: 0.0049731, 0.0460303, 0.0076161, 0.0478919, -0.0429188, 0.0384142
6: -0.0349180, 0.0442495, -0.0318324, 0.0494685, -0.0843865, 0.0760819
7: -0.0160732, 0.0126215, -0.0145913, 0.0107602, -0.0268334, 0.0272128
8: 0.7172979, 0.9457193, 0.7023503, 0.9393543, -0.1680372, 0.1894736
9: 0.0490810, 0.0904766, 0.0520359, 0.0921682, -0.0430872, 0.0384407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0413505, 0.0584358, -0.1025935, 0.0960693
1: -0.0152119, 0.0143234, -0.0135457, 0.0135796, -0.0287915, 0.0278691
2: -0.0034036, 0.0355157, -0.0017059, 0.0367558, -0.0381460, 0.0372216
3: -0.0049208, 0.0644339, -0.0023984, 0.0668044, -0.0536429, 0.0505283
4: -0.0250397, -0.0023599, -0.0241064, -0.0030614, -0.0219782, 0.0217464
5: 0.0049993, 0.0464991, 0.0068713, 0.0477711, -0.0427719, 0.0396278
6: -0.0348876, 0.0455635, -0.0327018, 0.0491298, -0.0840174, 0.0782653
7: -0.0160587, 0.0126032, -0.0150089, 0.0112847, -0.0273434, 0.0276121
8: 0.7135336, 0.9456578, 0.7033206, 0.9411473, -0.1723211, 0.1871364
9: 0.0491103, 0.0908419, 0.0512032, 0.0920525, -0.0429423, 0.0396387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0401982, 0.0587806, -0.1029403, 0.0949187
1: -0.0152119, 0.0143234, -0.0128829, 0.0132836, -0.0284955, 0.0272064
2: -0.0034036, 0.0355157, -0.0010305, 0.0368735, -0.0380357, 0.0365461
3: -0.0049208, 0.0644339, -0.0013947, 0.0670294, -0.0553089, 0.0517675
4: -0.0250397, -0.0023599, -0.0237351, -0.0032822, -0.0217575, 0.0213751
5: 0.0049993, 0.0464991, 0.0076161, 0.0478919, -0.0428927, 0.0388830
6: -0.0348876, 0.0455635, -0.0318324, 0.0494685, -0.0843561, 0.0773959
7: -0.0160587, 0.0126032, -0.0145913, 0.0107602, -0.0268189, 0.0271946
8: 0.7135336, 0.9456578, 0.7023503, 0.9393543, -0.1726165, 0.1907670
9: 0.0491103, 0.0908419, 0.0520359, 0.0921682, -0.0430580, 0.0388060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0413505, 0.0584358, -0.1026357, 0.0947535
1: -0.0152351, 0.0143338, -0.0135457, 0.0135796, -0.0288147, 0.0278795
2: -0.0034273, 0.0350586, -0.0017059, 0.0367558, -0.0383397, 0.0367645
3: -0.0049559, 0.0635683, -0.0023984, 0.0668044, -0.0556710, 0.0515055
4: -0.0250527, -0.0023502, -0.0241064, -0.0030614, -0.0219912, 0.0217561
5: 0.0049731, 0.0460303, 0.0068713, 0.0477711, -0.0427980, 0.0391590
6: -0.0349180, 0.0442495, -0.0327018, 0.0491298, -0.0840478, 0.0769513
7: -0.0160732, 0.0126215, -0.0150089, 0.0112847, -0.0273579, 0.0276304
8: 0.7172979, 0.9457193, 0.7033206, 0.9411473, -0.1712241, 0.1894178
9: 0.0490810, 0.0904766, 0.0512032, 0.0920525, -0.0429715, 0.0392734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0401982, 0.0587806, -0.1029809, 0.0936015
1: -0.0152351, 0.0143338, -0.0128829, 0.0132836, -0.0285187, 0.0272167
2: -0.0034273, 0.0350586, -0.0010305, 0.0368735, -0.0377323, 0.0360891
3: -0.0049559, 0.0635683, -0.0013947, 0.0670294, -0.0538581, 0.0493166
4: -0.0250527, -0.0023502, -0.0237351, -0.0032822, -0.0217705, 0.0213849
5: 0.0049731, 0.0460303, 0.0076161, 0.0478919, -0.0429188, 0.0384142
6: -0.0349180, 0.0442495, -0.0318324, 0.0494685, -0.0843865, 0.0760819
7: -0.0160732, 0.0126215, -0.0145913, 0.0107602, -0.0268334, 0.0272128
8: 0.7172979, 0.9457193, 0.7023503, 0.9393543, -0.1680372, 0.1894736
9: 0.0490810, 0.0904766, 0.0520359, 0.0921682, -0.0430872, 0.0384407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0413505, 0.0584358, -0.1025935, 0.0960693
1: -0.0152119, 0.0143234, -0.0135457, 0.0135796, -0.0287915, 0.0278691
2: -0.0034036, 0.0355157, -0.0017059, 0.0367558, -0.0381460, 0.0372216
3: -0.0049208, 0.0644339, -0.0023984, 0.0668044, -0.0536429, 0.0505283
4: -0.0250397, -0.0023599, -0.0241064, -0.0030614, -0.0219782, 0.0217464
5: 0.0049993, 0.0464991, 0.0068713, 0.0477711, -0.0427719, 0.0396278
6: -0.0348876, 0.0455635, -0.0327018, 0.0491298, -0.0840174, 0.0782653
7: -0.0160587, 0.0126032, -0.0150089, 0.0112847, -0.0273434, 0.0276121
8: 0.7135336, 0.9456578, 0.7033206, 0.9411473, -0.1723211, 0.1871364
9: 0.0491103, 0.0908419, 0.0512032, 0.0920525, -0.0429423, 0.0396387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0401982, 0.0587806, -0.1029403, 0.0949187
1: -0.0152119, 0.0143234, -0.0128829, 0.0132836, -0.0284955, 0.0272064
2: -0.0034036, 0.0355157, -0.0010305, 0.0368735, -0.0380357, 0.0365461
3: -0.0049208, 0.0644339, -0.0013947, 0.0670294, -0.0553089, 0.0517675
4: -0.0250397, -0.0023599, -0.0237351, -0.0032822, -0.0217575, 0.0213751
5: 0.0049993, 0.0464991, 0.0076161, 0.0478919, -0.0428927, 0.0388830
6: -0.0348876, 0.0455635, -0.0318324, 0.0494685, -0.0843561, 0.0773959
7: -0.0160587, 0.0126032, -0.0145913, 0.0107602, -0.0268189, 0.0271946
8: 0.7135336, 0.9456578, 0.7023503, 0.9393543, -0.1726165, 0.1907670
9: 0.0491103, 0.0908419, 0.0520359, 0.0921682, -0.0430580, 0.0388060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0413505, 0.0584358, -0.1026357, 0.0947535
1: -0.0152351, 0.0143338, -0.0135457, 0.0135796, -0.0288147, 0.0278795
2: -0.0034273, 0.0350586, -0.0017059, 0.0367558, -0.0383397, 0.0367645
3: -0.0049559, 0.0635683, -0.0023984, 0.0668044, -0.0556710, 0.0515055
4: -0.0250527, -0.0023502, -0.0241064, -0.0030614, -0.0219912, 0.0217561
5: 0.0049731, 0.0460303, 0.0068713, 0.0477711, -0.0427980, 0.0391590
6: -0.0349180, 0.0442495, -0.0327018, 0.0491298, -0.0840478, 0.0769513
7: -0.0160732, 0.0126215, -0.0150089, 0.0112847, -0.0273579, 0.0276304
8: 0.7172979, 0.9457193, 0.7033206, 0.9411473, -0.1712241, 0.1894178
9: 0.0490810, 0.0904766, 0.0512032, 0.0920525, -0.0429715, 0.0392734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0401982, 0.0587806, -0.1029809, 0.0936015
1: -0.0152351, 0.0143338, -0.0128829, 0.0132836, -0.0285187, 0.0272167
2: -0.0034273, 0.0350586, -0.0010305, 0.0368735, -0.0377323, 0.0360891
3: -0.0049559, 0.0635683, -0.0013947, 0.0670294, -0.0538581, 0.0493166
4: -0.0250527, -0.0023502, -0.0237351, -0.0032822, -0.0217705, 0.0213849
5: 0.0049731, 0.0460303, 0.0076161, 0.0478919, -0.0429188, 0.0384142
6: -0.0349180, 0.0442495, -0.0318324, 0.0494685, -0.0843865, 0.0760819
7: -0.0160732, 0.0126215, -0.0145913, 0.0107602, -0.0268334, 0.0272128
8: 0.7172979, 0.9457193, 0.7023503, 0.9393543, -0.1680372, 0.1894736
9: 0.0490810, 0.0904766, 0.0520359, 0.0921682, -0.0430872, 0.0384407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0413505, 0.0584358, -0.1025935, 0.0960693
1: -0.0152119, 0.0143234, -0.0135457, 0.0135796, -0.0287915, 0.0278691
2: -0.0034036, 0.0355157, -0.0017059, 0.0367558, -0.0381460, 0.0372216
3: -0.0049208, 0.0644339, -0.0023984, 0.0668044, -0.0536429, 0.0505283
4: -0.0250397, -0.0023599, -0.0241064, -0.0030614, -0.0219782, 0.0217464
5: 0.0049993, 0.0464991, 0.0068713, 0.0477711, -0.0427719, 0.0396278
6: -0.0348876, 0.0455635, -0.0327018, 0.0491298, -0.0840174, 0.0782653
7: -0.0160587, 0.0126032, -0.0150089, 0.0112847, -0.0273434, 0.0276121
8: 0.7135336, 0.9456578, 0.7033206, 0.9411473, -0.1723211, 0.1871364
9: 0.0491103, 0.0908419, 0.0512032, 0.0920525, -0.0429423, 0.0396387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0401982, 0.0587806, -0.1029403, 0.0949187
1: -0.0152119, 0.0143234, -0.0128829, 0.0132836, -0.0284955, 0.0272064
2: -0.0034036, 0.0355157, -0.0010305, 0.0368735, -0.0380357, 0.0365461
3: -0.0049208, 0.0644339, -0.0013947, 0.0670294, -0.0553089, 0.0517675
4: -0.0250397, -0.0023599, -0.0237351, -0.0032822, -0.0217575, 0.0213751
5: 0.0049993, 0.0464991, 0.0076161, 0.0478919, -0.0428927, 0.0388830
6: -0.0348876, 0.0455635, -0.0318324, 0.0494685, -0.0843561, 0.0773959
7: -0.0160587, 0.0126032, -0.0145913, 0.0107602, -0.0268189, 0.0271946
8: 0.7135336, 0.9456578, 0.7023503, 0.9393543, -0.1726165, 0.1907670
9: 0.0491103, 0.0908419, 0.0520359, 0.0921682, -0.0430580, 0.0388060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0413505, 0.0584358, -0.1026357, 0.0947535
1: -0.0152351, 0.0143338, -0.0135457, 0.0135796, -0.0288147, 0.0278795
2: -0.0034273, 0.0350586, -0.0017059, 0.0367558, -0.0383397, 0.0367645
3: -0.0049559, 0.0635683, -0.0023984, 0.0668044, -0.0556710, 0.0515055
4: -0.0250527, -0.0023502, -0.0241064, -0.0030614, -0.0219912, 0.0217561
5: 0.0049731, 0.0460303, 0.0068713, 0.0477711, -0.0427980, 0.0391590
6: -0.0349180, 0.0442495, -0.0327018, 0.0491298, -0.0840478, 0.0769513
7: -0.0160732, 0.0126215, -0.0150089, 0.0112847, -0.0273579, 0.0276304
8: 0.7172979, 0.9457193, 0.7033206, 0.9411473, -0.1712241, 0.1894178
9: 0.0490810, 0.0904766, 0.0512032, 0.0920525, -0.0429715, 0.0392734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0401982, 0.0587806, -0.1029809, 0.0936015
1: -0.0152351, 0.0143338, -0.0128829, 0.0132836, -0.0285187, 0.0272167
2: -0.0034273, 0.0350586, -0.0010305, 0.0368735, -0.0377323, 0.0360891
3: -0.0049559, 0.0635683, -0.0013947, 0.0670294, -0.0538581, 0.0493166
4: -0.0250527, -0.0023502, -0.0237351, -0.0032822, -0.0217705, 0.0213849
5: 0.0049731, 0.0460303, 0.0076161, 0.0478919, -0.0429188, 0.0384142
6: -0.0349180, 0.0442495, -0.0318324, 0.0494685, -0.0843865, 0.0760819
7: -0.0160732, 0.0126215, -0.0145913, 0.0107602, -0.0268334, 0.0272128
8: 0.7172979, 0.9457193, 0.7023503, 0.9393543, -0.1680372, 0.1894736
9: 0.0490810, 0.0904766, 0.0520359, 0.0921682, -0.0430872, 0.0384407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.35 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0413505, 0.0584358, -0.1025935, 0.0960693
1: -0.0152119, 0.0143234, -0.0135457, 0.0135796, -0.0287915, 0.0278691
2: -0.0034036, 0.0355157, -0.0017059, 0.0367558, -0.0381460, 0.0372216
3: -0.0049208, 0.0644339, -0.0023984, 0.0668044, -0.0536429, 0.0505283
4: -0.0250397, -0.0023599, -0.0241064, -0.0030614, -0.0219782, 0.0217464
5: 0.0049993, 0.0464991, 0.0068713, 0.0477711, -0.0427719, 0.0396278
6: -0.0348876, 0.0455635, -0.0327018, 0.0491298, -0.0840174, 0.0782653
7: -0.0160587, 0.0126032, -0.0150089, 0.0112847, -0.0273434, 0.0276121
8: 0.7135336, 0.9456578, 0.7033206, 0.9411473, -0.1723211, 0.1871364
9: 0.0491103, 0.0908419, 0.0512032, 0.0920525, -0.0429423, 0.0396387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0401982, 0.0587806, -0.1029403, 0.0949187
1: -0.0152119, 0.0143234, -0.0128829, 0.0132836, -0.0284955, 0.0272064
2: -0.0034036, 0.0355157, -0.0010305, 0.0368735, -0.0380357, 0.0365461
3: -0.0049208, 0.0644339, -0.0013947, 0.0670294, -0.0553089, 0.0517675
4: -0.0250397, -0.0023599, -0.0237351, -0.0032822, -0.0217575, 0.0213751
5: 0.0049993, 0.0464991, 0.0076161, 0.0478919, -0.0428927, 0.0388830
6: -0.0348876, 0.0455635, -0.0318324, 0.0494685, -0.0843561, 0.0773959
7: -0.0160587, 0.0126032, -0.0145913, 0.0107602, -0.0268189, 0.0271946
8: 0.7135336, 0.9456578, 0.7023503, 0.9393543, -0.1726165, 0.1907670
9: 0.0491103, 0.0908419, 0.0520359, 0.0921682, -0.0430580, 0.0388060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0413505, 0.0584358, -0.1026357, 0.0947535
1: -0.0152351, 0.0143338, -0.0135457, 0.0135796, -0.0288147, 0.0278795
2: -0.0034273, 0.0350586, -0.0017059, 0.0367558, -0.0383397, 0.0367645
3: -0.0049559, 0.0635683, -0.0023984, 0.0668044, -0.0556710, 0.0515055
4: -0.0250527, -0.0023502, -0.0241064, -0.0030614, -0.0219912, 0.0217561
5: 0.0049731, 0.0460303, 0.0068713, 0.0477711, -0.0427980, 0.0391590
6: -0.0349180, 0.0442495, -0.0327018, 0.0491298, -0.0840478, 0.0769513
7: -0.0160732, 0.0126215, -0.0150089, 0.0112847, -0.0273579, 0.0276304
8: 0.7172979, 0.9457193, 0.7033206, 0.9411473, -0.1712241, 0.1894178
9: 0.0490810, 0.0904766, 0.0512032, 0.0920525, -0.0429715, 0.0392734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0401982, 0.0587806, -0.1029809, 0.0936015
1: -0.0152351, 0.0143338, -0.0128829, 0.0132836, -0.0285187, 0.0272167
2: -0.0034273, 0.0350586, -0.0010305, 0.0368735, -0.0377323, 0.0360891
3: -0.0049559, 0.0635683, -0.0013947, 0.0670294, -0.0538581, 0.0493166
4: -0.0250527, -0.0023502, -0.0237351, -0.0032822, -0.0217705, 0.0213849
5: 0.0049731, 0.0460303, 0.0076161, 0.0478919, -0.0429188, 0.0384142
6: -0.0349180, 0.0442495, -0.0318324, 0.0494685, -0.0843865, 0.0760819
7: -0.0160732, 0.0126215, -0.0145913, 0.0107602, -0.0268334, 0.0272128
8: 0.7172979, 0.9457193, 0.7023503, 0.9393543, -0.1680372, 0.1894736
9: 0.0490810, 0.0904766, 0.0520359, 0.0921682, -0.0430872, 0.0384407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0413505, 0.0584358, -0.1025935, 0.0960693
1: -0.0152119, 0.0143234, -0.0135457, 0.0135796, -0.0287915, 0.0278691
2: -0.0034036, 0.0355157, -0.0017059, 0.0367558, -0.0381460, 0.0372216
3: -0.0049208, 0.0644339, -0.0023984, 0.0668044, -0.0536429, 0.0505283
4: -0.0250397, -0.0023599, -0.0241064, -0.0030614, -0.0219782, 0.0217464
5: 0.0049993, 0.0464991, 0.0068713, 0.0477711, -0.0427719, 0.0396278
6: -0.0348876, 0.0455635, -0.0327018, 0.0491298, -0.0840174, 0.0782653
7: -0.0160587, 0.0126032, -0.0150089, 0.0112847, -0.0273434, 0.0276121
8: 0.7135336, 0.9456578, 0.7033206, 0.9411473, -0.1723211, 0.1871364
9: 0.0491103, 0.0908419, 0.0512032, 0.0920525, -0.0429423, 0.0396387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0401982, 0.0587806, -0.1029403, 0.0949187
1: -0.0152119, 0.0143234, -0.0128829, 0.0132836, -0.0284955, 0.0272064
2: -0.0034036, 0.0355157, -0.0010305, 0.0368735, -0.0380357, 0.0365461
3: -0.0049208, 0.0644339, -0.0013947, 0.0670294, -0.0553089, 0.0517675
4: -0.0250397, -0.0023599, -0.0237351, -0.0032822, -0.0217575, 0.0213751
5: 0.0049993, 0.0464991, 0.0076161, 0.0478919, -0.0428927, 0.0388830
6: -0.0348876, 0.0455635, -0.0318324, 0.0494685, -0.0843561, 0.0773959
7: -0.0160587, 0.0126032, -0.0145913, 0.0107602, -0.0268189, 0.0271946
8: 0.7135336, 0.9456578, 0.7023503, 0.9393543, -0.1726165, 0.1907670
9: 0.0491103, 0.0908419, 0.0520359, 0.0921682, -0.0430580, 0.0388060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0413505, 0.0584358, -0.1026357, 0.0947535
1: -0.0152351, 0.0143338, -0.0135457, 0.0135796, -0.0288147, 0.0278795
2: -0.0034273, 0.0350586, -0.0017059, 0.0367558, -0.0383397, 0.0367645
3: -0.0049559, 0.0635683, -0.0023984, 0.0668044, -0.0556710, 0.0515055
4: -0.0250527, -0.0023502, -0.0241064, -0.0030614, -0.0219912, 0.0217561
5: 0.0049731, 0.0460303, 0.0068713, 0.0477711, -0.0427980, 0.0391590
6: -0.0349180, 0.0442495, -0.0327018, 0.0491298, -0.0840478, 0.0769513
7: -0.0160732, 0.0126215, -0.0150089, 0.0112847, -0.0273579, 0.0276304
8: 0.7172979, 0.9457193, 0.7033206, 0.9411473, -0.1712241, 0.1894178
9: 0.0490810, 0.0904766, 0.0512032, 0.0920525, -0.0429715, 0.0392734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0401982, 0.0587806, -0.1029809, 0.0936015
1: -0.0152351, 0.0143338, -0.0128829, 0.0132836, -0.0285187, 0.0272167
2: -0.0034273, 0.0350586, -0.0010305, 0.0368735, -0.0377323, 0.0360891
3: -0.0049559, 0.0635683, -0.0013947, 0.0670294, -0.0538581, 0.0493166
4: -0.0250527, -0.0023502, -0.0237351, -0.0032822, -0.0217705, 0.0213849
5: 0.0049731, 0.0460303, 0.0076161, 0.0478919, -0.0429188, 0.0384142
6: -0.0349180, 0.0442495, -0.0318324, 0.0494685, -0.0843865, 0.0760819
7: -0.0160732, 0.0126215, -0.0145913, 0.0107602, -0.0268334, 0.0272128
8: 0.7172979, 0.9457193, 0.7023503, 0.9393543, -0.1680372, 0.1894736
9: 0.0490810, 0.0904766, 0.0520359, 0.0921682, -0.0430872, 0.0384407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0413505, 0.0584358, -0.1025935, 0.0960693
1: -0.0152119, 0.0143234, -0.0135457, 0.0135796, -0.0287915, 0.0278691
2: -0.0034036, 0.0355157, -0.0017059, 0.0367558, -0.0381460, 0.0372216
3: -0.0049208, 0.0644339, -0.0023984, 0.0668044, -0.0536429, 0.0505283
4: -0.0250397, -0.0023599, -0.0241064, -0.0030614, -0.0219782, 0.0217464
5: 0.0049993, 0.0464991, 0.0068713, 0.0477711, -0.0427719, 0.0396278
6: -0.0348876, 0.0455635, -0.0327018, 0.0491298, -0.0840174, 0.0782653
7: -0.0160587, 0.0126032, -0.0150089, 0.0112847, -0.0273434, 0.0276121
8: 0.7135336, 0.9456578, 0.7033206, 0.9411473, -0.1723211, 0.1871364
9: 0.0491103, 0.0908419, 0.0512032, 0.0920525, -0.0429423, 0.0396387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0401982, 0.0587806, -0.1029403, 0.0949187
1: -0.0152119, 0.0143234, -0.0128829, 0.0132836, -0.0284955, 0.0272064
2: -0.0034036, 0.0355157, -0.0010305, 0.0368735, -0.0380357, 0.0365461
3: -0.0049208, 0.0644339, -0.0013947, 0.0670294, -0.0553089, 0.0517675
4: -0.0250397, -0.0023599, -0.0237351, -0.0032822, -0.0217575, 0.0213751
5: 0.0049993, 0.0464991, 0.0076161, 0.0478919, -0.0428927, 0.0388830
6: -0.0348876, 0.0455635, -0.0318324, 0.0494685, -0.0843561, 0.0773959
7: -0.0160587, 0.0126032, -0.0145913, 0.0107602, -0.0268189, 0.0271946
8: 0.7135336, 0.9456578, 0.7023503, 0.9393543, -0.1726165, 0.1907670
9: 0.0491103, 0.0908419, 0.0520359, 0.0921682, -0.0430580, 0.0388060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1109350
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0413505, 0.0584358, -0.1026357, 0.0947535
1: -0.0152351, 0.0143338, -0.0135457, 0.0135796, -0.0288147, 0.0278795
2: -0.0034273, 0.0350586, -0.0017059, 0.0367558, -0.0383397, 0.0367645
3: -0.0049559, 0.0635683, -0.0023984, 0.0668044, -0.0556710, 0.0515055
4: -0.0250527, -0.0023502, -0.0241064, -0.0030614, -0.0219912, 0.0217561
5: 0.0049731, 0.0460303, 0.0068713, 0.0477711, -0.0427980, 0.0391590
6: -0.0349180, 0.0442495, -0.0327018, 0.0491298, -0.0840478, 0.0769513
7: -0.0160732, 0.0126215, -0.0150089, 0.0112847, -0.0273579, 0.0276304
8: 0.7172979, 0.9457193, 0.7033206, 0.9411473, -0.1712241, 0.1894178
9: 0.0490810, 0.0904766, 0.0512032, 0.0920525, -0.0429715, 0.0392734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0401982, 0.0587806, -0.1029809, 0.0936015
1: -0.0152351, 0.0143338, -0.0128829, 0.0132836, -0.0285187, 0.0272167
2: -0.0034273, 0.0350586, -0.0010305, 0.0368735, -0.0377323, 0.0360891
3: -0.0049559, 0.0635683, -0.0013947, 0.0670294, -0.0538581, 0.0493166
4: -0.0250527, -0.0023502, -0.0237351, -0.0032822, -0.0217705, 0.0213849
5: 0.0049731, 0.0460303, 0.0076161, 0.0478919, -0.0429188, 0.0384142
6: -0.0349180, 0.0442495, -0.0318324, 0.0494685, -0.0843865, 0.0760819
7: -0.0160732, 0.0126215, -0.0145913, 0.0107602, -0.0268334, 0.0272128
8: 0.7172979, 0.9457193, 0.7023503, 0.9393543, -0.1680372, 0.1894736
9: 0.0490810, 0.0904766, 0.0520359, 0.0921682, -0.0430872, 0.0384407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0413505, 0.0584358, -0.1025935, 0.0960693
1: -0.0152119, 0.0143234, -0.0135457, 0.0135796, -0.0287915, 0.0278691
2: -0.0034036, 0.0355157, -0.0017059, 0.0367558, -0.0381460, 0.0372216
3: -0.0049208, 0.0644339, -0.0023984, 0.0668044, -0.0536429, 0.0505283
4: -0.0250397, -0.0023599, -0.0241064, -0.0030614, -0.0219782, 0.0217464
5: 0.0049993, 0.0464991, 0.0068713, 0.0477711, -0.0427719, 0.0396278
6: -0.0348876, 0.0455635, -0.0327018, 0.0491298, -0.0840174, 0.0782653
7: -0.0160587, 0.0126032, -0.0150089, 0.0112847, -0.0273434, 0.0276121
8: 0.7135336, 0.9456578, 0.7033206, 0.9411473, -0.1723211, 0.1871364
9: 0.0491103, 0.0908419, 0.0512032, 0.0920525, -0.0429423, 0.0396387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0401982, 0.0587806, -0.1029403, 0.0949187
1: -0.0152119, 0.0143234, -0.0128829, 0.0132836, -0.0284955, 0.0272064
2: -0.0034036, 0.0355157, -0.0010305, 0.0368735, -0.0380357, 0.0365461
3: -0.0049208, 0.0644339, -0.0013947, 0.0670294, -0.0553089, 0.0517675
4: -0.0250397, -0.0023599, -0.0237351, -0.0032822, -0.0217575, 0.0213751
5: 0.0049993, 0.0464991, 0.0076161, 0.0478919, -0.0428927, 0.0388830
6: -0.0348876, 0.0455635, -0.0318324, 0.0494685, -0.0843561, 0.0773959
7: -0.0160587, 0.0126032, -0.0145913, 0.0107602, -0.0268189, 0.0271946
8: 0.7135336, 0.9456578, 0.7023503, 0.9393543, -0.1726165, 0.1907670
9: 0.0491103, 0.0908419, 0.0520359, 0.0921682, -0.0430580, 0.0388060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0413505, 0.0584358, -0.1026357, 0.0947535
1: -0.0152351, 0.0143338, -0.0135457, 0.0135796, -0.0288147, 0.0278795
2: -0.0034273, 0.0350586, -0.0017059, 0.0367558, -0.0383397, 0.0367645
3: -0.0049559, 0.0635683, -0.0023984, 0.0668044, -0.0556710, 0.0515055
4: -0.0250527, -0.0023502, -0.0241064, -0.0030614, -0.0219912, 0.0217561
5: 0.0049731, 0.0460303, 0.0068713, 0.0477711, -0.0427980, 0.0391590
6: -0.0349180, 0.0442495, -0.0327018, 0.0491298, -0.0840478, 0.0769513
7: -0.0160732, 0.0126215, -0.0150089, 0.0112847, -0.0273579, 0.0276304
8: 0.7172979, 0.9457193, 0.7033206, 0.9411473, -0.1712241, 0.1894178
9: 0.0490810, 0.0904766, 0.0512032, 0.0920525, -0.0429715, 0.0392734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0401982, 0.0587806, -0.1029809, 0.0936015
1: -0.0152351, 0.0143338, -0.0128829, 0.0132836, -0.0285187, 0.0272167
2: -0.0034273, 0.0350586, -0.0010305, 0.0368735, -0.0377323, 0.0360891
3: -0.0049559, 0.0635683, -0.0013947, 0.0670294, -0.0538581, 0.0493166
4: -0.0250527, -0.0023502, -0.0237351, -0.0032822, -0.0217705, 0.0213849
5: 0.0049731, 0.0460303, 0.0076161, 0.0478919, -0.0429188, 0.0384142
6: -0.0349180, 0.0442495, -0.0318324, 0.0494685, -0.0843865, 0.0760819
7: -0.0160732, 0.0126215, -0.0145913, 0.0107602, -0.0268334, 0.0272128
8: 0.7172979, 0.9457193, 0.7023503, 0.9393543, -0.1680372, 0.1894736
9: 0.0490810, 0.0904766, 0.0520359, 0.0921682, -0.0430872, 0.0384407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0413505, 0.0584358, -0.1025935, 0.0960693
1: -0.0152119, 0.0143234, -0.0135457, 0.0135796, -0.0287915, 0.0278691
2: -0.0034036, 0.0355157, -0.0017059, 0.0367558, -0.0381460, 0.0372216
3: -0.0049208, 0.0644339, -0.0023984, 0.0668044, -0.0536429, 0.0505283
4: -0.0250397, -0.0023599, -0.0241064, -0.0030614, -0.0219782, 0.0217464
5: 0.0049993, 0.0464991, 0.0068713, 0.0477711, -0.0427719, 0.0396278
6: -0.0348876, 0.0455635, -0.0327018, 0.0491298, -0.0840174, 0.0782653
7: -0.0160587, 0.0126032, -0.0150089, 0.0112847, -0.0273434, 0.0276121
8: 0.7135336, 0.9456578, 0.7033206, 0.9411473, -0.1723211, 0.1871364
9: 0.0491103, 0.0908419, 0.0512032, 0.0920525, -0.0429423, 0.0396387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.35 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0442470, 0.0548080, -0.0401982, 0.0587806, -0.1029403, 0.0949187
1: -0.0152119, 0.0143234, -0.0128829, 0.0132836, -0.0284955, 0.0272064
2: -0.0034036, 0.0355157, -0.0010305, 0.0368735, -0.0380357, 0.0365461
3: -0.0049208, 0.0644339, -0.0013947, 0.0670294, -0.0553089, 0.0517675
4: -0.0250397, -0.0023599, -0.0237351, -0.0032822, -0.0217575, 0.0213751
5: 0.0049993, 0.0464991, 0.0076161, 0.0478919, -0.0428927, 0.0388830
6: -0.0348876, 0.0455635, -0.0318324, 0.0494685, -0.0843561, 0.0773959
7: -0.0160587, 0.0126032, -0.0145913, 0.0107602, -0.0268189, 0.0271946
8: 0.7135336, 0.9456578, 0.7023503, 0.9393543, -0.1726165, 0.1907670
9: 0.0491103, 0.0908419, 0.0520359, 0.0921682, -0.0430580, 0.0388060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1109350
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0413505, 0.0584358, -0.1026357, 0.0947535
1: -0.0152351, 0.0143338, -0.0135457, 0.0135796, -0.0288147, 0.0278795
2: -0.0034273, 0.0350586, -0.0017059, 0.0367558, -0.0383397, 0.0367645
3: -0.0049559, 0.0635683, -0.0023984, 0.0668044, -0.0556710, 0.0515055
4: -0.0250527, -0.0023502, -0.0241064, -0.0030614, -0.0219912, 0.0217561
5: 0.0049731, 0.0460303, 0.0068713, 0.0477711, -0.0427980, 0.0391590
6: -0.0349180, 0.0442495, -0.0327018, 0.0491298, -0.0840478, 0.0769513
7: -0.0160732, 0.0126215, -0.0150089, 0.0112847, -0.0273579, 0.0276304
8: 0.7172979, 0.9457193, 0.7033206, 0.9411473, -0.1712241, 0.1894178
9: 0.0490810, 0.0904766, 0.0512032, 0.0920525, -0.0429715, 0.0392734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0442871, 0.0534906, -0.0401982, 0.0587806, -0.1029809, 0.0936015
1: -0.0152351, 0.0143338, -0.0128829, 0.0132836, -0.0285187, 0.0272167
2: -0.0034273, 0.0350586, -0.0010305, 0.0368735, -0.0377323, 0.0360891
3: -0.0049559, 0.0635683, -0.0013947, 0.0670294, -0.0538581, 0.0493166
4: -0.0250527, -0.0023502, -0.0237351, -0.0032822, -0.0217705, 0.0213849
5: 0.0049731, 0.0460303, 0.0076161, 0.0478919, -0.0429188, 0.0384142
6: -0.0349180, 0.0442495, -0.0318324, 0.0494685, -0.0843865, 0.0760819
7: -0.0160732, 0.0126215, -0.0145913, 0.0107602, -0.0268334, 0.0272128
8: 0.7172979, 0.9457193, 0.7023503, 0.9393543, -0.1680372, 0.1894736
9: 0.0490810, 0.0904766, 0.0520359, 0.0921682, -0.0430872, 0.0384407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442470, 0.0548080, -0.0960693, 0.1025934
1: -0.0135457, 0.0135796, -0.0152119, 0.0143234, -0.0278691, 0.0287915
2: -0.0017059, 0.0367558, -0.0034036, 0.0355157, -0.0372216, 0.0381460
3: -0.0023984, 0.0668044, -0.0049208, 0.0644339, -0.0505283, 0.0536429
4: -0.0241064, -0.0030614, -0.0250397, -0.0023599, -0.0217464, 0.0219782
5: 0.0068713, 0.0477711, 0.0049993, 0.0464991, -0.0396278, 0.0427719
6: -0.0327018, 0.0491298, -0.0348876, 0.0455635, -0.0782653, 0.0840174
7: -0.0150089, 0.0112847, -0.0160587, 0.0126032, -0.0276121, 0.0273434
8: 0.7033206, 0.9411473, 0.7135336, 0.9456578, -0.1871364, 0.1723211
9: 0.0512032, 0.0920525, 0.0491103, 0.0908419, -0.0396387, 0.0429423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442871, 0.0534906, -0.0947534, 0.1026359
1: -0.0135457, 0.0135796, -0.0152351, 0.0143338, -0.0278795, 0.0288147
2: -0.0017059, 0.0367558, -0.0034273, 0.0350586, -0.0367645, 0.0383396
3: -0.0023984, 0.0668044, -0.0049559, 0.0635683, -0.0515055, 0.0556709
4: -0.0241064, -0.0030614, -0.0250527, -0.0023502, -0.0217561, 0.0219912
5: 0.0068713, 0.0477711, 0.0049731, 0.0460303, -0.0391590, 0.0427980
6: -0.0327018, 0.0491298, -0.0349180, 0.0442495, -0.0769513, 0.0840478
7: -0.0150089, 0.0112847, -0.0160732, 0.0126215, -0.0276304, 0.0273579
8: 0.7033206, 0.9411473, 0.7172979, 0.9457193, -0.1894178, 0.1712241
9: 0.0512032, 0.0920525, 0.0490810, 0.0904766, -0.0392734, 0.0429715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442470, 0.0548080, -0.0949187, 0.1029401
1: -0.0128829, 0.0132836, -0.0152119, 0.0143234, -0.0272064, 0.0284955
2: -0.0010305, 0.0368735, -0.0034036, 0.0355157, -0.0365461, 0.0380358
3: -0.0013947, 0.0670294, -0.0049208, 0.0644339, -0.0517675, 0.0553089
4: -0.0237351, -0.0032822, -0.0250397, -0.0023599, -0.0213751, 0.0217575
5: 0.0076161, 0.0478919, 0.0049993, 0.0464991, -0.0388830, 0.0428927
6: -0.0318324, 0.0494685, -0.0348876, 0.0455635, -0.0773959, 0.0843561
7: -0.0145913, 0.0107602, -0.0160587, 0.0126032, -0.0271946, 0.0268189
8: 0.7023503, 0.9393543, 0.7135336, 0.9456578, -0.1907670, 0.1726165
9: 0.0520359, 0.0921682, 0.0491103, 0.0908419, -0.0388060, 0.0430580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442871, 0.0534906, -0.0936016, 0.1029809
1: -0.0128829, 0.0132836, -0.0152351, 0.0143338, -0.0272167, 0.0285187
2: -0.0010305, 0.0368735, -0.0034273, 0.0350586, -0.0360891, 0.0377323
3: -0.0013947, 0.0670294, -0.0049559, 0.0635683, -0.0493166, 0.0538580
4: -0.0237351, -0.0032822, -0.0250527, -0.0023502, -0.0213849, 0.0217705
5: 0.0076161, 0.0478919, 0.0049731, 0.0460303, -0.0384142, 0.0429188
6: -0.0318324, 0.0494685, -0.0349180, 0.0442495, -0.0760819, 0.0843865
7: -0.0145913, 0.0107602, -0.0160732, 0.0126215, -0.0272128, 0.0268334
8: 0.7023503, 0.9393543, 0.7172979, 0.9457193, -0.1894736, 0.1680372
9: 0.0520359, 0.0921682, 0.0490810, 0.0904766, -0.0384407, 0.0430872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442470, 0.0548080, -0.0960693, 0.1025934
1: -0.0135457, 0.0135796, -0.0152119, 0.0143234, -0.0278691, 0.0287915
2: -0.0017059, 0.0367558, -0.0034036, 0.0355157, -0.0372216, 0.0381460
3: -0.0023984, 0.0668044, -0.0049208, 0.0644339, -0.0505283, 0.0536429
4: -0.0241064, -0.0030614, -0.0250397, -0.0023599, -0.0217464, 0.0219782
5: 0.0068713, 0.0477711, 0.0049993, 0.0464991, -0.0396278, 0.0427719
6: -0.0327018, 0.0491298, -0.0348876, 0.0455635, -0.0782653, 0.0840174
7: -0.0150089, 0.0112847, -0.0160587, 0.0126032, -0.0276121, 0.0273434
8: 0.7033206, 0.9411473, 0.7135336, 0.9456578, -0.1871364, 0.1723211
9: 0.0512032, 0.0920525, 0.0491103, 0.0908419, -0.0396387, 0.0429423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442871, 0.0534906, -0.0947534, 0.1026359
1: -0.0135457, 0.0135796, -0.0152351, 0.0143338, -0.0278795, 0.0288147
2: -0.0017059, 0.0367558, -0.0034273, 0.0350586, -0.0367645, 0.0383396
3: -0.0023984, 0.0668044, -0.0049559, 0.0635683, -0.0515055, 0.0556709
4: -0.0241064, -0.0030614, -0.0250527, -0.0023502, -0.0217561, 0.0219912
5: 0.0068713, 0.0477711, 0.0049731, 0.0460303, -0.0391590, 0.0427980
6: -0.0327018, 0.0491298, -0.0349180, 0.0442495, -0.0769513, 0.0840478
7: -0.0150089, 0.0112847, -0.0160732, 0.0126215, -0.0276304, 0.0273579
8: 0.7033206, 0.9411473, 0.7172979, 0.9457193, -0.1894178, 0.1712241
9: 0.0512032, 0.0920525, 0.0490810, 0.0904766, -0.0392734, 0.0429715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442470, 0.0548080, -0.0949187, 0.1029401
1: -0.0128829, 0.0132836, -0.0152119, 0.0143234, -0.0272064, 0.0284955
2: -0.0010305, 0.0368735, -0.0034036, 0.0355157, -0.0365461, 0.0380358
3: -0.0013947, 0.0670294, -0.0049208, 0.0644339, -0.0517675, 0.0553089
4: -0.0237351, -0.0032822, -0.0250397, -0.0023599, -0.0213751, 0.0217575
5: 0.0076161, 0.0478919, 0.0049993, 0.0464991, -0.0388830, 0.0428927
6: -0.0318324, 0.0494685, -0.0348876, 0.0455635, -0.0773959, 0.0843561
7: -0.0145913, 0.0107602, -0.0160587, 0.0126032, -0.0271946, 0.0268189
8: 0.7023503, 0.9393543, 0.7135336, 0.9456578, -0.1907670, 0.1726165
9: 0.0520359, 0.0921682, 0.0491103, 0.0908419, -0.0388060, 0.0430580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442871, 0.0534906, -0.0936016, 0.1029809
1: -0.0128829, 0.0132836, -0.0152351, 0.0143338, -0.0272167, 0.0285187
2: -0.0010305, 0.0368735, -0.0034273, 0.0350586, -0.0360891, 0.0377323
3: -0.0013947, 0.0670294, -0.0049559, 0.0635683, -0.0493166, 0.0538580
4: -0.0237351, -0.0032822, -0.0250527, -0.0023502, -0.0213849, 0.0217705
5: 0.0076161, 0.0478919, 0.0049731, 0.0460303, -0.0384142, 0.0429188
6: -0.0318324, 0.0494685, -0.0349180, 0.0442495, -0.0760819, 0.0843865
7: -0.0145913, 0.0107602, -0.0160732, 0.0126215, -0.0272128, 0.0268334
8: 0.7023503, 0.9393543, 0.7172979, 0.9457193, -0.1894736, 0.1680372
9: 0.0520359, 0.0921682, 0.0490810, 0.0904766, -0.0384407, 0.0430872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442470, 0.0548080, -0.0960693, 0.1025934
1: -0.0135457, 0.0135796, -0.0152119, 0.0143234, -0.0278691, 0.0287915
2: -0.0017059, 0.0367558, -0.0034036, 0.0355157, -0.0372216, 0.0381460
3: -0.0023984, 0.0668044, -0.0049208, 0.0644339, -0.0505283, 0.0536429
4: -0.0241064, -0.0030614, -0.0250397, -0.0023599, -0.0217464, 0.0219782
5: 0.0068713, 0.0477711, 0.0049993, 0.0464991, -0.0396278, 0.0427719
6: -0.0327018, 0.0491298, -0.0348876, 0.0455635, -0.0782653, 0.0840174
7: -0.0150089, 0.0112847, -0.0160587, 0.0126032, -0.0276121, 0.0273434
8: 0.7033206, 0.9411473, 0.7135336, 0.9456578, -0.1871364, 0.1723211
9: 0.0512032, 0.0920525, 0.0491103, 0.0908419, -0.0396387, 0.0429423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442871, 0.0534906, -0.0947534, 0.1026359
1: -0.0135457, 0.0135796, -0.0152351, 0.0143338, -0.0278795, 0.0288147
2: -0.0017059, 0.0367558, -0.0034273, 0.0350586, -0.0367645, 0.0383396
3: -0.0023984, 0.0668044, -0.0049559, 0.0635683, -0.0515055, 0.0556709
4: -0.0241064, -0.0030614, -0.0250527, -0.0023502, -0.0217561, 0.0219912
5: 0.0068713, 0.0477711, 0.0049731, 0.0460303, -0.0391590, 0.0427980
6: -0.0327018, 0.0491298, -0.0349180, 0.0442495, -0.0769513, 0.0840478
7: -0.0150089, 0.0112847, -0.0160732, 0.0126215, -0.0276304, 0.0273579
8: 0.7033206, 0.9411473, 0.7172979, 0.9457193, -0.1894178, 0.1712241
9: 0.0512032, 0.0920525, 0.0490810, 0.0904766, -0.0392734, 0.0429715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442470, 0.0548080, -0.0949187, 0.1029401
1: -0.0128829, 0.0132836, -0.0152119, 0.0143234, -0.0272064, 0.0284955
2: -0.0010305, 0.0368735, -0.0034036, 0.0355157, -0.0365461, 0.0380358
3: -0.0013947, 0.0670294, -0.0049208, 0.0644339, -0.0517675, 0.0553089
4: -0.0237351, -0.0032822, -0.0250397, -0.0023599, -0.0213751, 0.0217575
5: 0.0076161, 0.0478919, 0.0049993, 0.0464991, -0.0388830, 0.0428927
6: -0.0318324, 0.0494685, -0.0348876, 0.0455635, -0.0773959, 0.0843561
7: -0.0145913, 0.0107602, -0.0160587, 0.0126032, -0.0271946, 0.0268189
8: 0.7023503, 0.9393543, 0.7135336, 0.9456578, -0.1907670, 0.1726165
9: 0.0520359, 0.0921682, 0.0491103, 0.0908419, -0.0388060, 0.0430580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442871, 0.0534906, -0.0936016, 0.1029809
1: -0.0128829, 0.0132836, -0.0152351, 0.0143338, -0.0272167, 0.0285187
2: -0.0010305, 0.0368735, -0.0034273, 0.0350586, -0.0360891, 0.0377323
3: -0.0013947, 0.0670294, -0.0049559, 0.0635683, -0.0493166, 0.0538580
4: -0.0237351, -0.0032822, -0.0250527, -0.0023502, -0.0213849, 0.0217705
5: 0.0076161, 0.0478919, 0.0049731, 0.0460303, -0.0384142, 0.0429188
6: -0.0318324, 0.0494685, -0.0349180, 0.0442495, -0.0760819, 0.0843865
7: -0.0145913, 0.0107602, -0.0160732, 0.0126215, -0.0272128, 0.0268334
8: 0.7023503, 0.9393543, 0.7172979, 0.9457193, -0.1894736, 0.1680372
9: 0.0520359, 0.0921682, 0.0490810, 0.0904766, -0.0384407, 0.0430872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442470, 0.0548080, -0.0960693, 0.1025934
1: -0.0135457, 0.0135796, -0.0152119, 0.0143234, -0.0278691, 0.0287915
2: -0.0017059, 0.0367558, -0.0034036, 0.0355157, -0.0372216, 0.0381460
3: -0.0023984, 0.0668044, -0.0049208, 0.0644339, -0.0505283, 0.0536429
4: -0.0241064, -0.0030614, -0.0250397, -0.0023599, -0.0217464, 0.0219782
5: 0.0068713, 0.0477711, 0.0049993, 0.0464991, -0.0396278, 0.0427719
6: -0.0327018, 0.0491298, -0.0348876, 0.0455635, -0.0782653, 0.0840174
7: -0.0150089, 0.0112847, -0.0160587, 0.0126032, -0.0276121, 0.0273434
8: 0.7033206, 0.9411473, 0.7135336, 0.9456578, -0.1871364, 0.1723211
9: 0.0512032, 0.0920525, 0.0491103, 0.0908419, -0.0396387, 0.0429423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0413505, 0.0584358, -0.0442871, 0.0534906, -0.0947534, 0.1026359
1: -0.0135457, 0.0135796, -0.0152351, 0.0143338, -0.0278795, 0.0288147
2: -0.0017059, 0.0367558, -0.0034273, 0.0350586, -0.0367645, 0.0383396
3: -0.0023984, 0.0668044, -0.0049559, 0.0635683, -0.0515055, 0.0556709
4: -0.0241064, -0.0030614, -0.0250527, -0.0023502, -0.0217561, 0.0219912
5: 0.0068713, 0.0477711, 0.0049731, 0.0460303, -0.0391590, 0.0427980
6: -0.0327018, 0.0491298, -0.0349180, 0.0442495, -0.0769513, 0.0840478
7: -0.0150089, 0.0112847, -0.0160732, 0.0126215, -0.0276304, 0.0273579
8: 0.7033206, 0.9411473, 0.7172979, 0.9457193, -0.1894178, 0.1712241
9: 0.0512032, 0.0920525, 0.0490810, 0.0904766, -0.0392734, 0.0429715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442470, 0.0548080, -0.0949187, 0.1029401
1: -0.0128829, 0.0132836, -0.0152119, 0.0143234, -0.0272064, 0.0284955
2: -0.0010305, 0.0368735, -0.0034036, 0.0355157, -0.0365461, 0.0380358
3: -0.0013947, 0.0670294, -0.0049208, 0.0644339, -0.0517675, 0.0553089
4: -0.0237351, -0.0032822, -0.0250397, -0.0023599, -0.0213751, 0.0217575
5: 0.0076161, 0.0478919, 0.0049993, 0.0464991, -0.0388830, 0.0428927
6: -0.0318324, 0.0494685, -0.0348876, 0.0455635, -0.0773959, 0.0843561
7: -0.0145913, 0.0107602, -0.0160587, 0.0126032, -0.0271946, 0.0268189
8: 0.7023503, 0.9393543, 0.7135336, 0.9456578, -0.1907670, 0.1726165
9: 0.0520359, 0.0921682, 0.0491103, 0.0908419, -0.0388060, 0.0430580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0401982, 0.0587806, -0.0442871, 0.0534906, -0.0936016, 0.1029809
1: -0.0128829, 0.0132836, -0.0152351, 0.0143338, -0.0272167, 0.0285187
2: -0.0010305, 0.0368735, -0.0034273, 0.0350586, -0.0360891, 0.0377323
3: -0.0013947, 0.0670294, -0.0049559, 0.0635683, -0.0493166, 0.0538580
4: -0.0237351, -0.0032822, -0.0250527, -0.0023502, -0.0213849, 0.0217705
5: 0.0076161, 0.0478919, 0.0049731, 0.0460303, -0.0384142, 0.0429188
6: -0.0318324, 0.0494685, -0.0349180, 0.0442495, -0.0760819, 0.0843865
7: -0.0145913, 0.0107602, -0.0160732, 0.0126215, -0.0272128, 0.0268334
8: 0.7023503, 0.9393543, 0.7172979, 0.9457193, -0.1894736, 0.1680372
9: 0.0520359, 0.0921682, 0.0490810, 0.0904766, -0.0384407, 0.0430872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
time: 0.58 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 2.92 seconds
IS_A1_B2_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1109350
IS_A1_B2_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A1_B2_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A1_B2_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A1_B2_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1109350
IS_A1_B2_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A1_B2_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A1_B2_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A1_B2_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1109350
IS_A1_B2_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A1_B2_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A1_B2_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A1_B2_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1105825
IS_A1_B2_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1028533, upper bound: 0.1109350
IS_A1_B2_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A1_B2_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1105825
IS_A1_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A1_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.0999781, upper bound: 0.1109350
IS_A2_B1_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
IS_A2_B1_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
IS_A2_B1_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
IS_A2_B1_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
IS_A2_B1_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
IS_A2_B1_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
IS_A2_B1_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
IS_A2_B1_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1108777, upper bound: 0.0999781
IS_A2_B1_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
IS_A2_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
IS_A2_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
IS_A2_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
IS_A2_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
IS_A2_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
IS_A2_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
IS_A2_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1109350, upper bound: 0.1028533
IS_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
IS_A2_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
IS_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
IS_A2_B1_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
IS_A2_B1_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
IS_A2_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.1028533
IS_A2_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1109350, upper bound: 0.1028533
IS_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
IS_A2_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1105825, upper bound: 0.0999781
IS_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
IS_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1109350, upper bound: 0.0999781
IS_A2_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1074684, upper bound: 0.0915589
IS_A2_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1074684, upper bound: 0.0902061
IS_A2_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1074684, upper bound: 0.0902061
IS_A2_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1074684, upper bound: 0.0902061
IS_A2_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1074684, upper bound: 0.0915589
IS_A2_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1074684, upper bound: 0.0902061
IS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1121336, upper bound: 0.1068673
IS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.92
Output dim: 8, lower bound: -0.1121336, upper bound: 0.1068673

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 2.60 + 598.38 = 600.98 seconds
