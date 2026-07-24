## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.01061397


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758)
1: (-0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403)
2: (0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284)
3: (-0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0114900, 0.0114900)
4: (-0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521)
5: (0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548)
6: (-0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312)
7: (0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624)
8: (-0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763)
9: (-0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.21 + 2.03 = 3.24 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0117933, upper bound: 0.0117933

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0112680, upper bound: 0.0116051
time: 1.01 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0116051, upper bound: 0.0116051
time: 1.14 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.27 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.27
Output dim: 7, lower bound: -0.0112680, upper bound: 0.0116051
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.27
Output dim: 7, lower bound: -0.0116051, upper bound: 0.0116051

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0071846, 0.0051765, -0.0077501, 0.0051960, -0.0123805, 0.0129266
1: -0.0059552, -0.0010585, -0.0061792, -0.0010508, -0.0049045, 0.0051208
2: 0.0274520, 0.0395582, 0.0274338, 0.0405128, -0.0130608, 0.0121244
3: -0.0073805, 0.0050782, -0.0073994, 0.0056915, -0.0114386, 0.0108346
4: -0.0053052, 0.0055804, -0.0057272, 0.0056023, -0.0109075, 0.0113076
5: 0.0070966, 0.0167065, 0.0066894, 0.0167227, -0.0096261, 0.0100171
6: -0.0118714, 0.0020427, -0.0118981, 0.0026032, -0.0144746, 0.0139408
7: 0.9670542, 0.9843211, 0.9659423, 0.9843468, -0.0172926, 0.0183789
8: -0.0223939, -0.0011105, -0.0224391, -0.0004964, -0.0218975, 0.0213287
9: -0.0040051, 0.0089597, -0.0044215, 0.0089858, -0.0129909, 0.0133812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111158, upper bound: 0.0113506
time: 1.07 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111158, upper bound: 0.0114973
time: 1.09 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0074924, 0.0056362, -0.0077401, 0.0051947, -0.0126871, 0.0133763
1: -0.0060772, -0.0008764, -0.0061753, -0.0010513, -0.0050259, 0.0052989
2: 0.0270238, 0.0400779, 0.0274350, 0.0404960, -0.0134722, 0.0126429
3: -0.0078263, 0.0054121, -0.0073982, 0.0056807, -0.0119047, 0.0111757
4: -0.0055349, 0.0060971, -0.0057197, 0.0056009, -0.0111358, 0.0118169
5: 0.0068749, 0.0170887, 0.0066966, 0.0167216, -0.0098467, 0.0103921
6: -0.0125008, 0.0023478, -0.0118963, 0.0025934, -0.0150942, 0.0142442
7: 0.9664489, 0.9849313, 0.9659618, 0.9843452, -0.0178963, 0.0189696
8: -0.0234600, -0.0007762, -0.0224361, -0.0005072, -0.0229528, 0.0216599
9: -0.0042318, 0.0095767, -0.0044142, 0.0089841, -0.0132159, 0.0139909

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0114973, upper bound: 0.0113511
time: 1.16 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0114973, upper bound: 0.0114973
time: 1.08 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.42 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 3.42
Output dim: 7, lower bound: -0.0111158, upper bound: 0.0113506
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 3.42
Output dim: 7, lower bound: -0.0111158, upper bound: 0.0114973
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 3.42
Output dim: 7, lower bound: -0.0114973, upper bound: 0.0113511
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 3.42
Output dim: 7, lower bound: -0.0114973, upper bound: 0.0114973

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -0.0070223, 0.0045636, -0.0077498, 0.0051229, -0.0121452, 0.0123134
1: -0.0058909, -0.0013013, -0.0061791, -0.0010797, -0.0048112, 0.0048779
2: 0.0280229, 0.0392843, 0.0275019, 0.0405124, -0.0124895, 0.0117824
3: -0.0067861, 0.0049022, -0.0073285, 0.0056912, -0.0108401, 0.0105578
4: -0.0051841, 0.0048915, -0.0057270, 0.0055202, -0.0107042, 0.0106185
5: 0.0072134, 0.0161969, 0.0066896, 0.0166619, -0.0094485, 0.0095073
6: -0.0110322, 0.0018819, -0.0117980, 0.0026030, -0.0136351, 0.0136799
7: 0.9673733, 0.9835072, 0.9659428, 0.9842498, -0.0168765, 0.0175644
8: -0.0209725, -0.0012867, -0.0222696, -0.0004966, -0.0204759, 0.0209829
9: -0.0038856, 0.0081369, -0.0044213, 0.0088877, -0.0127733, 0.0125582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111158, upper bound: 0.0111305
time: 1.09 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111158, upper bound: 0.0113506
time: 1.06 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -0.0071827, 0.0048104, -0.0077499, 0.0051557, -0.0123384, 0.0125603
1: -0.0059545, -0.0012035, -0.0061792, -0.0010667, -0.0048878, 0.0049757
2: 0.0277929, 0.0395551, 0.0274713, 0.0405124, -0.0127195, 0.0120838
3: -0.0070255, 0.0050762, -0.0073604, 0.0056913, -0.0110177, 0.0107925
4: -0.0053038, 0.0051690, -0.0057270, 0.0055571, -0.0108608, 0.0108960
5: 0.0070979, 0.0164022, 0.0066896, 0.0166892, -0.0095913, 0.0097126
6: -0.0113701, 0.0020409, -0.0118429, 0.0026030, -0.0139731, 0.0138838
7: 0.9670579, 0.9838350, 0.9659427, 0.9842934, -0.0172355, 0.0178922
8: -0.0215449, -0.0011125, -0.0223457, -0.0004966, -0.0210484, 0.0212332
9: -0.0040037, 0.0084682, -0.0044213, 0.0089317, -0.0129355, 0.0128896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111158, upper bound: 0.0112349
time: 1.08 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111158, upper bound: 0.0114973
time: 1.09 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -0.0073250, 0.0050266, -0.0077399, 0.0051216, -0.0124465, 0.0127665
1: -0.0060108, -0.0011179, -0.0061752, -0.0010802, -0.0049306, 0.0050573
2: 0.0275916, 0.0397952, 0.0275031, 0.0404956, -0.0129040, 0.0122921
3: -0.0072352, 0.0052304, -0.0073273, 0.0056805, -0.0113044, 0.0108919
4: -0.0054099, 0.0054120, -0.0057196, 0.0055187, -0.0109286, 0.0111316
5: 0.0069955, 0.0165819, 0.0066968, 0.0166609, -0.0096653, 0.0098852
6: -0.0116662, 0.0021819, -0.0117962, 0.0025931, -0.0142593, 0.0139781
7: 0.9667782, 0.9841221, 0.9659623, 0.9842481, -0.0174699, 0.0181598
8: -0.0220464, -0.0009580, -0.0222666, -0.0005074, -0.0215390, 0.0213085
9: -0.0041085, 0.0087585, -0.0044140, 0.0088859, -0.0129944, 0.0131725

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0114973, upper bound: 0.0110083
time: 1.25 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0114973, upper bound: 0.0110216
time: 1.33 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -0.0074905, 0.0052772, -0.0077399, 0.0051544, -0.0126449, 0.0130171
1: -0.0060764, -0.0010186, -0.0061752, -0.0010672, -0.0050092, 0.0051566
2: 0.0273582, 0.0400746, 0.0274725, 0.0404956, -0.0131375, 0.0126021
3: -0.0074782, 0.0054100, -0.0073591, 0.0056805, -0.0114896, 0.0111337
4: -0.0055334, 0.0056936, -0.0057196, 0.0055556, -0.0110890, 0.0114132
5: 0.0068763, 0.0167902, 0.0066967, 0.0166882, -0.0098118, 0.0100935
6: -0.0120093, 0.0023459, -0.0118412, 0.0025932, -0.0146025, 0.0141871
7: 0.9664528, 0.9844546, 0.9659624, 0.9842917, -0.0178389, 0.0184923
8: -0.0226274, -0.0007783, -0.0223427, -0.0005074, -0.0221200, 0.0215644
9: -0.0042304, 0.0090948, -0.0044140, 0.0089300, -0.0131604, 0.0135088

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0114973, upper bound: 0.0111158
time: 1.14 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0114973, upper bound: 0.0111358
time: 1.11 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.41 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 7, lower bound: -0.0111158, upper bound: 0.0111305
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 7, lower bound: -0.0111158, upper bound: 0.0113506
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 7, lower bound: -0.0111158, upper bound: 0.0112349
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 7, lower bound: -0.0111158, upper bound: 0.0114973
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 7, lower bound: -0.0114973, upper bound: 0.0110083
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 7, lower bound: -0.0114973, upper bound: 0.0110216
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 7, lower bound: -0.0114973, upper bound: 0.0111158
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 7, lower bound: -0.0114973, upper bound: 0.0111358

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0070223, 0.0045636, -0.0071843, 0.0051033, -0.0121256, 0.0117479
1: -0.0058909, -0.0013013, -0.0059551, -0.0010875, -0.0048035, 0.0046538
2: 0.0280229, 0.0392843, 0.0275201, 0.0395579, -0.0115350, 0.0117642
3: -0.0067861, 0.0049022, -0.0073095, 0.0050780, -0.0102175, 0.0105391
4: -0.0051841, 0.0048915, -0.0053050, 0.0054982, -0.0106822, 0.0101965
5: 0.0072134, 0.0161969, 0.0070968, 0.0166456, -0.0094322, 0.0091002
6: -0.0110322, 0.0018819, -0.0117712, 0.0020425, -0.0130747, 0.0136531
7: 0.9673733, 0.9835072, 0.9670547, 0.9842239, -0.0168507, 0.0164525
8: -0.0209725, -0.0012867, -0.0222241, -0.0011107, -0.0198618, 0.0209375
9: -0.0038856, 0.0081369, -0.0040049, 0.0088614, -0.0127470, 0.0121418

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110489, upper bound: 0.0111305
time: 1.30 seconds

## Relational analysis of IS_A1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110489, upper bound: 0.0111305
time: 1.11 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0070223, 0.0045636, -0.0074922, 0.0055628, -0.0125851, 0.0120558
1: -0.0058909, -0.0013013, -0.0060771, -0.0009055, -0.0049855, 0.0047758
2: 0.0280229, 0.0392843, 0.0270921, 0.0400774, -0.0120546, 0.0121922
3: -0.0067861, 0.0049022, -0.0077552, 0.0054118, -0.0105902, 0.0110122
4: -0.0051841, 0.0048915, -0.0055347, 0.0060146, -0.0111987, 0.0104262
5: 0.0072134, 0.0161969, 0.0068751, 0.0170277, -0.0098142, 0.0093218
6: -0.0110322, 0.0018819, -0.0124003, 0.0023476, -0.0133798, 0.0142822
7: 0.9673733, 0.9835072, 0.9664494, 0.9848338, -0.0174606, 0.0170578
8: -0.0209725, -0.0012867, -0.0232897, -0.0007764, -0.0201961, 0.0220031
9: -0.0038856, 0.0081369, -0.0042316, 0.0094782, -0.0133639, 0.0123685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110489, upper bound: 0.0113506
time: 1.28 seconds

## Relational analysis of IS_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110489, upper bound: 0.0113506
time: 1.34 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0071827, 0.0048104, -0.0071844, 0.0051362, -0.0123189, 0.0119948
1: -0.0059545, -0.0012035, -0.0059551, -0.0010745, -0.0048800, 0.0047516
2: 0.0277929, 0.0395551, 0.0274895, 0.0395579, -0.0117650, 0.0120656
3: -0.0070255, 0.0050762, -0.0073414, 0.0050780, -0.0103957, 0.0107739
4: -0.0053038, 0.0051690, -0.0053050, 0.0055351, -0.0108389, 0.0104739
5: 0.0070979, 0.0164022, 0.0070967, 0.0166730, -0.0095750, 0.0093054
6: -0.0113701, 0.0020409, -0.0118162, 0.0020425, -0.0134127, 0.0138571
7: 0.9670579, 0.9838350, 0.9670546, 0.9842675, -0.0172095, 0.0167804
8: -0.0215449, -0.0011125, -0.0223004, -0.0011107, -0.0204343, 0.0211879
9: -0.0040037, 0.0084682, -0.0040050, 0.0089055, -0.0129092, 0.0124732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110083, upper bound: 0.0112349
time: 1.35 seconds

## Relational analysis of IS_A1_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110083, upper bound: 0.0112349
time: 1.47 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0071827, 0.0048104, -0.0074922, 0.0055966, -0.0127793, 0.0123026
1: -0.0059545, -0.0012035, -0.0060771, -0.0008921, -0.0050624, 0.0048736
2: 0.0277929, 0.0395551, 0.0270606, 0.0400775, -0.0122845, 0.0124945
3: -0.0070255, 0.0050762, -0.0077880, 0.0054118, -0.0107667, 0.0112485
4: -0.0053038, 0.0051690, -0.0055347, 0.0060526, -0.0113564, 0.0107037
5: 0.0070979, 0.0164022, 0.0068751, 0.0170558, -0.0099578, 0.0095271
6: -0.0113701, 0.0020409, -0.0124466, 0.0023476, -0.0137178, 0.0144875
7: 0.9670579, 0.9838350, 0.9664493, 0.9848787, -0.0178208, 0.0173857
8: -0.0215449, -0.0011125, -0.0233682, -0.0007764, -0.0207685, 0.0222557
9: -0.0040037, 0.0084682, -0.0042316, 0.0095236, -0.0135274, 0.0126999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110083, upper bound: 0.0114973
time: 1.12 seconds

## Relational analysis of IS_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110083, upper bound: 0.0114973
time: 1.23 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0073250, 0.0050266, -0.0071843, 0.0051033, -0.0124283, 0.0122110
1: -0.0060108, -0.0011179, -0.0059551, -0.0010875, -0.0049233, 0.0048373
2: 0.0275916, 0.0397952, 0.0275201, 0.0395579, -0.0119663, 0.0122750
3: -0.0072352, 0.0052304, -0.0073095, 0.0050780, -0.0106890, 0.0109045
4: -0.0054099, 0.0054120, -0.0053050, 0.0054982, -0.0109081, 0.0107170
5: 0.0069955, 0.0165819, 0.0070968, 0.0166456, -0.0096501, 0.0094852
6: -0.0116662, 0.0021819, -0.0117712, 0.0020425, -0.0137087, 0.0139530
7: 0.9667782, 0.9841221, 0.9670547, 0.9842239, -0.0174457, 0.0170674
8: -0.0220464, -0.0009580, -0.0222241, -0.0011107, -0.0209357, 0.0212661
9: -0.0041085, 0.0087585, -0.0040049, 0.0088614, -0.0129698, 0.0127634

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0114255, upper bound: 0.0110083
time: 1.35 seconds

## Relational analysis of IS_A2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0114255, upper bound: 0.0110083
time: 1.19 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0073250, 0.0050266, -0.0074922, 0.0055628, -0.0128877, 0.0125188
1: -0.0060108, -0.0011179, -0.0060771, -0.0009055, -0.0051054, 0.0049592
2: 0.0275916, 0.0397952, 0.0270921, 0.0400774, -0.0124859, 0.0127030
3: -0.0072352, 0.0052304, -0.0077552, 0.0054118, -0.0108912, 0.0112027
4: -0.0054099, 0.0054120, -0.0055347, 0.0060146, -0.0114245, 0.0109467
5: 0.0069955, 0.0165819, 0.0068751, 0.0170277, -0.0100321, 0.0097068
6: -0.0116662, 0.0021819, -0.0124003, 0.0023476, -0.0140138, 0.0145822
7: 0.9667782, 0.9841221, 0.9664494, 0.9848338, -0.0180557, 0.0176727
8: -0.0220464, -0.0009580, -0.0232897, -0.0007764, -0.0212699, 0.0223317
9: -0.0041085, 0.0087585, -0.0042316, 0.0094782, -0.0135867, 0.0129901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0114255, upper bound: 0.0110083
time: 1.27 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0114255, upper bound: 0.0110216
time: 1.35 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0074905, 0.0052772, -0.0071844, 0.0051362, -0.0126266, 0.0124616
1: -0.0060764, -0.0010186, -0.0059551, -0.0010745, -0.0050019, 0.0049365
2: 0.0273582, 0.0400746, 0.0274895, 0.0395579, -0.0121998, 0.0125851
3: -0.0074782, 0.0054100, -0.0073414, 0.0050780, -0.0108747, 0.0111464
4: -0.0055334, 0.0056936, -0.0053050, 0.0055351, -0.0110685, 0.0109986
5: 0.0068763, 0.0167902, 0.0070967, 0.0166730, -0.0097966, 0.0096935
6: -0.0120093, 0.0023459, -0.0118162, 0.0020425, -0.0140518, 0.0141621
7: 0.9664528, 0.9844546, 0.9670546, 0.9842675, -0.0178147, 0.0174000
8: -0.0226274, -0.0007783, -0.0223004, -0.0011107, -0.0215168, 0.0215221
9: -0.0042304, 0.0090948, -0.0040050, 0.0089055, -0.0131359, 0.0130998

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0113506, upper bound: 0.0111158
time: 1.72 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0113506, upper bound: 0.0111158
time: 1.17 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0074905, 0.0052772, -0.0074922, 0.0055966, -0.0130871, 0.0127694
1: -0.0060764, -0.0010186, -0.0060771, -0.0008921, -0.0051843, 0.0050585
2: 0.0273582, 0.0400746, 0.0270606, 0.0400775, -0.0127193, 0.0130140
3: -0.0074782, 0.0054100, -0.0077880, 0.0054118, -0.0110692, 0.0114452
4: -0.0055334, 0.0056936, -0.0055347, 0.0060526, -0.0115861, 0.0112283
5: 0.0068763, 0.0167902, 0.0068751, 0.0170558, -0.0101794, 0.0099151
6: -0.0120093, 0.0023459, -0.0124466, 0.0023476, -0.0143569, 0.0147926
7: 0.9664528, 0.9844546, 0.9664493, 0.9848787, -0.0184259, 0.0180054
8: -0.0226274, -0.0007783, -0.0233682, -0.0007764, -0.0218510, 0.0225899
9: -0.0042304, 0.0090948, -0.0042316, 0.0095236, -0.0137540, 0.0133265

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0113506, upper bound: 0.0111358
time: 1.21 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0113506, upper bound: 0.0111358
time: 1.12 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.51 seconds
IS_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 7, lower bound: -0.0110489, upper bound: 0.0111305
IS_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 7, lower bound: -0.0110489, upper bound: 0.0111305
IS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 7, lower bound: -0.0110489, upper bound: 0.0113506
IS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 7, lower bound: -0.0110489, upper bound: 0.0113506
IS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 7, lower bound: -0.0110083, upper bound: 0.0112349
IS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 7, lower bound: -0.0110083, upper bound: 0.0112349
IS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 7, lower bound: -0.0110083, upper bound: 0.0114973
IS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 7, lower bound: -0.0110083, upper bound: 0.0114973
IS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 7, lower bound: -0.0114255, upper bound: 0.0110083
IS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 7, lower bound: -0.0114255, upper bound: 0.0110083
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 7, lower bound: -0.0114255, upper bound: 0.0110083
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 7, lower bound: -0.0114255, upper bound: 0.0110216
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 7, lower bound: -0.0113506, upper bound: 0.0111158
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 7, lower bound: -0.0113506, upper bound: 0.0111158
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 7, lower bound: -0.0113506, upper bound: 0.0111358
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 7, lower bound: -0.0113506, upper bound: 0.0111358

## BFS IS instance: IS_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0070223, 0.0045636, -0.0070223, 0.0045636, -0.0115859, 0.0115859
1: -0.0058909, -0.0013013, -0.0058909, -0.0013013, -0.0045897, 0.0045897
2: 0.0280229, 0.0392843, 0.0280229, 0.0392843, -0.0112615, 0.0112615
3: -0.0067861, 0.0049022, -0.0067861, 0.0049022, -0.0100139, 0.0100139
4: -0.0051841, 0.0048915, -0.0051841, 0.0048915, -0.0100756, 0.0100756
5: 0.0072134, 0.0161969, 0.0072134, 0.0161969, -0.0089835, 0.0089835
6: -0.0110322, 0.0018819, -0.0110322, 0.0018819, -0.0129141, 0.0129141
7: 0.9673733, 0.9835072, 0.9673733, 0.9835072, -0.0161340, 0.0161340
8: -0.0209725, -0.0012867, -0.0209725, -0.0012867, -0.0196858, 0.0196858
9: -0.0038856, 0.0081369, -0.0038856, 0.0081369, -0.0120225, 0.0120225

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B1_B1_A1

### Relational analysis result of IS_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110108, upper bound: 0.0108874
time: 1.09 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2

### Relational analysis result of IS_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109340, upper bound: 0.0108800
time: 1.11 seconds

## BFS IS instance: IS_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0070223, 0.0045636, -0.0071827, 0.0048104, -0.0118327, 0.0117463
1: -0.0058909, -0.0013013, -0.0059545, -0.0012035, -0.0046874, 0.0046532
2: 0.0280229, 0.0392843, 0.0277929, 0.0395551, -0.0115322, 0.0114914
3: -0.0067861, 0.0049022, -0.0070255, 0.0050762, -0.0102142, 0.0102762
4: -0.0051841, 0.0048915, -0.0053038, 0.0051690, -0.0103530, 0.0101953
5: 0.0072134, 0.0161969, 0.0070979, 0.0164022, -0.0091887, 0.0090990
6: -0.0110322, 0.0018819, -0.0113701, 0.0020409, -0.0130731, 0.0132521
7: 0.9673733, 0.9835072, 0.9670579, 0.9838350, -0.0164617, 0.0164493
8: -0.0209725, -0.0012867, -0.0215449, -0.0011125, -0.0198600, 0.0202583
9: -0.0038856, 0.0081369, -0.0040037, 0.0084682, -0.0123539, 0.0121406

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B1_B2_B1

### Relational analysis result of IS_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109459, upper bound: 0.0109640
time: 1.03 seconds

## Relational analysis of IS_A1_A1_B1_B2_B2

### Relational analysis result of IS_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109340, upper bound: 0.0108800
time: 1.19 seconds

## BFS IS instance: IS_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0070223, 0.0045636, -0.0073250, 0.0050266, -0.0120489, 0.0118886
1: -0.0058909, -0.0013013, -0.0060108, -0.0011179, -0.0047731, 0.0047095
2: 0.0280229, 0.0392843, 0.0275916, 0.0397952, -0.0117723, 0.0116928
3: -0.0067861, 0.0049022, -0.0072352, 0.0052304, -0.0103793, 0.0104855
4: -0.0051841, 0.0048915, -0.0054099, 0.0054120, -0.0105961, 0.0103014
5: 0.0072134, 0.0161969, 0.0069955, 0.0165819, -0.0093685, 0.0092014
6: -0.0110322, 0.0018819, -0.0116662, 0.0021819, -0.0132140, 0.0135481
7: 0.9673733, 0.9835072, 0.9667782, 0.9841221, -0.0167488, 0.0167291
8: -0.0209725, -0.0012867, -0.0220464, -0.0009580, -0.0200145, 0.0207597
9: -0.0038856, 0.0081369, -0.0041085, 0.0087585, -0.0126441, 0.0122453

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108621, upper bound: 0.0111688
time: 1.15 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2

### Relational analysis result of IS_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108465, upper bound: 0.0110344
time: 1.15 seconds

## BFS IS instance: IS_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0070223, 0.0045636, -0.0074905, 0.0052772, -0.0122995, 0.0120541
1: -0.0058909, -0.0013013, -0.0060764, -0.0010186, -0.0048723, 0.0047751
2: 0.0280229, 0.0392843, 0.0273582, 0.0400746, -0.0120517, 0.0119262
3: -0.0067861, 0.0049022, -0.0074782, 0.0054100, -0.0105867, 0.0107565
4: -0.0051841, 0.0048915, -0.0055334, 0.0056936, -0.0108777, 0.0104250
5: 0.0072134, 0.0161969, 0.0068763, 0.0167902, -0.0095768, 0.0093206
6: -0.0110322, 0.0018819, -0.0120093, 0.0023459, -0.0133781, 0.0138912
7: 0.9673733, 0.9835072, 0.9664528, 0.9844546, -0.0170814, 0.0170544
8: -0.0209725, -0.0012867, -0.0226274, -0.0007783, -0.0201942, 0.0213408
9: -0.0038856, 0.0081369, -0.0042304, 0.0090948, -0.0129805, 0.0123672

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108621, upper bound: 0.0111688
time: 1.15 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108465, upper bound: 0.0110344
time: 1.11 seconds

## BFS IS instance: IS_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0071827, 0.0048104, -0.0070223, 0.0045636, -0.0117463, 0.0118327
1: -0.0059545, -0.0012035, -0.0058909, -0.0013013, -0.0046532, 0.0046874
2: 0.0277929, 0.0395551, 0.0280229, 0.0392843, -0.0114914, 0.0115322
3: -0.0070255, 0.0050762, -0.0067861, 0.0049022, -0.0102762, 0.0102142
4: -0.0053038, 0.0051690, -0.0051841, 0.0048915, -0.0101953, 0.0103530
5: 0.0070979, 0.0164022, 0.0072134, 0.0161969, -0.0090990, 0.0091887
6: -0.0113701, 0.0020409, -0.0110322, 0.0018819, -0.0132521, 0.0130731
7: 0.9670579, 0.9838350, 0.9673733, 0.9835072, -0.0164493, 0.0164617
8: -0.0215449, -0.0011125, -0.0209725, -0.0012867, -0.0202583, 0.0198600
9: -0.0040037, 0.0084682, -0.0038856, 0.0081369, -0.0121406, 0.0123539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B1_B1_A1

### Relational analysis result of IS_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109640, upper bound: 0.0109185
time: 1.10 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2

### Relational analysis result of IS_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108800, upper bound: 0.0109123
time: 1.14 seconds

## BFS IS instance: IS_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0071827, 0.0048104, -0.0071827, 0.0048104, -0.0119931, 0.0119931
1: -0.0059545, -0.0012035, -0.0059545, -0.0012035, -0.0047510, 0.0047510
2: 0.0277929, 0.0395551, 0.0277929, 0.0395551, -0.0117622, 0.0117622
3: -0.0070255, 0.0050762, -0.0070255, 0.0050762, -0.0103930, 0.0103930
4: -0.0053038, 0.0051690, -0.0053038, 0.0051690, -0.0104727, 0.0104727
5: 0.0070979, 0.0164022, 0.0070979, 0.0164022, -0.0093042, 0.0093042
6: -0.0113701, 0.0020409, -0.0113701, 0.0020409, -0.0134110, 0.0134110
7: 0.9670579, 0.9838350, 0.9670579, 0.9838350, -0.0167770, 0.0167770
8: -0.0215449, -0.0011125, -0.0215449, -0.0011125, -0.0204325, 0.0204325
9: -0.0040037, 0.0084682, -0.0040037, 0.0084682, -0.0124720, 0.0124720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B1_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108874, upper bound: 0.0110458
time: 1.12 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2

### Relational analysis result of IS_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108800, upper bound: 0.0109123
time: 1.05 seconds

## BFS IS instance: IS_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0071827, 0.0048104, -0.0073250, 0.0050266, -0.0122094, 0.0121354
1: -0.0059545, -0.0012035, -0.0060108, -0.0011179, -0.0048366, 0.0048073
2: 0.0277929, 0.0395551, 0.0275916, 0.0397952, -0.0120022, 0.0119635
3: -0.0070255, 0.0050762, -0.0072352, 0.0052304, -0.0106416, 0.0106858
4: -0.0053038, 0.0051690, -0.0054099, 0.0054120, -0.0107158, 0.0105788
5: 0.0070979, 0.0164022, 0.0069955, 0.0165819, -0.0094840, 0.0094066
6: -0.0113701, 0.0020409, -0.0116662, 0.0021819, -0.0135520, 0.0137071
7: 0.9670579, 0.9838350, 0.9667782, 0.9841221, -0.0170642, 0.0170568
8: -0.0215449, -0.0011125, -0.0220464, -0.0009580, -0.0205869, 0.0209339
9: -0.0040037, 0.0084682, -0.0041085, 0.0087585, -0.0127622, 0.0125767

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B2_B1_B1

### Relational analysis result of IS_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108131, upper bound: 0.0112625
time: 1.10 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2

### Relational analysis result of IS_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108023, upper bound: 0.0110585
time: 1.18 seconds

## BFS IS instance: IS_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0071827, 0.0048104, -0.0074905, 0.0052772, -0.0124599, 0.0123009
1: -0.0059545, -0.0012035, -0.0060764, -0.0010186, -0.0049359, 0.0048729
2: 0.0277929, 0.0395551, 0.0273582, 0.0400746, -0.0122816, 0.0121969
3: -0.0070255, 0.0050762, -0.0074782, 0.0054100, -0.0107639, 0.0108719
4: -0.0053038, 0.0051690, -0.0055334, 0.0056936, -0.0109974, 0.0107024
5: 0.0070979, 0.0164022, 0.0068763, 0.0167902, -0.0096923, 0.0095258
6: -0.0113701, 0.0020409, -0.0120093, 0.0023459, -0.0137161, 0.0140502
7: 0.9670579, 0.9838350, 0.9664528, 0.9844546, -0.0173967, 0.0173822
8: -0.0215449, -0.0011125, -0.0226274, -0.0007783, -0.0207667, 0.0215149
9: -0.0040037, 0.0084682, -0.0042304, 0.0090948, -0.0130986, 0.0126986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108131, upper bound: 0.0112625
time: 1.16 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108023, upper bound: 0.0110585
time: 1.10 seconds

## BFS IS instance: IS_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0073250, 0.0050266, -0.0070223, 0.0045636, -0.0118886, 0.0120489
1: -0.0060108, -0.0011179, -0.0058909, -0.0013013, -0.0047095, 0.0047731
2: 0.0275916, 0.0397952, 0.0280229, 0.0392843, -0.0116928, 0.0117723
3: -0.0072352, 0.0052304, -0.0067861, 0.0049022, -0.0104855, 0.0103793
4: -0.0054099, 0.0054120, -0.0051841, 0.0048915, -0.0103014, 0.0105961
5: 0.0069955, 0.0165819, 0.0072134, 0.0161969, -0.0092014, 0.0093685
6: -0.0116662, 0.0021819, -0.0110322, 0.0018819, -0.0135481, 0.0132140
7: 0.9667782, 0.9841221, 0.9673733, 0.9835072, -0.0167291, 0.0167488
8: -0.0220464, -0.0009580, -0.0209725, -0.0012867, -0.0207597, 0.0200145
9: -0.0041085, 0.0087585, -0.0038856, 0.0081369, -0.0122453, 0.0126441

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_A1_B1_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0112355, upper bound: 0.0108131
time: 1.05 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2

### Relational analysis result of IS_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110979, upper bound: 0.0108023
time: 1.10 seconds

## BFS IS instance: IS_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0073250, 0.0050266, -0.0071827, 0.0048104, -0.0121354, 0.0122094
1: -0.0060108, -0.0011179, -0.0059545, -0.0012035, -0.0048073, 0.0048366
2: 0.0275916, 0.0397952, 0.0277929, 0.0395551, -0.0119635, 0.0120022
3: -0.0072352, 0.0052304, -0.0070255, 0.0050762, -0.0106858, 0.0106416
4: -0.0054099, 0.0054120, -0.0053038, 0.0051690, -0.0105788, 0.0107158
5: 0.0069955, 0.0165819, 0.0070979, 0.0164022, -0.0094066, 0.0094840
6: -0.0116662, 0.0021819, -0.0113701, 0.0020409, -0.0137071, 0.0135520
7: 0.9667782, 0.9841221, 0.9670579, 0.9838350, -0.0170568, 0.0170642
8: -0.0220464, -0.0009580, -0.0215449, -0.0011125, -0.0209339, 0.0205869
9: -0.0041085, 0.0087585, -0.0040037, 0.0084682, -0.0125767, 0.0127622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_A1_B1_B2_A1

### Relational analysis result of IS_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0112355, upper bound: 0.0108131
time: 1.29 seconds

## Relational analysis of IS_A2_A1_B1_B2_A2

### Relational analysis result of IS_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110979, upper bound: 0.0108023
time: 1.07 seconds

## BFS IS instance: IS_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0073250, 0.0050266, -0.0073250, 0.0050266, -0.0123516, 0.0123516
1: -0.0060108, -0.0011179, -0.0060108, -0.0011179, -0.0048930, 0.0048930
2: 0.0275916, 0.0397952, 0.0275916, 0.0397952, -0.0122036, 0.0122036
3: -0.0072352, 0.0052304, -0.0072352, 0.0052304, -0.0106805, 0.0106805
4: -0.0054099, 0.0054120, -0.0054099, 0.0054120, -0.0108219, 0.0108219
5: 0.0069955, 0.0165819, 0.0069955, 0.0165819, -0.0095864, 0.0095864
6: -0.0116662, 0.0021819, -0.0116662, 0.0021819, -0.0138481, 0.0138481
7: 0.9667782, 0.9841221, 0.9667782, 0.9841221, -0.0173439, 0.0173439
8: -0.0220464, -0.0009580, -0.0220464, -0.0009580, -0.0210883, 0.0210883
9: -0.0041085, 0.0087585, -0.0041085, 0.0087585, -0.0128669, 0.0128669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_A1_B2_B1_B1

### Relational analysis result of IS_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111109, upper bound: 0.0108458
time: 1.02 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2

### Relational analysis result of IS_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110979, upper bound: 0.0108166
time: 1.30 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0073250, 0.0050266, -0.0074905, 0.0052772, -0.0126022, 0.0125171
1: -0.0060108, -0.0011179, -0.0060764, -0.0010186, -0.0049922, 0.0049585
2: 0.0275916, 0.0397952, 0.0273582, 0.0400746, -0.0124830, 0.0124370
3: -0.0072352, 0.0052304, -0.0074782, 0.0054100, -0.0108880, 0.0109473
4: -0.0054099, 0.0054120, -0.0055334, 0.0056936, -0.0111035, 0.0109454
5: 0.0069955, 0.0165819, 0.0068763, 0.0167902, -0.0097947, 0.0097056
6: -0.0116662, 0.0021819, -0.0120093, 0.0023459, -0.0140121, 0.0141912
7: 0.9667782, 0.9841221, 0.9664528, 0.9844546, -0.0176765, 0.0176693
8: -0.0220464, -0.0009580, -0.0226274, -0.0007783, -0.0212681, 0.0216694
9: -0.0041085, 0.0087585, -0.0042304, 0.0090948, -0.0132033, 0.0129889

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_A1_B2_B2_B1

### Relational analysis result of IS_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111109, upper bound: 0.0108458
time: 1.02 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110979, upper bound: 0.0108166
time: 1.30 seconds

## BFS IS instance: IS_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0074905, 0.0052772, -0.0070223, 0.0045636, -0.0120541, 0.0122995
1: -0.0060764, -0.0010186, -0.0058909, -0.0013013, -0.0047751, 0.0048723
2: 0.0273582, 0.0400746, 0.0280229, 0.0392843, -0.0119262, 0.0120517
3: -0.0074782, 0.0054100, -0.0067861, 0.0049022, -0.0107565, 0.0105867
4: -0.0055334, 0.0056936, -0.0051841, 0.0048915, -0.0104250, 0.0108777
5: 0.0068763, 0.0167902, 0.0072134, 0.0161969, -0.0093206, 0.0095768
6: -0.0120093, 0.0023459, -0.0110322, 0.0018819, -0.0138912, 0.0133781
7: 0.9664528, 0.9844546, 0.9673733, 0.9835072, -0.0170544, 0.0170814
8: -0.0226274, -0.0007783, -0.0209725, -0.0012867, -0.0213408, 0.0201942
9: -0.0042304, 0.0090948, -0.0038856, 0.0081369, -0.0123672, 0.0129805

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_A2_B1_B1_A1

### Relational analysis result of IS_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111688, upper bound: 0.0108832
time: 1.29 seconds

## Relational analysis of IS_A2_A2_B1_B1_A2

### Relational analysis result of IS_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110344, upper bound: 0.0108641
time: 1.08 seconds

## BFS IS instance: IS_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0074905, 0.0052772, -0.0071827, 0.0048104, -0.0123009, 0.0124599
1: -0.0060764, -0.0010186, -0.0059545, -0.0012035, -0.0048729, 0.0049359
2: 0.0273582, 0.0400746, 0.0277929, 0.0395551, -0.0121969, 0.0122816
3: -0.0074782, 0.0054100, -0.0070255, 0.0050762, -0.0108719, 0.0107639
4: -0.0055334, 0.0056936, -0.0053038, 0.0051690, -0.0107024, 0.0109974
5: 0.0068763, 0.0167902, 0.0070979, 0.0164022, -0.0095258, 0.0096923
6: -0.0120093, 0.0023459, -0.0113701, 0.0020409, -0.0140502, 0.0137161
7: 0.9664528, 0.9844546, 0.9670579, 0.9838350, -0.0173822, 0.0173967
8: -0.0226274, -0.0007783, -0.0215449, -0.0011125, -0.0215149, 0.0207667
9: -0.0042304, 0.0090948, -0.0040037, 0.0084682, -0.0126986, 0.0130986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_A2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111688, upper bound: 0.0108832
time: 1.38 seconds

## Relational analysis of IS_A2_A2_B1_B2_A2

### Relational analysis result of IS_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110344, upper bound: 0.0108641
time: 1.05 seconds

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0074905, 0.0052772, -0.0073250, 0.0050266, -0.0125171, 0.0126022
1: -0.0060764, -0.0010186, -0.0060108, -0.0011179, -0.0049585, 0.0049922
2: 0.0273582, 0.0400746, 0.0275916, 0.0397952, -0.0124370, 0.0124830
3: -0.0074782, 0.0054100, -0.0072352, 0.0052304, -0.0109473, 0.0108880
4: -0.0055334, 0.0056936, -0.0054099, 0.0054120, -0.0109454, 0.0111035
5: 0.0068763, 0.0167902, 0.0069955, 0.0165819, -0.0097056, 0.0097947
6: -0.0120093, 0.0023459, -0.0116662, 0.0021819, -0.0141912, 0.0140121
7: 0.9664528, 0.9844546, 0.9667782, 0.9841221, -0.0176693, 0.0176765
8: -0.0226274, -0.0007783, -0.0220464, -0.0009580, -0.0216694, 0.0212681
9: -0.0042304, 0.0090948, -0.0041085, 0.0087585, -0.0129889, 0.0132033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_A2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111688, upper bound: 0.0109118
time: 1.32 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108937
time: 1.30 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0074905, 0.0052772, -0.0074905, 0.0052772, -0.0127677, 0.0127677
1: -0.0060764, -0.0010186, -0.0060764, -0.0010186, -0.0050578, 0.0050578
2: 0.0273582, 0.0400746, 0.0273582, 0.0400746, -0.0127164, 0.0127164
3: -0.0074782, 0.0054100, -0.0074782, 0.0054100, -0.0110667, 0.0110667
4: -0.0055334, 0.0056936, -0.0055334, 0.0056936, -0.0112270, 0.0112270
5: 0.0068763, 0.0167902, 0.0068763, 0.0167902, -0.0099139, 0.0099139
6: -0.0120093, 0.0023459, -0.0120093, 0.0023459, -0.0143552, 0.0143552
7: 0.9664528, 0.9844546, 0.9664528, 0.9844546, -0.0180019, 0.0180019
8: -0.0226274, -0.0007783, -0.0226274, -0.0007783, -0.0218492, 0.0218492
9: -0.0042304, 0.0090948, -0.0042304, 0.0090948, -0.0133252, 0.0133252

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_A2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111688, upper bound: 0.0109118
time: 1.45 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108937
time: 1.05 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.73 seconds
IS_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 7, lower bound: -0.0110108, upper bound: 0.0108874
IS_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 7, lower bound: -0.0109340, upper bound: 0.0108800
IS_A1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 7, lower bound: -0.0109459, upper bound: 0.0109640
IS_A1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 7, lower bound: -0.0109340, upper bound: 0.0108800
IS_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 7, lower bound: -0.0108621, upper bound: 0.0111688
IS_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 7, lower bound: -0.0108465, upper bound: 0.0110344
IS_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 7, lower bound: -0.0108621, upper bound: 0.0111688
IS_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 7, lower bound: -0.0108465, upper bound: 0.0110344
IS_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 7, lower bound: -0.0109640, upper bound: 0.0109185
IS_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 7, lower bound: -0.0108800, upper bound: 0.0109123
IS_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 7, lower bound: -0.0108874, upper bound: 0.0110458
IS_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 7, lower bound: -0.0108800, upper bound: 0.0109123
IS_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 7, lower bound: -0.0108131, upper bound: 0.0112625
IS_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 7, lower bound: -0.0108023, upper bound: 0.0110585
IS_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 7, lower bound: -0.0108131, upper bound: 0.0112625
IS_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 7, lower bound: -0.0108023, upper bound: 0.0110585
IS_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 7, lower bound: -0.0112355, upper bound: 0.0108131
IS_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 7, lower bound: -0.0110979, upper bound: 0.0108023
IS_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 7, lower bound: -0.0112355, upper bound: 0.0108131
IS_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 7, lower bound: -0.0110979, upper bound: 0.0108023
IS_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 7, lower bound: -0.0111109, upper bound: 0.0108458
IS_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 7, lower bound: -0.0110979, upper bound: 0.0108166
IS_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 7, lower bound: -0.0111109, upper bound: 0.0108458
IS_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 7, lower bound: -0.0110979, upper bound: 0.0108166
IS_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 7, lower bound: -0.0111688, upper bound: 0.0108832
IS_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 7, lower bound: -0.0110344, upper bound: 0.0108641
IS_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 7, lower bound: -0.0111688, upper bound: 0.0108832
IS_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 7, lower bound: -0.0110344, upper bound: 0.0108641
IS_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 7, lower bound: -0.0111688, upper bound: 0.0109118
IS_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108937
IS_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 7, lower bound: -0.0111688, upper bound: 0.0109118
IS_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108937

## BFS IS instance: IS_A1_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0070136, 0.0043946, -0.0070223, 0.0045636, -0.0115772, 0.0114169
1: -0.0058875, -0.0013682, -0.0058909, -0.0013013, -0.0045862, 0.0045227
2: 0.0281803, 0.0392696, 0.0280229, 0.0392843, -0.0111040, 0.0112467
3: -0.0066222, 0.0048927, -0.0067861, 0.0049022, -0.0098479, 0.0100045
4: -0.0051775, 0.0047016, -0.0051841, 0.0048915, -0.0100690, 0.0098856
5: 0.0072197, 0.0160564, 0.0072134, 0.0161969, -0.0089772, 0.0088430
6: -0.0108007, 0.0018732, -0.0110322, 0.0018819, -0.0126827, 0.0129054
7: 0.9673904, 0.9832830, 0.9673733, 0.9835072, -0.0161168, 0.0159097
8: -0.0205806, -0.0012962, -0.0209725, -0.0012867, -0.0192939, 0.0196763
9: -0.0038792, 0.0079100, -0.0038856, 0.0081369, -0.0120160, 0.0117956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of IS_A1_A1_B1_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101751, upper bound: 0.0104632
time: 1.01 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_B2

### Relational analysis result of IS_A1_A1_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101530, upper bound: 0.0098948
time: 1.17 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0073856, 0.0042302, -0.0070196, 0.0044967, -0.0118823, 0.0112498
1: -0.0060349, -0.0014334, -0.0058898, -0.0013278, -0.0047071, 0.0044565
2: 0.0283335, 0.0398976, 0.0280852, 0.0392798, -0.0109463, 0.0118124
3: -0.0064628, 0.0052963, -0.0067212, 0.0048993, -0.0097639, 0.0103707
4: -0.0054552, 0.0045168, -0.0051820, 0.0048163, -0.0102715, 0.0096988
5: 0.0069518, 0.0159198, 0.0072154, 0.0161413, -0.0091895, 0.0087044
6: -0.0105756, 0.0022420, -0.0109405, 0.0018792, -0.0124548, 0.0131825
7: 0.9666588, 0.9830647, 0.9673786, 0.9834185, -0.0167597, 0.0156860
8: -0.0201993, -0.0008921, -0.0208173, -0.0012897, -0.0189096, 0.0199252
9: -0.0041531, 0.0076893, -0.0038836, 0.0080471, -0.0122002, 0.0115729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of IS_A1_A1_B1_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0104387
time: 1.20 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2_B2

### Relational analysis result of IS_A1_A1_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0098948
time: 1.08 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0070223, 0.0045636, -0.0071739, 0.0046427, -0.0116650, 0.0117375
1: -0.0058909, -0.0013013, -0.0059510, -0.0012699, -0.0046210, 0.0046497
2: 0.0280229, 0.0392843, 0.0279491, 0.0395402, -0.0115173, 0.0113352
3: -0.0067861, 0.0049022, -0.0068629, 0.0050667, -0.0102043, 0.0101080
4: -0.0051841, 0.0048915, -0.0052972, 0.0049805, -0.0101646, 0.0101887
5: 0.0072134, 0.0161969, 0.0071043, 0.0162627, -0.0090493, 0.0090927
6: -0.0110322, 0.0018819, -0.0111406, 0.0020322, -0.0130643, 0.0130225
7: 0.9673733, 0.9835072, 0.9670752, 0.9836124, -0.0162392, 0.0164320
8: -0.0209725, -0.0012867, -0.0211561, -0.0011221, -0.0198504, 0.0198694
9: -0.0038856, 0.0081369, -0.0039972, 0.0082431, -0.0121287, 0.0121341

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: B, layer: 3, pos: 106
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 13

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of IS_A1_A1_B1_B2_B1_B1

### Relational analysis result of IS_A1_A1_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0104947
time: 1.13 seconds

## Relational analysis of IS_A1_A1_B1_B2_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0101312
time: 1.06 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0070196, 0.0044967, -0.0075451, 0.0044799, -0.0114995, 0.0120418
1: -0.0058898, -0.0013278, -0.0060980, -0.0013344, -0.0045554, 0.0047702
2: 0.0280852, 0.0392798, 0.0281009, 0.0401667, -0.0120815, 0.0111789
3: -0.0067212, 0.0048993, -0.0067050, 0.0054692, -0.0105667, 0.0100296
4: -0.0051820, 0.0048163, -0.0055742, 0.0047974, -0.0099795, 0.0103905
5: 0.0072154, 0.0161413, 0.0068371, 0.0161274, -0.0089120, 0.0093043
6: -0.0109405, 0.0018792, -0.0109175, 0.0024000, -0.0133406, 0.0127968
7: 0.9673786, 0.9834185, 0.9663454, 0.9833962, -0.0160176, 0.0170732
8: -0.0208173, -0.0012897, -0.0207784, -0.0007190, -0.0200983, 0.0194887
9: -0.0038836, 0.0080471, -0.0042705, 0.0080245, -0.0119081, 0.0123176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 106

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of IS_A1_A1_B1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0103991
time: 1.22 seconds

## Relational analysis of IS_A1_A1_B1_B2_B2_B2

### Relational analysis result of IS_A1_A1_B1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0098920
time: 1.03 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0070223, 0.0045636, -0.0073164, 0.0048494, -0.0118717, 0.0118800
1: -0.0058909, -0.0013013, -0.0060075, -0.0011881, -0.0047029, 0.0047062
2: 0.0280229, 0.0392843, 0.0277567, 0.0397808, -0.0117579, 0.0115276
3: -0.0067861, 0.0049022, -0.0070633, 0.0052212, -0.0103700, 0.0103111
4: -0.0051841, 0.0048915, -0.0054036, 0.0052127, -0.0103968, 0.0102951
5: 0.0072134, 0.0161969, 0.0070017, 0.0164345, -0.0092211, 0.0091953
6: -0.0110322, 0.0018819, -0.0114235, 0.0021734, -0.0132056, 0.0133054
7: 0.9673733, 0.9835072, 0.9667950, 0.9838866, -0.0165133, 0.0167122
8: -0.0209725, -0.0012867, -0.0216352, -0.0009673, -0.0200052, 0.0203485
9: -0.0038856, 0.0081369, -0.0041022, 0.0085205, -0.0124061, 0.0122391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: B, layer: 3, pos: 106
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 13

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of IS_A1_A1_B2_B1_B1_A1

### Relational analysis result of IS_A1_A1_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104226, upper bound: 0.0105067
time: 1.11 seconds

## Relational analysis of IS_A1_A1_B2_B1_B1_A2

### Relational analysis result of IS_A1_A1_B2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0104737
time: 1.29 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0070196, 0.0044967, -0.0077128, 0.0047094, -0.0117289, 0.0122095
1: -0.0058898, -0.0013278, -0.0061645, -0.0012435, -0.0046463, 0.0048367
2: 0.0280852, 0.0392798, 0.0278871, 0.0404499, -0.0123647, 0.0113927
3: -0.0067212, 0.0048993, -0.0069275, 0.0056511, -0.0107649, 0.0102464
4: -0.0051820, 0.0048163, -0.0056994, 0.0050554, -0.0102374, 0.0105157
5: 0.0072154, 0.0161413, 0.0067163, 0.0163181, -0.0091027, 0.0094251
6: -0.0109405, 0.0018792, -0.0112318, 0.0025663, -0.0135068, 0.0131110
7: 0.9673786, 0.9834185, 0.9660155, 0.9837008, -0.0163222, 0.0174030
8: -0.0208173, -0.0012897, -0.0213106, -0.0005368, -0.0202805, 0.0200209
9: -0.0038836, 0.0080471, -0.0043940, 0.0083326, -0.0122162, 0.0124411

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 106

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of IS_A1_A1_B2_B1_B2_A1

### Relational analysis result of IS_A1_A1_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0103902, upper bound: 0.0101262
time: 1.09 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2_A2

### Relational analysis result of IS_A1_A1_B2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0101262
time: 1.14 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0070223, 0.0045636, -0.0074819, 0.0051025, -0.0121248, 0.0120455
1: -0.0058909, -0.0013013, -0.0060730, -0.0010878, -0.0048031, 0.0047717
2: 0.0280229, 0.0392843, 0.0275209, 0.0400601, -0.0120372, 0.0117634
3: -0.0067861, 0.0049022, -0.0073087, 0.0054006, -0.0105769, 0.0105821
4: -0.0051841, 0.0048915, -0.0055270, 0.0054972, -0.0106813, 0.0104185
5: 0.0072134, 0.0161969, 0.0068825, 0.0166450, -0.0094315, 0.0093144
6: -0.0110322, 0.0018819, -0.0117700, 0.0023374, -0.0133696, 0.0136519
7: 0.9673733, 0.9835072, 0.9664696, 0.9842228, -0.0168495, 0.0170376
8: -0.0209725, -0.0012867, -0.0222222, -0.0007876, -0.0201849, 0.0209356
9: -0.0038856, 0.0081369, -0.0042240, 0.0088603, -0.0127459, 0.0123609

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: A, layer: 3, pos: 106
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 13

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of IS_A1_A1_B2_B2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0107108
time: 1.12 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_B2

### Relational analysis result of IS_A1_A1_B2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0104304
time: 1.02 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0070196, 0.0044967, -0.0078688, 0.0049581, -0.0119777, 0.0123655
1: -0.0058898, -0.0013278, -0.0062263, -0.0011450, -0.0047449, 0.0048985
2: 0.0280852, 0.0392798, 0.0276553, 0.0407133, -0.0126280, 0.0116244
3: -0.0067212, 0.0048993, -0.0071688, 0.0058203, -0.0109690, 0.0105169
4: -0.0051820, 0.0048163, -0.0058158, 0.0053350, -0.0105170, 0.0106321
5: 0.0072154, 0.0161413, 0.0066039, 0.0165250, -0.0093096, 0.0095374
6: -0.0109405, 0.0018792, -0.0115724, 0.0027210, -0.0136615, 0.0134517
7: 0.9673786, 0.9834185, 0.9657087, 0.9840311, -0.0166525, 0.0177098
8: -0.0208173, -0.0012897, -0.0218875, -0.0003674, -0.0204499, 0.0205979
9: -0.0038836, 0.0080471, -0.0045089, 0.0086665, -0.0125502, 0.0125560

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 106

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of IS_A1_A1_B2_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0105697
time: 1.33 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0100928
time: 1.06 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0071739, 0.0046427, -0.0070223, 0.0045636, -0.0117375, 0.0116650
1: -0.0059510, -0.0012699, -0.0058909, -0.0013013, -0.0046497, 0.0046210
2: 0.0279491, 0.0395402, 0.0280229, 0.0392843, -0.0113352, 0.0115173
3: -0.0068629, 0.0050667, -0.0067861, 0.0049022, -0.0101080, 0.0102043
4: -0.0052972, 0.0049805, -0.0051841, 0.0048915, -0.0101887, 0.0101646
5: 0.0071043, 0.0162627, 0.0072134, 0.0161969, -0.0090927, 0.0090493
6: -0.0111406, 0.0020322, -0.0110322, 0.0018819, -0.0130225, 0.0130643
7: 0.9670752, 0.9836124, 0.9673733, 0.9835072, -0.0164320, 0.0162392
8: -0.0211561, -0.0011221, -0.0209725, -0.0012867, -0.0198694, 0.0198504
9: -0.0039972, 0.0082431, -0.0038856, 0.0081369, -0.0121341, 0.0121287

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: A, layer: 3, pos: 106
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 13

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of IS_A1_A2_B1_B1_A1_A1

### Relational analysis result of IS_A1_A2_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104947, upper bound: 0.0098948
time: 1.17 seconds

## Relational analysis of IS_A1_A2_B1_B1_A1_A2

### Relational analysis result of IS_A1_A2_B1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101312, upper bound: 0.0098948
time: 1.27 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0075451, 0.0044799, -0.0070196, 0.0044967, -0.0120418, 0.0114995
1: -0.0060980, -0.0013344, -0.0058898, -0.0013278, -0.0047702, 0.0045554
2: 0.0281009, 0.0401667, 0.0280852, 0.0392798, -0.0111789, 0.0120815
3: -0.0067050, 0.0054692, -0.0067212, 0.0048993, -0.0100296, 0.0105667
4: -0.0055742, 0.0047974, -0.0051820, 0.0048163, -0.0103905, 0.0099795
5: 0.0068371, 0.0161274, 0.0072154, 0.0161413, -0.0093043, 0.0089120
6: -0.0109175, 0.0024000, -0.0109405, 0.0018792, -0.0127968, 0.0133406
7: 0.9663454, 0.9833962, 0.9673786, 0.9834185, -0.0170732, 0.0160176
8: -0.0207784, -0.0007190, -0.0208173, -0.0012897, -0.0194887, 0.0200983
9: -0.0042705, 0.0080245, -0.0038836, 0.0080471, -0.0123176, 0.0119081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of IS_A1_A2_B1_B1_A2_A1

### Relational analysis result of IS_A1_A2_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0103991, upper bound: 0.0098948
time: 1.14 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2_A2

### Relational analysis result of IS_A1_A2_B1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098919, upper bound: 0.0098948
time: 1.05 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0071827, 0.0048104, -0.0071739, 0.0046427, -0.0118255, 0.0119843
1: -0.0059545, -0.0012035, -0.0059510, -0.0012699, -0.0046846, 0.0047475
2: 0.0277929, 0.0395551, 0.0279491, 0.0395402, -0.0117473, 0.0116060
3: -0.0070255, 0.0050762, -0.0068629, 0.0050667, -0.0103835, 0.0102272
4: -0.0053038, 0.0051690, -0.0052972, 0.0049805, -0.0102843, 0.0104661
5: 0.0070979, 0.0164022, 0.0071043, 0.0162627, -0.0091648, 0.0092979
6: -0.0113701, 0.0020409, -0.0111406, 0.0020322, -0.0134023, 0.0131814
7: 0.9670579, 0.9838350, 0.9670752, 0.9836124, -0.0165545, 0.0167598
8: -0.0215449, -0.0011125, -0.0211561, -0.0011221, -0.0204229, 0.0200436
9: -0.0040037, 0.0084682, -0.0039972, 0.0082431, -0.0122468, 0.0124655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of IS_A1_A2_B1_B2_B1_A1

### Relational analysis result of IS_A1_A2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104190, upper bound: 0.0101734
time: 1.05 seconds

## Relational analysis of IS_A1_A2_B1_B2_B1_A2

### Relational analysis result of IS_A1_A2_B1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098920, upper bound: 0.0101634
time: 1.35 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0071800, 0.0047423, -0.0075451, 0.0044799, -0.0116599, 0.0122874
1: -0.0059534, -0.0012305, -0.0060980, -0.0013344, -0.0046190, 0.0048675
2: 0.0278564, 0.0395505, 0.0281009, 0.0401667, -0.0123103, 0.0114496
3: -0.0069595, 0.0050732, -0.0067050, 0.0054692, -0.0107444, 0.0101438
4: -0.0053017, 0.0050924, -0.0055742, 0.0047974, -0.0100992, 0.0106666
5: 0.0070999, 0.0163455, 0.0068371, 0.0161274, -0.0090275, 0.0095085
6: -0.0112769, 0.0020382, -0.0109175, 0.0024000, -0.0136769, 0.0129557
7: 0.9670632, 0.9837446, 0.9663454, 0.9833962, -0.0163330, 0.0173992
8: -0.0213870, -0.0011154, -0.0207784, -0.0007190, -0.0206680, 0.0196629
9: -0.0040017, 0.0083768, -0.0042705, 0.0080245, -0.0120262, 0.0126473

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 106

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of IS_A1_A2_B1_B2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104017, upper bound: 0.0098920
time: 1.04 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098920, upper bound: 0.0098920
time: 1.27 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0071827, 0.0048104, -0.0073164, 0.0048494, -0.0120321, 0.0121269
1: -0.0059545, -0.0012035, -0.0060075, -0.0011881, -0.0047664, 0.0048040
2: 0.0277929, 0.0395551, 0.0277567, 0.0397808, -0.0119879, 0.0117984
3: -0.0070255, 0.0050762, -0.0070633, 0.0052212, -0.0106323, 0.0105114
4: -0.0053038, 0.0051690, -0.0054036, 0.0052127, -0.0105165, 0.0105725
5: 0.0070979, 0.0164022, 0.0070017, 0.0164345, -0.0093366, 0.0094005
6: -0.0113701, 0.0020409, -0.0114235, 0.0021734, -0.0135436, 0.0134644
7: 0.9670579, 0.9838350, 0.9667950, 0.9838866, -0.0168287, 0.0170400
8: -0.0215449, -0.0011125, -0.0216352, -0.0009673, -0.0205777, 0.0205227
9: -0.0040037, 0.0084682, -0.0041022, 0.0085205, -0.0125242, 0.0125704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of IS_A1_A2_B2_B1_B1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0103738, upper bound: 0.0105120
time: 1.18 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098920, upper bound: 0.0104913
time: 1.08 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0071800, 0.0047423, -0.0077128, 0.0047094, -0.0118893, 0.0124552
1: -0.0059534, -0.0012305, -0.0061645, -0.0012435, -0.0047099, 0.0049340
2: 0.0278564, 0.0395505, 0.0278871, 0.0404499, -0.0125935, 0.0116634
3: -0.0069595, 0.0050732, -0.0069275, 0.0056511, -0.0110278, 0.0104465
4: -0.0053017, 0.0050924, -0.0056994, 0.0050554, -0.0103571, 0.0107918
5: 0.0070999, 0.0163455, 0.0067163, 0.0163181, -0.0092182, 0.0096293
6: -0.0112769, 0.0020382, -0.0112318, 0.0025663, -0.0138432, 0.0132700
7: 0.9670632, 0.9837446, 0.9660155, 0.9837008, -0.0166376, 0.0177290
8: -0.0213870, -0.0011154, -0.0213106, -0.0005368, -0.0208501, 0.0201951
9: -0.0040017, 0.0083768, -0.0043940, 0.0083326, -0.0123343, 0.0127708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 106

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of IS_A1_A2_B2_B1_B2_A1

### Relational analysis result of IS_A1_A2_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0103486, upper bound: 0.0101262
time: 1.42 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_A2

### Relational analysis result of IS_A1_A2_B2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098878, upper bound: 0.0101262
time: 1.12 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0071827, 0.0048104, -0.0074819, 0.0051025, -0.0122852, 0.0122923
1: -0.0059545, -0.0012035, -0.0060730, -0.0010878, -0.0048667, 0.0048695
2: 0.0277929, 0.0395551, 0.0275209, 0.0400601, -0.0122671, 0.0120342
3: -0.0070255, 0.0050762, -0.0073087, 0.0054006, -0.0107546, 0.0106997
4: -0.0053038, 0.0051690, -0.0055270, 0.0054972, -0.0108010, 0.0106960
5: 0.0070979, 0.0164022, 0.0068825, 0.0166450, -0.0095470, 0.0095196
6: -0.0113701, 0.0020409, -0.0117700, 0.0023374, -0.0137076, 0.0138109
7: 0.9670579, 0.9838350, 0.9664696, 0.9842228, -0.0171648, 0.0173654
8: -0.0215449, -0.0011125, -0.0222222, -0.0007876, -0.0207573, 0.0211098
9: -0.0040037, 0.0084682, -0.0042240, 0.0088603, -0.0128640, 0.0126922

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: B, layer: 3, pos: 106
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of IS_A1_A2_B2_B2_B1_A1

### Relational analysis result of IS_A1_A2_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0103762, upper bound: 0.0104866
time: 1.15 seconds

## Relational analysis of IS_A1_A2_B2_B2_B1_A2

### Relational analysis result of IS_A1_A2_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098920, upper bound: 0.0104722
time: 1.11 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0071800, 0.0047423, -0.0078688, 0.0049581, -0.0121381, 0.0126112
1: -0.0059534, -0.0012305, -0.0062263, -0.0011450, -0.0048084, 0.0049958
2: 0.0278564, 0.0395505, 0.0276553, 0.0407133, -0.0128568, 0.0118952
3: -0.0069595, 0.0050732, -0.0071688, 0.0058203, -0.0111475, 0.0106291
4: -0.0053017, 0.0050924, -0.0058158, 0.0053350, -0.0106367, 0.0109082
5: 0.0070999, 0.0163455, 0.0066039, 0.0165250, -0.0094251, 0.0097416
6: -0.0112769, 0.0020382, -0.0115724, 0.0027210, -0.0139978, 0.0136106
7: 0.9670632, 0.9837446, 0.9657087, 0.9840311, -0.0169679, 0.0180358
8: -0.0213870, -0.0011154, -0.0218875, -0.0003674, -0.0210196, 0.0207721
9: -0.0040017, 0.0083768, -0.0045089, 0.0086665, -0.0126683, 0.0128857

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 106

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of IS_A1_A2_B2_B2_B2_A1

### Relational analysis result of IS_A1_A2_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0103513, upper bound: 0.0100928
time: 1.43 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_A2

### Relational analysis result of IS_A1_A2_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098919, upper bound: 0.0100928
time: 1.09 seconds

## BFS IS instance: IS_A2_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0073164, 0.0048494, -0.0070223, 0.0045636, -0.0118800, 0.0118717
1: -0.0060075, -0.0011881, -0.0058909, -0.0013013, -0.0047062, 0.0047029
2: 0.0277567, 0.0397808, 0.0280229, 0.0392843, -0.0115276, 0.0117579
3: -0.0070633, 0.0052212, -0.0067861, 0.0049022, -0.0103111, 0.0103700
4: -0.0054036, 0.0052127, -0.0051841, 0.0048915, -0.0102951, 0.0103968
5: 0.0070017, 0.0164345, 0.0072134, 0.0161969, -0.0091953, 0.0092211
6: -0.0114235, 0.0021734, -0.0110322, 0.0018819, -0.0133054, 0.0132056
7: 0.9667950, 0.9838866, 0.9673733, 0.9835072, -0.0167122, 0.0165133
8: -0.0216352, -0.0009673, -0.0209725, -0.0012867, -0.0203485, 0.0200052
9: -0.0041022, 0.0085205, -0.0038856, 0.0081369, -0.0122391, 0.0124061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 13

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of IS_A2_A1_B1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105067, upper bound: 0.0104226
time: 1.23 seconds

## Relational analysis of IS_A2_A1_B1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104737, upper bound: 0.0098948
time: 1.04 seconds

## BFS IS instance: IS_A2_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0077128, 0.0047094, -0.0070196, 0.0044967, -0.0122095, 0.0117289
1: -0.0061645, -0.0012435, -0.0058898, -0.0013278, -0.0048367, 0.0046463
2: 0.0278871, 0.0404499, 0.0280852, 0.0392798, -0.0113927, 0.0123647
3: -0.0069275, 0.0056511, -0.0067212, 0.0048993, -0.0102464, 0.0107649
4: -0.0056994, 0.0050554, -0.0051820, 0.0048163, -0.0105157, 0.0102374
5: 0.0067163, 0.0163181, 0.0072154, 0.0161413, -0.0094251, 0.0091027
6: -0.0112318, 0.0025663, -0.0109405, 0.0018792, -0.0131110, 0.0135068
7: 0.9660155, 0.9837008, 0.9673786, 0.9834185, -0.0174030, 0.0163222
8: -0.0213106, -0.0005368, -0.0208173, -0.0012897, -0.0200209, 0.0202805
9: -0.0043940, 0.0083326, -0.0038836, 0.0080471, -0.0124411, 0.0122162

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 106
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of IS_A2_A1_B1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0103902
time: 1.08 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0098948
time: 1.04 seconds

## BFS IS instance: IS_A2_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0073164, 0.0048494, -0.0071827, 0.0048104, -0.0121269, 0.0120321
1: -0.0060075, -0.0011881, -0.0059545, -0.0012035, -0.0048040, 0.0047664
2: 0.0277567, 0.0397808, 0.0277929, 0.0395551, -0.0117984, 0.0119879
3: -0.0070633, 0.0052212, -0.0070255, 0.0050762, -0.0105114, 0.0106323
4: -0.0054036, 0.0052127, -0.0053038, 0.0051690, -0.0105725, 0.0105165
5: 0.0070017, 0.0164345, 0.0070979, 0.0164022, -0.0094005, 0.0093366
6: -0.0114235, 0.0021734, -0.0113701, 0.0020409, -0.0134644, 0.0135436
7: 0.9667950, 0.9838866, 0.9670579, 0.9838350, -0.0170400, 0.0168287
8: -0.0216352, -0.0009673, -0.0215449, -0.0011125, -0.0205227, 0.0205777
9: -0.0041022, 0.0085205, -0.0040037, 0.0084682, -0.0125704, 0.0125242

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 106
type: B, layer: 3, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of IS_A2_A1_B1_B2_A1_B1

### Relational analysis result of IS_A2_A1_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105120, upper bound: 0.0103738
time: 1.23 seconds

## Relational analysis of IS_A2_A1_B1_B2_A1_B2

### Relational analysis result of IS_A2_A1_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104913, upper bound: 0.0098920
time: 1.01 seconds

## BFS IS instance: IS_A2_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0077128, 0.0047094, -0.0071800, 0.0047423, -0.0124552, 0.0118893
1: -0.0061645, -0.0012435, -0.0059534, -0.0012305, -0.0049340, 0.0047099
2: 0.0278871, 0.0404499, 0.0278564, 0.0395505, -0.0116634, 0.0125935
3: -0.0069275, 0.0056511, -0.0069595, 0.0050732, -0.0104465, 0.0110278
4: -0.0056994, 0.0050554, -0.0053017, 0.0050924, -0.0107918, 0.0103571
5: 0.0067163, 0.0163181, 0.0070999, 0.0163455, -0.0096293, 0.0092182
6: -0.0112318, 0.0025663, -0.0112769, 0.0020382, -0.0132700, 0.0138432
7: 0.9660155, 0.9837008, 0.9670632, 0.9837446, -0.0177290, 0.0166376
8: -0.0213106, -0.0005368, -0.0213870, -0.0011154, -0.0201951, 0.0208501
9: -0.0043940, 0.0083326, -0.0040017, 0.0083768, -0.0127708, 0.0123343

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 106
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of IS_A2_A1_B1_B2_A2_B1

### Relational analysis result of IS_A2_A1_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0103486
time: 1.05 seconds

## Relational analysis of IS_A2_A1_B1_B2_A2_B2

### Relational analysis result of IS_A2_A1_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0098878
time: 1.06 seconds

## BFS IS instance: IS_A2_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0073250, 0.0050266, -0.0073164, 0.0048494, -0.0121743, 0.0123431
1: -0.0060108, -0.0011179, -0.0060075, -0.0011881, -0.0048227, 0.0048896
2: 0.0275916, 0.0397952, 0.0277567, 0.0397808, -0.0121892, 0.0120385
3: -0.0072352, 0.0052304, -0.0070633, 0.0052212, -0.0106713, 0.0105092
4: -0.0054099, 0.0054120, -0.0054036, 0.0052127, -0.0106226, 0.0108156
5: 0.0069955, 0.0165819, 0.0070017, 0.0164345, -0.0094390, 0.0095803
6: -0.0116662, 0.0021819, -0.0114235, 0.0021734, -0.0138397, 0.0136053
7: 0.9667782, 0.9841221, 0.9667950, 0.9838866, -0.0171084, 0.0173271
8: -0.0220464, -0.0009580, -0.0216352, -0.0009673, -0.0210791, 0.0206772
9: -0.0041085, 0.0087585, -0.0041022, 0.0085205, -0.0126289, 0.0128607

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of IS_A2_A1_B2_B1_B1_A1

### Relational analysis result of IS_A2_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106452, upper bound: 0.0102343
time: 1.24 seconds

## Relational analysis of IS_A2_A1_B2_B1_B1_A2

### Relational analysis result of IS_A2_A1_B2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0102230
time: 0.97 seconds

## BFS IS instance: IS_A2_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0073222, 0.0049614, -0.0077128, 0.0047094, -0.0120315, 0.0126742
1: -0.0060097, -0.0011437, -0.0061645, -0.0012435, -0.0047662, 0.0050208
2: 0.0276524, 0.0397905, 0.0278871, 0.0404499, -0.0127975, 0.0119034
3: -0.0071719, 0.0052275, -0.0069275, 0.0056511, -0.0110636, 0.0104450
4: -0.0054078, 0.0053386, -0.0056994, 0.0050554, -0.0104632, 0.0110380
5: 0.0069975, 0.0165276, 0.0067163, 0.0163181, -0.0093206, 0.0098114
6: -0.0115768, 0.0021791, -0.0112318, 0.0025663, -0.0141431, 0.0134109
7: 0.9667836, 0.9840354, 0.9660155, 0.9837008, -0.0169172, 0.0180199
8: -0.0218950, -0.0009610, -0.0213106, -0.0005368, -0.0213581, 0.0203495
9: -0.0041064, 0.0086709, -0.0043940, 0.0083326, -0.0124390, 0.0130649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 106

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of IS_A2_A1_B2_B1_B2_A1

### Relational analysis result of IS_A2_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106214, upper bound: 0.0100658
time: 1.25 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2_A2

### Relational analysis result of IS_A2_A1_B2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0100500
time: 1.03 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0073250, 0.0050266, -0.0074819, 0.0051025, -0.0124274, 0.0125085
1: -0.0060108, -0.0011179, -0.0060730, -0.0010878, -0.0049230, 0.0049551
2: 0.0275916, 0.0397952, 0.0275209, 0.0400601, -0.0124685, 0.0122742
3: -0.0072352, 0.0052304, -0.0073087, 0.0054006, -0.0108785, 0.0107762
4: -0.0054099, 0.0054120, -0.0055270, 0.0054972, -0.0109071, 0.0109390
5: 0.0069955, 0.0165819, 0.0068825, 0.0166450, -0.0096494, 0.0096994
6: -0.0116662, 0.0021819, -0.0117700, 0.0023374, -0.0140036, 0.0139519
7: 0.9667782, 0.9841221, 0.9664696, 0.9842228, -0.0174446, 0.0176525
8: -0.0220464, -0.0009580, -0.0222222, -0.0007876, -0.0212588, 0.0212642
9: -0.0041085, 0.0087585, -0.0042240, 0.0088603, -0.0129688, 0.0129825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: B, layer: 3, pos: 106
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 13

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of IS_A2_A1_B2_B2_B1_B1

### Relational analysis result of IS_A2_A1_B2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0104584
time: 1.23 seconds

## Relational analysis of IS_A2_A1_B2_B2_B1_B2

### Relational analysis result of IS_A2_A1_B2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0101946
time: 0.98 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0073222, 0.0049614, -0.0078688, 0.0049581, -0.0122803, 0.0128302
1: -0.0060097, -0.0011437, -0.0062263, -0.0011450, -0.0048648, 0.0050826
2: 0.0276524, 0.0397905, 0.0276553, 0.0407133, -0.0130609, 0.0121352
3: -0.0071719, 0.0052275, -0.0071688, 0.0058203, -0.0112662, 0.0107129
4: -0.0054078, 0.0053386, -0.0058158, 0.0053350, -0.0107429, 0.0111544
5: 0.0069975, 0.0165276, 0.0066039, 0.0165250, -0.0095275, 0.0099237
6: -0.0115768, 0.0021791, -0.0115724, 0.0027210, -0.0142978, 0.0137516
7: 0.9667836, 0.9840354, 0.9657087, 0.9840311, -0.0172475, 0.0183267
8: -0.0218950, -0.0009610, -0.0218875, -0.0003674, -0.0215276, 0.0209265
9: -0.0041064, 0.0086709, -0.0045089, 0.0086665, -0.0127730, 0.0131798

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 106

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of IS_A2_A1_B2_B2_B2_B1

### Relational analysis result of IS_A2_A1_B2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0103851
time: 1.38 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0100220
time: 1.06 seconds

## BFS IS instance: IS_A2_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0074819, 0.0051025, -0.0070223, 0.0045636, -0.0120455, 0.0121248
1: -0.0060730, -0.0010878, -0.0058909, -0.0013013, -0.0047717, 0.0048031
2: 0.0275209, 0.0400601, 0.0280229, 0.0392843, -0.0117634, 0.0120372
3: -0.0073087, 0.0054006, -0.0067861, 0.0049022, -0.0105821, 0.0105769
4: -0.0055270, 0.0054972, -0.0051841, 0.0048915, -0.0104185, 0.0106813
5: 0.0068825, 0.0166450, 0.0072134, 0.0161969, -0.0093144, 0.0094315
6: -0.0117700, 0.0023374, -0.0110322, 0.0018819, -0.0136519, 0.0133696
7: 0.9664696, 0.9842228, 0.9673733, 0.9835072, -0.0170376, 0.0168495
8: -0.0222222, -0.0007876, -0.0209725, -0.0012867, -0.0209356, 0.0201849
9: -0.0042240, 0.0088603, -0.0038856, 0.0081369, -0.0123609, 0.0127459

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: B, layer: 3, pos: 106
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 13

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of IS_A2_A2_B1_B1_A1_A1

### Relational analysis result of IS_A2_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107108, upper bound: 0.0098948
time: 1.04 seconds

## Relational analysis of IS_A2_A2_B1_B1_A1_A2

### Relational analysis result of IS_A2_A2_B1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104304, upper bound: 0.0098948
time: 1.07 seconds

## BFS IS instance: IS_A2_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0078688, 0.0049581, -0.0070196, 0.0044967, -0.0123655, 0.0119777
1: -0.0062263, -0.0011450, -0.0058898, -0.0013278, -0.0048985, 0.0047449
2: 0.0276553, 0.0407133, 0.0280852, 0.0392798, -0.0116244, 0.0126280
3: -0.0071688, 0.0058203, -0.0067212, 0.0048993, -0.0105169, 0.0109690
4: -0.0058158, 0.0053350, -0.0051820, 0.0048163, -0.0106321, 0.0105170
5: 0.0066039, 0.0165250, 0.0072154, 0.0161413, -0.0095374, 0.0093096
6: -0.0115724, 0.0027210, -0.0109405, 0.0018792, -0.0134517, 0.0136615
7: 0.9657087, 0.9840311, 0.9673786, 0.9834185, -0.0177098, 0.0166525
8: -0.0218875, -0.0003674, -0.0208173, -0.0012897, -0.0205979, 0.0204499
9: -0.0045089, 0.0086665, -0.0038836, 0.0080471, -0.0125560, 0.0125502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 106
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of IS_A2_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_A2_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105697, upper bound: 0.0098948
time: 1.10 seconds

## Relational analysis of IS_A2_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_A2_B1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0098948
time: 0.97 seconds

## BFS IS instance: IS_A2_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0074819, 0.0051025, -0.0071827, 0.0048104, -0.0122923, 0.0122852
1: -0.0060730, -0.0010878, -0.0059545, -0.0012035, -0.0048695, 0.0048667
2: 0.0275209, 0.0400601, 0.0277929, 0.0395551, -0.0120342, 0.0122671
3: -0.0073087, 0.0054006, -0.0070255, 0.0050762, -0.0106997, 0.0107546
4: -0.0055270, 0.0054972, -0.0053038, 0.0051690, -0.0106960, 0.0108010
5: 0.0068825, 0.0166450, 0.0070979, 0.0164022, -0.0095196, 0.0095470
6: -0.0117700, 0.0023374, -0.0113701, 0.0020409, -0.0138109, 0.0137076
7: 0.9664696, 0.9842228, 0.9670579, 0.9838350, -0.0173654, 0.0171648
8: -0.0222222, -0.0007876, -0.0215449, -0.0011125, -0.0211098, 0.0207573
9: -0.0042240, 0.0088603, -0.0040037, 0.0084682, -0.0126922, 0.0128640

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 13

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of IS_A2_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_A2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104618, upper bound: 0.0105344
time: 1.02 seconds

## Relational analysis of IS_A2_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_A2_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104329, upper bound: 0.0098920
time: 1.22 seconds

## BFS IS instance: IS_A2_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0078688, 0.0049581, -0.0071800, 0.0047423, -0.0126112, 0.0121381
1: -0.0062263, -0.0011450, -0.0059534, -0.0012305, -0.0049958, 0.0048084
2: 0.0276553, 0.0407133, 0.0278564, 0.0395505, -0.0118952, 0.0128568
3: -0.0071688, 0.0058203, -0.0069595, 0.0050732, -0.0106291, 0.0111475
4: -0.0058158, 0.0053350, -0.0053017, 0.0050924, -0.0109082, 0.0106367
5: 0.0066039, 0.0165250, 0.0070999, 0.0163455, -0.0097416, 0.0094251
6: -0.0115724, 0.0027210, -0.0112769, 0.0020382, -0.0136106, 0.0139978
7: 0.9657087, 0.9840311, 0.9670632, 0.9837446, -0.0180358, 0.0169679
8: -0.0218875, -0.0003674, -0.0213870, -0.0011154, -0.0207721, 0.0210196
9: -0.0045089, 0.0086665, -0.0040017, 0.0083768, -0.0128857, 0.0126683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of IS_A2_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_A2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0105045
time: 1.21 seconds

## Relational analysis of IS_A2_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_A2_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0098920
time: 1.06 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0074819, 0.0051025, -0.0073250, 0.0050266, -0.0125085, 0.0124274
1: -0.0060730, -0.0010878, -0.0060108, -0.0011179, -0.0049551, 0.0049230
2: 0.0275209, 0.0400601, 0.0275916, 0.0397952, -0.0122742, 0.0124685
3: -0.0073087, 0.0054006, -0.0072352, 0.0052304, -0.0107762, 0.0108785
4: -0.0055270, 0.0054972, -0.0054099, 0.0054120, -0.0109390, 0.0109071
5: 0.0068825, 0.0166450, 0.0069955, 0.0165819, -0.0096994, 0.0096494
6: -0.0117700, 0.0023374, -0.0116662, 0.0021819, -0.0139519, 0.0140036
7: 0.9664696, 0.9842228, 0.9667782, 0.9841221, -0.0176525, 0.0174446
8: -0.0222222, -0.0007876, -0.0220464, -0.0009580, -0.0212642, 0.0212588
9: -0.0042240, 0.0088603, -0.0041085, 0.0087585, -0.0129825, 0.0129688

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: A, layer: 3, pos: 106
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 13

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of IS_A2_A2_B2_B1_A1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107108, upper bound: 0.0100987
time: 1.35 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104304, upper bound: 0.0100926
time: 1.06 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0078688, 0.0049581, -0.0073222, 0.0049614, -0.0128302, 0.0122803
1: -0.0062263, -0.0011450, -0.0060097, -0.0011437, -0.0050826, 0.0048648
2: 0.0276553, 0.0407133, 0.0276524, 0.0397905, -0.0121352, 0.0130609
3: -0.0071688, 0.0058203, -0.0071719, 0.0052275, -0.0107129, 0.0112662
4: -0.0058158, 0.0053350, -0.0054078, 0.0053386, -0.0111544, 0.0107429
5: 0.0066039, 0.0165250, 0.0069975, 0.0165276, -0.0099237, 0.0095275
6: -0.0115724, 0.0027210, -0.0115768, 0.0021791, -0.0137516, 0.0142978
7: 0.9657087, 0.9840311, 0.9667836, 0.9840354, -0.0183267, 0.0172475
8: -0.0218875, -0.0003674, -0.0218950, -0.0009610, -0.0209265, 0.0215276
9: -0.0045089, 0.0086665, -0.0041064, 0.0086709, -0.0131798, 0.0127730

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of IS_A2_A2_B2_B1_A2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105697, upper bound: 0.0100734
time: 1.18 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0100654
time: 1.05 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0074819, 0.0051025, -0.0074905, 0.0052772, -0.0127591, 0.0125929
1: -0.0060730, -0.0010878, -0.0060764, -0.0010186, -0.0050544, 0.0049886
2: 0.0275209, 0.0400601, 0.0273582, 0.0400746, -0.0125537, 0.0127019
3: -0.0073087, 0.0054006, -0.0074782, 0.0054100, -0.0108962, 0.0110575
4: -0.0055270, 0.0054972, -0.0055334, 0.0056936, -0.0112206, 0.0110306
5: 0.0068825, 0.0166450, 0.0068763, 0.0167902, -0.0099077, 0.0097686
6: -0.0117700, 0.0023374, -0.0120093, 0.0023459, -0.0141159, 0.0143467
7: 0.9664696, 0.9842228, 0.9664528, 0.9844546, -0.0179850, 0.0177700
8: -0.0222222, -0.0007876, -0.0226274, -0.0007783, -0.0214440, 0.0218398
9: -0.0042240, 0.0088603, -0.0042304, 0.0090948, -0.0133188, 0.0130907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of IS_A2_A2_B2_B2_A1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104618, upper bound: 0.0105673
time: 1.06 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104329, upper bound: 0.0100748
time: 1.24 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0078688, 0.0049581, -0.0074877, 0.0052109, -0.0130797, 0.0124459
1: -0.0062263, -0.0011450, -0.0060753, -0.0010449, -0.0051814, 0.0049303
2: 0.0276553, 0.0407133, 0.0274199, 0.0400699, -0.0124146, 0.0132933
3: -0.0071688, 0.0058203, -0.0074139, 0.0054070, -0.0108305, 0.0114451
4: -0.0058158, 0.0053350, -0.0055314, 0.0056191, -0.0114349, 0.0108664
5: 0.0066039, 0.0165250, 0.0068783, 0.0167351, -0.0101312, 0.0096466
6: -0.0115724, 0.0027210, -0.0119185, 0.0023432, -0.0139156, 0.0146395
7: 0.9657087, 0.9840311, 0.9664581, 0.9843667, -0.0186580, 0.0175730
8: -0.0218875, -0.0003674, -0.0224736, -0.0007813, -0.0211063, 0.0221063
9: -0.0045089, 0.0086665, -0.0042283, 0.0090058, -0.0135148, 0.0128949

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of IS_A2_A2_B2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0105346
time: 1.07 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0100521
time: 0.96 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.38 seconds
IS_A1_A1_B1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0101751, upper bound: 0.0104632
IS_A1_A1_B1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0101530, upper bound: 0.0098948
IS_A1_A1_B1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0104387
IS_A1_A1_B1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0098948
IS_A1_A1_B1_B2_B1_B1, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0104947
IS_A1_A1_B1_B2_B1_B2, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0101312
IS_A1_A1_B1_B2_B2_B1, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0103991
IS_A1_A1_B1_B2_B2_B2, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0098920
IS_A1_A1_B2_B1_B1_A1, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0104226, upper bound: 0.0105067
IS_A1_A1_B2_B1_B1_A2, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0104737
IS_A1_A1_B2_B1_B2_A1, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0103902, upper bound: 0.0101262
IS_A1_A1_B2_B1_B2_A2, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0101262
IS_A1_A1_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0107108
IS_A1_A1_B2_B2_B1_B2, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0104304
IS_A1_A1_B2_B2_B2_B1, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0105697
IS_A1_A1_B2_B2_B2_B2, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0100928
IS_A1_A2_B1_B1_A1_A1, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0104947, upper bound: 0.0098948
IS_A1_A2_B1_B1_A1_A2, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0101312, upper bound: 0.0098948
IS_A1_A2_B1_B1_A2_A1, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0103991, upper bound: 0.0098948
IS_A1_A2_B1_B1_A2_A2, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0098919, upper bound: 0.0098948
IS_A1_A2_B1_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0104190, upper bound: 0.0101734
IS_A1_A2_B1_B2_B1_A2, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0098920, upper bound: 0.0101634
IS_A1_A2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0104017, upper bound: 0.0098920
IS_A1_A2_B1_B2_B2_A2, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0098920, upper bound: 0.0098920
IS_A1_A2_B2_B1_B1_A1, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0103738, upper bound: 0.0105120
IS_A1_A2_B2_B1_B1_A2, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0098920, upper bound: 0.0104913
IS_A1_A2_B2_B1_B2_A1, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0103486, upper bound: 0.0101262
IS_A1_A2_B2_B1_B2_A2, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0098878, upper bound: 0.0101262
IS_A1_A2_B2_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0103762, upper bound: 0.0104866
IS_A1_A2_B2_B2_B1_A2, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0098920, upper bound: 0.0104722
IS_A1_A2_B2_B2_B2_A1, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0103513, upper bound: 0.0100928
IS_A1_A2_B2_B2_B2_A2, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0098919, upper bound: 0.0100928
IS_A2_A1_B1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0105067, upper bound: 0.0104226
IS_A2_A1_B1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0104737, upper bound: 0.0098948
IS_A2_A1_B1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0103902
IS_A2_A1_B1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0098948
IS_A2_A1_B1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0105120, upper bound: 0.0103738
IS_A2_A1_B1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0104913, upper bound: 0.0098920
IS_A2_A1_B1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0103486
IS_A2_A1_B1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0098878
IS_A2_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0106452, upper bound: 0.0102343
IS_A2_A1_B2_B1_B1_A2, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0102230
IS_A2_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0106214, upper bound: 0.0100658
IS_A2_A1_B2_B1_B2_A2, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0100500
IS_A2_A1_B2_B2_B1_B1, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0104584
IS_A2_A1_B2_B2_B1_B2, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0101946
IS_A2_A1_B2_B2_B2_B1, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0103851
IS_A2_A1_B2_B2_B2_B2, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0100220
IS_A2_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0107108, upper bound: 0.0098948
IS_A2_A2_B1_B1_A1_A2, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0104304, upper bound: 0.0098948
IS_A2_A2_B1_B1_A2_A1, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0105697, upper bound: 0.0098948
IS_A2_A2_B1_B1_A2_A2, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0098948
IS_A2_A2_B1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0104618, upper bound: 0.0105344
IS_A2_A2_B1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0104329, upper bound: 0.0098920
IS_A2_A2_B1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0105045
IS_A2_A2_B1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0098920
IS_A2_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0107108, upper bound: 0.0100987
IS_A2_A2_B2_B1_A1_A2, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0104304, upper bound: 0.0100926
IS_A2_A2_B2_B1_A2_A1, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0105697, upper bound: 0.0100734
IS_A2_A2_B2_B1_A2_A2, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0100654
IS_A2_A2_B2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0104618, upper bound: 0.0105673
IS_A2_A2_B2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0104329, upper bound: 0.0100748
IS_A2_A2_B2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0105346
IS_A2_A2_B2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.38
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0100521

## BFS IS instance: IS_A1_A1_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0070223, 0.0045636, -0.0074819, 0.0048139, -0.0118362, 0.0120455
1: -0.0058909, -0.0013013, -0.0060730, -0.0012021, -0.0046888, 0.0047717
2: 0.0280229, 0.0392843, 0.0277897, 0.0400601, -0.0120372, 0.0114946
3: -0.0067861, 0.0049022, -0.0070288, 0.0054006, -0.0105754, 0.0102973
4: -0.0051841, 0.0048915, -0.0055270, 0.0051728, -0.0103569, 0.0104185
5: 0.0072134, 0.0161969, 0.0068825, 0.0164050, -0.0091916, 0.0093144
6: -0.0110322, 0.0018819, -0.0113749, 0.0023374, -0.0133696, 0.0132568
7: 0.9673733, 0.9835072, 0.9664697, 0.9838396, -0.0164663, 0.0170375
8: -0.0209725, -0.0012867, -0.0215529, -0.0007876, -0.0201849, 0.0202663
9: -0.0038856, 0.0081369, -0.0042240, 0.0084728, -0.0123585, 0.0123609

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: B, layer: 3, pos: 106
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of IS_A1_A1_B2_B2_B1_B1_A1

### Relational analysis result of IS_A1_A1_B2_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0104304
time: 1.01 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_B1_A2

### Relational analysis result of IS_A1_A1_B2_B2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0104304
time: 1.19 seconds

## BFS IS instance: IS_A2_A1_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0073249, 0.0047968, -0.0073164, 0.0048494, -0.0121743, 0.0121132
1: -0.0060108, -0.0012089, -0.0060075, -0.0011881, -0.0048227, 0.0047986
2: 0.0278056, 0.0397952, 0.0277567, 0.0397808, -0.0119752, 0.0120385
3: -0.0070123, 0.0052304, -0.0070633, 0.0052212, -0.0104468, 0.0105087
4: -0.0054099, 0.0051536, -0.0054036, 0.0052127, -0.0106226, 0.0105572
5: 0.0069955, 0.0163908, 0.0070017, 0.0164345, -0.0094390, 0.0093892
6: -0.0113515, 0.0021818, -0.0114235, 0.0021734, -0.0135249, 0.0136053
7: 0.9667783, 0.9838169, 0.9667950, 0.9838866, -0.0171083, 0.0170220
8: -0.0215133, -0.0009580, -0.0216352, -0.0009673, -0.0205460, 0.0206772
9: -0.0041085, 0.0084499, -0.0041022, 0.0085205, -0.0126290, 0.0125521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: B, layer: 3, pos: 106
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of IS_A2_A1_B2_B1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B2_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0102230
time: 1.05 seconds

## Relational analysis of IS_A2_A1_B2_B1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B2_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0102230
time: 1.05 seconds

## BFS IS instance: IS_A2_A1_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0073222, 0.0047302, -0.0077128, 0.0047094, -0.0120316, 0.0124430
1: -0.0060097, -0.0012353, -0.0061645, -0.0012435, -0.0047662, 0.0049292
2: 0.0278677, 0.0397905, 0.0278871, 0.0404499, -0.0125821, 0.0119034
3: -0.0069477, 0.0052275, -0.0069275, 0.0056511, -0.0108381, 0.0104445
4: -0.0054078, 0.0050787, -0.0056994, 0.0050554, -0.0104632, 0.0107781
5: 0.0069975, 0.0163354, 0.0067163, 0.0163181, -0.0093206, 0.0096192
6: -0.0112602, 0.0021791, -0.0112318, 0.0025663, -0.0138265, 0.0134109
7: 0.9667836, 0.9837284, 0.9660155, 0.9837008, -0.0169172, 0.0177129
8: -0.0213588, -0.0009610, -0.0213106, -0.0005368, -0.0208219, 0.0203495
9: -0.0041064, 0.0083604, -0.0043940, 0.0083326, -0.0124390, 0.0127545

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 106

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of IS_A2_A1_B2_B1_B2_A1_B1

### Relational analysis result of IS_A2_A1_B2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0100500
time: 1.06 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2_A1_B2

### Relational analysis result of IS_A2_A1_B2_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0100500
time: 1.07 seconds

## BFS IS instance: IS_A2_A2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0074819, 0.0048139, -0.0070223, 0.0045636, -0.0120455, 0.0118362
1: -0.0060730, -0.0012021, -0.0058909, -0.0013013, -0.0047717, 0.0046888
2: 0.0277897, 0.0400601, 0.0280229, 0.0392843, -0.0114946, 0.0120372
3: -0.0070288, 0.0054006, -0.0067861, 0.0049022, -0.0102973, 0.0105754
4: -0.0055270, 0.0051728, -0.0051841, 0.0048915, -0.0104185, 0.0103569
5: 0.0068825, 0.0164050, 0.0072134, 0.0161969, -0.0093144, 0.0091916
6: -0.0113749, 0.0023374, -0.0110322, 0.0018819, -0.0132568, 0.0133696
7: 0.9664697, 0.9838396, 0.9673733, 0.9835072, -0.0170375, 0.0164663
8: -0.0215529, -0.0007876, -0.0209725, -0.0012867, -0.0202663, 0.0201849
9: -0.0042240, 0.0084728, -0.0038856, 0.0081369, -0.0123609, 0.0123585

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of IS_A2_A2_B1_B1_A1_A1_B1

### Relational analysis result of IS_A2_A2_B1_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104304, upper bound: 0.0098948
time: 1.30 seconds

## Relational analysis of IS_A2_A2_B1_B1_A1_A1_B2

### Relational analysis result of IS_A2_A2_B1_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104304, upper bound: 0.0098948
time: 1.30 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0074819, 0.0048139, -0.0073250, 0.0050266, -0.0125085, 0.0121388
1: -0.0060730, -0.0012021, -0.0060108, -0.0011179, -0.0049551, 0.0048087
2: 0.0277897, 0.0400601, 0.0275916, 0.0397952, -0.0120054, 0.0124685
3: -0.0070288, 0.0054006, -0.0072352, 0.0052304, -0.0104878, 0.0108773
4: -0.0055270, 0.0051728, -0.0054099, 0.0054120, -0.0109390, 0.0105827
5: 0.0068825, 0.0164050, 0.0069955, 0.0165819, -0.0096994, 0.0094095
6: -0.0113749, 0.0023374, -0.0116662, 0.0021819, -0.0135567, 0.0140036
7: 0.9664697, 0.9838396, 0.9667782, 0.9841221, -0.0176524, 0.0170614
8: -0.0215529, -0.0007876, -0.0220464, -0.0009580, -0.0205949, 0.0212588
9: -0.0042240, 0.0084728, -0.0041085, 0.0087585, -0.0129825, 0.0125813

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104304, upper bound: 0.0100926
time: 1.09 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104304, upper bound: 0.0100926
time: 1.08 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.41 seconds
IS_A1_A1_B2_B2_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0104304
IS_A1_A1_B2_B2_B1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0104304
IS_A2_A1_B2_B1_B1_A1_B1, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0102230
IS_A2_A1_B2_B1_B1_A1_B2, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0102230
IS_A2_A1_B2_B1_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0100500
IS_A2_A1_B2_B1_B2_A1_B2, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0100500
IS_A2_A2_B1_B1_A1_A1_B1, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 7, lower bound: -0.0104304, upper bound: 0.0098948
IS_A2_A2_B1_B1_A1_A1_B2, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 7, lower bound: -0.0104304, upper bound: 0.0098948
IS_A2_A2_B2_B1_A1_A1_B1, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 7, lower bound: -0.0104304, upper bound: 0.0100926
IS_A2_A2_B2_B1_A1_A1_B2, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 7, lower bound: -0.0104304, upper bound: 0.0100926

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 3.24 + 241.64 = 244.88 seconds
