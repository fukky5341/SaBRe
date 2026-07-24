## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.01061397


## IAR start

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
execution time: IAR + RelationalAnalysis = 1.11 + 1.91 = 3.02 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0117933, upper bound: 0.0117933

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0112680, upper bound: 0.0116051
time: 0.97 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0116051, upper bound: 0.0116051
time: 1.05 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.10 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.10
Output dim: 7, lower bound: -0.0112680, upper bound: 0.0116051
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.10
Output dim: 7, lower bound: -0.0116051, upper bound: 0.0116051

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0112680, upper bound: 0.0112680
time: 1.08 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0112680, upper bound: 0.0116051
time: 1.01 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0113511, upper bound: 0.0114973
time: 1.24 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0114973, upper bound: 0.0114973
time: 1.02 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.31 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 7, lower bound: -0.0112680, upper bound: 0.0112680
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 7, lower bound: -0.0112680, upper bound: 0.0116051
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 7, lower bound: -0.0113511, upper bound: 0.0114973
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 7, lower bound: -0.0114973, upper bound: 0.0114973

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0071846, 0.0051765, -0.0071846, 0.0051765, -0.0123611, 0.0123611
1: -0.0059552, -0.0010585, -0.0059552, -0.0010585, -0.0048967, 0.0048967
2: 0.0274520, 0.0395582, 0.0274520, 0.0395582, -0.0121063, 0.0121063
3: -0.0073805, 0.0050782, -0.0073805, 0.0050782, -0.0108160, 0.0108160
4: -0.0053052, 0.0055804, -0.0053052, 0.0055804, -0.0108856, 0.0108856
5: 0.0070966, 0.0167065, 0.0070966, 0.0167065, -0.0096099, 0.0096099
6: -0.0118714, 0.0020427, -0.0118714, 0.0020427, -0.0139142, 0.0139142
7: 0.9670542, 0.9843211, 0.9670542, 0.9843211, -0.0172669, 0.0172669
8: -0.0223939, -0.0011105, -0.0223939, -0.0011105, -0.0212835, 0.0212835
9: -0.0040051, 0.0089597, -0.0040051, 0.0089597, -0.0129648, 0.0129648

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111158, upper bound: 0.0111305
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111158, upper bound: 0.0112349
time: 0.99 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0071846, 0.0051765, -0.0074924, 0.0056362, -0.0128207, 0.0126689
1: -0.0059552, -0.0010585, -0.0060772, -0.0008764, -0.0050788, 0.0050187
2: 0.0274520, 0.0395582, 0.0270238, 0.0400779, -0.0126259, 0.0125345
3: -0.0073805, 0.0050782, -0.0078263, 0.0054121, -0.0111887, 0.0112893
4: -0.0053052, 0.0055804, -0.0055349, 0.0060971, -0.0114023, 0.0111153
5: 0.0070966, 0.0167065, 0.0068749, 0.0170887, -0.0099921, 0.0098316
6: -0.0118714, 0.0020427, -0.0125008, 0.0023478, -0.0142193, 0.0145436
7: 0.9670542, 0.9843211, 0.9664489, 0.9849313, -0.0178772, 0.0178722
8: -0.0223939, -0.0011105, -0.0234600, -0.0007762, -0.0216177, 0.0223495
9: -0.0040051, 0.0089597, -0.0042318, 0.0095767, -0.0135819, 0.0131914

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111158, upper bound: 0.0113506
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111158, upper bound: 0.0114973
time: 1.00 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0074922, 0.0055628, -0.0075762, 0.0045825, -0.0120747, 0.0131390
1: -0.0060771, -0.0009055, -0.0061104, -0.0012938, -0.0047833, 0.0052049
2: 0.0270921, 0.0400774, 0.0280052, 0.0402193, -0.0131272, 0.0120722
3: -0.0077552, 0.0054118, -0.0068045, 0.0055030, -0.0116229, 0.0105781
4: -0.0055347, 0.0060146, -0.0055974, 0.0049128, -0.0104475, 0.0116121
5: 0.0068751, 0.0170277, 0.0068146, 0.0162127, -0.0093376, 0.0102131
6: -0.0124003, 0.0023476, -0.0110581, 0.0024309, -0.0148312, 0.0134057
7: 0.9664494, 0.9848338, 0.9662841, 0.9835325, -0.0170831, 0.0185497
8: -0.0232897, -0.0007764, -0.0210164, -0.0006851, -0.0226046, 0.0202400
9: -0.0042316, 0.0094782, -0.0042935, 0.0081623, -0.0123939, 0.0137717

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0113511, upper bound: 0.0113511
time: 1.27 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0113511, upper bound: 0.0114973
time: 1.15 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0074922, 0.0055966, -0.0077382, 0.0048289, -0.0123211, 0.0133348
1: -0.0060771, -0.0008921, -0.0061745, -0.0011962, -0.0048809, 0.0052825
2: 0.0270606, 0.0400775, 0.0277758, 0.0404928, -0.0134321, 0.0123017
3: -0.0077880, 0.0054118, -0.0070434, 0.0056786, -0.0118639, 0.0107532
4: -0.0055347, 0.0060526, -0.0057183, 0.0051897, -0.0107244, 0.0117710
5: 0.0068751, 0.0170558, 0.0066980, 0.0164175, -0.0095424, 0.0103578
6: -0.0124466, 0.0023476, -0.0113954, 0.0025915, -0.0150381, 0.0137430
7: 0.9664493, 0.9848787, 0.9659656, 0.9838594, -0.0174102, 0.0189131
8: -0.0233682, -0.0007764, -0.0215877, -0.0005092, -0.0228590, 0.0208113
9: -0.0042316, 0.0095236, -0.0044128, 0.0084930, -0.0127246, 0.0139364

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0114973, upper bound: 0.0113511
time: 1.21 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0114973, upper bound: 0.0114973
time: 0.99 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.30 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 7, lower bound: -0.0111158, upper bound: 0.0111305
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 7, lower bound: -0.0111158, upper bound: 0.0112349
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 7, lower bound: -0.0111158, upper bound: 0.0113506
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 7, lower bound: -0.0111158, upper bound: 0.0114973
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 7, lower bound: -0.0113511, upper bound: 0.0113511
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 7, lower bound: -0.0113511, upper bound: 0.0114973
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 7, lower bound: -0.0114973, upper bound: 0.0113511
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 7, lower bound: -0.0114973, upper bound: 0.0114973

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111305, upper bound: 0.0111305
time: 0.99 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111305, upper bound: 0.0111305
time: 1.03 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111305, upper bound: 0.0112349
time: 1.13 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111305, upper bound: 0.0112349
time: 1.11 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110083, upper bound: 0.0113506
time: 1.11 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110083, upper bound: 0.0113506
time: 1.21 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110083, upper bound: 0.0114973
time: 1.05 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110083, upper bound: 0.0114973
time: 1.25 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0073250, 0.0050266, -0.0075762, 0.0045825, -0.0119075, 0.0126029
1: -0.0060108, -0.0011179, -0.0061104, -0.0012938, -0.0047170, 0.0049925
2: 0.0275916, 0.0397952, 0.0280052, 0.0402193, -0.0126278, 0.0117899
3: -0.0072352, 0.0052304, -0.0068045, 0.0055030, -0.0110961, 0.0103675
4: -0.0054099, 0.0054120, -0.0055974, 0.0049128, -0.0103227, 0.0110094
5: 0.0069955, 0.0165819, 0.0068146, 0.0162127, -0.0092172, 0.0097673
6: -0.0116662, 0.0021819, -0.0110581, 0.0024309, -0.0140971, 0.0132399
7: 0.9667782, 0.9841221, 0.9662841, 0.9835325, -0.0167543, 0.0178380
8: -0.0220464, -0.0009580, -0.0210164, -0.0006851, -0.0213612, 0.0200584
9: -0.0041085, 0.0087585, -0.0042935, 0.0081623, -0.0122707, 0.0130520

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0113506, upper bound: 0.0110489
time: 1.04 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0113506, upper bound: 0.0110635
time: 1.01 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0074905, 0.0052772, -0.0075762, 0.0045825, -0.0120730, 0.0128534
1: -0.0060764, -0.0010186, -0.0061104, -0.0012938, -0.0047826, 0.0050918
2: 0.0273582, 0.0400746, 0.0280052, 0.0402193, -0.0128612, 0.0120693
3: -0.0074782, 0.0054100, -0.0068045, 0.0055030, -0.0113672, 0.0105750
4: -0.0055334, 0.0056936, -0.0055974, 0.0049128, -0.0104462, 0.0112910
5: 0.0068763, 0.0167902, 0.0068146, 0.0162127, -0.0093363, 0.0099756
6: -0.0120093, 0.0023459, -0.0110581, 0.0024309, -0.0144402, 0.0134040
7: 0.9664528, 0.9844546, 0.9662841, 0.9835325, -0.0170797, 0.0181705
8: -0.0226274, -0.0007783, -0.0210164, -0.0006851, -0.0219423, 0.0202382
9: -0.0042304, 0.0090948, -0.0042935, 0.0081623, -0.0123926, 0.0133883

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0113506, upper bound: 0.0111158
time: 1.23 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0113506, upper bound: 0.0111358
time: 1.27 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0073250, 0.0050266, -0.0077382, 0.0048289, -0.0121538, 0.0127649
1: -0.0060108, -0.0011179, -0.0061745, -0.0011962, -0.0048146, 0.0050567
2: 0.0275916, 0.0397952, 0.0277758, 0.0404928, -0.0129012, 0.0120194
3: -0.0072352, 0.0052304, -0.0070434, 0.0056786, -0.0113012, 0.0106288
4: -0.0054099, 0.0054120, -0.0057183, 0.0051897, -0.0105996, 0.0111303
5: 0.0069955, 0.0165819, 0.0066980, 0.0164175, -0.0094220, 0.0098840
6: -0.0116662, 0.0021819, -0.0113954, 0.0025915, -0.0142577, 0.0135772
7: 0.9667782, 0.9841221, 0.9659656, 0.9838594, -0.0170813, 0.0181565
8: -0.0220464, -0.0009580, -0.0215877, -0.0005092, -0.0215372, 0.0206297
9: -0.0041085, 0.0087585, -0.0044128, 0.0084930, -0.0126014, 0.0131713

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0113506, upper bound: 0.0110083
time: 1.31 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0113506, upper bound: 0.0110216
time: 1.01 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0074905, 0.0052772, -0.0077382, 0.0048289, -0.0123193, 0.0130154
1: -0.0060764, -0.0010186, -0.0061745, -0.0011962, -0.0048802, 0.0051560
2: 0.0273582, 0.0400746, 0.0277758, 0.0404928, -0.0131346, 0.0122988
3: -0.0074782, 0.0054100, -0.0070434, 0.0056786, -0.0114868, 0.0107508
4: -0.0055334, 0.0056936, -0.0057183, 0.0051897, -0.0107231, 0.0114119
5: 0.0068763, 0.0167902, 0.0066980, 0.0164175, -0.0095411, 0.0100923
6: -0.0120093, 0.0023459, -0.0113954, 0.0025915, -0.0146008, 0.0137413
7: 0.9664528, 0.9844546, 0.9659656, 0.9838594, -0.0174066, 0.0184891
8: -0.0226274, -0.0007783, -0.0215877, -0.0005092, -0.0221182, 0.0208094
9: -0.0042304, 0.0090948, -0.0044128, 0.0084930, -0.0127233, 0.0135076

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0113506, upper bound: 0.0111158
time: 1.33 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0113506, upper bound: 0.0111358
time: 1.03 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.44 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 7, lower bound: -0.0111305, upper bound: 0.0111305
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 7, lower bound: -0.0111305, upper bound: 0.0111305
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 7, lower bound: -0.0111305, upper bound: 0.0112349
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 7, lower bound: -0.0111305, upper bound: 0.0112349
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 7, lower bound: -0.0110083, upper bound: 0.0113506
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 7, lower bound: -0.0110083, upper bound: 0.0113506
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 7, lower bound: -0.0110083, upper bound: 0.0114973
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 7, lower bound: -0.0110083, upper bound: 0.0114973
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 7, lower bound: -0.0113506, upper bound: 0.0110489
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 7, lower bound: -0.0113506, upper bound: 0.0110635
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 7, lower bound: -0.0113506, upper bound: 0.0111158
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 7, lower bound: -0.0113506, upper bound: 0.0111358
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 7, lower bound: -0.0113506, upper bound: 0.0110083
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 7, lower bound: -0.0113506, upper bound: 0.0110216
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 7, lower bound: -0.0113506, upper bound: 0.0111158
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 7, lower bound: -0.0113506, upper bound: 0.0111358

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110108, upper bound: 0.0108874
time: 1.02 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109340, upper bound: 0.0108800
time: 1.01 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110108, upper bound: 0.0108874
time: 1.02 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109340, upper bound: 0.0108800
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109640, upper bound: 0.0109185
time: 1.02 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108800, upper bound: 0.0109123
time: 1.04 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109640, upper bound: 0.0109184
time: 1.39 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108800, upper bound: 0.0109123
time: 1.00 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108739, upper bound: 0.0110436
time: 1.03 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108465, upper bound: 0.0110344
time: 1.03 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108739, upper bound: 0.0110436
time: 0.99 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108465, upper bound: 0.0110344
time: 1.06 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108381, upper bound: 0.0110646
time: 1.11 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108023, upper bound: 0.0110585
time: 1.08 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108381, upper bound: 0.0110646
time: 1.11 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108023, upper bound: 0.0110585
time: 1.13 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0112357, upper bound: 0.0108624
time: 0.99 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110980, upper bound: 0.0108467
time: 1.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0112357, upper bound: 0.0108775
time: 0.94 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110980, upper bound: 0.0108662
time: 0.97 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111688, upper bound: 0.0108897
time: 1.21 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108710
time: 1.24 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111688, upper bound: 0.0109124
time: 1.28 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108955
time: 0.94 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0112658, upper bound: 0.0108131
time: 1.18 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111035, upper bound: 0.0108023
time: 1.25 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0112658, upper bound: 0.0108269
time: 1.18 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111035, upper bound: 0.0108166
time: 1.21 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111688, upper bound: 0.0108832
time: 1.32 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108641
time: 1.14 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111688, upper bound: 0.0109118
time: 1.24 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108937
time: 1.13 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.49 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 7, lower bound: -0.0110108, upper bound: 0.0108874
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 7, lower bound: -0.0109340, upper bound: 0.0108800
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 7, lower bound: -0.0110108, upper bound: 0.0108874
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 7, lower bound: -0.0109340, upper bound: 0.0108800
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 7, lower bound: -0.0109640, upper bound: 0.0109185
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 7, lower bound: -0.0108800, upper bound: 0.0109123
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 7, lower bound: -0.0109640, upper bound: 0.0109184
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 7, lower bound: -0.0108800, upper bound: 0.0109123
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 7, lower bound: -0.0108739, upper bound: 0.0110436
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 7, lower bound: -0.0108465, upper bound: 0.0110344
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 7, lower bound: -0.0108739, upper bound: 0.0110436
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 7, lower bound: -0.0108465, upper bound: 0.0110344
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 7, lower bound: -0.0108381, upper bound: 0.0110646
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 7, lower bound: -0.0108023, upper bound: 0.0110585
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 7, lower bound: -0.0108381, upper bound: 0.0110646
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 7, lower bound: -0.0108023, upper bound: 0.0110585
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 7, lower bound: -0.0112357, upper bound: 0.0108624
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 7, lower bound: -0.0110980, upper bound: 0.0108467
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 7, lower bound: -0.0112357, upper bound: 0.0108775
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 7, lower bound: -0.0110980, upper bound: 0.0108662
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 7, lower bound: -0.0111688, upper bound: 0.0108897
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108710
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 7, lower bound: -0.0111688, upper bound: 0.0109124
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108955
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 7, lower bound: -0.0112658, upper bound: 0.0108131
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 7, lower bound: -0.0111035, upper bound: 0.0108023
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 7, lower bound: -0.0112658, upper bound: 0.0108269
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 7, lower bound: -0.0111035, upper bound: 0.0108166
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 7, lower bound: -0.0111688, upper bound: 0.0108832
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108641
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 7, lower bound: -0.0111688, upper bound: 0.0109118
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108937

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109340, upper bound: 0.0109340
time: 0.97 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109340, upper bound: 0.0109340
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109340, upper bound: 0.0109340
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109340, upper bound: 0.0109340
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0070136, 0.0043946, -0.0071827, 0.0048104, -0.0118240, 0.0115773
1: -0.0058875, -0.0013682, -0.0059545, -0.0012035, -0.0046840, 0.0045863
2: 0.0281803, 0.0392696, 0.0277929, 0.0395551, -0.0113748, 0.0114766
3: -0.0066222, 0.0048927, -0.0070255, 0.0050762, -0.0100482, 0.0102668
4: -0.0051775, 0.0047016, -0.0053038, 0.0051690, -0.0103465, 0.0100053
5: 0.0072197, 0.0160564, 0.0070979, 0.0164022, -0.0091824, 0.0089585
6: -0.0108007, 0.0018732, -0.0113701, 0.0020409, -0.0128416, 0.0132434
7: 0.9673904, 0.9832830, 0.9670579, 0.9838350, -0.0164446, 0.0162250
8: -0.0205806, -0.0012962, -0.0215449, -0.0011125, -0.0194681, 0.0202488
9: -0.0038792, 0.0079100, -0.0040037, 0.0084682, -0.0123474, 0.0119137

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109430, upper bound: 0.0108800
time: 1.02 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109430, upper bound: 0.0108800
time: 1.04 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0073856, 0.0042302, -0.0071800, 0.0047423, -0.0121279, 0.0114102
1: -0.0060349, -0.0014334, -0.0059534, -0.0012305, -0.0048044, 0.0045200
2: 0.0283335, 0.0398976, 0.0278564, 0.0395505, -0.0112170, 0.0120412
3: -0.0064628, 0.0052963, -0.0069595, 0.0050732, -0.0099640, 0.0106336
4: -0.0054552, 0.0045168, -0.0053017, 0.0050924, -0.0105476, 0.0098185
5: 0.0069518, 0.0159198, 0.0070999, 0.0163455, -0.0093937, 0.0088199
6: -0.0105756, 0.0022420, -0.0112769, 0.0020382, -0.0126138, 0.0135189
7: 0.9666588, 0.9830647, 0.9670632, 0.9837446, -0.0170857, 0.0160014
8: -0.0201993, -0.0008921, -0.0213870, -0.0011154, -0.0190838, 0.0204948
9: -0.0041531, 0.0076893, -0.0040017, 0.0083768, -0.0125299, 0.0116910

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109430, upper bound: 0.0108800
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109430, upper bound: 0.0108800
time: 1.01 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108800, upper bound: 0.0109430
time: 1.06 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108800, upper bound: 0.0109430
time: 1.04 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108800, upper bound: 0.0109430
time: 1.08 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108800, upper bound: 0.0109430
time: 1.07 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0071739, 0.0046427, -0.0071827, 0.0048104, -0.0119843, 0.0118255
1: -0.0059510, -0.0012699, -0.0059545, -0.0012035, -0.0047475, 0.0046846
2: 0.0279491, 0.0395402, 0.0277929, 0.0395551, -0.0116060, 0.0117473
3: -0.0068629, 0.0050667, -0.0070255, 0.0050762, -0.0102272, 0.0103835
4: -0.0052972, 0.0049805, -0.0053038, 0.0051690, -0.0104661, 0.0102843
5: 0.0071043, 0.0162627, 0.0070979, 0.0164022, -0.0092979, 0.0091648
6: -0.0111406, 0.0020322, -0.0113701, 0.0020409, -0.0131814, 0.0134023
7: 0.9670752, 0.9836124, 0.9670579, 0.9838350, -0.0167598, 0.0165545
8: -0.0211561, -0.0011221, -0.0215449, -0.0011125, -0.0200436, 0.0204229
9: -0.0039972, 0.0082431, -0.0040037, 0.0084682, -0.0124655, 0.0122468

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108800, upper bound: 0.0109123
time: 0.99 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108800, upper bound: 0.0109123
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0075451, 0.0044799, -0.0071800, 0.0047423, -0.0122874, 0.0116599
1: -0.0060980, -0.0013344, -0.0059534, -0.0012305, -0.0048675, 0.0046190
2: 0.0281009, 0.0401667, 0.0278564, 0.0395505, -0.0114496, 0.0123103
3: -0.0067050, 0.0054692, -0.0069595, 0.0050732, -0.0101438, 0.0107443
4: -0.0055742, 0.0047974, -0.0053017, 0.0050924, -0.0106666, 0.0100992
5: 0.0068371, 0.0161274, 0.0070999, 0.0163455, -0.0095085, 0.0090275
6: -0.0109175, 0.0024000, -0.0112769, 0.0020382, -0.0129557, 0.0136769
7: 0.9663454, 0.9833962, 0.9670632, 0.9837446, -0.0173992, 0.0163330
8: -0.0207784, -0.0007190, -0.0213870, -0.0011154, -0.0196629, 0.0206680
9: -0.0042705, 0.0080245, -0.0040017, 0.0083768, -0.0126473, 0.0120262

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108800, upper bound: 0.0109123
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108800, upper bound: 0.0109123
time: 1.05 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0070136, 0.0043946, -0.0073250, 0.0050266, -0.0120402, 0.0117196
1: -0.0058875, -0.0013682, -0.0060108, -0.0011179, -0.0047696, 0.0046426
2: 0.0281803, 0.0392696, 0.0275916, 0.0397952, -0.0116149, 0.0116780
3: -0.0066222, 0.0048927, -0.0072352, 0.0052304, -0.0102132, 0.0104760
4: -0.0051775, 0.0047016, -0.0054099, 0.0054120, -0.0105895, 0.0101115
5: 0.0072197, 0.0160564, 0.0069955, 0.0165819, -0.0093622, 0.0090609
6: -0.0108007, 0.0018732, -0.0116662, 0.0021819, -0.0129826, 0.0135394
7: 0.9673904, 0.9832830, 0.9667782, 0.9841221, -0.0167317, 0.0165048
8: -0.0205806, -0.0012962, -0.0220464, -0.0009580, -0.0196225, 0.0207502
9: -0.0038792, 0.0079100, -0.0041085, 0.0087585, -0.0126377, 0.0120184

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108467, upper bound: 0.0110980
time: 1.10 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108467, upper bound: 0.0110980
time: 1.10 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0073856, 0.0042302, -0.0073222, 0.0049614, -0.0123470, 0.0115524
1: -0.0060349, -0.0014334, -0.0060097, -0.0011437, -0.0048911, 0.0045764
2: 0.0283335, 0.0398976, 0.0276524, 0.0397905, -0.0114571, 0.0122453
3: -0.0064628, 0.0052963, -0.0071719, 0.0052275, -0.0101291, 0.0108455
4: -0.0054552, 0.0045168, -0.0054078, 0.0053386, -0.0107938, 0.0099246
5: 0.0069518, 0.0159198, 0.0069975, 0.0165276, -0.0095758, 0.0089222
6: -0.0105756, 0.0022420, -0.0115768, 0.0021791, -0.0127547, 0.0138188
7: 0.9666588, 0.9830647, 0.9667836, 0.9840354, -0.0173766, 0.0162810
8: -0.0201993, -0.0008921, -0.0218950, -0.0009610, -0.0192382, 0.0210028
9: -0.0041531, 0.0076893, -0.0041064, 0.0086709, -0.0128240, 0.0117957

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108467, upper bound: 0.0110980
time: 1.03 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108467, upper bound: 0.0110980
time: 1.04 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0070136, 0.0043946, -0.0074905, 0.0052772, -0.0122908, 0.0118851
1: -0.0058875, -0.0013682, -0.0060764, -0.0010186, -0.0048689, 0.0047082
2: 0.0281803, 0.0392696, 0.0273582, 0.0400746, -0.0118943, 0.0119114
3: -0.0066222, 0.0048927, -0.0074782, 0.0054100, -0.0104207, 0.0107470
4: -0.0051775, 0.0047016, -0.0055334, 0.0056936, -0.0108711, 0.0102350
5: 0.0072197, 0.0160564, 0.0068763, 0.0167902, -0.0095705, 0.0091801
6: -0.0108007, 0.0018732, -0.0120093, 0.0023459, -0.0131467, 0.0138825
7: 0.9673904, 0.9832830, 0.9664528, 0.9844546, -0.0170642, 0.0168302
8: -0.0205806, -0.0012962, -0.0226274, -0.0007783, -0.0198023, 0.0213312
9: -0.0038792, 0.0079100, -0.0042304, 0.0090948, -0.0129740, 0.0121404

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108710, upper bound: 0.0110344
time: 1.03 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108710, upper bound: 0.0110344
time: 1.04 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0073856, 0.0042302, -0.0074877, 0.0052109, -0.0125965, 0.0117179
1: -0.0060349, -0.0014334, -0.0060753, -0.0010449, -0.0049900, 0.0046419
2: 0.0283335, 0.0398976, 0.0274199, 0.0400699, -0.0117365, 0.0124777
3: -0.0064628, 0.0052963, -0.0074139, 0.0054070, -0.0103364, 0.0111161
4: -0.0054552, 0.0045168, -0.0055314, 0.0056191, -0.0110743, 0.0100481
5: 0.0069518, 0.0159198, 0.0068783, 0.0167351, -0.0097832, 0.0090414
6: -0.0105756, 0.0022420, -0.0119185, 0.0023432, -0.0129188, 0.0141605
7: 0.9666588, 0.9830647, 0.9664581, 0.9843667, -0.0177079, 0.0166065
8: -0.0201993, -0.0008921, -0.0224736, -0.0007813, -0.0194180, 0.0215815
9: -0.0041531, 0.0076893, -0.0042283, 0.0090058, -0.0131590, 0.0119176

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108710, upper bound: 0.0110344
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108710, upper bound: 0.0110344
time: 0.97 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0071739, 0.0046427, -0.0073250, 0.0050266, -0.0122005, 0.0119677
1: -0.0059510, -0.0012699, -0.0060108, -0.0011179, -0.0048331, 0.0047409
2: 0.0279491, 0.0395402, 0.0275916, 0.0397952, -0.0118460, 0.0119487
3: -0.0068629, 0.0050667, -0.0072352, 0.0052304, -0.0104734, 0.0106759
4: -0.0052972, 0.0049805, -0.0054099, 0.0054120, -0.0107092, 0.0103904
5: 0.0071043, 0.0162627, 0.0069955, 0.0165819, -0.0094777, 0.0092672
6: -0.0111406, 0.0020322, -0.0116662, 0.0021819, -0.0133224, 0.0136984
7: 0.9670752, 0.9836124, 0.9667782, 0.9841221, -0.0170469, 0.0168343
8: -0.0211561, -0.0011221, -0.0220464, -0.0009580, -0.0201980, 0.0209243
9: -0.0039972, 0.0082431, -0.0041085, 0.0087585, -0.0127557, 0.0123516

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108023, upper bound: 0.0111036
time: 1.05 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108023, upper bound: 0.0111036
time: 1.06 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0075451, 0.0044799, -0.0073222, 0.0049614, -0.0125064, 0.0118021
1: -0.0060980, -0.0013344, -0.0060097, -0.0011437, -0.0049543, 0.0046753
2: 0.0281009, 0.0401667, 0.0276524, 0.0397905, -0.0116897, 0.0125144
3: -0.0067050, 0.0054692, -0.0071719, 0.0052275, -0.0103949, 0.0110416
4: -0.0055742, 0.0047974, -0.0054078, 0.0053386, -0.0109128, 0.0102053
5: 0.0068371, 0.0161274, 0.0069975, 0.0165276, -0.0096906, 0.0091298
6: -0.0109175, 0.0024000, -0.0115768, 0.0021791, -0.0130967, 0.0139769
7: 0.9663454, 0.9833962, 0.9667836, 0.9840354, -0.0176901, 0.0166126
8: -0.0207784, -0.0007190, -0.0218950, -0.0009610, -0.0198173, 0.0211760
9: -0.0042705, 0.0080245, -0.0041064, 0.0086709, -0.0129414, 0.0121309

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108023, upper bound: 0.0111036
time: 1.07 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108023, upper bound: 0.0111036
time: 1.05 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0071739, 0.0046427, -0.0074905, 0.0052772, -0.0124511, 0.0121332
1: -0.0059510, -0.0012699, -0.0060764, -0.0010186, -0.0049324, 0.0048065
2: 0.0279491, 0.0395402, 0.0273582, 0.0400746, -0.0121255, 0.0121821
3: -0.0068629, 0.0050667, -0.0074782, 0.0054100, -0.0105981, 0.0108624
4: -0.0052972, 0.0049805, -0.0055334, 0.0056936, -0.0109908, 0.0105139
5: 0.0071043, 0.0162627, 0.0068763, 0.0167902, -0.0096860, 0.0093864
6: -0.0111406, 0.0020322, -0.0120093, 0.0023459, -0.0134865, 0.0140415
7: 0.9670752, 0.9836124, 0.9664528, 0.9844546, -0.0173795, 0.0171596
8: -0.0211561, -0.0011221, -0.0226274, -0.0007783, -0.0203778, 0.0215054
9: -0.0039972, 0.0082431, -0.0042304, 0.0090948, -0.0130921, 0.0124735

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108023, upper bound: 0.0110585
time: 1.17 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108023, upper bound: 0.0110585
time: 1.08 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0075451, 0.0044799, -0.0074877, 0.0052109, -0.0127559, 0.0119676
1: -0.0060980, -0.0013344, -0.0060753, -0.0010449, -0.0050532, 0.0047409
2: 0.0281009, 0.0401667, 0.0274199, 0.0400699, -0.0119691, 0.0127468
3: -0.0067050, 0.0054692, -0.0074139, 0.0054070, -0.0105147, 0.0112259
4: -0.0055742, 0.0047974, -0.0055314, 0.0056191, -0.0111933, 0.0103288
5: 0.0068371, 0.0161274, 0.0068783, 0.0167351, -0.0098980, 0.0092490
6: -0.0109175, 0.0024000, -0.0119185, 0.0023432, -0.0132607, 0.0143185
7: 0.9663454, 0.9833962, 0.9664581, 0.9843667, -0.0180213, 0.0169381
8: -0.0207784, -0.0007190, -0.0224736, -0.0007813, -0.0199971, 0.0217547
9: -0.0042705, 0.0080245, -0.0042283, 0.0090058, -0.0132763, 0.0122528

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108023, upper bound: 0.0110585
time: 1.04 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108023, upper bound: 0.0110585
time: 1.03 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110980, upper bound: 0.0108467
time: 1.02 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110980, upper bound: 0.0108467
time: 1.01 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110980, upper bound: 0.0108467
time: 1.00 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110980, upper bound: 0.0108467
time: 0.97 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0073164, 0.0048494, -0.0073250, 0.0050266, -0.0123431, 0.0121743
1: -0.0060075, -0.0011881, -0.0060108, -0.0011179, -0.0048896, 0.0048227
2: 0.0277567, 0.0397808, 0.0275916, 0.0397952, -0.0120385, 0.0121892
3: -0.0070633, 0.0052212, -0.0072352, 0.0052304, -0.0105092, 0.0106713
4: -0.0054036, 0.0052127, -0.0054099, 0.0054120, -0.0108156, 0.0106226
5: 0.0070017, 0.0164345, 0.0069955, 0.0165819, -0.0095803, 0.0094390
6: -0.0114235, 0.0021734, -0.0116662, 0.0021819, -0.0136053, 0.0138397
7: 0.9667950, 0.9838866, 0.9667782, 0.9841221, -0.0173271, 0.0171084
8: -0.0216352, -0.0009673, -0.0220464, -0.0009580, -0.0206772, 0.0210791
9: -0.0041022, 0.0085205, -0.0041085, 0.0087585, -0.0128607, 0.0126289

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110980, upper bound: 0.0108662
time: 1.16 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110980, upper bound: 0.0108662
time: 1.34 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0077128, 0.0047094, -0.0073222, 0.0049614, -0.0126742, 0.0120315
1: -0.0061645, -0.0012435, -0.0060097, -0.0011437, -0.0050208, 0.0047662
2: 0.0278871, 0.0404499, 0.0276524, 0.0397905, -0.0119034, 0.0127975
3: -0.0069275, 0.0056511, -0.0071719, 0.0052275, -0.0104450, 0.0110636
4: -0.0056994, 0.0050554, -0.0054078, 0.0053386, -0.0110380, 0.0104632
5: 0.0067163, 0.0163181, 0.0069975, 0.0165276, -0.0098114, 0.0093206
6: -0.0112318, 0.0025663, -0.0115768, 0.0021791, -0.0134109, 0.0141431
7: 0.9660155, 0.9837008, 0.9667836, 0.9840354, -0.0180199, 0.0169172
8: -0.0213106, -0.0005368, -0.0218950, -0.0009610, -0.0203495, 0.0213581
9: -0.0043940, 0.0083326, -0.0041064, 0.0086709, -0.0130649, 0.0124390

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110980, upper bound: 0.0108662
time: 1.01 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110980, upper bound: 0.0108662
time: 1.00 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110344, upper bound: 0.0108710
time: 0.92 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110344, upper bound: 0.0108710
time: 1.15 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110344, upper bound: 0.0108710
time: 1.13 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110344, upper bound: 0.0108710
time: 1.15 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108955
time: 1.11 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108955
time: 1.27 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108955
time: 1.03 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108955
time: 1.04 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111036, upper bound: 0.0108023
time: 1.17 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111036, upper bound: 0.0108023
time: 1.40 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111036, upper bound: 0.0108023
time: 1.08 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111036, upper bound: 0.0108023
time: 1.15 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0073164, 0.0048494, -0.0074905, 0.0052772, -0.0125936, 0.0123398
1: -0.0060075, -0.0011881, -0.0060764, -0.0010186, -0.0049889, 0.0048883
2: 0.0277567, 0.0397808, 0.0273582, 0.0400746, -0.0123179, 0.0124226
3: -0.0070633, 0.0052212, -0.0074782, 0.0054100, -0.0107167, 0.0109381
4: -0.0054036, 0.0052127, -0.0055334, 0.0056936, -0.0110972, 0.0107461
5: 0.0070017, 0.0164345, 0.0068763, 0.0167902, -0.0097886, 0.0095582
6: -0.0114235, 0.0021734, -0.0120093, 0.0023459, -0.0137694, 0.0141827
7: 0.9667950, 0.9838866, 0.9664528, 0.9844546, -0.0176597, 0.0174338
8: -0.0216352, -0.0009673, -0.0226274, -0.0007783, -0.0208569, 0.0216602
9: -0.0041022, 0.0085205, -0.0042304, 0.0090948, -0.0131970, 0.0127509

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111035, upper bound: 0.0108165
time: 1.11 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111035, upper bound: 0.0108165
time: 1.20 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0077128, 0.0047094, -0.0074877, 0.0052109, -0.0129237, 0.0121971
1: -0.0061645, -0.0012435, -0.0060753, -0.0010449, -0.0051196, 0.0048318
2: 0.0278871, 0.0404499, 0.0274199, 0.0400699, -0.0121828, 0.0130300
3: -0.0069275, 0.0056511, -0.0074139, 0.0054070, -0.0106524, 0.0113309
4: -0.0056994, 0.0050554, -0.0055314, 0.0056191, -0.0113185, 0.0105867
5: 0.0067163, 0.0163181, 0.0068783, 0.0167351, -0.0100188, 0.0094398
6: -0.0112318, 0.0025663, -0.0119185, 0.0023432, -0.0135750, 0.0144848
7: 0.9660155, 0.9837008, 0.9664581, 0.9843667, -0.0183512, 0.0172427
8: -0.0213106, -0.0005368, -0.0224736, -0.0007813, -0.0205293, 0.0219368
9: -0.0043940, 0.0083326, -0.0042283, 0.0090058, -0.0133999, 0.0125609

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111035, upper bound: 0.0108166
time: 1.06 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111035, upper bound: 0.0108166
time: 1.13 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110344, upper bound: 0.0108641
time: 0.98 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110344, upper bound: 0.0108641
time: 1.19 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110344, upper bound: 0.0108641
time: 0.99 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110344, upper bound: 0.0108641
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108937
time: 1.08 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108937
time: 1.26 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108937
time: 1.12 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108937
time: 1.08 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.47 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0109340, upper bound: 0.0109340
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0109340, upper bound: 0.0109340
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0109340, upper bound: 0.0109340
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0109340, upper bound: 0.0109340
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0109430, upper bound: 0.0108800
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0109430, upper bound: 0.0108800
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0109430, upper bound: 0.0108800
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0109430, upper bound: 0.0108800
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0108800, upper bound: 0.0109430
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0108800, upper bound: 0.0109430
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0108800, upper bound: 0.0109430
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0108800, upper bound: 0.0109430
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0108800, upper bound: 0.0109123
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0108800, upper bound: 0.0109123
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0108800, upper bound: 0.0109123
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0108800, upper bound: 0.0109123
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0108467, upper bound: 0.0110980
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0108467, upper bound: 0.0110980
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0108467, upper bound: 0.0110980
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0108467, upper bound: 0.0110980
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0108710, upper bound: 0.0110344
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0108710, upper bound: 0.0110344
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0108710, upper bound: 0.0110344
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0108710, upper bound: 0.0110344
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0108023, upper bound: 0.0111036
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0108023, upper bound: 0.0111036
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0108023, upper bound: 0.0111036
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0108023, upper bound: 0.0111036
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0108023, upper bound: 0.0110585
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0108023, upper bound: 0.0110585
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0108023, upper bound: 0.0110585
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0108023, upper bound: 0.0110585
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0110980, upper bound: 0.0108467
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0110980, upper bound: 0.0108467
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0110980, upper bound: 0.0108467
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0110980, upper bound: 0.0108467
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0110980, upper bound: 0.0108662
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0110980, upper bound: 0.0108662
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0110980, upper bound: 0.0108662
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0110980, upper bound: 0.0108662
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0110344, upper bound: 0.0108710
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0110344, upper bound: 0.0108710
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0110344, upper bound: 0.0108710
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0110344, upper bound: 0.0108710
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108955
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108955
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108955
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108955
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0111036, upper bound: 0.0108023
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0111036, upper bound: 0.0108023
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0111036, upper bound: 0.0108023
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0111036, upper bound: 0.0108023
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0111035, upper bound: 0.0108165
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0111035, upper bound: 0.0108165
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0111035, upper bound: 0.0108166
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0111035, upper bound: 0.0108166
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0110344, upper bound: 0.0108641
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0110344, upper bound: 0.0108641
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0110344, upper bound: 0.0108641
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0110344, upper bound: 0.0108641
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108937
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108937
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108937
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108937

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0070136, 0.0043946, -0.0070136, 0.0043946, -0.0114081, 0.0114081
1: -0.0058875, -0.0013682, -0.0058875, -0.0013682, -0.0045192, 0.0045192
2: 0.0281803, 0.0392696, 0.0281803, 0.0392696, -0.0110893, 0.0110893
3: -0.0066222, 0.0048927, -0.0066222, 0.0048927, -0.0098384, 0.0098384
4: -0.0051775, 0.0047016, -0.0051775, 0.0047016, -0.0098791, 0.0098791
5: 0.0072197, 0.0160564, 0.0072197, 0.0160564, -0.0088367, 0.0088367
6: -0.0108007, 0.0018732, -0.0108007, 0.0018732, -0.0126740, 0.0126740
7: 0.9673904, 0.9832830, 0.9673904, 0.9832830, -0.0158926, 0.0158926
8: -0.0205806, -0.0012962, -0.0205806, -0.0012962, -0.0192844, 0.0192844
9: -0.0038792, 0.0079100, -0.0038792, 0.0079100, -0.0117892, 0.0117892

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105332, upper bound: 0.0098948
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101530, upper bound: 0.0098948
time: 0.94 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0070136, 0.0043946, -0.0073856, 0.0042302, -0.0112437, 0.0117802
1: -0.0058875, -0.0013682, -0.0060349, -0.0014334, -0.0044541, 0.0046666
2: 0.0281803, 0.0392696, 0.0283335, 0.0398976, -0.0117173, 0.0109361
3: -0.0066222, 0.0048927, -0.0064628, 0.0052963, -0.0102690, 0.0096960
4: -0.0051775, 0.0047016, -0.0054552, 0.0045168, -0.0096943, 0.0101568
5: 0.0072197, 0.0160564, 0.0069518, 0.0159198, -0.0087000, 0.0091046
6: -0.0108007, 0.0018732, -0.0105756, 0.0022420, -0.0130427, 0.0124488
7: 0.9673904, 0.9832830, 0.9666588, 0.9830647, -0.0156742, 0.0166242
8: -0.0205806, -0.0012962, -0.0201993, -0.0008921, -0.0196884, 0.0189031
9: -0.0038792, 0.0079100, -0.0041531, 0.0076893, -0.0115685, 0.0120631

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105332, upper bound: 0.0098948
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101530, upper bound: 0.0098948
time: 0.97 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0073856, 0.0042302, -0.0070136, 0.0043946, -0.0117802, 0.0112437
1: -0.0060349, -0.0014334, -0.0058875, -0.0013682, -0.0046666, 0.0044541
2: 0.0283335, 0.0398976, 0.0281803, 0.0392696, -0.0109361, 0.0117173
3: -0.0064628, 0.0052963, -0.0066222, 0.0048927, -0.0096960, 0.0102690
4: -0.0054552, 0.0045168, -0.0051775, 0.0047016, -0.0101568, 0.0096943
5: 0.0069518, 0.0159198, 0.0072197, 0.0160564, -0.0091046, 0.0087000
6: -0.0105756, 0.0022420, -0.0108007, 0.0018732, -0.0124488, 0.0130427
7: 0.9666588, 0.9830647, 0.9673904, 0.9832830, -0.0166242, 0.0156742
8: -0.0201993, -0.0008921, -0.0205806, -0.0012962, -0.0189031, 0.0196884
9: -0.0041531, 0.0076893, -0.0038792, 0.0079100, -0.0120631, 0.0115685

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104387, upper bound: 0.0098948
time: 1.07 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0098948
time: 0.97 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0073856, 0.0042302, -0.0073856, 0.0042302, -0.0116158, 0.0116158
1: -0.0060349, -0.0014334, -0.0060349, -0.0014334, -0.0046015, 0.0046015
2: 0.0283335, 0.0398976, 0.0283335, 0.0398976, -0.0115642, 0.0115642
3: -0.0064628, 0.0052963, -0.0064628, 0.0052963, -0.0100729, 0.0100729
4: -0.0054552, 0.0045168, -0.0054552, 0.0045168, -0.0099719, 0.0099719
5: 0.0069518, 0.0159198, 0.0069518, 0.0159198, -0.0089679, 0.0089679
6: -0.0105756, 0.0022420, -0.0105756, 0.0022420, -0.0128176, 0.0128176
7: 0.9666588, 0.9830647, 0.9666588, 0.9830647, -0.0164058, 0.0164058
8: -0.0201993, -0.0008921, -0.0201993, -0.0008921, -0.0193071, 0.0193071
9: -0.0041531, 0.0076893, -0.0041531, 0.0076893, -0.0118424, 0.0118424

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104387, upper bound: 0.0098948
time: 1.09 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0098948
time: 1.24 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0070136, 0.0043946, -0.0071739, 0.0046427, -0.0116563, 0.0115685
1: -0.0058875, -0.0013682, -0.0059510, -0.0012699, -0.0046175, 0.0045828
2: 0.0281803, 0.0392696, 0.0279491, 0.0395402, -0.0113599, 0.0113204
3: -0.0066222, 0.0048927, -0.0068629, 0.0050667, -0.0100383, 0.0100986
4: -0.0051775, 0.0047016, -0.0052972, 0.0049805, -0.0101580, 0.0099988
5: 0.0072197, 0.0160564, 0.0071043, 0.0162627, -0.0090430, 0.0089522
6: -0.0108007, 0.0018732, -0.0111406, 0.0020322, -0.0128329, 0.0130138
7: 0.9673904, 0.9832830, 0.9670752, 0.9836124, -0.0162220, 0.0162078
8: -0.0205806, -0.0012962, -0.0211561, -0.0011221, -0.0194585, 0.0198599
9: -0.0038792, 0.0079100, -0.0039972, 0.0082431, -0.0121223, 0.0119072

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106501, upper bound: 0.0098920
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101671, upper bound: 0.0098920
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0070136, 0.0043946, -0.0075451, 0.0044799, -0.0114935, 0.0119397
1: -0.0058875, -0.0013682, -0.0060980, -0.0013344, -0.0045530, 0.0047298
2: 0.0281803, 0.0392696, 0.0281009, 0.0401667, -0.0119864, 0.0111687
3: -0.0066222, 0.0048927, -0.0067050, 0.0054692, -0.0104651, 0.0099622
4: -0.0051775, 0.0047016, -0.0055742, 0.0047974, -0.0099750, 0.0102757
5: 0.0072197, 0.0160564, 0.0068371, 0.0161274, -0.0089076, 0.0092194
6: -0.0108007, 0.0018732, -0.0109175, 0.0024000, -0.0132008, 0.0127908
7: 0.9673904, 0.9832830, 0.9663454, 0.9833962, -0.0160058, 0.0169376
8: -0.0205806, -0.0012962, -0.0207784, -0.0007190, -0.0198616, 0.0194822
9: -0.0038792, 0.0079100, -0.0042705, 0.0080245, -0.0119037, 0.0121805

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106501, upper bound: 0.0098920
time: 1.02 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101671, upper bound: 0.0098920
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0073856, 0.0042302, -0.0071739, 0.0046427, -0.0120283, 0.0114041
1: -0.0060349, -0.0014334, -0.0059510, -0.0012699, -0.0047649, 0.0045176
2: 0.0283335, 0.0398976, 0.0279491, 0.0395402, -0.0112068, 0.0119485
3: -0.0064628, 0.0052963, -0.0068629, 0.0050667, -0.0098959, 0.0105292
4: -0.0054552, 0.0045168, -0.0052972, 0.0049805, -0.0104357, 0.0098139
5: 0.0069518, 0.0159198, 0.0071043, 0.0162627, -0.0093109, 0.0088155
6: -0.0105756, 0.0022420, -0.0111406, 0.0020322, -0.0126078, 0.0133825
7: 0.9666588, 0.9830647, 0.9670752, 0.9836124, -0.0169536, 0.0159895
8: -0.0201993, -0.0008921, -0.0211561, -0.0011221, -0.0190772, 0.0202639
9: -0.0041531, 0.0076893, -0.0039972, 0.0082431, -0.0123962, 0.0116865

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105598, upper bound: 0.0098920
time: 1.10 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0098920
time: 1.06 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0073856, 0.0042302, -0.0075451, 0.0044799, -0.0118655, 0.0117753
1: -0.0060349, -0.0014334, -0.0060980, -0.0013344, -0.0047004, 0.0046647
2: 0.0283335, 0.0398976, 0.0281009, 0.0401667, -0.0118333, 0.0117968
3: -0.0064628, 0.0052963, -0.0067050, 0.0054692, -0.0102711, 0.0103386
4: -0.0054552, 0.0045168, -0.0055742, 0.0047974, -0.0102526, 0.0100909
5: 0.0069518, 0.0159198, 0.0068371, 0.0161274, -0.0091755, 0.0090827
6: -0.0105756, 0.0022420, -0.0109175, 0.0024000, -0.0129756, 0.0131595
7: 0.9666588, 0.9830647, 0.9663454, 0.9833962, -0.0167374, 0.0167193
8: -0.0201993, -0.0008921, -0.0207784, -0.0007190, -0.0194803, 0.0198862
9: -0.0041531, 0.0076893, -0.0042705, 0.0080245, -0.0121776, 0.0119598

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105598, upper bound: 0.0098920
time: 1.13 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0098920
time: 1.03 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0071739, 0.0046427, -0.0070136, 0.0043946, -0.0115685, 0.0116563
1: -0.0059510, -0.0012699, -0.0058875, -0.0013682, -0.0045828, 0.0046175
2: 0.0279491, 0.0395402, 0.0281803, 0.0392696, -0.0113204, 0.0113599
3: -0.0068629, 0.0050667, -0.0066222, 0.0048927, -0.0100986, 0.0100383
4: -0.0052972, 0.0049805, -0.0051775, 0.0047016, -0.0099988, 0.0101580
5: 0.0071043, 0.0162627, 0.0072197, 0.0160564, -0.0089522, 0.0090430
6: -0.0111406, 0.0020322, -0.0108007, 0.0018732, -0.0130138, 0.0128329
7: 0.9670752, 0.9836124, 0.9673904, 0.9832830, -0.0162078, 0.0162220
8: -0.0211561, -0.0011221, -0.0205806, -0.0012962, -0.0198599, 0.0194585
9: -0.0039972, 0.0082431, -0.0038792, 0.0079100, -0.0119072, 0.0121223

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104947, upper bound: 0.0098948
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101312, upper bound: 0.0098948
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0071739, 0.0046427, -0.0073856, 0.0042302, -0.0114041, 0.0120283
1: -0.0059510, -0.0012699, -0.0060349, -0.0014334, -0.0045176, 0.0047649
2: 0.0279491, 0.0395402, 0.0283335, 0.0398976, -0.0119485, 0.0112068
3: -0.0068629, 0.0050667, -0.0064628, 0.0052963, -0.0105292, 0.0098959
4: -0.0052972, 0.0049805, -0.0054552, 0.0045168, -0.0098139, 0.0104357
5: 0.0071043, 0.0162627, 0.0069518, 0.0159198, -0.0088155, 0.0093109
6: -0.0111406, 0.0020322, -0.0105756, 0.0022420, -0.0133825, 0.0126078
7: 0.9670752, 0.9836124, 0.9666588, 0.9830647, -0.0159895, 0.0169536
8: -0.0211561, -0.0011221, -0.0201993, -0.0008921, -0.0202639, 0.0190772
9: -0.0039972, 0.0082431, -0.0041531, 0.0076893, -0.0116865, 0.0123962

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104947, upper bound: 0.0098948
time: 1.02 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101312, upper bound: 0.0098948
time: 1.17 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0075451, 0.0044799, -0.0070136, 0.0043946, -0.0119397, 0.0114935
1: -0.0060980, -0.0013344, -0.0058875, -0.0013682, -0.0047298, 0.0045530
2: 0.0281009, 0.0401667, 0.0281803, 0.0392696, -0.0111687, 0.0119864
3: -0.0067050, 0.0054692, -0.0066222, 0.0048927, -0.0099622, 0.0104651
4: -0.0055742, 0.0047974, -0.0051775, 0.0047016, -0.0102757, 0.0099750
5: 0.0068371, 0.0161274, 0.0072197, 0.0160564, -0.0092194, 0.0089076
6: -0.0109175, 0.0024000, -0.0108007, 0.0018732, -0.0127908, 0.0132008
7: 0.9663454, 0.9833962, 0.9673904, 0.9832830, -0.0169376, 0.0160058
8: -0.0207784, -0.0007190, -0.0205806, -0.0012962, -0.0194822, 0.0198616
9: -0.0042705, 0.0080245, -0.0038792, 0.0079100, -0.0121805, 0.0119037

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0103991, upper bound: 0.0098948
time: 1.11 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098920, upper bound: 0.0098948
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0075451, 0.0044799, -0.0073856, 0.0042302, -0.0117753, 0.0118655
1: -0.0060980, -0.0013344, -0.0060349, -0.0014334, -0.0046647, 0.0047004
2: 0.0281009, 0.0401667, 0.0283335, 0.0398976, -0.0117968, 0.0118333
3: -0.0067050, 0.0054692, -0.0064628, 0.0052963, -0.0103386, 0.0102711
4: -0.0055742, 0.0047974, -0.0054552, 0.0045168, -0.0100909, 0.0102526
5: 0.0068371, 0.0161274, 0.0069518, 0.0159198, -0.0090827, 0.0091755
6: -0.0109175, 0.0024000, -0.0105756, 0.0022420, -0.0131595, 0.0129756
7: 0.9663454, 0.9833962, 0.9666588, 0.9830647, -0.0167193, 0.0167374
8: -0.0207784, -0.0007190, -0.0201993, -0.0008921, -0.0198862, 0.0194803
9: -0.0042705, 0.0080245, -0.0041531, 0.0076893, -0.0119598, 0.0121776

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0103991, upper bound: 0.0098948
time: 1.11 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098920, upper bound: 0.0098948
time: 1.05 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0071739, 0.0046427, -0.0071739, 0.0046427, -0.0118166, 0.0118166
1: -0.0059510, -0.0012699, -0.0059510, -0.0012699, -0.0046811, 0.0046811
2: 0.0279491, 0.0395402, 0.0279491, 0.0395402, -0.0115911, 0.0115911
3: -0.0068629, 0.0050667, -0.0068629, 0.0050667, -0.0102177, 0.0102177
4: -0.0052972, 0.0049805, -0.0052972, 0.0049805, -0.0102777, 0.0102777
5: 0.0071043, 0.0162627, 0.0071043, 0.0162627, -0.0091585, 0.0091585
6: -0.0111406, 0.0020322, -0.0111406, 0.0020322, -0.0131727, 0.0131727
7: 0.9670752, 0.9836124, 0.9670752, 0.9836124, -0.0165372, 0.0165372
8: -0.0211561, -0.0011221, -0.0211561, -0.0011221, -0.0200340, 0.0200340
9: -0.0039972, 0.0082431, -0.0039972, 0.0082431, -0.0122403, 0.0122403

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104972, upper bound: 0.0098920
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101443, upper bound: 0.0098920
time: 1.08 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0071739, 0.0046427, -0.0075451, 0.0044799, -0.0116538, 0.0121878
1: -0.0059510, -0.0012699, -0.0060980, -0.0013344, -0.0046166, 0.0048281
2: 0.0279491, 0.0395402, 0.0281009, 0.0401667, -0.0122176, 0.0114394
3: -0.0068629, 0.0050667, -0.0067050, 0.0054692, -0.0106430, 0.0100777
4: -0.0052972, 0.0049805, -0.0055742, 0.0047974, -0.0100946, 0.0105547
5: 0.0071043, 0.0162627, 0.0068371, 0.0161274, -0.0090231, 0.0094257
6: -0.0111406, 0.0020322, -0.0109175, 0.0024000, -0.0135406, 0.0129497
7: 0.9670752, 0.9836124, 0.9663454, 0.9833962, -0.0163211, 0.0172670
8: -0.0211561, -0.0011221, -0.0207784, -0.0007190, -0.0204371, 0.0196563
9: -0.0039972, 0.0082431, -0.0042705, 0.0080245, -0.0120218, 0.0125136

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104972, upper bound: 0.0098920
time: 1.07 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101443, upper bound: 0.0098920
time: 1.14 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0075451, 0.0044799, -0.0071739, 0.0046427, -0.0121878, 0.0116538
1: -0.0060980, -0.0013344, -0.0059510, -0.0012699, -0.0048281, 0.0046166
2: 0.0281009, 0.0401667, 0.0279491, 0.0395402, -0.0114394, 0.0122176
3: -0.0067050, 0.0054692, -0.0068629, 0.0050667, -0.0100777, 0.0106430
4: -0.0055742, 0.0047974, -0.0052972, 0.0049805, -0.0105547, 0.0100946
5: 0.0068371, 0.0161274, 0.0071043, 0.0162627, -0.0094257, 0.0090231
6: -0.0109175, 0.0024000, -0.0111406, 0.0020322, -0.0129497, 0.0135406
7: 0.9663454, 0.9833962, 0.9670752, 0.9836124, -0.0172670, 0.0163211
8: -0.0207784, -0.0007190, -0.0211561, -0.0011221, -0.0196563, 0.0204371
9: -0.0042705, 0.0080245, -0.0039972, 0.0082431, -0.0125136, 0.0120218

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104017, upper bound: 0.0098920
time: 1.09 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098920, upper bound: 0.0098920
time: 1.12 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0075451, 0.0044799, -0.0075451, 0.0044799, -0.0120250, 0.0120250
1: -0.0060980, -0.0013344, -0.0060980, -0.0013344, -0.0047636, 0.0047636
2: 0.0281009, 0.0401667, 0.0281009, 0.0401667, -0.0120659, 0.0120659
3: -0.0067050, 0.0054692, -0.0067050, 0.0054692, -0.0104494, 0.0104494
4: -0.0055742, 0.0047974, -0.0055742, 0.0047974, -0.0103716, 0.0103716
5: 0.0068371, 0.0161274, 0.0068371, 0.0161274, -0.0092903, 0.0092903
6: -0.0109175, 0.0024000, -0.0109175, 0.0024000, -0.0133176, 0.0133176
7: 0.9663454, 0.9833962, 0.9663454, 0.9833962, -0.0170509, 0.0170509
8: -0.0207784, -0.0007190, -0.0207784, -0.0007190, -0.0200594, 0.0200594
9: -0.0042705, 0.0080245, -0.0042705, 0.0080245, -0.0122950, 0.0122950

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104017, upper bound: 0.0098920
time: 1.16 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098920, upper bound: 0.0098920
time: 1.02 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0070136, 0.0043946, -0.0073164, 0.0048494, -0.0118629, 0.0117110
1: -0.0058875, -0.0013682, -0.0060075, -0.0011881, -0.0046994, 0.0046392
2: 0.0281803, 0.0392696, 0.0277567, 0.0397808, -0.0116005, 0.0115129
3: -0.0066222, 0.0048927, -0.0070633, 0.0052212, -0.0102039, 0.0103016
4: -0.0051775, 0.0047016, -0.0054036, 0.0052127, -0.0103902, 0.0101051
5: 0.0072197, 0.0160564, 0.0070017, 0.0164345, -0.0092148, 0.0090548
6: -0.0108007, 0.0018732, -0.0114235, 0.0021734, -0.0129742, 0.0132967
7: 0.9673904, 0.9832830, 0.9667950, 0.9838866, -0.0164962, 0.0164880
8: -0.0205806, -0.0012962, -0.0216352, -0.0009673, -0.0196133, 0.0203390
9: -0.0038792, 0.0079100, -0.0041022, 0.0085205, -0.0123997, 0.0120122

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104510, upper bound: 0.0101262
time: 1.09 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101112, upper bound: 0.0101262
time: 1.00 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0070136, 0.0043946, -0.0077128, 0.0047094, -0.0117229, 0.0121074
1: -0.0058875, -0.0013682, -0.0061645, -0.0012435, -0.0046439, 0.0047962
2: 0.0281803, 0.0392696, 0.0278871, 0.0404499, -0.0122696, 0.0113825
3: -0.0066222, 0.0048927, -0.0069275, 0.0056511, -0.0106632, 0.0101844
4: -0.0051775, 0.0047016, -0.0056994, 0.0050554, -0.0102329, 0.0104009
5: 0.0072197, 0.0160564, 0.0067163, 0.0163181, -0.0090984, 0.0093402
6: -0.0108007, 0.0018732, -0.0112318, 0.0025663, -0.0133671, 0.0131050
7: 0.9673904, 0.9832830, 0.9660155, 0.9837008, -0.0163104, 0.0172675
8: -0.0205806, -0.0012962, -0.0213106, -0.0005368, -0.0200437, 0.0200144
9: -0.0038792, 0.0079100, -0.0043940, 0.0083326, -0.0122118, 0.0123040

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104510, upper bound: 0.0101262
time: 1.10 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101112, upper bound: 0.0101262
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0073856, 0.0042302, -0.0073164, 0.0048494, -0.0122350, 0.0115466
1: -0.0060349, -0.0014334, -0.0060075, -0.0011881, -0.0048468, 0.0045741
2: 0.0283335, 0.0398976, 0.0277567, 0.0397808, -0.0114474, 0.0121409
3: -0.0064628, 0.0052963, -0.0070633, 0.0052212, -0.0100615, 0.0107323
4: -0.0054552, 0.0045168, -0.0054036, 0.0052127, -0.0106679, 0.0099203
5: 0.0069518, 0.0159198, 0.0070017, 0.0164345, -0.0094827, 0.0089181
6: -0.0105756, 0.0022420, -0.0114235, 0.0021734, -0.0127490, 0.0136655
7: 0.9666588, 0.9830647, 0.9667950, 0.9838866, -0.0172278, 0.0162697
8: -0.0201993, -0.0008921, -0.0216352, -0.0009673, -0.0192320, 0.0207430
9: -0.0041531, 0.0076893, -0.0041022, 0.0085205, -0.0126736, 0.0117915

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0103902, upper bound: 0.0101262
time: 1.07 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0101262
time: 1.21 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0073856, 0.0042302, -0.0077128, 0.0047094, -0.0120950, 0.0119430
1: -0.0060349, -0.0014334, -0.0061645, -0.0012435, -0.0047913, 0.0047311
2: 0.0283335, 0.0398976, 0.0278871, 0.0404499, -0.0121164, 0.0120105
3: -0.0064628, 0.0052963, -0.0069275, 0.0056511, -0.0104597, 0.0105554
4: -0.0054552, 0.0045168, -0.0056994, 0.0050554, -0.0105106, 0.0102161
5: 0.0069518, 0.0159198, 0.0067163, 0.0163181, -0.0093663, 0.0092035
6: -0.0105756, 0.0022420, -0.0112318, 0.0025663, -0.0131419, 0.0134738
7: 0.9666588, 0.9830647, 0.9660155, 0.9837008, -0.0170420, 0.0170491
8: -0.0201993, -0.0008921, -0.0213106, -0.0005368, -0.0196624, 0.0204184
9: -0.0041531, 0.0076893, -0.0043940, 0.0083326, -0.0124857, 0.0120833

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0103902, upper bound: 0.0101262
time: 1.07 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0101262
time: 0.99 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0070136, 0.0043946, -0.0074819, 0.0051025, -0.0121160, 0.0118765
1: -0.0058875, -0.0013682, -0.0060730, -0.0010878, -0.0047997, 0.0047048
2: 0.0281803, 0.0392696, 0.0275209, 0.0400601, -0.0118797, 0.0117486
3: -0.0066222, 0.0048927, -0.0073087, 0.0054006, -0.0104108, 0.0105727
4: -0.0051775, 0.0047016, -0.0055270, 0.0054972, -0.0106747, 0.0102286
5: 0.0072197, 0.0160564, 0.0068825, 0.0166450, -0.0094252, 0.0091739
6: -0.0108007, 0.0018732, -0.0117700, 0.0023374, -0.0131382, 0.0136432
7: 0.9673904, 0.9832830, 0.9664696, 0.9842228, -0.0168324, 0.0168134
8: -0.0205806, -0.0012962, -0.0222222, -0.0007876, -0.0197929, 0.0209261
9: -0.0038792, 0.0079100, -0.0042240, 0.0088603, -0.0127395, 0.0121340

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105589, upper bound: 0.0100928
time: 1.09 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101259, upper bound: 0.0100928
time: 1.01 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0070136, 0.0043946, -0.0078688, 0.0049581, -0.0119717, 0.0122634
1: -0.0058875, -0.0013682, -0.0062263, -0.0011450, -0.0047425, 0.0048581
2: 0.0281803, 0.0392696, 0.0276553, 0.0407133, -0.0125329, 0.0116142
3: -0.0066222, 0.0048927, -0.0071688, 0.0058203, -0.0108673, 0.0104563
4: -0.0051775, 0.0047016, -0.0058158, 0.0053350, -0.0105125, 0.0105174
5: 0.0072197, 0.0160564, 0.0066039, 0.0165250, -0.0093053, 0.0094525
6: -0.0108007, 0.0018732, -0.0115724, 0.0027210, -0.0135217, 0.0134457
7: 0.9673904, 0.9832830, 0.9657087, 0.9840311, -0.0166407, 0.0175743
8: -0.0205806, -0.0012962, -0.0218875, -0.0003674, -0.0202132, 0.0205914
9: -0.0038792, 0.0079100, -0.0045089, 0.0086665, -0.0125457, 0.0124189

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105589, upper bound: 0.0100928
time: 1.08 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101259, upper bound: 0.0100928
time: 1.02 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0073856, 0.0042302, -0.0074819, 0.0051025, -0.0124881, 0.0117121
1: -0.0060349, -0.0014334, -0.0060730, -0.0010878, -0.0049470, 0.0046396
2: 0.0283335, 0.0398976, 0.0275209, 0.0400601, -0.0117266, 0.0123767
3: -0.0064628, 0.0052963, -0.0073087, 0.0054006, -0.0102685, 0.0110033
4: -0.0054552, 0.0045168, -0.0055270, 0.0054972, -0.0109524, 0.0100438
5: 0.0069518, 0.0159198, 0.0068825, 0.0166450, -0.0096931, 0.0090372
6: -0.0105756, 0.0022420, -0.0117700, 0.0023374, -0.0129130, 0.0140120
7: 0.9666588, 0.9830647, 0.9664696, 0.9842228, -0.0175639, 0.0165951
8: -0.0201993, -0.0008921, -0.0222222, -0.0007876, -0.0194116, 0.0213301
9: -0.0041531, 0.0076893, -0.0042240, 0.0088603, -0.0130134, 0.0119133

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105045, upper bound: 0.0100928
time: 1.25 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0100928
time: 0.94 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0073856, 0.0042302, -0.0078688, 0.0049581, -0.0123438, 0.0120990
1: -0.0060349, -0.0014334, -0.0062263, -0.0011450, -0.0048899, 0.0047929
2: 0.0283335, 0.0398976, 0.0276553, 0.0407133, -0.0123798, 0.0122423
3: -0.0064628, 0.0052963, -0.0071688, 0.0058203, -0.0106618, 0.0108259
4: -0.0054552, 0.0045168, -0.0058158, 0.0053350, -0.0107902, 0.0103326
5: 0.0069518, 0.0159198, 0.0066039, 0.0165250, -0.0095731, 0.0093158
6: -0.0105756, 0.0022420, -0.0115724, 0.0027210, -0.0132966, 0.0138144
7: 0.9666588, 0.9830647, 0.9657087, 0.9840311, -0.0173723, 0.0173559
8: -0.0201993, -0.0008921, -0.0218875, -0.0003674, -0.0198319, 0.0209954
9: -0.0041531, 0.0076893, -0.0045089, 0.0086665, -0.0128197, 0.0121982

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105045, upper bound: 0.0100928
time: 1.31 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0100928
time: 1.02 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0071739, 0.0046427, -0.0073164, 0.0048494, -0.0120233, 0.0119592
1: -0.0059510, -0.0012699, -0.0060075, -0.0011881, -0.0047629, 0.0047375
2: 0.0279491, 0.0395402, 0.0277567, 0.0397808, -0.0118317, 0.0117835
3: -0.0068629, 0.0050667, -0.0070633, 0.0052212, -0.0104641, 0.0105015
4: -0.0052972, 0.0049805, -0.0054036, 0.0052127, -0.0105099, 0.0103840
5: 0.0071043, 0.0162627, 0.0070017, 0.0164345, -0.0093303, 0.0092611
6: -0.0111406, 0.0020322, -0.0114235, 0.0021734, -0.0133140, 0.0134556
7: 0.9670752, 0.9836124, 0.9667950, 0.9838866, -0.0168114, 0.0168175
8: -0.0211561, -0.0011221, -0.0216352, -0.0009673, -0.0201888, 0.0205131
9: -0.0039972, 0.0082431, -0.0041022, 0.0085205, -0.0125177, 0.0123453

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104227, upper bound: 0.0101262
time: 1.12 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100930, upper bound: 0.0101262
time: 1.18 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0071739, 0.0046427, -0.0077128, 0.0047094, -0.0118833, 0.0123556
1: -0.0059510, -0.0012699, -0.0061645, -0.0012435, -0.0047075, 0.0048946
2: 0.0279491, 0.0395402, 0.0278871, 0.0404499, -0.0125008, 0.0116531
3: -0.0068629, 0.0050667, -0.0069275, 0.0056511, -0.0109234, 0.0103843
4: -0.0052972, 0.0049805, -0.0056994, 0.0050554, -0.0103526, 0.0106799
5: 0.0071043, 0.0162627, 0.0067163, 0.0163181, -0.0092139, 0.0095465
6: -0.0111406, 0.0020322, -0.0112318, 0.0025663, -0.0137069, 0.0132639
7: 0.9670752, 0.9836124, 0.9660155, 0.9837008, -0.0166256, 0.0175969
8: -0.0211561, -0.0011221, -0.0213106, -0.0005368, -0.0206192, 0.0201885
9: -0.0039972, 0.0082431, -0.0043940, 0.0083326, -0.0123298, 0.0126371

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104227, upper bound: 0.0101262
time: 1.42 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100930, upper bound: 0.0101262
time: 1.26 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0075451, 0.0044799, -0.0073164, 0.0048494, -0.0123944, 0.0117964
1: -0.0060980, -0.0013344, -0.0060075, -0.0011881, -0.0049099, 0.0046730
2: 0.0281009, 0.0401667, 0.0277567, 0.0397808, -0.0116800, 0.0124100
3: -0.0067050, 0.0054692, -0.0070633, 0.0052212, -0.0103277, 0.0109283
4: -0.0055742, 0.0047974, -0.0054036, 0.0052127, -0.0107869, 0.0102010
5: 0.0068371, 0.0161274, 0.0070017, 0.0164345, -0.0095975, 0.0091257
6: -0.0109175, 0.0024000, -0.0114235, 0.0021734, -0.0130910, 0.0138235
7: 0.9663454, 0.9833962, 0.9667950, 0.9838866, -0.0175412, 0.0166013
8: -0.0207784, -0.0007190, -0.0216352, -0.0009673, -0.0198111, 0.0209162
9: -0.0042705, 0.0080245, -0.0041022, 0.0085205, -0.0127910, 0.0121267

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0103486, upper bound: 0.0101262
time: 1.33 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098878, upper bound: 0.0101262
time: 1.18 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0075451, 0.0044799, -0.0077128, 0.0047094, -0.0122544, 0.0121927
1: -0.0060980, -0.0013344, -0.0061645, -0.0012435, -0.0048545, 0.0048301
2: 0.0281009, 0.0401667, 0.0278871, 0.0404499, -0.0123490, 0.0122796
3: -0.0067050, 0.0054692, -0.0069275, 0.0056511, -0.0107255, 0.0107536
4: -0.0055742, 0.0047974, -0.0056994, 0.0050554, -0.0106295, 0.0104968
5: 0.0068371, 0.0161274, 0.0067163, 0.0163181, -0.0094811, 0.0094111
6: -0.0109175, 0.0024000, -0.0112318, 0.0025663, -0.0134839, 0.0136318
7: 0.9663454, 0.9833962, 0.9660155, 0.9837008, -0.0173554, 0.0173807
8: -0.0207784, -0.0007190, -0.0213106, -0.0005368, -0.0202415, 0.0205916
9: -0.0042705, 0.0080245, -0.0043940, 0.0083326, -0.0126031, 0.0124186

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0103486, upper bound: 0.0101262
time: 1.17 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098878, upper bound: 0.0101262
time: 1.10 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0071739, 0.0046427, -0.0074819, 0.0051025, -0.0122764, 0.0121246
1: -0.0059510, -0.0012699, -0.0060730, -0.0010878, -0.0048632, 0.0048031
2: 0.0279491, 0.0395402, 0.0275209, 0.0400601, -0.0121109, 0.0120193
3: -0.0068629, 0.0050667, -0.0073087, 0.0054006, -0.0105888, 0.0106902
4: -0.0052972, 0.0049805, -0.0055270, 0.0054972, -0.0107944, 0.0105075
5: 0.0071043, 0.0162627, 0.0068825, 0.0166450, -0.0095407, 0.0093802
6: -0.0111406, 0.0020322, -0.0117700, 0.0023374, -0.0134780, 0.0138022
7: 0.9670752, 0.9836124, 0.9664696, 0.9842228, -0.0171476, 0.0171428
8: -0.0211561, -0.0011221, -0.0222222, -0.0007876, -0.0203684, 0.0211002
9: -0.0039972, 0.0082431, -0.0042240, 0.0088603, -0.0128576, 0.0124671

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104256, upper bound: 0.0100928
time: 1.13 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101060, upper bound: 0.0100928
time: 1.40 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0071739, 0.0046427, -0.0078688, 0.0049581, -0.0121320, 0.0125116
1: -0.0059510, -0.0012699, -0.0062263, -0.0011450, -0.0048060, 0.0049564
2: 0.0279491, 0.0395402, 0.0276553, 0.0407133, -0.0127641, 0.0118849
3: -0.0068629, 0.0050667, -0.0071688, 0.0058203, -0.0110462, 0.0105708
4: -0.0052972, 0.0049805, -0.0058158, 0.0053350, -0.0106322, 0.0107963
5: 0.0071043, 0.0162627, 0.0066039, 0.0165250, -0.0094207, 0.0096588
6: -0.0111406, 0.0020322, -0.0115724, 0.0027210, -0.0138615, 0.0136046
7: 0.9670752, 0.9836124, 0.9657087, 0.9840311, -0.0169560, 0.0179037
8: -0.0211561, -0.0011221, -0.0218875, -0.0003674, -0.0207887, 0.0207655
9: -0.0039972, 0.0082431, -0.0045089, 0.0086665, -0.0126638, 0.0127520

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104256, upper bound: 0.0100928
time: 1.07 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101060, upper bound: 0.0100928
time: 1.34 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0075451, 0.0044799, -0.0074819, 0.0051025, -0.0126475, 0.0119618
1: -0.0060980, -0.0013344, -0.0060730, -0.0010878, -0.0050102, 0.0047386
2: 0.0281009, 0.0401667, 0.0275209, 0.0400601, -0.0119592, 0.0126458
3: -0.0067050, 0.0054692, -0.0073087, 0.0054006, -0.0104488, 0.0111154
4: -0.0055742, 0.0047974, -0.0055270, 0.0054972, -0.0110714, 0.0103245
5: 0.0068371, 0.0161274, 0.0068825, 0.0166450, -0.0098079, 0.0092448
6: -0.0109175, 0.0024000, -0.0117700, 0.0023374, -0.0132550, 0.0141701
7: 0.9663454, 0.9833962, 0.9664696, 0.9842228, -0.0178774, 0.0169266
8: -0.0207784, -0.0007190, -0.0222222, -0.0007876, -0.0199907, 0.0215033
9: -0.0042705, 0.0080245, -0.0042240, 0.0088603, -0.0131308, 0.0122485

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0103513, upper bound: 0.0100928
time: 1.21 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098919, upper bound: 0.0100928
time: 1.28 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0075451, 0.0044799, -0.0078688, 0.0049581, -0.0125032, 0.0123488
1: -0.0060980, -0.0013344, -0.0062263, -0.0011450, -0.0049530, 0.0048919
2: 0.0281009, 0.0401667, 0.0276553, 0.0407133, -0.0126124, 0.0125114
3: -0.0067050, 0.0054692, -0.0071688, 0.0058203, -0.0108418, 0.0109347
4: -0.0055742, 0.0047974, -0.0058158, 0.0053350, -0.0109092, 0.0106133
5: 0.0068371, 0.0161274, 0.0066039, 0.0165250, -0.0096879, 0.0095234
6: -0.0109175, 0.0024000, -0.0115724, 0.0027210, -0.0136385, 0.0139725
7: 0.9663454, 0.9833962, 0.9657087, 0.9840311, -0.0176858, 0.0176875
8: -0.0207784, -0.0007190, -0.0218875, -0.0003674, -0.0204110, 0.0211685
9: -0.0042705, 0.0080245, -0.0045089, 0.0086665, -0.0129371, 0.0125335

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0103513, upper bound: 0.0100928
time: 1.61 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098919, upper bound: 0.0100928
time: 1.48 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0073164, 0.0048494, -0.0070136, 0.0043946, -0.0117110, 0.0118629
1: -0.0060075, -0.0011881, -0.0058875, -0.0013682, -0.0046392, 0.0046994
2: 0.0277567, 0.0397808, 0.0281803, 0.0392696, -0.0115129, 0.0116005
3: -0.0070633, 0.0052212, -0.0066222, 0.0048927, -0.0103016, 0.0102039
4: -0.0054036, 0.0052127, -0.0051775, 0.0047016, -0.0101051, 0.0103902
5: 0.0070017, 0.0164345, 0.0072197, 0.0160564, -0.0090548, 0.0092148
6: -0.0114235, 0.0021734, -0.0108007, 0.0018732, -0.0132967, 0.0129742
7: 0.9667950, 0.9838866, 0.9673904, 0.9832830, -0.0164880, 0.0164962
8: -0.0216352, -0.0009673, -0.0205806, -0.0012962, -0.0203390, 0.0196133
9: -0.0041022, 0.0085205, -0.0038792, 0.0079100, -0.0120122, 0.0123997

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107712, upper bound: 0.0098948
time: 1.08 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104737, upper bound: 0.0098948
time: 1.10 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0073164, 0.0048494, -0.0073856, 0.0042302, -0.0115466, 0.0122350
1: -0.0060075, -0.0011881, -0.0060349, -0.0014334, -0.0045741, 0.0048468
2: 0.0277567, 0.0397808, 0.0283335, 0.0398976, -0.0121409, 0.0114474
3: -0.0070633, 0.0052212, -0.0064628, 0.0052963, -0.0107323, 0.0100615
4: -0.0054036, 0.0052127, -0.0054552, 0.0045168, -0.0099203, 0.0106679
5: 0.0070017, 0.0164345, 0.0069518, 0.0159198, -0.0089181, 0.0094827
6: -0.0114235, 0.0021734, -0.0105756, 0.0022420, -0.0136655, 0.0127490
7: 0.9667950, 0.9838866, 0.9666588, 0.9830647, -0.0162697, 0.0172278
8: -0.0216352, -0.0009673, -0.0201993, -0.0008921, -0.0207430, 0.0192320
9: -0.0041022, 0.0085205, -0.0041531, 0.0076893, -0.0117915, 0.0126736

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107712, upper bound: 0.0098948
time: 0.98 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104737, upper bound: 0.0098948
time: 0.99 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0077128, 0.0047094, -0.0070136, 0.0043946, -0.0121074, 0.0117229
1: -0.0061645, -0.0012435, -0.0058875, -0.0013682, -0.0047962, 0.0046439
2: 0.0278871, 0.0404499, 0.0281803, 0.0392696, -0.0113825, 0.0122696
3: -0.0069275, 0.0056511, -0.0066222, 0.0048927, -0.0101844, 0.0106632
4: -0.0056994, 0.0050554, -0.0051775, 0.0047016, -0.0104009, 0.0102329
5: 0.0067163, 0.0163181, 0.0072197, 0.0160564, -0.0093402, 0.0090984
6: -0.0112318, 0.0025663, -0.0108007, 0.0018732, -0.0131050, 0.0133671
7: 0.9660155, 0.9837008, 0.9673904, 0.9832830, -0.0172675, 0.0163104
8: -0.0213106, -0.0005368, -0.0205806, -0.0012962, -0.0200144, 0.0200437
9: -0.0043940, 0.0083326, -0.0038792, 0.0079100, -0.0123040, 0.0122118

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106214, upper bound: 0.0098948
time: 1.01 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0098948
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0077128, 0.0047094, -0.0073856, 0.0042302, -0.0119430, 0.0120950
1: -0.0061645, -0.0012435, -0.0060349, -0.0014334, -0.0047311, 0.0047913
2: 0.0278871, 0.0404499, 0.0283335, 0.0398976, -0.0120105, 0.0121164
3: -0.0069275, 0.0056511, -0.0064628, 0.0052963, -0.0105554, 0.0104597
4: -0.0056994, 0.0050554, -0.0054552, 0.0045168, -0.0102161, 0.0105106
5: 0.0067163, 0.0163181, 0.0069518, 0.0159198, -0.0092035, 0.0093663
6: -0.0112318, 0.0025663, -0.0105756, 0.0022420, -0.0134738, 0.0131419
7: 0.9660155, 0.9837008, 0.9666588, 0.9830647, -0.0170491, 0.0170420
8: -0.0213106, -0.0005368, -0.0201993, -0.0008921, -0.0204184, 0.0196624
9: -0.0043940, 0.0083326, -0.0041531, 0.0076893, -0.0120833, 0.0124857

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106214, upper bound: 0.0098948
time: 1.13 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0098948
time: 1.11 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0073164, 0.0048494, -0.0073164, 0.0048494, -0.0121658, 0.0121658
1: -0.0060075, -0.0011881, -0.0060075, -0.0011881, -0.0048194, 0.0048194
2: 0.0277567, 0.0397808, 0.0277567, 0.0397808, -0.0120241, 0.0120241
3: -0.0070633, 0.0052212, -0.0070633, 0.0052212, -0.0105000, 0.0105000
4: -0.0054036, 0.0052127, -0.0054036, 0.0052127, -0.0106163, 0.0106163
5: 0.0070017, 0.0164345, 0.0070017, 0.0164345, -0.0094329, 0.0094329
6: -0.0114235, 0.0021734, -0.0114235, 0.0021734, -0.0135969, 0.0135969
7: 0.9667950, 0.9838866, 0.9667950, 0.9838866, -0.0170916, 0.0170916
8: -0.0216352, -0.0009673, -0.0216352, -0.0009673, -0.0206679, 0.0206679
9: -0.0041022, 0.0085205, -0.0041022, 0.0085205, -0.0126227, 0.0126227

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107712, upper bound: 0.0100928
time: 1.03 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104737, upper bound: 0.0100825
time: 1.07 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0073164, 0.0048494, -0.0077128, 0.0047094, -0.0120258, 0.0125622
1: -0.0060075, -0.0011881, -0.0061645, -0.0012435, -0.0047639, 0.0049764
2: 0.0277567, 0.0397808, 0.0278871, 0.0404499, -0.0126932, 0.0118937
3: -0.0070633, 0.0052212, -0.0069275, 0.0056511, -0.0109541, 0.0103766
4: -0.0054036, 0.0052127, -0.0056994, 0.0050554, -0.0104589, 0.0109121
5: 0.0070017, 0.0164345, 0.0067163, 0.0163181, -0.0093165, 0.0097183
6: -0.0114235, 0.0021734, -0.0112318, 0.0025663, -0.0139898, 0.0134052
7: 0.9667950, 0.9838866, 0.9660155, 0.9837008, -0.0169058, 0.0178711
8: -0.0216352, -0.0009673, -0.0213106, -0.0005368, -0.0210984, 0.0203433
9: -0.0041022, 0.0085205, -0.0043940, 0.0083326, -0.0124348, 0.0129145

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107712, upper bound: 0.0100928
time: 0.97 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104737, upper bound: 0.0100825
time: 1.09 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0077128, 0.0047094, -0.0073164, 0.0048494, -0.0125622, 0.0120258
1: -0.0061645, -0.0012435, -0.0060075, -0.0011881, -0.0049764, 0.0047639
2: 0.0278871, 0.0404499, 0.0277567, 0.0397808, -0.0118937, 0.0126932
3: -0.0069275, 0.0056511, -0.0070633, 0.0052212, -0.0103765, 0.0109541
4: -0.0056994, 0.0050554, -0.0054036, 0.0052127, -0.0109121, 0.0104589
5: 0.0067163, 0.0163181, 0.0070017, 0.0164345, -0.0097183, 0.0093165
6: -0.0112318, 0.0025663, -0.0114235, 0.0021734, -0.0134052, 0.0139898
7: 0.9660155, 0.9837008, 0.9667950, 0.9838866, -0.0178711, 0.0169058
8: -0.0213106, -0.0005368, -0.0216352, -0.0009673, -0.0203433, 0.0210984
9: -0.0043940, 0.0083326, -0.0041022, 0.0085205, -0.0129145, 0.0124348

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106214, upper bound: 0.0100658
time: 0.98 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0100500
time: 0.98 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0077128, 0.0047094, -0.0077128, 0.0047094, -0.0124222, 0.0124222
1: -0.0061645, -0.0012435, -0.0061645, -0.0012435, -0.0049210, 0.0049210
2: 0.0278871, 0.0404499, 0.0278871, 0.0404499, -0.0125628, 0.0125628
3: -0.0069275, 0.0056511, -0.0069275, 0.0056511, -0.0107727, 0.0107727
4: -0.0056994, 0.0050554, -0.0056994, 0.0050554, -0.0107547, 0.0107547
5: 0.0067163, 0.0163181, 0.0067163, 0.0163181, -0.0096019, 0.0096019
6: -0.0112318, 0.0025663, -0.0112318, 0.0025663, -0.0137981, 0.0137981
7: 0.9660155, 0.9837008, 0.9660155, 0.9837008, -0.0176853, 0.0176853
8: -0.0213106, -0.0005368, -0.0213106, -0.0005368, -0.0207737, 0.0207737
9: -0.0043940, 0.0083326, -0.0043940, 0.0083326, -0.0127266, 0.0127266

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106214, upper bound: 0.0100658
time: 1.00 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0100500
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0074819, 0.0051025, -0.0070136, 0.0043946, -0.0118765, 0.0121160
1: -0.0060730, -0.0010878, -0.0058875, -0.0013682, -0.0047048, 0.0047997
2: 0.0275209, 0.0400601, 0.0281803, 0.0392696, -0.0117486, 0.0118797
3: -0.0073087, 0.0054006, -0.0066222, 0.0048927, -0.0105727, 0.0104108
4: -0.0055270, 0.0054972, -0.0051775, 0.0047016, -0.0102286, 0.0106747
5: 0.0068825, 0.0166450, 0.0072197, 0.0160564, -0.0091739, 0.0094252
6: -0.0117700, 0.0023374, -0.0108007, 0.0018732, -0.0136432, 0.0131382
7: 0.9664696, 0.9842228, 0.9673904, 0.9832830, -0.0168134, 0.0168324
8: -0.0222222, -0.0007876, -0.0205806, -0.0012962, -0.0209261, 0.0197929
9: -0.0042240, 0.0088603, -0.0038792, 0.0079100, -0.0121340, 0.0127395

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107108, upper bound: 0.0098948
time: 1.03 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104304, upper bound: 0.0098948
time: 1.04 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0074819, 0.0051025, -0.0073856, 0.0042302, -0.0117121, 0.0124881
1: -0.0060730, -0.0010878, -0.0060349, -0.0014334, -0.0046396, 0.0049470
2: 0.0275209, 0.0400601, 0.0283335, 0.0398976, -0.0123767, 0.0117266
3: -0.0073087, 0.0054006, -0.0064628, 0.0052963, -0.0110033, 0.0102685
4: -0.0055270, 0.0054972, -0.0054552, 0.0045168, -0.0100438, 0.0109524
5: 0.0068825, 0.0166450, 0.0069518, 0.0159198, -0.0090372, 0.0096931
6: -0.0117700, 0.0023374, -0.0105756, 0.0022420, -0.0140120, 0.0129130
7: 0.9664696, 0.9842228, 0.9666588, 0.9830647, -0.0165951, 0.0175639
8: -0.0222222, -0.0007876, -0.0201993, -0.0008921, -0.0213301, 0.0194116
9: -0.0042240, 0.0088603, -0.0041531, 0.0076893, -0.0119133, 0.0130134

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107108, upper bound: 0.0098948
time: 1.07 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104304, upper bound: 0.0098948
time: 1.04 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0078688, 0.0049581, -0.0070136, 0.0043946, -0.0122634, 0.0119717
1: -0.0062263, -0.0011450, -0.0058875, -0.0013682, -0.0048581, 0.0047425
2: 0.0276553, 0.0407133, 0.0281803, 0.0392696, -0.0116142, 0.0125329
3: -0.0071688, 0.0058203, -0.0066222, 0.0048927, -0.0104563, 0.0108673
4: -0.0058158, 0.0053350, -0.0051775, 0.0047016, -0.0105174, 0.0105125
5: 0.0066039, 0.0165250, 0.0072197, 0.0160564, -0.0094525, 0.0093053
6: -0.0115724, 0.0027210, -0.0108007, 0.0018732, -0.0134457, 0.0135217
7: 0.9657087, 0.9840311, 0.9673904, 0.9832830, -0.0175743, 0.0166407
8: -0.0218875, -0.0003674, -0.0205806, -0.0012962, -0.0205914, 0.0202132
9: -0.0045089, 0.0086665, -0.0038792, 0.0079100, -0.0124189, 0.0125457

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105697, upper bound: 0.0098948
time: 1.20 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0098948
time: 1.07 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0078688, 0.0049581, -0.0073856, 0.0042302, -0.0120990, 0.0123438
1: -0.0062263, -0.0011450, -0.0060349, -0.0014334, -0.0047929, 0.0048899
2: 0.0276553, 0.0407133, 0.0283335, 0.0398976, -0.0122423, 0.0123798
3: -0.0071688, 0.0058203, -0.0064628, 0.0052963, -0.0108259, 0.0106618
4: -0.0058158, 0.0053350, -0.0054552, 0.0045168, -0.0103326, 0.0107902
5: 0.0066039, 0.0165250, 0.0069518, 0.0159198, -0.0093158, 0.0095731
6: -0.0115724, 0.0027210, -0.0105756, 0.0022420, -0.0138144, 0.0132966
7: 0.9657087, 0.9840311, 0.9666588, 0.9830647, -0.0173559, 0.0173723
8: -0.0218875, -0.0003674, -0.0201993, -0.0008921, -0.0209954, 0.0198319
9: -0.0045089, 0.0086665, -0.0041531, 0.0076893, -0.0121982, 0.0128197

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105697, upper bound: 0.0098948
time: 1.01 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0098948
time: 0.92 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0074819, 0.0051025, -0.0073164, 0.0048494, -0.0123312, 0.0124189
1: -0.0060730, -0.0010878, -0.0060075, -0.0011881, -0.0048849, 0.0049196
2: 0.0275209, 0.0400601, 0.0277567, 0.0397808, -0.0122599, 0.0123034
3: -0.0073087, 0.0054006, -0.0070633, 0.0052212, -0.0107670, 0.0107072
4: -0.0055270, 0.0054972, -0.0054036, 0.0052127, -0.0107397, 0.0109008
5: 0.0068825, 0.0166450, 0.0070017, 0.0164345, -0.0095520, 0.0096433
6: -0.0117700, 0.0023374, -0.0114235, 0.0021734, -0.0139435, 0.0137609
7: 0.9664696, 0.9842228, 0.9667950, 0.9838866, -0.0174170, 0.0174278
8: -0.0222222, -0.0007876, -0.0216352, -0.0009673, -0.0212550, 0.0208476
9: -0.0042240, 0.0088603, -0.0041022, 0.0085205, -0.0127445, 0.0129625

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107108, upper bound: 0.0100987
time: 1.12 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104304, upper bound: 0.0100926
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0074819, 0.0051025, -0.0077128, 0.0047094, -0.0121913, 0.0128153
1: -0.0060730, -0.0010878, -0.0061645, -0.0012435, -0.0048295, 0.0050767
2: 0.0275209, 0.0400601, 0.0278871, 0.0404499, -0.0129290, 0.0121730
3: -0.0073087, 0.0054006, -0.0069275, 0.0056511, -0.0112211, 0.0105837
4: -0.0055270, 0.0054972, -0.0056994, 0.0050554, -0.0105824, 0.0111966
5: 0.0068825, 0.0166450, 0.0067163, 0.0163181, -0.0094356, 0.0099287
6: -0.0117700, 0.0023374, -0.0112318, 0.0025663, -0.0143363, 0.0135692
7: 0.9664696, 0.9842228, 0.9660155, 0.9837008, -0.0172312, 0.0182073
8: -0.0222222, -0.0007876, -0.0213106, -0.0005368, -0.0216854, 0.0205229
9: -0.0042240, 0.0088603, -0.0043940, 0.0083326, -0.0125566, 0.0132544

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107108, upper bound: 0.0100987
time: 1.31 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104304, upper bound: 0.0100926
time: 0.98 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0078688, 0.0049581, -0.0073164, 0.0048494, -0.0127182, 0.0122746
1: -0.0062263, -0.0011450, -0.0060075, -0.0011881, -0.0050382, 0.0048625
2: 0.0276553, 0.0407133, 0.0277567, 0.0397808, -0.0121255, 0.0129565
3: -0.0071688, 0.0058203, -0.0070633, 0.0052212, -0.0106451, 0.0111567
4: -0.0058158, 0.0053350, -0.0054036, 0.0052127, -0.0110285, 0.0107386
5: 0.0066039, 0.0165250, 0.0070017, 0.0164345, -0.0098306, 0.0095233
6: -0.0115724, 0.0027210, -0.0114235, 0.0021734, -0.0137459, 0.0141444
7: 0.9657087, 0.9840311, 0.9667950, 0.9838866, -0.0181779, 0.0172362
8: -0.0218875, -0.0003674, -0.0216352, -0.0009673, -0.0209203, 0.0212678
9: -0.0045089, 0.0086665, -0.0041022, 0.0085205, -0.0130294, 0.0127687

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105697, upper bound: 0.0100734
time: 1.06 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0100654
time: 1.16 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0078688, 0.0049581, -0.0077128, 0.0047094, -0.0125782, 0.0126710
1: -0.0062263, -0.0011450, -0.0061645, -0.0012435, -0.0049828, 0.0050195
2: 0.0276553, 0.0407133, 0.0278871, 0.0404499, -0.0127946, 0.0128262
3: -0.0071688, 0.0058203, -0.0069275, 0.0056511, -0.0110406, 0.0109744
4: -0.0058158, 0.0053350, -0.0056994, 0.0050554, -0.0108712, 0.0110344
5: 0.0066039, 0.0165250, 0.0067163, 0.0163181, -0.0097142, 0.0098087
6: -0.0115724, 0.0027210, -0.0112318, 0.0025663, -0.0141387, 0.0139527
7: 0.9657087, 0.9840311, 0.9660155, 0.9837008, -0.0179921, 0.0180156
8: -0.0218875, -0.0003674, -0.0213106, -0.0005368, -0.0213507, 0.0209432
9: -0.0045089, 0.0086665, -0.0043940, 0.0083326, -0.0128415, 0.0130606

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105697, upper bound: 0.0100734
time: 0.99 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0100654
time: 1.10 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0073164, 0.0048494, -0.0071739, 0.0046427, -0.0119592, 0.0120233
1: -0.0060075, -0.0011881, -0.0059510, -0.0012699, -0.0047375, 0.0047629
2: 0.0277567, 0.0397808, 0.0279491, 0.0395402, -0.0117835, 0.0118317
3: -0.0070633, 0.0052212, -0.0068629, 0.0050667, -0.0105015, 0.0104641
4: -0.0054036, 0.0052127, -0.0052972, 0.0049805, -0.0103840, 0.0105099
5: 0.0070017, 0.0164345, 0.0071043, 0.0162627, -0.0092611, 0.0093303
6: -0.0114235, 0.0021734, -0.0111406, 0.0020322, -0.0134556, 0.0133140
7: 0.9667950, 0.9838866, 0.9670752, 0.9836124, -0.0168175, 0.0168114
8: -0.0216352, -0.0009673, -0.0211561, -0.0011221, -0.0205131, 0.0201888
9: -0.0041022, 0.0085205, -0.0039972, 0.0082431, -0.0123453, 0.0125177

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108766, upper bound: 0.0098920
time: 1.08 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104913, upper bound: 0.0098920
time: 1.01 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0073164, 0.0048494, -0.0075451, 0.0044799, -0.0117964, 0.0123944
1: -0.0060075, -0.0011881, -0.0060980, -0.0013344, -0.0046730, 0.0049099
2: 0.0277567, 0.0397808, 0.0281009, 0.0401667, -0.0124100, 0.0116800
3: -0.0070633, 0.0052212, -0.0067050, 0.0054692, -0.0109283, 0.0103277
4: -0.0054036, 0.0052127, -0.0055742, 0.0047974, -0.0102010, 0.0107869
5: 0.0070017, 0.0164345, 0.0068371, 0.0161274, -0.0091257, 0.0095975
6: -0.0114235, 0.0021734, -0.0109175, 0.0024000, -0.0138235, 0.0130910
7: 0.9667950, 0.9838866, 0.9663454, 0.9833962, -0.0166013, 0.0175412
8: -0.0216352, -0.0009673, -0.0207784, -0.0007190, -0.0209162, 0.0198111
9: -0.0041022, 0.0085205, -0.0042705, 0.0080245, -0.0121267, 0.0127910

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108766, upper bound: 0.0098920
time: 1.03 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104913, upper bound: 0.0098920
time: 1.15 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0077128, 0.0047094, -0.0071739, 0.0046427, -0.0123556, 0.0118833
1: -0.0061645, -0.0012435, -0.0059510, -0.0012699, -0.0048946, 0.0047075
2: 0.0278871, 0.0404499, 0.0279491, 0.0395402, -0.0116531, 0.0125008
3: -0.0069275, 0.0056511, -0.0068629, 0.0050667, -0.0103843, 0.0109234
4: -0.0056994, 0.0050554, -0.0052972, 0.0049805, -0.0106799, 0.0103526
5: 0.0067163, 0.0163181, 0.0071043, 0.0162627, -0.0095465, 0.0092139
6: -0.0112318, 0.0025663, -0.0111406, 0.0020322, -0.0132639, 0.0137069
7: 0.9660155, 0.9837008, 0.9670752, 0.9836124, -0.0175969, 0.0166256
8: -0.0213106, -0.0005368, -0.0211561, -0.0011221, -0.0201885, 0.0206192
9: -0.0043940, 0.0083326, -0.0039972, 0.0082431, -0.0126371, 0.0123298

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107205, upper bound: 0.0098920
time: 1.02 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0098878
time: 1.03 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0077128, 0.0047094, -0.0075451, 0.0044799, -0.0121927, 0.0122544
1: -0.0061645, -0.0012435, -0.0060980, -0.0013344, -0.0048301, 0.0048545
2: 0.0278871, 0.0404499, 0.0281009, 0.0401667, -0.0122796, 0.0123490
3: -0.0069275, 0.0056511, -0.0067050, 0.0054692, -0.0107536, 0.0107255
4: -0.0056994, 0.0050554, -0.0055742, 0.0047974, -0.0104968, 0.0106295
5: 0.0067163, 0.0163181, 0.0068371, 0.0161274, -0.0094111, 0.0094811
6: -0.0112318, 0.0025663, -0.0109175, 0.0024000, -0.0136318, 0.0134839
7: 0.9660155, 0.9837008, 0.9663454, 0.9833962, -0.0173807, 0.0173554
8: -0.0213106, -0.0005368, -0.0207784, -0.0007190, -0.0205916, 0.0202415
9: -0.0043940, 0.0083326, -0.0042705, 0.0080245, -0.0124186, 0.0126031

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107205, upper bound: 0.0098920
time: 1.07 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0098878
time: 0.98 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0073164, 0.0048494, -0.0074819, 0.0051025, -0.0124189, 0.0123312
1: -0.0060075, -0.0011881, -0.0060730, -0.0010878, -0.0049196, 0.0048849
2: 0.0277567, 0.0397808, 0.0275209, 0.0400601, -0.0123034, 0.0122599
3: -0.0070633, 0.0052212, -0.0073087, 0.0054006, -0.0107072, 0.0107670
4: -0.0054036, 0.0052127, -0.0055270, 0.0054972, -0.0109008, 0.0107397
5: 0.0070017, 0.0164345, 0.0068825, 0.0166450, -0.0096433, 0.0095520
6: -0.0114235, 0.0021734, -0.0117700, 0.0023374, -0.0137609, 0.0139435
7: 0.9667950, 0.9838866, 0.9664696, 0.9842228, -0.0174278, 0.0174170
8: -0.0216352, -0.0009673, -0.0222222, -0.0007876, -0.0208476, 0.0212550
9: -0.0041022, 0.0085205, -0.0042240, 0.0088603, -0.0129625, 0.0127445

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108766, upper bound: 0.0100628
time: 1.04 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104913, upper bound: 0.0100514
time: 1.15 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0073164, 0.0048494, -0.0078688, 0.0049581, -0.0122746, 0.0127182
1: -0.0060075, -0.0011881, -0.0062263, -0.0011450, -0.0048625, 0.0050382
2: 0.0277567, 0.0397808, 0.0276553, 0.0407133, -0.0129565, 0.0121255
3: -0.0070633, 0.0052212, -0.0071688, 0.0058203, -0.0111567, 0.0106451
4: -0.0054036, 0.0052127, -0.0058158, 0.0053350, -0.0107386, 0.0110285
5: 0.0070017, 0.0164345, 0.0066039, 0.0165250, -0.0095233, 0.0098306
6: -0.0114235, 0.0021734, -0.0115724, 0.0027210, -0.0141444, 0.0137459
7: 0.9667950, 0.9838866, 0.9657087, 0.9840311, -0.0172362, 0.0181779
8: -0.0216352, -0.0009673, -0.0218875, -0.0003674, -0.0212678, 0.0209203
9: -0.0041022, 0.0085205, -0.0045089, 0.0086665, -0.0127687, 0.0130294

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108766, upper bound: 0.0100628
time: 1.06 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104913, upper bound: 0.0100514
time: 1.17 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0077128, 0.0047094, -0.0074819, 0.0051025, -0.0128153, 0.0121913
1: -0.0061645, -0.0012435, -0.0060730, -0.0010878, -0.0050767, 0.0048295
2: 0.0278871, 0.0404499, 0.0275209, 0.0400601, -0.0121730, 0.0129290
3: -0.0069275, 0.0056511, -0.0073087, 0.0054006, -0.0105837, 0.0112211
4: -0.0056994, 0.0050554, -0.0055270, 0.0054972, -0.0111966, 0.0105824
5: 0.0067163, 0.0163181, 0.0068825, 0.0166450, -0.0099287, 0.0094356
6: -0.0112318, 0.0025663, -0.0117700, 0.0023374, -0.0135692, 0.0143363
7: 0.9660155, 0.9837008, 0.9664696, 0.9842228, -0.0182073, 0.0172312
8: -0.0213106, -0.0005368, -0.0222222, -0.0007876, -0.0205229, 0.0216854
9: -0.0043940, 0.0083326, -0.0042240, 0.0088603, -0.0132544, 0.0125566

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107205, upper bound: 0.0100364
time: 1.02 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0100220
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0077128, 0.0047094, -0.0078688, 0.0049581, -0.0126710, 0.0125782
1: -0.0061645, -0.0012435, -0.0062263, -0.0011450, -0.0050195, 0.0049828
2: 0.0278871, 0.0404499, 0.0276553, 0.0407133, -0.0128262, 0.0127946
3: -0.0069275, 0.0056511, -0.0071688, 0.0058203, -0.0109744, 0.0110406
4: -0.0056994, 0.0050554, -0.0058158, 0.0053350, -0.0110344, 0.0108712
5: 0.0067163, 0.0163181, 0.0066039, 0.0165250, -0.0098087, 0.0097142
6: -0.0112318, 0.0025663, -0.0115724, 0.0027210, -0.0139527, 0.0141387
7: 0.9660155, 0.9837008, 0.9657087, 0.9840311, -0.0180156, 0.0179921
8: -0.0213106, -0.0005368, -0.0218875, -0.0003674, -0.0209432, 0.0213507
9: -0.0043940, 0.0083326, -0.0045089, 0.0086665, -0.0130606, 0.0128415

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107205, upper bound: 0.0100364
time: 1.17 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0100220
time: 0.98 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0074819, 0.0051025, -0.0071739, 0.0046427, -0.0121246, 0.0122764
1: -0.0060730, -0.0010878, -0.0059510, -0.0012699, -0.0048031, 0.0048632
2: 0.0275209, 0.0400601, 0.0279491, 0.0395402, -0.0120193, 0.0121109
3: -0.0073087, 0.0054006, -0.0068629, 0.0050667, -0.0106902, 0.0105888
4: -0.0055270, 0.0054972, -0.0052972, 0.0049805, -0.0105075, 0.0107944
5: 0.0068825, 0.0166450, 0.0071043, 0.0162627, -0.0093802, 0.0095407
6: -0.0117700, 0.0023374, -0.0111406, 0.0020322, -0.0138022, 0.0134780
7: 0.9664696, 0.9842228, 0.9670752, 0.9836124, -0.0171428, 0.0171476
8: -0.0222222, -0.0007876, -0.0211561, -0.0011221, -0.0211002, 0.0203684
9: -0.0042240, 0.0088603, -0.0039972, 0.0082431, -0.0124671, 0.0128576

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107108, upper bound: 0.0098920
time: 1.02 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104329, upper bound: 0.0098920
time: 1.13 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0074819, 0.0051025, -0.0075451, 0.0044799, -0.0119618, 0.0126475
1: -0.0060730, -0.0010878, -0.0060980, -0.0013344, -0.0047386, 0.0050102
2: 0.0275209, 0.0400601, 0.0281009, 0.0401667, -0.0126458, 0.0119592
3: -0.0073087, 0.0054006, -0.0067050, 0.0054692, -0.0111154, 0.0104488
4: -0.0055270, 0.0054972, -0.0055742, 0.0047974, -0.0103245, 0.0110714
5: 0.0068825, 0.0166450, 0.0068371, 0.0161274, -0.0092448, 0.0098079
6: -0.0117700, 0.0023374, -0.0109175, 0.0024000, -0.0141701, 0.0132550
7: 0.9664696, 0.9842228, 0.9663454, 0.9833962, -0.0169266, 0.0178774
8: -0.0222222, -0.0007876, -0.0207784, -0.0007190, -0.0215033, 0.0199907
9: -0.0042240, 0.0088603, -0.0042705, 0.0080245, -0.0122485, 0.0131308

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107108, upper bound: 0.0098920
time: 1.06 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104329, upper bound: 0.0098920
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0078688, 0.0049581, -0.0071739, 0.0046427, -0.0125116, 0.0121320
1: -0.0062263, -0.0011450, -0.0059510, -0.0012699, -0.0049564, 0.0048060
2: 0.0276553, 0.0407133, 0.0279491, 0.0395402, -0.0118849, 0.0127641
3: -0.0071688, 0.0058203, -0.0068629, 0.0050667, -0.0105708, 0.0110462
4: -0.0058158, 0.0053350, -0.0052972, 0.0049805, -0.0107963, 0.0106322
5: 0.0066039, 0.0165250, 0.0071043, 0.0162627, -0.0096588, 0.0094207
6: -0.0115724, 0.0027210, -0.0111406, 0.0020322, -0.0136046, 0.0138615
7: 0.9657087, 0.9840311, 0.9670752, 0.9836124, -0.0179037, 0.0169560
8: -0.0218875, -0.0003674, -0.0211561, -0.0011221, -0.0207655, 0.0207887
9: -0.0045089, 0.0086665, -0.0039972, 0.0082431, -0.0127520, 0.0126638

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105698, upper bound: 0.0098920
time: 1.07 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0098920
time: 1.08 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0078688, 0.0049581, -0.0075451, 0.0044799, -0.0123488, 0.0125032
1: -0.0062263, -0.0011450, -0.0060980, -0.0013344, -0.0048919, 0.0049530
2: 0.0276553, 0.0407133, 0.0281009, 0.0401667, -0.0125114, 0.0126124
3: -0.0071688, 0.0058203, -0.0067050, 0.0054692, -0.0109347, 0.0108418
4: -0.0058158, 0.0053350, -0.0055742, 0.0047974, -0.0106133, 0.0109092
5: 0.0066039, 0.0165250, 0.0068371, 0.0161274, -0.0095234, 0.0096879
6: -0.0115724, 0.0027210, -0.0109175, 0.0024000, -0.0139725, 0.0136385
7: 0.9657087, 0.9840311, 0.9663454, 0.9833962, -0.0176875, 0.0176858
8: -0.0218875, -0.0003674, -0.0207784, -0.0007190, -0.0211685, 0.0204110
9: -0.0045089, 0.0086665, -0.0042705, 0.0080245, -0.0125335, 0.0129371

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105698, upper bound: 0.0098920
time: 1.28 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0098920
time: 1.11 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0074819, 0.0051025, -0.0074819, 0.0051025, -0.0125843, 0.0125843
1: -0.0060730, -0.0010878, -0.0060730, -0.0010878, -0.0049852, 0.0049852
2: 0.0275209, 0.0400601, 0.0275209, 0.0400601, -0.0125391, 0.0125391
3: -0.0073087, 0.0054006, -0.0073087, 0.0054006, -0.0108871, 0.0108871
4: -0.0055270, 0.0054972, -0.0055270, 0.0054972, -0.0110242, 0.0110242
5: 0.0068825, 0.0166450, 0.0068825, 0.0166450, -0.0097624, 0.0097624
6: -0.0117700, 0.0023374, -0.0117700, 0.0023374, -0.0141074, 0.0141074
7: 0.9664696, 0.9842228, 0.9664696, 0.9842228, -0.0177532, 0.0177532
8: -0.0222222, -0.0007876, -0.0222222, -0.0007876, -0.0214346, 0.0214346
9: -0.0042240, 0.0088603, -0.0042240, 0.0088603, -0.0130843, 0.0130843

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107109, upper bound: 0.0100776
time: 0.98 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104329, upper bound: 0.0100748
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0074819, 0.0051025, -0.0078688, 0.0049581, -0.0124400, 0.0129713
1: -0.0060730, -0.0010878, -0.0062263, -0.0011450, -0.0049280, 0.0051385
2: 0.0275209, 0.0400601, 0.0276553, 0.0407133, -0.0131923, 0.0124047
3: -0.0073087, 0.0054006, -0.0071688, 0.0058203, -0.0113363, 0.0107632
4: -0.0055270, 0.0054972, -0.0058158, 0.0053350, -0.0108620, 0.0113130
5: 0.0068825, 0.0166450, 0.0066039, 0.0165250, -0.0096424, 0.0100410
6: -0.0117700, 0.0023374, -0.0115724, 0.0027210, -0.0144910, 0.0139098
7: 0.9664696, 0.9842228, 0.9657087, 0.9840311, -0.0175616, 0.0185140
8: -0.0222222, -0.0007876, -0.0218875, -0.0003674, -0.0218549, 0.0210999
9: -0.0042240, 0.0088603, -0.0045089, 0.0086665, -0.0128905, 0.0133692

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107109, upper bound: 0.0100776
time: 0.98 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104329, upper bound: 0.0100748
time: 1.15 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0078688, 0.0049581, -0.0074819, 0.0051025, -0.0129713, 0.0124400
1: -0.0062263, -0.0011450, -0.0060730, -0.0010878, -0.0051385, 0.0049280
2: 0.0276553, 0.0407133, 0.0275209, 0.0400601, -0.0124047, 0.0131923
3: -0.0071688, 0.0058203, -0.0073087, 0.0054006, -0.0107632, 0.0113363
4: -0.0058158, 0.0053350, -0.0055270, 0.0054972, -0.0113130, 0.0108620
5: 0.0066039, 0.0165250, 0.0068825, 0.0166450, -0.0100410, 0.0096424
6: -0.0115724, 0.0027210, -0.0117700, 0.0023374, -0.0139098, 0.0144910
7: 0.9657087, 0.9840311, 0.9664696, 0.9842228, -0.0185140, 0.0175616
8: -0.0218875, -0.0003674, -0.0222222, -0.0007876, -0.0210999, 0.0218549
9: -0.0045089, 0.0086665, -0.0042240, 0.0088603, -0.0133692, 0.0128905

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105698, upper bound: 0.0100562
time: 1.17 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0100521
time: 1.08 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0078688, 0.0049581, -0.0078688, 0.0049581, -0.0128270, 0.0128270
1: -0.0062263, -0.0011450, -0.0062263, -0.0011450, -0.0050813, 0.0050813
2: 0.0276553, 0.0407133, 0.0276553, 0.0407133, -0.0130579, 0.0130579
3: -0.0071688, 0.0058203, -0.0071688, 0.0058203, -0.0111522, 0.0111522
4: -0.0058158, 0.0053350, -0.0058158, 0.0053350, -0.0111508, 0.0111508
5: 0.0066039, 0.0165250, 0.0066039, 0.0165250, -0.0099211, 0.0099211
6: -0.0115724, 0.0027210, -0.0115724, 0.0027210, -0.0142934, 0.0142934
7: 0.9657087, 0.9840311, 0.9657087, 0.9840311, -0.0183224, 0.0183224
8: -0.0218875, -0.0003674, -0.0218875, -0.0003674, -0.0215201, 0.0215201
9: -0.0045089, 0.0086665, -0.0045089, 0.0086665, -0.0131755, 0.0131755

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 106

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105698, upper bound: 0.0100562
time: 1.01 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0100521
time: 1.03 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.35 seconds
NS_A1_B1_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0105332, upper bound: 0.0098948
NS_A1_B1_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0101530, upper bound: 0.0098948
NS_A1_B1_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0105332, upper bound: 0.0098948
NS_A1_B1_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0101530, upper bound: 0.0098948
NS_A1_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0104387, upper bound: 0.0098948
NS_A1_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0098948
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0104387, upper bound: 0.0098948
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0098948
NS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0106501, upper bound: 0.0098920
NS_A1_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0101671, upper bound: 0.0098920
NS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0106501, upper bound: 0.0098920
NS_A1_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0101671, upper bound: 0.0098920
NS_A1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0105598, upper bound: 0.0098920
NS_A1_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0098920
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0105598, upper bound: 0.0098920
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0098920
NS_A1_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0104947, upper bound: 0.0098948
NS_A1_B1_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0101312, upper bound: 0.0098948
NS_A1_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0104947, upper bound: 0.0098948
NS_A1_B1_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0101312, upper bound: 0.0098948
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0103991, upper bound: 0.0098948
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0098920, upper bound: 0.0098948
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0103991, upper bound: 0.0098948
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0098920, upper bound: 0.0098948
NS_A1_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0104972, upper bound: 0.0098920
NS_A1_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0101443, upper bound: 0.0098920
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0104972, upper bound: 0.0098920
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0101443, upper bound: 0.0098920
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0104017, upper bound: 0.0098920
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0098920, upper bound: 0.0098920
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0104017, upper bound: 0.0098920
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0098920, upper bound: 0.0098920
NS_A1_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0104510, upper bound: 0.0101262
NS_A1_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0101112, upper bound: 0.0101262
NS_A1_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0104510, upper bound: 0.0101262
NS_A1_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0101112, upper bound: 0.0101262
NS_A1_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0103902, upper bound: 0.0101262
NS_A1_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0101262
NS_A1_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0103902, upper bound: 0.0101262
NS_A1_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0101262
NS_A1_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0105589, upper bound: 0.0100928
NS_A1_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0101259, upper bound: 0.0100928
NS_A1_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0105589, upper bound: 0.0100928
NS_A1_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0101259, upper bound: 0.0100928
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0105045, upper bound: 0.0100928
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0100928
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0105045, upper bound: 0.0100928
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0098948, upper bound: 0.0100928
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0104227, upper bound: 0.0101262
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0100930, upper bound: 0.0101262
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0104227, upper bound: 0.0101262
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0100930, upper bound: 0.0101262
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0103486, upper bound: 0.0101262
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0098878, upper bound: 0.0101262
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0103486, upper bound: 0.0101262
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0098878, upper bound: 0.0101262
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0104256, upper bound: 0.0100928
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0101060, upper bound: 0.0100928
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0104256, upper bound: 0.0100928
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0101060, upper bound: 0.0100928
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0103513, upper bound: 0.0100928
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0098919, upper bound: 0.0100928
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0103513, upper bound: 0.0100928
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0098919, upper bound: 0.0100928
NS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0107712, upper bound: 0.0098948
NS_A2_B1_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0104737, upper bound: 0.0098948
NS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0107712, upper bound: 0.0098948
NS_A2_B1_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0104737, upper bound: 0.0098948
NS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0106214, upper bound: 0.0098948
NS_A2_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0098948
NS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0106214, upper bound: 0.0098948
NS_A2_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0098948
NS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0107712, upper bound: 0.0100928
NS_A2_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0104737, upper bound: 0.0100825
NS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0107712, upper bound: 0.0100928
NS_A2_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0104737, upper bound: 0.0100825
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0106214, upper bound: 0.0100658
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0100500
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0106214, upper bound: 0.0100658
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0100500
NS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0107108, upper bound: 0.0098948
NS_A2_B1_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0104304, upper bound: 0.0098948
NS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0107108, upper bound: 0.0098948
NS_A2_B1_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0104304, upper bound: 0.0098948
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0105697, upper bound: 0.0098948
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0098948
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0105697, upper bound: 0.0098948
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0098948
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0107108, upper bound: 0.0100987
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0104304, upper bound: 0.0100926
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0107108, upper bound: 0.0100987
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0104304, upper bound: 0.0100926
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0105697, upper bound: 0.0100734
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0100654
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0105697, upper bound: 0.0100734
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0100654
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0108766, upper bound: 0.0098920
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0104913, upper bound: 0.0098920
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0108766, upper bound: 0.0098920
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0104913, upper bound: 0.0098920
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0107205, upper bound: 0.0098920
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0098878
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0107205, upper bound: 0.0098920
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0098878
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0108766, upper bound: 0.0100628
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0104913, upper bound: 0.0100514
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0108766, upper bound: 0.0100628
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0104913, upper bound: 0.0100514
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0107205, upper bound: 0.0100364
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0100220
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0107205, upper bound: 0.0100364
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0100220
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0107108, upper bound: 0.0098920
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0104329, upper bound: 0.0098920
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0107108, upper bound: 0.0098920
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0104329, upper bound: 0.0098920
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0105698, upper bound: 0.0098920
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0098920
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0105698, upper bound: 0.0098920
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0098920
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0107109, upper bound: 0.0100776
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0104329, upper bound: 0.0100748
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0107109, upper bound: 0.0100776
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0104329, upper bound: 0.0100748
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0105698, upper bound: 0.0100562
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0100521
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0105698, upper bound: 0.0100562
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0100521

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0070135, 0.0041515, -0.0071739, 0.0046427, -0.0116563, 0.0113254
1: -0.0058875, -0.0014645, -0.0059510, -0.0012699, -0.0046175, 0.0044865
2: 0.0284068, 0.0392696, 0.0279491, 0.0395402, -0.0111335, 0.0113204
3: -0.0063865, 0.0048927, -0.0068629, 0.0050667, -0.0097993, 0.0100981
4: -0.0051775, 0.0044283, -0.0052972, 0.0049805, -0.0101580, 0.0097255
5: 0.0072197, 0.0158543, 0.0071043, 0.0162627, -0.0090430, 0.0087500
6: -0.0104678, 0.0018732, -0.0111406, 0.0020322, -0.0125000, 0.0130138
7: 0.9673906, 0.9829602, 0.9670752, 0.9836124, -0.0162218, 0.0158850
8: -0.0200167, -0.0012962, -0.0211561, -0.0011221, -0.0188947, 0.0198599
9: -0.0038792, 0.0075837, -0.0039972, 0.0082431, -0.0121223, 0.0115809

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: B, layer: 3, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0102154, upper bound: 0.0102076
time: 1.11 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0102154, upper bound: 0.0102076
time: 1.12 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0070135, 0.0041515, -0.0075451, 0.0044799, -0.0114935, 0.0116966
1: -0.0058875, -0.0014645, -0.0060980, -0.0013344, -0.0045530, 0.0046335
2: 0.0284068, 0.0392696, 0.0281009, 0.0401667, -0.0117600, 0.0111687
3: -0.0063865, 0.0048927, -0.0067050, 0.0054692, -0.0102261, 0.0099617
4: -0.0051775, 0.0044283, -0.0055742, 0.0047974, -0.0099749, 0.0100025
5: 0.0072197, 0.0158543, 0.0068371, 0.0161274, -0.0089076, 0.0090173
6: -0.0104678, 0.0018732, -0.0109175, 0.0024000, -0.0128679, 0.0127908
7: 0.9673906, 0.9829602, 0.9663454, 0.9833962, -0.0160056, 0.0166148
8: -0.0200167, -0.0012962, -0.0207784, -0.0007190, -0.0192977, 0.0194822
9: -0.0038792, 0.0075837, -0.0042705, 0.0080245, -0.0119037, 0.0118542

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 106

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101671, upper bound: 0.0098920
time: 0.98 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101671, upper bound: 0.0098920
time: 1.01 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0073164, 0.0046110, -0.0070136, 0.0043946, -0.0117110, 0.0116245
1: -0.0060074, -0.0012825, -0.0058875, -0.0013682, -0.0046392, 0.0046050
2: 0.0279787, 0.0397808, 0.0281803, 0.0392696, -0.0112908, 0.0116005
3: -0.0068321, 0.0052212, -0.0066222, 0.0048927, -0.0100708, 0.0102032
4: -0.0054036, 0.0049448, -0.0051775, 0.0047016, -0.0101051, 0.0101223
5: 0.0070017, 0.0162363, 0.0072197, 0.0160564, -0.0090548, 0.0090166
6: -0.0110970, 0.0021734, -0.0108007, 0.0018732, -0.0129703, 0.0129742
7: 0.9667950, 0.9835703, 0.9673904, 0.9832830, -0.0164880, 0.0161799
8: -0.0210824, -0.0009673, -0.0205806, -0.0012962, -0.0197862, 0.0196133
9: -0.0041022, 0.0082005, -0.0038792, 0.0079100, -0.0120122, 0.0120797

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: B, layer: 3, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106361, upper bound: 0.0102147
time: 1.01 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106361, upper bound: 0.0102147
time: 1.02 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0073164, 0.0046110, -0.0073856, 0.0042302, -0.0115466, 0.0119966
1: -0.0060074, -0.0012825, -0.0060349, -0.0014334, -0.0045741, 0.0047523
2: 0.0279787, 0.0397808, 0.0283335, 0.0398976, -0.0119189, 0.0114474
3: -0.0068321, 0.0052212, -0.0064628, 0.0052963, -0.0105015, 0.0100609
4: -0.0054036, 0.0049448, -0.0054552, 0.0045168, -0.0099203, 0.0104000
5: 0.0070017, 0.0162363, 0.0069518, 0.0159198, -0.0089181, 0.0092845
6: -0.0110970, 0.0021734, -0.0105756, 0.0022420, -0.0133390, 0.0127490
7: 0.9667950, 0.9835703, 0.9666588, 0.9830647, -0.0162697, 0.0169114
8: -0.0210824, -0.0009673, -0.0201993, -0.0008921, -0.0201903, 0.0192320
9: -0.0041022, 0.0082005, -0.0041531, 0.0076893, -0.0117915, 0.0123536

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 106

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104737, upper bound: 0.0098948
time: 1.12 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104737, upper bound: 0.0098948
time: 1.13 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0077128, 0.0044311, -0.0070136, 0.0043946, -0.0121074, 0.0114446
1: -0.0061645, -0.0013538, -0.0058875, -0.0013682, -0.0047962, 0.0045337
2: 0.0281463, 0.0404499, 0.0281803, 0.0392696, -0.0111233, 0.0122696
3: -0.0066576, 0.0056511, -0.0066222, 0.0048927, -0.0099179, 0.0106619
4: -0.0056994, 0.0047426, -0.0051775, 0.0047016, -0.0104009, 0.0099201
5: 0.0067163, 0.0160868, 0.0072197, 0.0160564, -0.0093402, 0.0088671
6: -0.0108507, 0.0025663, -0.0108007, 0.0018732, -0.0127239, 0.0133671
7: 0.9660155, 0.9833314, 0.9673904, 0.9832830, -0.0172675, 0.0159410
8: -0.0206652, -0.0005368, -0.0205806, -0.0012962, -0.0193690, 0.0200437
9: -0.0043941, 0.0079590, -0.0038792, 0.0079100, -0.0123041, 0.0118382

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: B, layer: 3, pos: 13

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0101112
time: 0.96 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0101112
time: 1.00 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0077128, 0.0044311, -0.0073856, 0.0042302, -0.0119430, 0.0118167
1: -0.0061645, -0.0013538, -0.0060349, -0.0014334, -0.0047311, 0.0046811
2: 0.0281463, 0.0404499, 0.0283335, 0.0398976, -0.0117513, 0.0121164
3: -0.0066576, 0.0056511, -0.0064628, 0.0052963, -0.0102849, 0.0104588
4: -0.0056994, 0.0047426, -0.0054552, 0.0045168, -0.0102161, 0.0101978
5: 0.0067163, 0.0160868, 0.0069518, 0.0159198, -0.0092035, 0.0091350
6: -0.0108507, 0.0025663, -0.0105756, 0.0022420, -0.0130927, 0.0131419
7: 0.9660155, 0.9833314, 0.9666588, 0.9830647, -0.0170491, 0.0166726
8: -0.0206652, -0.0005368, -0.0201993, -0.0008921, -0.0197730, 0.0196624
9: -0.0043941, 0.0079590, -0.0041531, 0.0076893, -0.0120833, 0.0121121

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 106

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0098948
time: 1.01 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0098948
time: 0.99 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0073164, 0.0046110, -0.0073164, 0.0048494, -0.0121658, 0.0119274
1: -0.0060074, -0.0012825, -0.0060075, -0.0011881, -0.0048194, 0.0047249
2: 0.0279787, 0.0397808, 0.0277567, 0.0397808, -0.0118021, 0.0120241
3: -0.0068321, 0.0052212, -0.0070633, 0.0052212, -0.0102663, 0.0104995
4: -0.0054036, 0.0049448, -0.0054036, 0.0052127, -0.0106163, 0.0103484
5: 0.0070017, 0.0162363, 0.0070017, 0.0164345, -0.0094329, 0.0092347
6: -0.0110970, 0.0021734, -0.0114235, 0.0021734, -0.0132705, 0.0135969
7: 0.9667950, 0.9835703, 0.9667950, 0.9838866, -0.0170916, 0.0167753
8: -0.0210824, -0.0009673, -0.0216352, -0.0009673, -0.0201152, 0.0206679
9: -0.0041022, 0.0082005, -0.0041022, 0.0085205, -0.0126227, 0.0123027

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: B, layer: 3, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106379, upper bound: 0.0104440
time: 0.93 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106379, upper bound: 0.0104440
time: 0.92 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0073164, 0.0046110, -0.0077128, 0.0047094, -0.0120258, 0.0123238
1: -0.0060074, -0.0012825, -0.0061645, -0.0012435, -0.0047639, 0.0048820
2: 0.0279787, 0.0397808, 0.0278871, 0.0404499, -0.0124712, 0.0118937
3: -0.0068321, 0.0052212, -0.0069275, 0.0056511, -0.0107204, 0.0103760
4: -0.0054036, 0.0049448, -0.0056994, 0.0050554, -0.0104589, 0.0106442
5: 0.0070017, 0.0162363, 0.0067163, 0.0163181, -0.0093165, 0.0095201
6: -0.0110970, 0.0021734, -0.0112318, 0.0025663, -0.0136634, 0.0134052
7: 0.9667950, 0.9835703, 0.9660155, 0.9837008, -0.0169058, 0.0175548
8: -0.0210824, -0.0009673, -0.0213106, -0.0005368, -0.0205456, 0.0203433
9: -0.0041022, 0.0082005, -0.0043940, 0.0083326, -0.0124348, 0.0125945

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 106

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104737, upper bound: 0.0100825
time: 1.02 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104737, upper bound: 0.0100825
time: 1.00 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0077128, 0.0044311, -0.0073164, 0.0048494, -0.0125622, 0.0117475
1: -0.0061645, -0.0013538, -0.0060075, -0.0011881, -0.0049764, 0.0046537
2: 0.0281463, 0.0404499, 0.0277567, 0.0397808, -0.0116345, 0.0126932
3: -0.0066576, 0.0056511, -0.0070633, 0.0052212, -0.0101070, 0.0109534
4: -0.0056994, 0.0047426, -0.0054036, 0.0052127, -0.0109121, 0.0101461
5: 0.0067163, 0.0160868, 0.0070017, 0.0164345, -0.0097183, 0.0090851
6: -0.0108507, 0.0025663, -0.0114235, 0.0021734, -0.0130242, 0.0139898
7: 0.9660155, 0.9833314, 0.9667950, 0.9838866, -0.0178711, 0.0165365
8: -0.0206652, -0.0005368, -0.0216352, -0.0009673, -0.0196979, 0.0210984
9: -0.0043941, 0.0079590, -0.0041022, 0.0085205, -0.0129146, 0.0120612

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: B, layer: 3, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0102230
time: 0.98 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0102230
time: 1.05 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0077128, 0.0044311, -0.0077128, 0.0047094, -0.0124222, 0.0121439
1: -0.0061645, -0.0013538, -0.0061645, -0.0012435, -0.0049209, 0.0048107
2: 0.0281463, 0.0404499, 0.0278871, 0.0404499, -0.0123036, 0.0125628
3: -0.0066576, 0.0056511, -0.0069275, 0.0056511, -0.0105008, 0.0107722
4: -0.0056994, 0.0047426, -0.0056994, 0.0050554, -0.0107547, 0.0104419
5: 0.0067163, 0.0160868, 0.0067163, 0.0163181, -0.0096019, 0.0093705
6: -0.0108507, 0.0025663, -0.0112318, 0.0025663, -0.0134170, 0.0137981
7: 0.9660155, 0.9833314, 0.9660155, 0.9837008, -0.0176853, 0.0173159
8: -0.0206652, -0.0005368, -0.0213106, -0.0005368, -0.0201283, 0.0207737
9: -0.0043941, 0.0079590, -0.0043940, 0.0083326, -0.0127266, 0.0123530

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 106

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0100500
time: 0.97 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0100500
time: 0.95 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0074819, 0.0048139, -0.0070136, 0.0043946, -0.0118764, 0.0118274
1: -0.0060730, -0.0012021, -0.0058875, -0.0013682, -0.0047048, 0.0046853
2: 0.0277897, 0.0400601, 0.0281803, 0.0392696, -0.0114798, 0.0118797
3: -0.0070288, 0.0054006, -0.0066222, 0.0048927, -0.0102878, 0.0104094
4: -0.0055270, 0.0051728, -0.0051775, 0.0047016, -0.0102286, 0.0103504
5: 0.0068825, 0.0164050, 0.0072197, 0.0160564, -0.0091739, 0.0091853
6: -0.0113749, 0.0023374, -0.0108007, 0.0018732, -0.0132481, 0.0131381
7: 0.9664697, 0.9838396, 0.9673904, 0.9832830, -0.0168133, 0.0164492
8: -0.0215529, -0.0007876, -0.0205806, -0.0012962, -0.0202568, 0.0197929
9: -0.0042240, 0.0084728, -0.0038792, 0.0079100, -0.0121340, 0.0123520

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: B, layer: 3, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105839, upper bound: 0.0102154
time: 1.08 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105839, upper bound: 0.0102154
time: 1.01 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0074819, 0.0048139, -0.0073856, 0.0042302, -0.0117120, 0.0121995
1: -0.0060730, -0.0012021, -0.0060349, -0.0014334, -0.0046396, 0.0048327
2: 0.0277897, 0.0400601, 0.0283335, 0.0398976, -0.0121079, 0.0117266
3: -0.0070288, 0.0054006, -0.0064628, 0.0052963, -0.0107185, 0.0102670
4: -0.0055270, 0.0051728, -0.0054552, 0.0045168, -0.0100438, 0.0106280
5: 0.0068825, 0.0164050, 0.0069518, 0.0159198, -0.0090372, 0.0094532
6: -0.0113749, 0.0023374, -0.0105756, 0.0022420, -0.0136169, 0.0129130
7: 0.9664697, 0.9838396, 0.9666588, 0.9830647, -0.0165949, 0.0171807
8: -0.0215529, -0.0007876, -0.0201993, -0.0008921, -0.0206608, 0.0194116
9: -0.0042240, 0.0084728, -0.0041531, 0.0076893, -0.0119133, 0.0126260

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 106

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104304, upper bound: 0.0098948
time: 1.12 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104304, upper bound: 0.0098948
time: 1.11 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0074819, 0.0048139, -0.0073164, 0.0048494, -0.0123312, 0.0121303
1: -0.0060730, -0.0012021, -0.0060075, -0.0011881, -0.0048849, 0.0048053
2: 0.0277897, 0.0400601, 0.0277567, 0.0397808, -0.0119911, 0.0123034
3: -0.0070288, 0.0054006, -0.0070633, 0.0052212, -0.0104786, 0.0107061
4: -0.0055270, 0.0051728, -0.0054036, 0.0052127, -0.0107397, 0.0105764
5: 0.0068825, 0.0164050, 0.0070017, 0.0164345, -0.0095520, 0.0094034
6: -0.0113749, 0.0023374, -0.0114235, 0.0021734, -0.0135483, 0.0137609
7: 0.9664697, 0.9838396, 0.9667950, 0.9838866, -0.0174169, 0.0170446
8: -0.0215529, -0.0007876, -0.0216352, -0.0009673, -0.0205857, 0.0208476
9: -0.0042240, 0.0084728, -0.0041022, 0.0085205, -0.0127445, 0.0125750

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: B, layer: 3, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105882, upper bound: 0.0104603
time: 0.95 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105882, upper bound: 0.0104603
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0074819, 0.0048139, -0.0077128, 0.0047094, -0.0121912, 0.0125267
1: -0.0060730, -0.0012021, -0.0061645, -0.0012435, -0.0048295, 0.0049623
2: 0.0277897, 0.0400601, 0.0278871, 0.0404499, -0.0126602, 0.0121730
3: -0.0070288, 0.0054006, -0.0069275, 0.0056511, -0.0109327, 0.0105826
4: -0.0055270, 0.0051728, -0.0056994, 0.0050554, -0.0105824, 0.0108722
5: 0.0068825, 0.0164050, 0.0067163, 0.0163181, -0.0094356, 0.0096888
6: -0.0113749, 0.0023374, -0.0112318, 0.0025663, -0.0139412, 0.0135692
7: 0.9664697, 0.9838396, 0.9660155, 0.9837008, -0.0172311, 0.0178241
8: -0.0215529, -0.0007876, -0.0213106, -0.0005368, -0.0210161, 0.0205230
9: -0.0042240, 0.0084728, -0.0043940, 0.0083326, -0.0125566, 0.0128669

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 106

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104304, upper bound: 0.0100926
time: 1.09 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104304, upper bound: 0.0100926
time: 1.10 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0073164, 0.0046110, -0.0071739, 0.0046427, -0.0119592, 0.0117849
1: -0.0060074, -0.0012825, -0.0059510, -0.0012699, -0.0047375, 0.0046685
2: 0.0279787, 0.0397808, 0.0279491, 0.0395402, -0.0115615, 0.0118317
3: -0.0068321, 0.0052212, -0.0068629, 0.0050667, -0.0102707, 0.0104634
4: -0.0054036, 0.0049448, -0.0052972, 0.0049805, -0.0103840, 0.0102420
5: 0.0070017, 0.0162363, 0.0071043, 0.0162627, -0.0092611, 0.0091321
6: -0.0110970, 0.0021734, -0.0111406, 0.0020322, -0.0131292, 0.0133140
7: 0.9667950, 0.9835703, 0.9670752, 0.9836124, -0.0168175, 0.0164951
8: -0.0210824, -0.0009673, -0.0211561, -0.0011221, -0.0199604, 0.0201888
9: -0.0041022, 0.0082005, -0.0039972, 0.0082431, -0.0123453, 0.0121977

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: B, layer: 3, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106415, upper bound: 0.0101985
time: 0.96 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106415, upper bound: 0.0101985
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0073164, 0.0046110, -0.0075451, 0.0044799, -0.0117963, 0.0121561
1: -0.0060074, -0.0012825, -0.0060980, -0.0013344, -0.0046730, 0.0048155
2: 0.0279787, 0.0397808, 0.0281009, 0.0401667, -0.0121880, 0.0116800
3: -0.0068321, 0.0052212, -0.0067050, 0.0054692, -0.0106975, 0.0103270
4: -0.0054036, 0.0049448, -0.0055742, 0.0047974, -0.0102010, 0.0105190
5: 0.0070017, 0.0162363, 0.0068371, 0.0161274, -0.0091257, 0.0093993
6: -0.0110970, 0.0021734, -0.0109175, 0.0024000, -0.0134971, 0.0130910
7: 0.9667950, 0.9835703, 0.9663454, 0.9833962, -0.0166013, 0.0172249
8: -0.0210824, -0.0009673, -0.0207784, -0.0007190, -0.0203634, 0.0198111
9: -0.0041022, 0.0082005, -0.0042705, 0.0080245, -0.0121267, 0.0124710

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 106

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104913, upper bound: 0.0098920
time: 1.13 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104913, upper bound: 0.0098920
time: 1.15 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0077128, 0.0044311, -0.0071739, 0.0046427, -0.0123556, 0.0116050
1: -0.0061645, -0.0013538, -0.0059510, -0.0012699, -0.0048945, 0.0045972
2: 0.0281463, 0.0404499, 0.0279491, 0.0395402, -0.0113939, 0.0125007
3: -0.0066576, 0.0056511, -0.0068629, 0.0050667, -0.0101178, 0.0109221
4: -0.0056994, 0.0047426, -0.0052972, 0.0049805, -0.0106799, 0.0100398
5: 0.0067163, 0.0160868, 0.0071043, 0.0162627, -0.0095465, 0.0089825
6: -0.0108507, 0.0025663, -0.0111406, 0.0020322, -0.0128829, 0.0137069
7: 0.9660155, 0.9833314, 0.9670752, 0.9836124, -0.0175969, 0.0162563
8: -0.0206652, -0.0005368, -0.0211561, -0.0011221, -0.0195431, 0.0206192
9: -0.0043941, 0.0079590, -0.0039972, 0.0082431, -0.0126372, 0.0119562

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: B, layer: 3, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0100930
time: 0.94 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0100930
time: 0.93 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0077128, 0.0044311, -0.0075451, 0.0044799, -0.0121928, 0.0119762
1: -0.0061645, -0.0013538, -0.0060980, -0.0013344, -0.0048300, 0.0047443
2: 0.0281463, 0.0404499, 0.0281009, 0.0401667, -0.0120204, 0.0123490
3: -0.0066576, 0.0056511, -0.0067050, 0.0054692, -0.0104831, 0.0107246
4: -0.0056994, 0.0047426, -0.0055742, 0.0047974, -0.0104968, 0.0103167
5: 0.0067163, 0.0160868, 0.0068371, 0.0161274, -0.0094111, 0.0092497
6: -0.0108507, 0.0025663, -0.0109175, 0.0024000, -0.0132508, 0.0134839
7: 0.9660155, 0.9833314, 0.9663454, 0.9833962, -0.0173807, 0.0169861
8: -0.0206652, -0.0005368, -0.0207784, -0.0007190, -0.0199462, 0.0202415
9: -0.0043941, 0.0079590, -0.0042705, 0.0080245, -0.0124186, 0.0122295

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 106

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0098878
time: 0.94 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0098878
time: 0.97 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0073164, 0.0046110, -0.0074819, 0.0051025, -0.0124189, 0.0120929
1: -0.0060074, -0.0012825, -0.0060730, -0.0010878, -0.0049196, 0.0047905
2: 0.0279787, 0.0397808, 0.0275209, 0.0400601, -0.0120813, 0.0122599
3: -0.0068321, 0.0052212, -0.0073087, 0.0054006, -0.0104734, 0.0107665
4: -0.0054036, 0.0049448, -0.0055270, 0.0054972, -0.0109008, 0.0104718
5: 0.0070017, 0.0162363, 0.0068825, 0.0166450, -0.0096433, 0.0093538
6: -0.0110970, 0.0021734, -0.0117700, 0.0023374, -0.0134345, 0.0139434
7: 0.9667950, 0.9835703, 0.9664696, 0.9842228, -0.0174278, 0.0171007
8: -0.0210824, -0.0009673, -0.0222222, -0.0007876, -0.0202948, 0.0212550
9: -0.0041022, 0.0082005, -0.0042240, 0.0088603, -0.0129625, 0.0124245

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: B, layer: 3, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106423, upper bound: 0.0104075
time: 1.01 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106423, upper bound: 0.0104075
time: 1.14 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0073164, 0.0046110, -0.0078688, 0.0049581, -0.0122746, 0.0124798
1: -0.0060074, -0.0012825, -0.0062263, -0.0011450, -0.0048625, 0.0049438
2: 0.0279787, 0.0397808, 0.0276553, 0.0407133, -0.0127345, 0.0121255
3: -0.0068321, 0.0052212, -0.0071688, 0.0058203, -0.0109230, 0.0106446
4: -0.0054036, 0.0049448, -0.0058158, 0.0053350, -0.0107386, 0.0107606
5: 0.0070017, 0.0162363, 0.0066039, 0.0165250, -0.0095233, 0.0096324
6: -0.0110970, 0.0021734, -0.0115724, 0.0027210, -0.0138180, 0.0137459
7: 0.9667950, 0.9835703, 0.9657087, 0.9840311, -0.0172362, 0.0178615
8: -0.0210824, -0.0009673, -0.0218875, -0.0003674, -0.0207150, 0.0209203
9: -0.0041022, 0.0082005, -0.0045089, 0.0086665, -0.0127687, 0.0127094

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 106

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104913, upper bound: 0.0100514
time: 0.94 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104913, upper bound: 0.0100514
time: 0.93 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0077128, 0.0044311, -0.0074819, 0.0051025, -0.0128153, 0.0119130
1: -0.0061645, -0.0013538, -0.0060730, -0.0010878, -0.0050767, 0.0047192
2: 0.0281463, 0.0404499, 0.0275209, 0.0400601, -0.0119138, 0.0129289
3: -0.0066576, 0.0056511, -0.0073087, 0.0054006, -0.0103142, 0.0112203
4: -0.0056994, 0.0047426, -0.0055270, 0.0054972, -0.0111966, 0.0102696
5: 0.0067163, 0.0160868, 0.0068825, 0.0166450, -0.0099287, 0.0092042
6: -0.0108507, 0.0025663, -0.0117700, 0.0023374, -0.0131881, 0.0143363
7: 0.9660155, 0.9833314, 0.9664696, 0.9842228, -0.0182073, 0.0168619
8: -0.0206652, -0.0005368, -0.0222222, -0.0007876, -0.0198776, 0.0216854
9: -0.0043941, 0.0079590, -0.0042240, 0.0088603, -0.0132544, 0.0121830

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: B, layer: 3, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0101946
time: 0.90 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0101946
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0077128, 0.0044311, -0.0078688, 0.0049581, -0.0126710, 0.0122999
1: -0.0061645, -0.0013538, -0.0062263, -0.0011450, -0.0050195, 0.0048725
2: 0.0281463, 0.0404499, 0.0276553, 0.0407133, -0.0125669, 0.0127945
3: -0.0066576, 0.0056511, -0.0071688, 0.0058203, -0.0107025, 0.0110402
4: -0.0056994, 0.0047426, -0.0058158, 0.0053350, -0.0110344, 0.0105584
5: 0.0067163, 0.0160868, 0.0066039, 0.0165250, -0.0098087, 0.0094829
6: -0.0108507, 0.0025663, -0.0115724, 0.0027210, -0.0135717, 0.0141387
7: 0.9660155, 0.9833314, 0.9657087, 0.9840311, -0.0180156, 0.0176227
8: -0.0206652, -0.0005368, -0.0218875, -0.0003674, -0.0202978, 0.0213507
9: -0.0043941, 0.0079590, -0.0045089, 0.0086665, -0.0130606, 0.0124679

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 106

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0100220
time: 0.92 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0100220
time: 1.04 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0074819, 0.0048139, -0.0071739, 0.0046427, -0.0121246, 0.0119878
1: -0.0060730, -0.0012021, -0.0059510, -0.0012699, -0.0048031, 0.0047489
2: 0.0277897, 0.0400601, 0.0279491, 0.0395402, -0.0117505, 0.0121109
3: -0.0070288, 0.0054006, -0.0068629, 0.0050667, -0.0104055, 0.0105881
4: -0.0055270, 0.0051728, -0.0052972, 0.0049805, -0.0105075, 0.0104700
5: 0.0068825, 0.0164050, 0.0071043, 0.0162627, -0.0093802, 0.0093007
6: -0.0113749, 0.0023374, -0.0111406, 0.0020322, -0.0134070, 0.0134779
7: 0.9664697, 0.9838396, 0.9670752, 0.9836124, -0.0171427, 0.0167644
8: -0.0215529, -0.0007876, -0.0211561, -0.0011221, -0.0204309, 0.0203684
9: -0.0042240, 0.0084728, -0.0039972, 0.0082431, -0.0124671, 0.0124701

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: B, layer: 3, pos: 13

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105839, upper bound: 0.0102115
time: 1.07 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105839, upper bound: 0.0102115
time: 1.00 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0074819, 0.0048139, -0.0075451, 0.0044799, -0.0119618, 0.0123590
1: -0.0060730, -0.0012021, -0.0060980, -0.0013344, -0.0047386, 0.0048959
2: 0.0277897, 0.0400601, 0.0281009, 0.0401667, -0.0123770, 0.0119592
3: -0.0070288, 0.0054006, -0.0067050, 0.0054692, -0.0108307, 0.0104480
4: -0.0055270, 0.0051728, -0.0055742, 0.0047974, -0.0103245, 0.0107470
5: 0.0068825, 0.0164050, 0.0068371, 0.0161274, -0.0092448, 0.0095680
6: -0.0113749, 0.0023374, -0.0109175, 0.0024000, -0.0137749, 0.0132549
7: 0.9664697, 0.9838396, 0.9663454, 0.9833962, -0.0169265, 0.0174942
8: -0.0215529, -0.0007876, -0.0207784, -0.0007190, -0.0208339, 0.0199907
9: -0.0042240, 0.0084728, -0.0042705, 0.0080245, -0.0122485, 0.0127434

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 106

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104329, upper bound: 0.0098920
time: 1.21 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104329, upper bound: 0.0098920
time: 1.09 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0074819, 0.0048139, -0.0074819, 0.0051025, -0.0125843, 0.0122958
1: -0.0060730, -0.0012021, -0.0060730, -0.0010878, -0.0049852, 0.0048709
2: 0.0277897, 0.0400601, 0.0275209, 0.0400601, -0.0122703, 0.0125391
3: -0.0070288, 0.0054006, -0.0073087, 0.0054006, -0.0106023, 0.0108866
4: -0.0055270, 0.0051728, -0.0055270, 0.0054972, -0.0110242, 0.0106999
5: 0.0068825, 0.0164050, 0.0068825, 0.0166450, -0.0097624, 0.0095225
6: -0.0113749, 0.0023374, -0.0117700, 0.0023374, -0.0137123, 0.0141074
7: 0.9664697, 0.9838396, 0.9664696, 0.9842228, -0.0177531, 0.0173700
8: -0.0215529, -0.0007876, -0.0222222, -0.0007876, -0.0207653, 0.0214346
9: -0.0042240, 0.0084728, -0.0042240, 0.0088603, -0.0130843, 0.0126968

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: B, layer: 3, pos: 13

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105882, upper bound: 0.0104500
time: 1.13 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105882, upper bound: 0.0104500
time: 1.13 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0074819, 0.0048139, -0.0078688, 0.0049581, -0.0124400, 0.0126827
1: -0.0060730, -0.0012021, -0.0062263, -0.0011450, -0.0049280, 0.0050242
2: 0.0277897, 0.0400601, 0.0276553, 0.0407133, -0.0129235, 0.0124047
3: -0.0070288, 0.0054006, -0.0071688, 0.0058203, -0.0110516, 0.0107627
4: -0.0055270, 0.0051728, -0.0058158, 0.0053350, -0.0108620, 0.0109887
5: 0.0068825, 0.0164050, 0.0066039, 0.0165250, -0.0096424, 0.0098011
6: -0.0113749, 0.0023374, -0.0115724, 0.0027210, -0.0140958, 0.0139098
7: 0.9664697, 0.9838396, 0.9657087, 0.9840311, -0.0175614, 0.0181308
8: -0.0215529, -0.0007876, -0.0218875, -0.0003674, -0.0211855, 0.0210999
9: -0.0042240, 0.0084728, -0.0045089, 0.0086665, -0.0128905, 0.0129818

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 106

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104329, upper bound: 0.0100748
time: 1.05 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104329, upper bound: 0.0100748
time: 1.06 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 3.68 seconds
NS_A1_B1_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0102154, upper bound: 0.0102076
NS_A1_B1_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0102154, upper bound: 0.0102076
NS_A1_B1_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0101671, upper bound: 0.0098920
NS_A1_B1_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0101671, upper bound: 0.0098920
NS_A2_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0106361, upper bound: 0.0102147
NS_A2_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0106361, upper bound: 0.0102147
NS_A2_B1_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0104737, upper bound: 0.0098948
NS_A2_B1_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0104737, upper bound: 0.0098948
NS_A2_B1_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0101112
NS_A2_B1_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0101112
NS_A2_B1_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0098948
NS_A2_B1_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0098948
NS_A2_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0106379, upper bound: 0.0104440
NS_A2_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0106379, upper bound: 0.0104440
NS_A2_B1_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0104737, upper bound: 0.0100825
NS_A2_B1_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0104737, upper bound: 0.0100825
NS_A2_B1_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0102230
NS_A2_B1_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0102230
NS_A2_B1_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0100500
NS_A2_B1_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0100500
NS_A2_B1_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0105839, upper bound: 0.0102154
NS_A2_B1_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0105839, upper bound: 0.0102154
NS_A2_B1_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0104304, upper bound: 0.0098948
NS_A2_B1_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0104304, upper bound: 0.0098948
NS_A2_B1_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0105882, upper bound: 0.0104603
NS_A2_B1_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0105882, upper bound: 0.0104603
NS_A2_B1_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0104304, upper bound: 0.0100926
NS_A2_B1_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0104304, upper bound: 0.0100926
NS_A2_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0106415, upper bound: 0.0101985
NS_A2_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0106415, upper bound: 0.0101985
NS_A2_B2_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0104913, upper bound: 0.0098920
NS_A2_B2_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0104913, upper bound: 0.0098920
NS_A2_B2_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0100930
NS_A2_B2_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0100930
NS_A2_B2_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0098878
NS_A2_B2_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0098878
NS_A2_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0106423, upper bound: 0.0104075
NS_A2_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0106423, upper bound: 0.0104075
NS_A2_B2_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0104913, upper bound: 0.0100514
NS_A2_B2_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0104913, upper bound: 0.0100514
NS_A2_B2_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0101946
NS_A2_B2_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0101946
NS_A2_B2_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0100220
NS_A2_B2_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0101262, upper bound: 0.0100220
NS_A2_B2_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0105839, upper bound: 0.0102115
NS_A2_B2_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0105839, upper bound: 0.0102115
NS_A2_B2_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0104329, upper bound: 0.0098920
NS_A2_B2_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0104329, upper bound: 0.0098920
NS_A2_B2_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0105882, upper bound: 0.0104500
NS_A2_B2_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0105882, upper bound: 0.0104500
NS_A2_B2_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0104329, upper bound: 0.0100748
NS_A2_B2_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.68
Output dim: 7, lower bound: -0.0104329, upper bound: 0.0100748

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0073164, 0.0046110, -0.0070135, 0.0041515, -0.0114679, 0.0116245
1: -0.0060074, -0.0012825, -0.0058875, -0.0014645, -0.0045429, 0.0046050
2: 0.0279787, 0.0397808, 0.0284068, 0.0392696, -0.0112908, 0.0113740
3: -0.0068321, 0.0052212, -0.0063865, 0.0048927, -0.0100703, 0.0099643
4: -0.0054036, 0.0049448, -0.0051775, 0.0044283, -0.0098319, 0.0101223
5: 0.0070017, 0.0162363, 0.0072197, 0.0158543, -0.0088527, 0.0090166
6: -0.0110970, 0.0021734, -0.0104678, 0.0018732, -0.0129703, 0.0126413
7: 0.9667950, 0.9835703, 0.9673906, 0.9829602, -0.0161652, 0.0161797
8: -0.0210824, -0.0009673, -0.0200167, -0.0012962, -0.0197862, 0.0190495
9: -0.0041022, 0.0082005, -0.0038792, 0.0075837, -0.0116859, 0.0120797

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109508, upper bound: 0.0102154
time: 1.35 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106361, upper bound: 0.0102147
time: 1.22 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0073164, 0.0046110, -0.0077420, 0.0038635, -0.0111799, 0.0123530
1: -0.0060074, -0.0012825, -0.0061760, -0.0015786, -0.0044288, 0.0048935
2: 0.0279787, 0.0397808, 0.0286751, 0.0404991, -0.0125204, 0.0111057
3: -0.0068321, 0.0052212, -0.0061071, 0.0056827, -0.0108937, 0.0097161
4: -0.0054036, 0.0049448, -0.0057211, 0.0041045, -0.0095081, 0.0106659
5: 0.0070017, 0.0162363, 0.0066953, 0.0156148, -0.0086132, 0.0095411
6: -0.0110970, 0.0021734, -0.0100734, 0.0025952, -0.0136922, 0.0122468
7: 0.9667950, 0.9835703, 0.9659583, 0.9825777, -0.0157827, 0.0176120
8: -0.0210824, -0.0009673, -0.0193487, -0.0005052, -0.0205773, 0.0183814
9: -0.0041022, 0.0082005, -0.0044155, 0.0071969, -0.0112991, 0.0126160

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109508, upper bound: 0.0102154
time: 1.27 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106361, upper bound: 0.0102147
time: 1.08 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0073164, 0.0046110, -0.0073164, 0.0046110, -0.0119274, 0.0119274
1: -0.0060074, -0.0012825, -0.0060074, -0.0012825, -0.0047249, 0.0047249
2: 0.0279787, 0.0397808, 0.0279787, 0.0397808, -0.0118021, 0.0118021
3: -0.0068321, 0.0052212, -0.0068321, 0.0052212, -0.0102658, 0.0102658
4: -0.0054036, 0.0049448, -0.0054036, 0.0049448, -0.0103483, 0.0103483
5: 0.0070017, 0.0162363, 0.0070017, 0.0162363, -0.0092347, 0.0092347
6: -0.0110970, 0.0021734, -0.0110970, 0.0021734, -0.0132705, 0.0132705
7: 0.9667950, 0.9835703, 0.9667950, 0.9835703, -0.0167753, 0.0167753
8: -0.0210824, -0.0009673, -0.0210824, -0.0009673, -0.0201151, 0.0201151
9: -0.0041022, 0.0082005, -0.0041022, 0.0082005, -0.0123027, 0.0123027

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109561, upper bound: 0.0104459
time: 1.01 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106379, upper bound: 0.0104440
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0073164, 0.0046110, -0.0080136, 0.0042946, -0.0116110, 0.0126246
1: -0.0060074, -0.0012825, -0.0062836, -0.0014078, -0.0045996, 0.0050011
2: 0.0279787, 0.0397808, 0.0282735, 0.0409575, -0.0129788, 0.0115073
3: -0.0068321, 0.0052212, -0.0065252, 0.0059773, -0.0110530, 0.0099861
4: -0.0054036, 0.0049448, -0.0059238, 0.0045891, -0.0099927, 0.0108686
5: 0.0070017, 0.0162363, 0.0064997, 0.0159733, -0.0089716, 0.0097366
6: -0.0110970, 0.0021734, -0.0106638, 0.0028644, -0.0139614, 0.0128372
7: 0.9667950, 0.9835703, 0.9654242, 0.9831502, -0.0163552, 0.0181461
8: -0.0210824, -0.0009673, -0.0203486, -0.0002102, -0.0208722, 0.0193813
9: -0.0041022, 0.0082005, -0.0046155, 0.0077757, -0.0118779, 0.0128160

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109561, upper bound: 0.0104459
time: 1.05 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106379, upper bound: 0.0104440
time: 0.97 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0073164, 0.0046110, -0.0071739, 0.0043496, -0.0116660, 0.0117849
1: -0.0060074, -0.0012825, -0.0059510, -0.0013860, -0.0046214, 0.0046685
2: 0.0279787, 0.0397808, 0.0282222, 0.0395402, -0.0115615, 0.0115586
3: -0.0068321, 0.0052212, -0.0065786, 0.0050667, -0.0102695, 0.0101708
4: -0.0054036, 0.0049448, -0.0052972, 0.0046510, -0.0100545, 0.0102420
5: 0.0070017, 0.0162363, 0.0071043, 0.0160190, -0.0090174, 0.0091321
6: -0.0110970, 0.0021734, -0.0107391, 0.0020322, -0.0131292, 0.0129125
7: 0.9667950, 0.9835703, 0.9670752, 0.9832233, -0.0164283, 0.0164950
8: -0.0210824, -0.0009673, -0.0204762, -0.0011220, -0.0199604, 0.0195089
9: -0.0041022, 0.0082005, -0.0039973, 0.0078496, -0.0119518, 0.0121977

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110650, upper bound: 0.0102087
time: 1.16 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106415, upper bound: 0.0101985
time: 1.14 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0073164, 0.0046110, -0.0079710, 0.0041366, -0.0114530, 0.0125820
1: -0.0060074, -0.0012825, -0.0062668, -0.0014704, -0.0045370, 0.0049843
2: 0.0279787, 0.0397808, 0.0284206, 0.0408856, -0.0129069, 0.0113602
3: -0.0068321, 0.0052212, -0.0063720, 0.0059311, -0.0111860, 0.0100113
4: -0.0054036, 0.0049448, -0.0058920, 0.0044116, -0.0098151, 0.0108368
5: 0.0070017, 0.0162363, 0.0065304, 0.0158419, -0.0088403, 0.0097060
6: -0.0110970, 0.0021734, -0.0104475, 0.0028222, -0.0139192, 0.0126209
7: 0.9667950, 0.9835703, 0.9655080, 0.9829404, -0.0161455, 0.0180623
8: -0.0210824, -0.0009673, -0.0199822, -0.0002565, -0.0208260, 0.0190149
9: -0.0041022, 0.0082005, -0.0045842, 0.0075637, -0.0116659, 0.0127846

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110650, upper bound: 0.0102087
time: 1.08 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106415, upper bound: 0.0101985
time: 0.96 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0073164, 0.0046110, -0.0074819, 0.0048139, -0.0121303, 0.0120928
1: -0.0060074, -0.0012825, -0.0060730, -0.0012021, -0.0048053, 0.0047905
2: 0.0279787, 0.0397808, 0.0277897, 0.0400601, -0.0120813, 0.0119911
3: -0.0068321, 0.0052212, -0.0070288, 0.0054006, -0.0104723, 0.0104781
4: -0.0054036, 0.0049448, -0.0055270, 0.0051728, -0.0105764, 0.0104718
5: 0.0070017, 0.0162363, 0.0068825, 0.0164050, -0.0094034, 0.0093538
6: -0.0110970, 0.0021734, -0.0113749, 0.0023374, -0.0134344, 0.0135483
7: 0.9667950, 0.9835703, 0.9664697, 0.9838396, -0.0170446, 0.0171006
8: -0.0210824, -0.0009673, -0.0215529, -0.0007876, -0.0202948, 0.0205857
9: -0.0041022, 0.0082005, -0.0042240, 0.0084728, -0.0125750, 0.0124245

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110659, upper bound: 0.0104107
time: 1.13 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106423, upper bound: 0.0104075
time: 1.24 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0073164, 0.0046110, -0.0082616, 0.0045814, -0.0118978, 0.0128726
1: -0.0060074, -0.0012825, -0.0063819, -0.0012942, -0.0047132, 0.0050994
2: 0.0279787, 0.0397808, 0.0280063, 0.0413762, -0.0133975, 0.0117745
3: -0.0068321, 0.0052212, -0.0068034, 0.0062463, -0.0113668, 0.0102985
4: -0.0054036, 0.0049448, -0.0061089, 0.0049115, -0.0103151, 0.0110537
5: 0.0070017, 0.0162363, 0.0063211, 0.0162117, -0.0092101, 0.0099152
6: -0.0110970, 0.0021734, -0.0110565, 0.0031102, -0.0142073, 0.0132299
7: 0.9667950, 0.9835703, 0.9649364, 0.9835309, -0.0167359, 0.0186338
8: -0.0210824, -0.0009673, -0.0210137, 0.0000591, -0.0211416, 0.0200465
9: -0.0041022, 0.0082005, -0.0047981, 0.0081607, -0.0122629, 0.0129986

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 106
type: A, layer: 3, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110659, upper bound: 0.0104107
time: 1.31 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106423, upper bound: 0.0104075
time: 0.98 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 3.76 seconds
NS_A2_B1_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 7, lower bound: -0.0109508, upper bound: 0.0102154
NS_A2_B1_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 7, lower bound: -0.0106361, upper bound: 0.0102147
NS_A2_B1_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 7, lower bound: -0.0109508, upper bound: 0.0102154
NS_A2_B1_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 7, lower bound: -0.0106361, upper bound: 0.0102147
NS_A2_B1_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 7, lower bound: -0.0109561, upper bound: 0.0104459
NS_A2_B1_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 7, lower bound: -0.0106379, upper bound: 0.0104440
NS_A2_B1_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 7, lower bound: -0.0109561, upper bound: 0.0104459
NS_A2_B1_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 7, lower bound: -0.0106379, upper bound: 0.0104440
NS_A2_B2_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 7, lower bound: -0.0110650, upper bound: 0.0102087
NS_A2_B2_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 7, lower bound: -0.0106415, upper bound: 0.0101985
NS_A2_B2_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 7, lower bound: -0.0110650, upper bound: 0.0102087
NS_A2_B2_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 7, lower bound: -0.0106415, upper bound: 0.0101985
NS_A2_B2_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 7, lower bound: -0.0110659, upper bound: 0.0104107
NS_A2_B2_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 7, lower bound: -0.0106423, upper bound: 0.0104075
NS_A2_B2_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 7, lower bound: -0.0110659, upper bound: 0.0104107
NS_A2_B2_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.76
Output dim: 7, lower bound: -0.0106423, upper bound: 0.0104075

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0073164, 0.0046110, -0.0070135, 0.0041515, -0.0114679, 0.0116245
1: -0.0060074, -0.0012825, -0.0058875, -0.0014645, -0.0045429, 0.0046050
2: 0.0279787, 0.0397808, 0.0284068, 0.0392696, -0.0112908, 0.0113740
3: -0.0068321, 0.0052212, -0.0063865, 0.0048927, -0.0100703, 0.0099643
4: -0.0054036, 0.0049448, -0.0051775, 0.0044283, -0.0098319, 0.0101223
5: 0.0070017, 0.0162363, 0.0072197, 0.0158543, -0.0088527, 0.0090166
6: -0.0110970, 0.0021734, -0.0104678, 0.0018732, -0.0129703, 0.0126413
7: 0.9667950, 0.9835703, 0.9673906, 0.9829602, -0.0161652, 0.0161797
8: -0.0210824, -0.0009673, -0.0200167, -0.0012962, -0.0197862, 0.0190495
9: -0.0041022, 0.0082005, -0.0038792, 0.0075837, -0.0116859, 0.0120797

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: B, layer: 3, pos: 13

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106445, upper bound: 0.0105998
time: 1.15 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106361, upper bound: 0.0102147
time: 1.24 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0080136, 0.0042946, -0.0070135, 0.0041515, -0.0121651, 0.0113081
1: -0.0062836, -0.0014078, -0.0058875, -0.0014645, -0.0048191, 0.0044796
2: 0.0282735, 0.0409575, 0.0284068, 0.0392696, -0.0109961, 0.0125508
3: -0.0065252, 0.0059773, -0.0063865, 0.0048927, -0.0097835, 0.0107397
4: -0.0059238, 0.0045891, -0.0051775, 0.0044283, -0.0103521, 0.0097666
5: 0.0064997, 0.0159733, 0.0072197, 0.0158543, -0.0093546, 0.0087535
6: -0.0106638, 0.0028644, -0.0104678, 0.0018732, -0.0125370, 0.0133322
7: 0.9654242, 0.9831502, 0.9673906, 0.9829602, -0.0175360, 0.0157596
8: -0.0203486, -0.0002102, -0.0200167, -0.0012962, -0.0190524, 0.0198065
9: -0.0046155, 0.0077757, -0.0038792, 0.0075837, -0.0121992, 0.0116549

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: B, layer: 3, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106445, upper bound: 0.0105998
time: 1.17 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106361, upper bound: 0.0102147
time: 1.08 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0073164, 0.0046110, -0.0077420, 0.0038635, -0.0111799, 0.0123530
1: -0.0060074, -0.0012825, -0.0061760, -0.0015786, -0.0044288, 0.0048935
2: 0.0279787, 0.0397808, 0.0286751, 0.0404991, -0.0125204, 0.0111057
3: -0.0068321, 0.0052212, -0.0061071, 0.0056827, -0.0108937, 0.0097161
4: -0.0054036, 0.0049448, -0.0057211, 0.0041045, -0.0095081, 0.0106659
5: 0.0070017, 0.0162363, 0.0066953, 0.0156148, -0.0086132, 0.0095411
6: -0.0110970, 0.0021734, -0.0100734, 0.0025952, -0.0136922, 0.0122468
7: 0.9667950, 0.9835703, 0.9659583, 0.9825777, -0.0157827, 0.0176120
8: -0.0210824, -0.0009673, -0.0193487, -0.0005052, -0.0205773, 0.0183814
9: -0.0041022, 0.0082005, -0.0044155, 0.0071969, -0.0112991, 0.0126160

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: B, layer: 3, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106361, upper bound: 0.0102147
time: 1.23 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106361, upper bound: 0.0102147
time: 1.01 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0080136, 0.0042946, -0.0077420, 0.0038635, -0.0118770, 0.0120366
1: -0.0062836, -0.0014078, -0.0061760, -0.0015786, -0.0047050, 0.0047682
2: 0.0282735, 0.0409575, 0.0286751, 0.0404991, -0.0122256, 0.0122825
3: -0.0065252, 0.0059773, -0.0061071, 0.0056827, -0.0104791, 0.0103708
4: -0.0059238, 0.0045891, -0.0057211, 0.0041045, -0.0100283, 0.0103102
5: 0.0064997, 0.0159733, 0.0066953, 0.0156148, -0.0091151, 0.0092780
6: -0.0106638, 0.0028644, -0.0100734, 0.0025952, -0.0132590, 0.0129378
7: 0.9654242, 0.9831502, 0.9659583, 0.9825777, -0.0171535, 0.0171919
8: -0.0203486, -0.0002102, -0.0193487, -0.0005052, -0.0198434, 0.0191385
9: -0.0046155, 0.0077757, -0.0044155, 0.0071969, -0.0118124, 0.0121912

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: B, layer: 3, pos: 13

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106361, upper bound: 0.0102147
time: 1.09 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106361, upper bound: 0.0102147
time: 0.99 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0073164, 0.0046110, -0.0073164, 0.0046110, -0.0119274, 0.0119274
1: -0.0060074, -0.0012825, -0.0060074, -0.0012825, -0.0047249, 0.0047249
2: 0.0279787, 0.0397808, 0.0279787, 0.0397808, -0.0118021, 0.0118021
3: -0.0068321, 0.0052212, -0.0068321, 0.0052212, -0.0102658, 0.0102658
4: -0.0054036, 0.0049448, -0.0054036, 0.0049448, -0.0103483, 0.0103483
5: 0.0070017, 0.0162363, 0.0070017, 0.0162363, -0.0092347, 0.0092347
6: -0.0110970, 0.0021734, -0.0110970, 0.0021734, -0.0132705, 0.0132705
7: 0.9667950, 0.9835703, 0.9667950, 0.9835703, -0.0167753, 0.0167753
8: -0.0210824, -0.0009673, -0.0210824, -0.0009673, -0.0201151, 0.0201151
9: -0.0041022, 0.0082005, -0.0041022, 0.0082005, -0.0123027, 0.0123027

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: B, layer: 3, pos: 13

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106445, upper bound: 0.0106530
time: 1.24 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106379, upper bound: 0.0104440
time: 0.97 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0080136, 0.0042946, -0.0073164, 0.0046110, -0.0126246, 0.0116110
1: -0.0062836, -0.0014078, -0.0060074, -0.0012825, -0.0050011, 0.0045996
2: 0.0282735, 0.0409575, 0.0279787, 0.0397808, -0.0115073, 0.0129788
3: -0.0065252, 0.0059773, -0.0068321, 0.0052212, -0.0099861, 0.0110530
4: -0.0059238, 0.0045891, -0.0054036, 0.0049448, -0.0108686, 0.0099927
5: 0.0064997, 0.0159733, 0.0070017, 0.0162363, -0.0097366, 0.0089716
6: -0.0106638, 0.0028644, -0.0110970, 0.0021734, -0.0128372, 0.0139614
7: 0.9654242, 0.9831502, 0.9667950, 0.9835703, -0.0181461, 0.0163552
8: -0.0203486, -0.0002102, -0.0210824, -0.0009673, -0.0193813, 0.0208722
9: -0.0046155, 0.0077757, -0.0041022, 0.0082005, -0.0128160, 0.0118779

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: B, layer: 3, pos: 13

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106445, upper bound: 0.0106530
time: 1.18 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106379, upper bound: 0.0104440
time: 0.99 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0073164, 0.0046110, -0.0080136, 0.0042946, -0.0116110, 0.0126246
1: -0.0060074, -0.0012825, -0.0062836, -0.0014078, -0.0045996, 0.0050011
2: 0.0279787, 0.0397808, 0.0282735, 0.0409575, -0.0129788, 0.0115073
3: -0.0068321, 0.0052212, -0.0065252, 0.0059773, -0.0110530, 0.0099861
4: -0.0054036, 0.0049448, -0.0059238, 0.0045891, -0.0099927, 0.0108686
5: 0.0070017, 0.0162363, 0.0064997, 0.0159733, -0.0089716, 0.0097366
6: -0.0110970, 0.0021734, -0.0106638, 0.0028644, -0.0139614, 0.0128372
7: 0.9667950, 0.9835703, 0.9654242, 0.9831502, -0.0163552, 0.0181461
8: -0.0210824, -0.0009673, -0.0203486, -0.0002102, -0.0208722, 0.0193813
9: -0.0041022, 0.0082005, -0.0046155, 0.0077757, -0.0118779, 0.0128160

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: B, layer: 3, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106379, upper bound: 0.0104440
time: 0.96 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106379, upper bound: 0.0104440
time: 1.03 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0080136, 0.0042946, -0.0080136, 0.0042946, -0.0123082, 0.0123082
1: -0.0062836, -0.0014078, -0.0062836, -0.0014078, -0.0048758, 0.0048758
2: 0.0282735, 0.0409575, 0.0282735, 0.0409575, -0.0126840, 0.0126840
3: -0.0065252, 0.0059773, -0.0065252, 0.0059773, -0.0106532, 0.0106532
4: -0.0059238, 0.0045891, -0.0059238, 0.0045891, -0.0105129, 0.0105129
5: 0.0064997, 0.0159733, 0.0064997, 0.0159733, -0.0094736, 0.0094736
6: -0.0106638, 0.0028644, -0.0106638, 0.0028644, -0.0135282, 0.0135282
7: 0.9654242, 0.9831502, 0.9654242, 0.9831502, -0.0177260, 0.0177260
8: -0.0203486, -0.0002102, -0.0203486, -0.0002102, -0.0201383, 0.0201383
9: -0.0046155, 0.0077757, -0.0046155, 0.0077757, -0.0123912, 0.0123912

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: B, layer: 3, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106379, upper bound: 0.0104440
time: 0.96 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106379, upper bound: 0.0104440
time: 1.09 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0073164, 0.0046110, -0.0071739, 0.0043496, -0.0116660, 0.0117849
1: -0.0060074, -0.0012825, -0.0059510, -0.0013860, -0.0046214, 0.0046685
2: 0.0279787, 0.0397808, 0.0282222, 0.0395402, -0.0115615, 0.0115586
3: -0.0068321, 0.0052212, -0.0065786, 0.0050667, -0.0102695, 0.0101708
4: -0.0054036, 0.0049448, -0.0052972, 0.0046510, -0.0100545, 0.0102420
5: 0.0070017, 0.0162363, 0.0071043, 0.0160190, -0.0090174, 0.0091321
6: -0.0110970, 0.0021734, -0.0107391, 0.0020322, -0.0131292, 0.0129125
7: 0.9667950, 0.9835703, 0.9670752, 0.9832233, -0.0164283, 0.0164950
8: -0.0210824, -0.0009673, -0.0204762, -0.0011220, -0.0199604, 0.0195089
9: -0.0041022, 0.0082005, -0.0039973, 0.0078496, -0.0119518, 0.0121977

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: B, layer: 3, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106445, upper bound: 0.0105584
time: 1.19 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106415, upper bound: 0.0101985
time: 0.99 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0080136, 0.0042946, -0.0071739, 0.0043496, -0.0123632, 0.0114685
1: -0.0062836, -0.0014078, -0.0059510, -0.0013860, -0.0048976, 0.0045431
2: 0.0282735, 0.0409575, 0.0282222, 0.0395402, -0.0112667, 0.0127353
3: -0.0065252, 0.0059773, -0.0065786, 0.0050667, -0.0099826, 0.0109462
4: -0.0059238, 0.0045891, -0.0052972, 0.0046510, -0.0105748, 0.0098863
5: 0.0064997, 0.0159733, 0.0071043, 0.0160190, -0.0095193, 0.0088690
6: -0.0106638, 0.0028644, -0.0107391, 0.0020322, -0.0126959, 0.0136035
7: 0.9654242, 0.9831502, 0.9670752, 0.9832233, -0.0177991, 0.0160750
8: -0.0203486, -0.0002102, -0.0204762, -0.0011220, -0.0192265, 0.0202660
9: -0.0046155, 0.0077757, -0.0039973, 0.0078496, -0.0124651, 0.0117730

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: B, layer: 3, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106445, upper bound: 0.0105584
time: 1.17 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106415, upper bound: 0.0101985
time: 0.97 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0073164, 0.0046110, -0.0079710, 0.0041366, -0.0114530, 0.0125820
1: -0.0060074, -0.0012825, -0.0062668, -0.0014704, -0.0045370, 0.0049843
2: 0.0279787, 0.0397808, 0.0284206, 0.0408856, -0.0129069, 0.0113602
3: -0.0068321, 0.0052212, -0.0063720, 0.0059311, -0.0111860, 0.0100113
4: -0.0054036, 0.0049448, -0.0058920, 0.0044116, -0.0098151, 0.0108368
5: 0.0070017, 0.0162363, 0.0065304, 0.0158419, -0.0088403, 0.0097060
6: -0.0110970, 0.0021734, -0.0104475, 0.0028222, -0.0139192, 0.0126209
7: 0.9667950, 0.9835703, 0.9655080, 0.9829404, -0.0161455, 0.0180623
8: -0.0210824, -0.0009673, -0.0199822, -0.0002565, -0.0208260, 0.0190149
9: -0.0041022, 0.0082005, -0.0045842, 0.0075637, -0.0116659, 0.0127846

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: B, layer: 3, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106415, upper bound: 0.0101985
time: 1.02 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106415, upper bound: 0.0101985
time: 1.18 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0080136, 0.0042946, -0.0079710, 0.0041366, -0.0121502, 0.0122656
1: -0.0062836, -0.0014078, -0.0062668, -0.0014704, -0.0048132, 0.0048589
2: 0.0282735, 0.0409575, 0.0284206, 0.0408856, -0.0126121, 0.0125369
3: -0.0065252, 0.0059773, -0.0063720, 0.0059311, -0.0107795, 0.0106726
4: -0.0059238, 0.0045891, -0.0058920, 0.0044116, -0.0103354, 0.0104811
5: 0.0064997, 0.0159733, 0.0065304, 0.0158419, -0.0093422, 0.0094429
6: -0.0106638, 0.0028644, -0.0104475, 0.0028222, -0.0134859, 0.0133119
7: 0.9654242, 0.9831502, 0.9655080, 0.9829404, -0.0175163, 0.0176422
8: -0.0203486, -0.0002102, -0.0199822, -0.0002565, -0.0200921, 0.0197720
9: -0.0046155, 0.0077757, -0.0045842, 0.0075637, -0.0121792, 0.0123599

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: B, layer: 3, pos: 13

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106415, upper bound: 0.0101985
time: 1.09 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106415, upper bound: 0.0101985
time: 1.23 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0073164, 0.0046110, -0.0074819, 0.0048139, -0.0121303, 0.0120928
1: -0.0060074, -0.0012825, -0.0060730, -0.0012021, -0.0048053, 0.0047905
2: 0.0279787, 0.0397808, 0.0277897, 0.0400601, -0.0120813, 0.0119911
3: -0.0068321, 0.0052212, -0.0070288, 0.0054006, -0.0104723, 0.0104781
4: -0.0054036, 0.0049448, -0.0055270, 0.0051728, -0.0105764, 0.0104718
5: 0.0070017, 0.0162363, 0.0068825, 0.0164050, -0.0094034, 0.0093538
6: -0.0110970, 0.0021734, -0.0113749, 0.0023374, -0.0134344, 0.0135483
7: 0.9667950, 0.9835703, 0.9664697, 0.9838396, -0.0170446, 0.0171006
8: -0.0210824, -0.0009673, -0.0215529, -0.0007876, -0.0202948, 0.0205857
9: -0.0041022, 0.0082005, -0.0042240, 0.0084728, -0.0125750, 0.0124245

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: B, layer: 3, pos: 13

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106445, upper bound: 0.0106088
time: 1.30 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106423, upper bound: 0.0104075
time: 1.27 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0080136, 0.0042946, -0.0074819, 0.0048139, -0.0128275, 0.0117764
1: -0.0062836, -0.0014078, -0.0060730, -0.0012021, -0.0050815, 0.0046651
2: 0.0282735, 0.0409575, 0.0277897, 0.0400601, -0.0117866, 0.0131678
3: -0.0065252, 0.0059773, -0.0070288, 0.0054006, -0.0101926, 0.0112653
4: -0.0059238, 0.0045891, -0.0055270, 0.0051728, -0.0110967, 0.0101161
5: 0.0064997, 0.0159733, 0.0068825, 0.0164050, -0.0099053, 0.0090907
6: -0.0106638, 0.0028644, -0.0113749, 0.0023374, -0.0130012, 0.0142393
7: 0.9654242, 0.9831502, 0.9664697, 0.9838396, -0.0184154, 0.0166805
8: -0.0203486, -0.0002102, -0.0215529, -0.0007876, -0.0195609, 0.0213427
9: -0.0046155, 0.0077757, -0.0042240, 0.0084728, -0.0130883, 0.0119997

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 106
type: B, layer: 3, pos: 13

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 241

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106445, upper bound: 0.0106088
time: 1.30 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106423, upper bound: 0.0104075
time: 1.31 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.02 + 599.40 = 602.42 seconds
