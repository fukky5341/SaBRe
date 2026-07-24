## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.542703008


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.7778639, 0.6806120, -0.7778639, 0.6806120, -1.4584759, 1.4584759)
1: (-0.6040506, 0.8626654, -0.6040506, 0.8626654, -1.4667161, 1.4667161)
2: (-0.5841001, 0.8418483, -0.5841001, 0.8418483, -1.4259484, 1.4259484)
3: (-0.6007968, 0.6236554, -0.6007968, 0.6236554, -1.2244523, 1.2244523)
4: (-0.7511072, 0.7287243, -0.7511072, 0.7287243, -1.4798315, 1.4798315)
5: (-0.5852865, 1.1952124, -0.5852865, 1.1952124, -1.7804989, 1.7804989)
6: (-0.4886691, 0.6989943, -0.4886691, 0.6989943, -1.1876633, 1.1876633)
7: (-0.6124615, 0.7924250, -0.6124615, 0.7924250, -1.4048865, 1.4048865)
8: (-0.6664105, 0.8338409, -0.6664105, 0.8338409, -1.5002514, 1.5002514)
9: (-0.6932293, 0.7872544, -0.6932293, 0.7872544, -1.4804838, 1.4804838)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.83 + 3.05 = 4.88 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -1.6069825, upper bound: 1.6069825

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6069825, upper bound: 1.6034144
time: 1.61 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6069825, upper bound: 1.6069825
time: 1.54 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 3.29 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 3.29
Output dim: 5, lower bound: -1.6069825, upper bound: 1.6034144
NS_A2, status: Status.UNKNOWN, split count: 1, time: 3.29
Output dim: 5, lower bound: -1.6069825, upper bound: 1.6069825

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.6690523, 0.6263713, -0.7594505, 0.6713563, -1.3404086, 1.3858218
1: -0.5400118, 0.7853968, -0.5931420, 0.8476349, -1.3876467, 1.3785387
2: -0.4943116, 0.7711002, -0.5687314, 0.8294503, -1.3237619, 1.3398316
3: -0.4797419, 0.5831993, -0.5795134, 0.6166465, -1.0963885, 1.1627127
4: -0.6595763, 0.6474200, -0.7350866, 0.7150322, -1.3746085, 1.3825066
5: -0.4520540, 1.1636808, -0.5617863, 1.1898608, -1.6419148, 1.7254671
6: -0.4175576, 0.6317892, -0.4763883, 0.6869192, -1.1044768, 1.1081775
7: -0.5387849, 0.7089120, -0.5998350, 0.7779419, -1.3167267, 1.3087469
8: -0.5664127, 0.7809650, -0.6493624, 0.8241662, -1.3905790, 1.4303274
9: -0.6160759, 0.7162293, -0.6793324, 0.7748278, -1.3909037, 1.3955617

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5933409, upper bound: 1.5335729
time: 1.77 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6006319, upper bound: 1.5960725
time: 1.68 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.7327614, 0.6581467, -0.7778639, 0.6806120, -1.4133735, 1.4360106
1: -0.5771269, 0.8276222, -0.6040506, 0.8626654, -1.4397924, 1.4316728
2: -0.5463867, 0.8121164, -0.5841001, 0.8418483, -1.3882351, 1.3962165
3: -0.5495321, 0.6067683, -0.6007968, 0.6236554, -1.1731875, 1.2075651
4: -0.7122568, 0.6951182, -0.7511072, 0.7287243, -1.4409811, 1.4462254
5: -0.5293463, 1.1866642, -0.5852865, 1.1952124, -1.7245587, 1.7719507
6: -0.4586701, 0.6701459, -0.4886691, 0.6989943, -1.1576644, 1.1588150
7: -0.5816387, 0.7571935, -0.6124615, 0.7924250, -1.3740637, 1.3696550
8: -0.6244087, 0.8113776, -0.6664105, 0.8338409, -1.4582496, 1.4777881
9: -0.6603124, 0.7568712, -0.6932293, 0.7872544, -1.4475667, 1.4501005

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6034144, upper bound: 1.6069825
time: 2.25 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6034144, upper bound: 1.6069825
time: 1.97 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.83 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.83
Output dim: 5, lower bound: -1.5933409, upper bound: 1.5335729
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.83
Output dim: 5, lower bound: -1.6006319, upper bound: 1.5960725
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.83
Output dim: 5, lower bound: -1.6034144, upper bound: 1.6069825
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.83
Output dim: 5, lower bound: -1.6034144, upper bound: 1.6069825

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.5880563, 0.5834422, -0.4353095, 0.4575914, -1.0456477, 1.0187516
1: -0.4928609, 0.7026590, -0.3834313, 0.4234885, -0.9163494, 1.0860903
2: -0.4295314, 0.7146801, -0.3101644, 0.5421547, -0.9716861, 1.0248444
3: -0.3920290, 0.5538729, -0.2500201, 0.4638594, -0.8558884, 0.8038930
4: -0.5943097, 0.5833899, -0.4602029, 0.4465656, -1.0408752, 1.0435929
5: -0.3338335, 1.1584902, 0.0271171, 1.1721300, -1.5059634, 1.1313732
6: -0.3624999, 0.5845274, -0.2629426, 0.4778900, -0.8403898, 0.8474700
7: -0.4822837, 0.6464243, -0.3561301, 0.4907650, -0.9730486, 1.0025544
8: -0.4948626, 0.7394735, -0.3613905, 0.6128445, -1.1077070, 1.1008639
9: -0.5617329, 0.6611689, -0.4292725, 0.5236725, -1.0854053, 1.0904413

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5845202, upper bound: 1.5051724
time: 2.58 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5784411, upper bound: 1.5057209
time: 2.15 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.6690523, 0.6263713, -0.5942481, 0.5875452, -1.2565975, 1.2206194
1: -0.5400118, 0.7853968, -0.4958497, 0.7007765, -1.2407882, 1.2812464
2: -0.4943116, 0.7711002, -0.4337596, 0.7189949, -1.2133064, 1.2048597
3: -0.4797419, 0.5831993, -0.3976308, 0.5567792, -1.0365212, 0.9808301
4: -0.6595763, 0.6474200, -0.5984763, 0.5877365, -1.2473128, 1.2458963
5: -0.4520540, 1.1636808, -0.3387501, 1.1819992, -1.6340532, 1.5024309
6: -0.4175576, 0.6317892, -0.3665989, 0.5874977, -1.0050553, 0.9983881
7: -0.5387849, 0.7089120, -0.4859794, 0.6536972, -1.1924820, 1.1948913
8: -0.5664127, 0.7809650, -0.4994365, 0.7429436, -1.3093563, 1.2804015
9: -0.6160759, 0.7162293, -0.5655066, 0.6651008, -1.2811768, 1.2817359

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 8

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5335729, upper bound: 1.5871688
time: 1.73 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5335729, upper bound: 1.5960725
time: 1.71 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.7327614, 0.6581467, -0.6690523, 0.6263713, -1.3591326, 1.3271990
1: -0.5771269, 0.8276222, -0.5400118, 0.7853968, -1.3625237, 1.3676341
2: -0.5463867, 0.8121164, -0.4943116, 0.7711002, -1.3174869, 1.3064280
3: -0.5495321, 0.6067683, -0.4797419, 0.5831993, -1.1327314, 1.0865102
4: -0.7122568, 0.6951182, -0.6595763, 0.6474200, -1.3596768, 1.3546945
5: -0.5293463, 1.1866642, -0.4520540, 1.1636808, -1.6930271, 1.6387182
6: -0.4586701, 0.6701459, -0.4175576, 0.6317892, -1.0904593, 1.0877035
7: -0.5816387, 0.7571935, -0.5387849, 0.7089120, -1.2905507, 1.2959783
8: -0.6244087, 0.8113776, -0.5664127, 0.7809650, -1.4053736, 1.3777902
9: -0.6603124, 0.7568712, -0.6160759, 0.7162293, -1.3765417, 1.3729472

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5335729, upper bound: 1.5933409
time: 1.53 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5960725, upper bound: 1.6006319
time: 2.15 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.7327614, 0.6581467, -0.7327614, 0.6581467, -1.3909080, 1.3909080
1: -0.5771269, 0.8276222, -0.5771269, 0.8276222, -1.4047492, 1.4047492
2: -0.5463867, 0.8121164, -0.5463867, 0.8121164, -1.3585031, 1.3585031
3: -0.5495321, 0.6067683, -0.5495321, 0.6067683, -1.1563004, 1.1563004
4: -0.7122568, 0.6951182, -0.7122568, 0.6951182, -1.4073750, 1.4073750
5: -0.5293463, 1.1866642, -0.5293463, 1.1866642, -1.7160106, 1.7160106
6: -0.4586701, 0.6701459, -0.4586701, 0.6701459, -1.1288161, 1.1288161
7: -0.5816387, 0.7571935, -0.5816387, 0.7571935, -1.3388321, 1.3388321
8: -0.6244087, 0.8113776, -0.6244087, 0.8113776, -1.4357862, 1.4357862
9: -0.6603124, 0.7568712, -0.6603124, 0.7568712, -1.4171836, 1.4171836

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5335729, upper bound: 1.5933409
time: 1.68 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5960725, upper bound: 1.6006319
time: 2.16 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.63 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.63
Output dim: 5, lower bound: -1.5845202, upper bound: 1.5051724
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.63
Output dim: 5, lower bound: -1.5784411, upper bound: 1.5057209
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.63
Output dim: 5, lower bound: -1.5335729, upper bound: 1.5871688
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.63
Output dim: 5, lower bound: -1.5335729, upper bound: 1.5960725
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.63
Output dim: 5, lower bound: -1.5335729, upper bound: 1.5933409
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.63
Output dim: 5, lower bound: -1.5960725, upper bound: 1.6006319
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.63
Output dim: 5, lower bound: -1.5335729, upper bound: 1.5933409
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.63
Output dim: 5, lower bound: -1.5960725, upper bound: 1.6006319

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.5266285, 0.5453887, -0.4353095, 0.4575914, -0.9842199, 0.9806982
1: -0.4571671, 0.6573631, -0.3834313, 0.4234885, -0.8806556, 1.0407944
2: -0.3830554, 0.6647385, -0.3101644, 0.5421547, -0.9252101, 0.9749029
3: -0.3286504, 0.5287566, -0.2500201, 0.4638594, -0.7925098, 0.7787766
4: -0.5463063, 0.5355127, -0.4602029, 0.4465656, -0.9928718, 0.9957156
5: -0.2596720, 1.1396759, 0.0271171, 1.1721300, -1.4318020, 1.1125588
6: -0.3258136, 0.5480757, -0.2629426, 0.4778900, -0.8037035, 0.8110182
7: -0.4365955, 0.5957977, -0.3561301, 0.4907650, -0.9273605, 0.9519278
8: -0.4451951, 0.7030770, -0.3613905, 0.6128445, -1.0580397, 1.0644674
9: -0.5162947, 0.6176199, -0.4292725, 0.5236725, -1.0399672, 1.0468924

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 131

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5783356, upper bound: 1.5051724
time: 1.96 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5783356, upper bound: 1.5051724
time: 1.91 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.8631139, 0.7372196, -0.4321812, 0.4540422, -1.3171561, 1.1694009
1: -0.6564021, 0.8717753, -0.3805148, 0.4194371, -1.0758393, 1.2522900
2: -0.6477537, 0.9167081, -0.3079228, 0.5371533, -1.1849070, 1.2246308
3: -0.6959582, 0.6527142, -0.2478611, 0.4614923, -1.1574504, 0.9005753
4: -0.8151168, 0.8005915, -0.4569366, 0.4434488, -1.2585657, 1.2575281
5: -0.6403569, 1.1529043, 0.0340992, 1.1704886, -1.8108455, 1.1188051
6: -0.5604084, 0.7439091, -0.2607099, 0.4751419, -1.0355502, 1.0046191
7: -0.6641110, 0.8826647, -0.3536389, 0.4865561, -1.1506670, 1.2363036
8: -0.7382755, 0.8849251, -0.3583214, 0.6093422, -1.3476176, 1.2432466
9: -0.7576594, 0.8534768, -0.4259813, 0.5201769, -1.2778363, 1.2794580

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 131

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5705206, upper bound: 1.5057209
time: 2.10 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5705206, upper bound: 1.5057209
time: 2.20 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.4026899, 0.4191411, -0.5942481, 0.5875452, -0.9902351, 1.0133892
1: -0.3521138, 0.3794717, -0.4958497, 0.7007765, -1.0528903, 0.8753213
2: -0.2859992, 0.4886915, -0.4337596, 0.7189949, -1.0049942, 0.9224511
3: -0.2267117, 0.4386235, -0.3976308, 0.5567792, -0.7834909, 0.8362544
4: -0.4259376, 0.4128717, -0.5984763, 0.5877365, -1.0136740, 1.0113480
5: 0.1034426, 1.1441865, -0.3387501, 1.1819992, -1.0785565, 1.4829366
6: -0.2389207, 0.4489258, -0.3665989, 0.5874977, -0.8264184, 0.8155247
7: -0.3291392, 0.4452140, -0.4859794, 0.6536972, -0.9828364, 0.9311934
8: -0.3282850, 0.5760045, -0.4994365, 0.7429436, -1.0712286, 1.0754410
9: -0.3940009, 0.4859381, -0.5655066, 0.6651008, -1.0591017, 1.0514448

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 8

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5060568, upper bound: 1.5780317
time: 1.72 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5060568, upper bound: 1.5705009
time: 1.96 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.5346003, 0.5504180, -0.5942481, 0.5875452, -1.1221454, 1.1446661
1: -0.4613726, 0.6573250, -0.4958497, 0.7007765, -1.1621491, 1.1531746
2: -0.3883922, 0.6712835, -0.4337596, 0.7189949, -1.1073872, 1.1050432
3: -0.3360583, 0.5322127, -0.3976308, 0.5567792, -0.8928375, 0.9298435
4: -0.5515785, 0.5412838, -0.5984763, 0.5877365, -1.1393150, 1.1397601
5: -0.2655362, 1.1559851, -0.3387501, 1.1819992, -1.4475354, 1.4947352
6: -0.3297866, 0.5524096, -0.3665989, 0.5874977, -0.9172843, 0.9190086
7: -0.4420976, 0.6037713, -0.4859794, 0.6536972, -1.0957948, 1.0897508
8: -0.4507531, 0.7079566, -0.4994365, 0.7429436, -1.1936967, 1.2073931
9: -0.5216522, 0.6228814, -0.5655066, 0.6651008, -1.1867530, 1.1883880

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5060568, upper bound: 1.5819786
time: 1.71 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5060568, upper bound: 1.5056951
time: 1.80 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.4273857, 0.4485906, -0.5880563, 0.5834422, -1.0108279, 1.0366470
1: -0.3760331, 0.4125217, -0.4928609, 0.7026590, -1.0786921, 0.9053825
2: -0.3044457, 0.5294700, -0.4295314, 0.7146801, -1.0191258, 0.9590014
3: -0.2445247, 0.4578295, -0.3920290, 0.5538729, -0.7983976, 0.8498585
4: -0.4519042, 0.4386348, -0.5943097, 0.5833899, -1.0352941, 1.0329444
5: 0.0454088, 1.1700767, -0.3338335, 1.1584902, -1.1130813, 1.5039102
6: -0.2572561, 0.4709196, -0.3624999, 0.5845274, -0.8417835, 0.8334195
7: -0.3497696, 0.4801137, -0.4822837, 0.6464243, -0.9961939, 0.9623974
8: -0.3535936, 0.6039436, -0.4948626, 0.7394735, -1.0930672, 1.0988061
9: -0.4209062, 0.5147840, -0.5617329, 0.6611689, -1.0820751, 1.0765169

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5845202
time: 2.12 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5057209, upper bound: 1.5784411
time: 2.01 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.5738508, 0.5759040, -0.6690523, 0.6263713, -1.2002220, 1.2449563
1: -0.4839600, 0.6847404, -0.5400118, 0.7853968, -1.2693567, 1.2247522
2: -0.4179204, 0.7037759, -0.4943116, 0.7711002, -1.1890206, 1.1980875
3: -0.3756537, 0.5489379, -0.4797419, 0.5831993, -0.9588531, 1.0286798
4: -0.5821376, 0.5716643, -0.6595763, 0.6474200, -1.2295578, 1.2312406
5: -0.3128855, 1.1789155, -0.4520540, 1.1636808, -1.4765663, 1.6309695
6: -0.3524698, 0.5755888, -0.4175576, 0.6317892, -0.9842590, 0.9931463
7: -0.4721063, 0.6370016, -0.5387849, 0.7089120, -1.1810184, 1.1757865
8: -0.4816706, 0.7319694, -0.5664127, 0.7809650, -1.2626355, 1.2983820
9: -0.5507526, 0.6506543, -0.6160759, 0.7162293, -1.2669820, 1.2667303

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5871688, upper bound: 1.5335729
time: 2.06 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5871688, upper bound: 1.6006319
time: 5.16 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.4273857, 0.4485906, -0.6471902, 0.6155454, -1.0429311, 1.0957808
1: -0.3760331, 0.4125217, -0.5268880, 0.7442890, -1.1203221, 0.9394097
2: -0.3044457, 0.5294700, -0.4754967, 0.7560903, -1.0605361, 1.0049667
3: -0.2445247, 0.4578295, -0.4540398, 0.5750561, -0.8195809, 0.9118693
4: -0.4519042, 0.4386348, -0.6405200, 0.6295185, -1.0814227, 1.0791547
5: 0.0454088, 1.1700767, -0.4067716, 1.1817236, -1.1363147, 1.5768483
6: -0.2572561, 0.4709196, -0.4024995, 0.6182482, -0.8755043, 0.8734192
7: -0.3497696, 0.4801137, -0.5225303, 0.6926979, -1.0424676, 1.0026441
8: -0.3535936, 0.6039436, -0.5463733, 0.7696142, -1.1232078, 1.1503170
9: -0.4209062, 0.5147840, -0.6008417, 0.7007641, -1.1216702, 1.1156256

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5845202
time: 2.16 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5057209, upper bound: 1.5784411
time: 1.98 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.5738508, 0.5759040, -0.7327614, 0.6581467, -1.2319975, 1.3086654
1: -0.4839600, 0.6847404, -0.5771269, 0.8276222, -1.3115822, 1.2618673
2: -0.4179204, 0.7037759, -0.5463867, 0.8121164, -1.2300369, 1.2501626
3: -0.3756537, 0.5489379, -0.5495321, 0.6067683, -0.9824221, 1.0984700
4: -0.5821376, 0.5716643, -0.7122568, 0.6951182, -1.2772558, 1.2839211
5: -0.3128855, 1.1789155, -0.5293463, 1.1866642, -1.4995497, 1.7082618
6: -0.3524698, 0.5755888, -0.4586701, 0.6701459, -1.0226157, 1.0342588
7: -0.4721063, 0.6370016, -0.5816387, 0.7571935, -1.2292998, 1.2186403
8: -0.4816706, 0.7319694, -0.6244087, 0.8113776, -1.2930481, 1.3563781
9: -0.5507526, 0.6506543, -0.6603124, 0.7568712, -1.3076239, 1.3109667

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5871688, upper bound: 1.5335729
time: 1.98 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5871688, upper bound: 1.6006319
time: 2.08 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 5.72 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.72
Output dim: 5, lower bound: -1.5783356, upper bound: 1.5051724
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.72
Output dim: 5, lower bound: -1.5783356, upper bound: 1.5051724
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.72
Output dim: 5, lower bound: -1.5705206, upper bound: 1.5057209
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.72
Output dim: 5, lower bound: -1.5705206, upper bound: 1.5057209
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.72
Output dim: 5, lower bound: -1.5060568, upper bound: 1.5780317
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.72
Output dim: 5, lower bound: -1.5060568, upper bound: 1.5705009
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.72
Output dim: 5, lower bound: -1.5060568, upper bound: 1.5819786
NS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 5.72
Output dim: 5, lower bound: -1.5060568, upper bound: 1.5056951
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.72
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5845202
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.72
Output dim: 5, lower bound: -1.5057209, upper bound: 1.5784411
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.72
Output dim: 5, lower bound: -1.5871688, upper bound: 1.5335729
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.72
Output dim: 5, lower bound: -1.5871688, upper bound: 1.6006319
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.72
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5845202
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.72
Output dim: 5, lower bound: -1.5057209, upper bound: 1.5784411
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.72
Output dim: 5, lower bound: -1.5871688, upper bound: 1.5335729
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.72
Output dim: 5, lower bound: -1.5871688, upper bound: 1.6006319

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.5266285, 0.5453887, -0.4142428, 0.4336945, -0.9603230, 0.9596315
1: -0.4571671, 0.6573631, -0.3638630, 0.3969387, -0.8541057, 1.0212260
2: -0.3830554, 0.6647385, -0.2951300, 0.5085570, -0.8916124, 0.9598686
3: -0.3286504, 0.5287566, -0.2355444, 0.4479266, -0.7765771, 0.7643009
4: -0.5463063, 0.5355127, -0.4382889, 0.4256445, -0.9719508, 0.9738016
5: -0.2596720, 1.1396759, 0.0736146, 1.1543233, -1.4139954, 1.0660613
6: -0.3258136, 0.5480757, -0.2479472, 0.4594488, -0.7852623, 0.7960229
7: -0.4365955, 0.5957977, -0.3394207, 0.4621921, -0.8987876, 0.9352184
8: -0.4451951, 0.7030770, -0.3407990, 0.5893049, -1.0345000, 1.0438759
9: -0.5162947, 0.6176199, -0.4071892, 0.5002099, -1.0165045, 1.0248091

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5328025, upper bound: 1.5051724
time: 1.91 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5328025, upper bound: 1.5051724
time: 2.06 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.5266285, 0.5453887, -0.5338054, 0.5689092, -1.0955377, 1.0791941
1: -0.4571671, 0.6573631, -0.4777257, 0.5175952, -0.9747623, 1.1350888
2: -0.3830554, 0.6647385, -0.3798169, 0.7022961, -1.0853516, 1.0445554
3: -0.3286504, 0.5287566, -0.3183542, 0.5356417, -0.8642921, 0.8471108
4: -0.5463063, 0.5355127, -0.5645366, 0.5447255, -1.0910318, 1.1000493
5: -0.2596720, 1.1396759, -0.1597792, 1.1618428, -1.4215149, 1.2994550
6: -0.3258136, 0.5480757, -0.3322386, 0.5666288, -0.8924424, 0.8803143
7: -0.4365955, 0.5957977, -0.4329141, 0.6207537, -1.0573492, 1.0287118
8: -0.4451951, 0.7030770, -0.4594866, 0.7227409, -1.1679361, 1.1625636
9: -0.5162947, 0.6176199, -0.5340111, 0.6340293, -1.1503241, 1.1516309

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5328025, upper bound: 1.5051724
time: 1.72 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5328025, upper bound: 1.5051724
time: 1.87 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.8631139, 0.7372196, -0.4001870, 0.4156592, -1.2787731, 1.1374066
1: -0.6564021, 0.8717753, -0.3492570, 0.3760026, -1.0324048, 1.2210323
2: -0.6477537, 0.9167081, -0.2838330, 0.4840446, -1.1317983, 1.2005411
3: -0.6959582, 0.6527142, -0.2245906, 0.4365792, -1.1325374, 0.8773048
4: -0.8151168, 0.8005915, -0.4232208, 0.4098117, -1.2249285, 1.2238123
5: -0.6403569, 1.1529043, 0.1099933, 1.1426980, -1.7830549, 1.0429111
6: -0.5604084, 0.7439091, -0.2367814, 0.4465680, -1.0069764, 0.9806905
7: -0.6641110, 0.8826647, -0.3267075, 0.4412563, -1.1053672, 1.2093722
8: -0.7382755, 0.8849251, -0.3252604, 0.5730474, -1.3113229, 1.2101855
9: -0.7576594, 0.8534768, -0.3909015, 0.4825158, -1.2401752, 1.2443783

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 131

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5056952, upper bound: 1.5056951
time: 1.64 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5056952, upper bound: 1.5057209
time: 1.84 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.8631139, 0.7372196, -0.4242224, 0.4450007, -1.3081145, 1.1614420
1: -0.6564021, 0.8717753, -0.3730846, 0.4083835, -1.0647856, 1.2448599
2: -0.6477537, 0.9167081, -0.3021763, 0.5244129, -1.1721666, 1.2188843
3: -0.6959582, 0.6527142, -0.2423404, 0.4554326, -1.1513908, 0.8950546
4: -0.8151168, 0.8005915, -0.4486009, 0.4354814, -1.2505982, 1.2491925
5: -0.6403569, 1.1529043, 0.0525133, 1.1684123, -1.8087692, 1.1003910
6: -0.5604084, 0.7439091, -0.2549961, 0.4681417, -1.0285500, 0.9989052
7: -0.6641110, 0.8826647, -0.3472473, 0.4758478, -1.1399589, 1.2299120
8: -0.7382755, 0.8849251, -0.3504899, 0.6003999, -1.3386755, 1.2354150
9: -0.7576594, 0.8534768, -0.4175776, 0.5112474, -1.2689068, 1.2710543

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 131

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5056952, upper bound: 1.5056951
time: 1.97 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5056952, upper bound: 1.5057209
time: 1.85 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.4026899, 0.4191411, -0.5354338, 0.5511386, -0.9538285, 0.9545749
1: -0.3521138, 0.3794717, -0.4616147, 0.6556986, -1.0078124, 0.8410864
2: -0.2859992, 0.4886915, -0.3888087, 0.6718566, -0.9578558, 0.8775002
3: -0.2267117, 0.4386235, -0.3365193, 0.5327753, -0.7594869, 0.7751428
4: -0.4259376, 0.4128717, -0.5519553, 0.5417447, -0.9676823, 0.9648271
5: 0.1034426, 1.1441865, -0.2655265, 1.1636064, -1.0601637, 1.4097130
6: -0.2389207, 0.4489258, -0.3301950, 0.5526736, -0.7915943, 0.7791208
7: -0.3291392, 0.4452140, -0.4425724, 0.6047803, -0.9339195, 0.8877863
8: -0.3282850, 0.5760045, -0.4511241, 0.7085128, -1.0367979, 1.0271287
9: -0.3940009, 0.4859381, -0.5220423, 0.6233384, -1.0173393, 1.0079803

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 131

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5060568, upper bound: 1.5702700
time: 1.70 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5060568, upper bound: 1.5705009
time: 1.69 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.4001870, 0.4156592, -0.8155174, 0.7116473, -1.1118343, 1.2311766
1: -0.3492570, 0.3760026, -0.6295613, 0.8341336, -1.1833906, 1.0055640
2: -0.2838330, 0.4840446, -0.6094971, 0.8815447, -1.1653777, 1.0935416
3: -0.2245906, 0.4365792, -0.6431277, 0.6360282, -0.8606188, 1.0797069
4: -0.4232208, 0.4098117, -0.7761424, 0.7665916, -1.1898124, 1.1859541
5: 0.1099933, 1.1426980, -0.5844316, 1.1739504, -1.0639572, 1.7271297
6: -0.2367814, 0.4465680, -0.5257284, 0.7238255, -0.9606069, 0.9722964
7: -0.3267075, 0.4412563, -0.6352049, 0.8414713, -1.1681788, 1.0764612
8: -0.3252604, 0.5730474, -0.7016140, 0.8597875, -1.1850479, 1.2746613
9: -0.3909015, 0.4825158, -0.7231329, 0.8206913, -1.2115928, 1.2056487

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 131

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5060568, upper bound: 1.5702700
time: 1.73 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5060568, upper bound: 1.5705009
time: 1.67 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.5346003, 0.5504180, -0.5354338, 0.5511386, -1.0857389, 1.0858518
1: -0.4613726, 0.6573250, -0.4616147, 0.6556986, -1.1170712, 1.1189396
2: -0.3883922, 0.6712835, -0.3888087, 0.6718566, -1.0602489, 1.0600922
3: -0.3360583, 0.5322127, -0.3365193, 0.5327753, -0.8688336, 0.8687320
4: -0.5515785, 0.5412838, -0.5519553, 0.5417447, -1.0933232, 1.0932392
5: -0.2655362, 1.1559851, -0.2655265, 1.1636064, -1.4291426, 1.4215117
6: -0.3297866, 0.5524096, -0.3301950, 0.5526736, -0.8824602, 0.8826046
7: -0.4420976, 0.6037713, -0.4425724, 0.6047803, -1.0468779, 1.0463437
8: -0.4507531, 0.7079566, -0.4511241, 0.7085128, -1.1592660, 1.1590807
9: -0.5216522, 0.6228814, -0.5220423, 0.6233384, -1.1449906, 1.1449237

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5935493, upper bound: 1.5764483
time: 2.01 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5935493, upper bound: 1.5790562
time: 2.40 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.4273857, 0.4485906, -0.5266285, 0.5453887, -0.9727744, 0.9752191
1: -0.3760331, 0.4125217, -0.4571671, 0.6573631, -1.0333961, 0.8696887
2: -0.3044457, 0.5294700, -0.3830554, 0.6647385, -0.9691843, 0.9125254
3: -0.2445247, 0.4578295, -0.3286504, 0.5287566, -0.7732813, 0.7864799
4: -0.4519042, 0.4386348, -0.5463063, 0.5355127, -0.9874169, 0.9849411
5: 0.0454088, 1.1700767, -0.2596720, 1.1396759, -1.0942670, 1.4297488
6: -0.2572561, 0.4709196, -0.3258136, 0.5480757, -0.8053318, 0.7967332
7: -0.3497696, 0.4801137, -0.4365955, 0.5957977, -0.9455673, 0.9167092
8: -0.3535936, 0.6039436, -0.4451951, 0.7030770, -1.0566707, 1.0491388
9: -0.4209062, 0.5147840, -0.5162947, 0.6176199, -1.0385261, 1.0310787

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 131

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5783356
time: 2.11 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5784411
time: 1.82 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.4242224, 0.4450007, -0.8631139, 0.7372196, -1.1614420, 1.3081145
1: -0.3730846, 0.4083835, -0.6564021, 0.8717753, -1.2448599, 1.0647856
2: -0.3021763, 0.5244129, -0.6477537, 0.9167081, -1.2188843, 1.1721666
3: -0.2423404, 0.4554326, -0.6959582, 0.6527142, -0.8950546, 1.1513908
4: -0.4486009, 0.4354814, -0.8151168, 0.8005915, -1.2491925, 1.2505982
5: 0.0525133, 1.1684123, -0.6403569, 1.1529043, -1.1003910, 1.8087692
6: -0.2549961, 0.4681417, -0.5604084, 0.7439091, -0.9989052, 1.0285500
7: -0.3472473, 0.4758478, -0.6641110, 0.8826647, -1.2299120, 1.1399589
8: -0.3504899, 0.6003999, -0.7382755, 0.8849251, -1.2354150, 1.3386755
9: -0.4175776, 0.5112474, -0.7576594, 0.8534768, -1.2710543, 1.2689068

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 131

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5057209, upper bound: 1.5783356
time: 1.99 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5057209, upper bound: 1.5784411
time: 2.66 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.5738508, 0.5759040, -0.4026899, 0.4191411, -0.9929919, 0.9785939
1: -0.4839600, 0.6847404, -0.3521138, 0.3794717, -0.8634316, 1.0368541
2: -0.4179204, 0.7037759, -0.2859992, 0.4886915, -0.9066119, 0.9897751
3: -0.3756537, 0.5489379, -0.2267117, 0.4386235, -0.8142773, 0.7756495
4: -0.5821376, 0.5716643, -0.4259376, 0.4128717, -0.9950094, 0.9976019
5: -0.3128855, 1.1789155, 0.1034426, 1.1441865, -1.4570720, 1.0754728
6: -0.3524698, 0.5755888, -0.2389207, 0.4489258, -0.8013955, 0.8145094
7: -0.4721063, 0.6370016, -0.3291392, 0.4452140, -0.9173203, 0.9661407
8: -0.4816706, 0.7319694, -0.3282850, 0.5760045, -1.0576751, 1.0602545
9: -0.5507526, 0.6506543, -0.3940009, 0.4859381, -1.0366907, 1.0446552

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5780317, upper bound: 1.5060568
time: 2.25 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5705009, upper bound: 1.5060568
time: 1.70 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.5738508, 0.5759040, -0.5346003, 0.5504180, -1.1242688, 1.1105043
1: -0.4839600, 0.6847404, -0.4613726, 0.6573250, -1.1412849, 1.1461129
2: -0.4179204, 0.7037759, -0.3883922, 0.6712835, -1.0892040, 1.0921681
3: -0.3756537, 0.5489379, -0.3360583, 0.5322127, -0.9078664, 0.8849962
4: -0.5821376, 0.5716643, -0.5515785, 0.5412838, -1.1234214, 1.1232429
5: -0.3128855, 1.1789155, -0.2655362, 1.1559851, -1.4688706, 1.4444517
6: -0.3524698, 0.5755888, -0.3297866, 0.5524096, -0.9048795, 0.9053754
7: -0.4721063, 0.6370016, -0.4420976, 0.6037713, -1.0758777, 1.0790992
8: -0.4816706, 0.7319694, -0.4507531, 0.7079566, -1.1896272, 1.1827226
9: -0.5507526, 0.6506543, -0.5216522, 0.6228814, -1.1736341, 1.1723065

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5780317, upper bound: 1.5884389
time: 8.06 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5705009, upper bound: 1.5902935
time: 2.02 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.4273857, 0.4485906, -0.5735637, 0.5754527, -1.0028384, 1.0221543
1: -0.3760331, 0.4125217, -0.4841228, 0.6916738, -1.0677069, 0.8966445
2: -0.3044457, 0.5294700, -0.4182384, 0.7036884, -1.0081341, 0.9477084
3: -0.2445247, 0.4578295, -0.3760577, 0.5487165, -0.7932412, 0.8338872
4: -0.4519042, 0.4386348, -0.5823968, 0.5718654, -1.0237696, 1.0210316
5: 0.0454088, 1.1700767, -0.3174203, 1.1639028, -1.1184939, 1.4874971
6: -0.2572561, 0.4709196, -0.3525820, 0.5757295, -0.8329856, 0.8235016
7: -0.3497696, 0.4801137, -0.4724032, 0.6351199, -0.9848895, 0.9525169
8: -0.3535936, 0.6039436, -0.4819312, 0.7318722, -1.0854659, 1.0858748
9: -0.4209062, 0.5147840, -0.5509792, 0.6508268, -1.0717330, 1.0657632

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 131

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5783356
time: 1.77 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5784411
time: 1.85 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.4242224, 0.4450007, -0.9077567, 0.7489418, -1.1731641, 1.3527575
1: -0.3730846, 0.4083835, -0.6828596, 0.8937833, -1.2668680, 1.0912431
2: -0.3021763, 0.5244129, -0.6783161, 0.9376457, -1.2398219, 1.2027290
3: -0.2423404, 0.4554326, -0.7361358, 0.6580015, -0.9003419, 1.1915684
4: -0.4486009, 0.4354814, -0.8468796, 0.8335834, -1.2821844, 1.2823610
5: 0.0525133, 1.1684123, -0.6771894, 1.1738083, -1.1212951, 1.8456018
6: -0.2549961, 0.4681417, -0.5737228, 0.7708660, -1.0258621, 1.0418645
7: -0.3472473, 0.4758478, -0.7008318, 0.8944464, -1.2416937, 1.1766796
8: -0.3504899, 0.6003999, -0.7795265, 0.8966410, -1.2471309, 1.3799264
9: -0.4175776, 0.5112474, -0.7760882, 0.8712727, -1.2888503, 1.2873356

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 131

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5057209, upper bound: 1.5783356
time: 2.16 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5057209, upper bound: 1.5784411
time: 2.04 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.5738508, 0.5759040, -0.4273857, 0.4485906, -1.0224414, 1.0032897
1: -0.4839600, 0.6847404, -0.3760331, 0.4125217, -0.8964816, 1.0607735
2: -0.4179204, 0.7037759, -0.3044457, 0.5294700, -0.9473904, 1.0082216
3: -0.3756537, 0.5489379, -0.2445247, 0.4578295, -0.8334833, 0.7934626
4: -0.5821376, 0.5716643, -0.4519042, 0.4386348, -1.0207725, 1.0235685
5: -0.3128855, 1.1789155, 0.0454088, 1.1700767, -1.4829623, 1.1335067
6: -0.3524698, 0.5755888, -0.2572561, 0.4709196, -0.8233894, 0.8328449
7: -0.4721063, 0.6370016, -0.3497696, 0.4801137, -0.9522201, 0.9867712
8: -0.4816706, 0.7319694, -0.3535936, 0.6039436, -1.0856142, 1.0855631
9: -0.5507526, 0.6506543, -0.4209062, 0.5147840, -1.0655366, 1.0715605

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 8

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5780317, upper bound: 1.5060568
time: 2.60 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5705009, upper bound: 1.5060568
time: 2.06 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.5738508, 0.5759040, -0.5738508, 0.5759040, -1.1497548, 1.1497548
1: -0.4839600, 0.6847404, -0.4839600, 0.6847404, -1.1687003, 1.1687003
2: -0.4179204, 0.7037759, -0.4179204, 0.7037759, -1.1216964, 1.1216964
3: -0.3756537, 0.5489379, -0.3756537, 0.5489379, -0.9245917, 0.9245917
4: -0.5821376, 0.5716643, -0.5821376, 0.5716643, -1.1538019, 1.1538019
5: -0.3128855, 1.1789155, -0.3128855, 1.1789155, -1.4918010, 1.4918010
6: -0.3524698, 0.5755888, -0.3524698, 0.5755888, -0.9280585, 0.9280585
7: -0.4721063, 0.6370016, -0.4721063, 0.6370016, -1.1091080, 1.1091080
8: -0.4816706, 0.7319694, -0.4816706, 0.7319694, -1.2136400, 1.2136400
9: -0.5507526, 0.6506543, -0.5507526, 0.6506543, -1.2014070, 1.2014070

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 8

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5780317, upper bound: 1.5060568
time: 2.48 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5705009, upper bound: 1.5060568
time: 2.21 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 6.82 seconds
NS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.82
Output dim: 5, lower bound: -1.5328025, upper bound: 1.5051724
NS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 6.82
Output dim: 5, lower bound: -1.5328025, upper bound: 1.5051724
NS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 6.82
Output dim: 5, lower bound: -1.5328025, upper bound: 1.5051724
NS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 6.82
Output dim: 5, lower bound: -1.5328025, upper bound: 1.5051724
NS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.82
Output dim: 5, lower bound: -1.5056952, upper bound: 1.5056951
NS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 6.82
Output dim: 5, lower bound: -1.5056952, upper bound: 1.5057209
NS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 6.82
Output dim: 5, lower bound: -1.5056952, upper bound: 1.5056951
NS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 6.82
Output dim: 5, lower bound: -1.5056952, upper bound: 1.5057209
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.82
Output dim: 5, lower bound: -1.5060568, upper bound: 1.5702700
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.82
Output dim: 5, lower bound: -1.5060568, upper bound: 1.5705009
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.82
Output dim: 5, lower bound: -1.5060568, upper bound: 1.5702700
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.82
Output dim: 5, lower bound: -1.5060568, upper bound: 1.5705009
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.82
Output dim: 5, lower bound: -1.5935493, upper bound: 1.5764483
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.82
Output dim: 5, lower bound: -1.5935493, upper bound: 1.5790562
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.82
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5783356
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.82
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5784411
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.82
Output dim: 5, lower bound: -1.5057209, upper bound: 1.5783356
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.82
Output dim: 5, lower bound: -1.5057209, upper bound: 1.5784411
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.82
Output dim: 5, lower bound: -1.5780317, upper bound: 1.5060568
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.82
Output dim: 5, lower bound: -1.5705009, upper bound: 1.5060568
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.82
Output dim: 5, lower bound: -1.5780317, upper bound: 1.5884389
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.82
Output dim: 5, lower bound: -1.5705009, upper bound: 1.5902935
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.82
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5783356
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.82
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5784411
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.82
Output dim: 5, lower bound: -1.5057209, upper bound: 1.5783356
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.82
Output dim: 5, lower bound: -1.5057209, upper bound: 1.5784411
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.82
Output dim: 5, lower bound: -1.5780317, upper bound: 1.5060568
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.82
Output dim: 5, lower bound: -1.5705009, upper bound: 1.5060568
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.82
Output dim: 5, lower bound: -1.5780317, upper bound: 1.5060568
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.82
Output dim: 5, lower bound: -1.5705009, upper bound: 1.5060568

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.3846650, 0.3941466, -0.5354338, 0.5511386, -0.9358037, 0.9295805
1: -0.3319941, 0.3536625, -0.4616147, 0.6556986, -0.9876927, 0.8152772
2: -0.2704943, 0.4554276, -0.3888087, 0.6718566, -0.9423509, 0.8442364
3: -0.2115196, 0.4238317, -0.3365193, 0.5327753, -0.7442949, 0.7603510
4: -0.4065562, 0.3909165, -0.5519553, 0.5417447, -0.9483010, 0.9428718
5: 0.1512546, 1.1255324, -0.2655265, 1.1636064, -1.0123518, 1.3910589
6: -0.2236446, 0.4320862, -0.3301950, 0.5526736, -0.7763183, 0.7622812
7: -0.3116984, 0.4164688, -0.4425724, 0.6047803, -0.9164788, 0.8590412
8: -0.3066485, 0.5550406, -0.4511241, 0.7085128, -1.0151613, 1.0061648
9: -0.3718071, 0.4616198, -0.5220423, 0.6233384, -0.9951456, 0.9836621

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 8

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5780317
time: 2.46 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5780317
time: 2.07 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.4976699, 0.5514714, -0.5354338, 0.5511386, -1.0488086, 1.0869052
1: -0.4630894, 0.4839199, -0.4616147, 0.6556986, -1.1187880, 0.9455346
2: -0.3678207, 0.6678746, -0.3888087, 0.6718566, -1.0396774, 1.0566833
3: -0.3079270, 0.5140205, -0.3365193, 0.5327753, -0.8407022, 0.8505398
4: -0.5304418, 0.5295544, -0.5519553, 0.5417447, -1.0721865, 1.0815097
5: -0.1186955, 1.1348166, -0.2655265, 1.1636064, -1.2823019, 1.4003432
6: -0.3196089, 0.5404234, -0.3301950, 0.5526736, -0.8722825, 0.8706183
7: -0.4204883, 0.5938017, -0.4425724, 0.6047803, -1.0252686, 1.0363741
8: -0.4448276, 0.6881066, -0.4511241, 0.7085128, -1.1533403, 1.1392307
9: -0.5130015, 0.6167052, -0.5220423, 0.6233384, -1.1363399, 1.1387475

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5780317
time: 2.19 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5780317
time: 2.33 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.3846650, 0.3941466, -0.8155174, 0.7116473, -1.0963123, 1.2096641
1: -0.3319941, 0.3536625, -0.6295613, 0.8341336, -1.1661278, 0.9832238
2: -0.2704943, 0.4554276, -0.6094971, 0.8815447, -1.1520389, 1.0649247
3: -0.2115196, 0.4238317, -0.6431277, 0.6360282, -0.8475478, 1.0669594
4: -0.4065562, 0.3909165, -0.7761424, 0.7665916, -1.1731479, 1.1670589
5: 0.1512546, 1.1255324, -0.5844316, 1.1739504, -1.0226958, 1.7099640
6: -0.2236446, 0.4320862, -0.5257284, 0.7238255, -0.9474701, 0.9578146
7: -0.3116984, 0.4164688, -0.6352049, 0.8414713, -1.1531696, 1.0516737
8: -0.3066485, 0.5550406, -0.7016140, 0.8597875, -1.1664360, 1.2566545
9: -0.3718071, 0.4616198, -0.7231329, 0.8206913, -1.1924984, 1.1847527

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 8

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5702700
time: 2.31 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5702700
time: 2.63 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.4976699, 0.5514714, -0.8155174, 0.7116473, -1.2093172, 1.3669888
1: -0.4630894, 0.4839199, -0.6295613, 0.8341336, -1.2972231, 1.1134812
2: -0.3678207, 0.6678746, -0.6094971, 0.8815447, -1.2493654, 1.2773716
3: -0.3079270, 0.5140205, -0.6431277, 0.6360282, -0.9439551, 1.1571481
4: -0.5304418, 0.5295544, -0.7761424, 0.7665916, -1.2970333, 1.3056967
5: -0.1186955, 1.1348166, -0.5844316, 1.1739504, -1.2926459, 1.7192483
6: -0.3196089, 0.5404234, -0.5257284, 0.7238255, -1.0434344, 1.0661519
7: -0.4204883, 0.5938017, -0.6352049, 0.8414713, -1.2619596, 1.2290066
8: -0.4448276, 0.6881066, -0.7016140, 0.8597875, -1.3046150, 1.3897206
9: -0.5130015, 0.6167052, -0.7231329, 0.8206913, -1.3336928, 1.3398381

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5705009
time: 1.86 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5705009
time: 2.04 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.4897729, 0.5190340, -0.5354338, 0.5511386, -1.0409114, 1.0544679
1: -0.4321097, 0.6187906, -0.4616147, 0.6556986, -1.0878084, 1.0804052
2: -0.3550040, 0.6257194, -0.3888087, 0.6718566, -1.0268606, 1.0145280
3: -0.2907113, 0.5107954, -0.3365193, 0.5327753, -0.8234866, 0.8473147
4: -0.5177408, 0.5034260, -0.5519553, 0.5417447, -1.0594854, 1.0553813
5: -0.2044711, 1.1373084, -0.2655265, 1.1636064, -1.3680775, 1.4028349
6: -0.3058346, 0.5231596, -0.3301950, 0.5526736, -0.8585082, 0.8533546
7: -0.4072961, 0.5635933, -0.4425724, 0.6047803, -1.0120764, 1.0061656
8: -0.4149080, 0.6760141, -0.4511241, 0.7085128, -1.1234208, 1.1271383
9: -0.4871923, 0.5877865, -0.5220423, 0.6233384, -1.1105307, 1.1098287

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5835800, upper bound: 1.5816530
time: 1.83 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5835800, upper bound: 1.5816530
time: 2.29 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.7496231, 0.6859171, -0.5354338, 0.5511386, -1.3007617, 1.2213509
1: -0.5881135, 0.7931735, -0.4616147, 0.6556986, -1.2438121, 1.2547882
2: -0.5522286, 0.8548584, -0.3888087, 0.6718566, -1.2240852, 1.2436671
3: -0.5610748, 0.6149958, -0.3365193, 0.5327753, -1.0938501, 0.9515151
4: -0.7187871, 0.7081248, -0.5519553, 0.5417447, -1.2605319, 1.2600801
5: -0.4895747, 1.1498928, -0.2655265, 1.1636064, -1.6531811, 1.4154193
6: -0.4465384, 0.6813953, -0.3301950, 0.5526736, -0.9992121, 1.0115902
7: -0.6118792, 0.7767581, -0.4425724, 0.6047803, -1.2166595, 1.2193304
8: -0.6199242, 0.8393949, -0.4511241, 0.7085128, -1.3284371, 1.2905190
9: -0.6868015, 0.7742674, -0.5220423, 0.6233384, -1.3101399, 1.2963097

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5835800, upper bound: 1.5819786
time: 1.89 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5835800, upper bound: 1.5819786
time: 1.89 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.4071087, 0.4252596, -0.5266285, 0.5453887, -0.9524974, 0.9518881
1: -0.3569512, 0.3870762, -0.4571671, 0.6573631, -1.0143142, 0.8442433
2: -0.2898312, 0.4966636, -0.3830554, 0.6647385, -0.9545697, 0.8797190
3: -0.2303881, 0.4423752, -0.3286504, 0.5287566, -0.7591447, 0.7710256
4: -0.4306142, 0.4182070, -0.5463063, 0.5355127, -0.9661269, 0.9645133
5: 0.0903304, 1.1526667, -0.2596720, 1.1396759, -1.0493455, 1.4123387
6: -0.2427156, 0.4529262, -0.3258136, 0.5480757, -0.7907912, 0.7787398
7: -0.3334745, 0.4523583, -0.4365955, 0.5957977, -0.9292722, 0.8889538
8: -0.3334717, 0.5812222, -0.4451951, 0.7030770, -1.0365486, 1.0264174
9: -0.3993464, 0.4918939, -0.5162947, 0.6176199, -1.0169663, 1.0081886

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5328025
time: 2.16 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5328025
time: 1.80 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.5261132, 0.5601768, -0.5266285, 0.5453887, -1.0715019, 1.0868053
1: -0.4705340, 0.5071646, -0.4571671, 0.6573631, -1.1278970, 0.9643317
2: -0.3742779, 0.6899731, -0.3830554, 0.6647385, -1.0390164, 1.0730284
3: -0.3130228, 0.5298095, -0.3286504, 0.5287566, -0.8417794, 0.8584599
4: -0.5564782, 0.5370346, -0.5463063, 0.5355127, -1.0919909, 1.0833409
5: -0.1423194, 1.1591973, -0.2596720, 1.1396759, -1.2819953, 1.4188694
6: -0.3267299, 0.5598539, -0.3258136, 0.5480757, -0.8748056, 0.8856674
7: -0.4267581, 0.6103408, -0.4365955, 0.5957977, -1.0225558, 1.0469363
8: -0.4519157, 0.7141070, -0.4451951, 0.7030770, -1.1549927, 1.1593022
9: -0.5258906, 0.6254055, -0.5162947, 0.6176199, -1.1435106, 1.1417003

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5328025
time: 2.06 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5328025
time: 1.74 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.4071087, 0.4252596, -0.8631139, 0.7372196, -1.1443284, 1.2883735
1: -0.3569512, 0.3870762, -0.6564021, 0.8717753, -1.2287264, 1.0434784
2: -0.2898312, 0.4966636, -0.6477537, 0.9167081, -1.2065393, 1.1444173
3: -0.2303881, 0.4423752, -0.6959582, 0.6527142, -0.8831023, 1.1383333
4: -0.4306142, 0.4182070, -0.8151168, 0.8005915, -1.2312057, 1.2333238
5: 0.0903304, 1.1526667, -0.6403569, 1.1529043, -1.0625739, 1.7930236
6: -0.2427156, 0.4529262, -0.5604084, 0.7439091, -0.9866247, 1.0133345
7: -0.3334745, 0.4523583, -0.6641110, 0.8826647, -1.2161392, 1.1164693
8: -0.3334717, 0.5812222, -0.7382755, 0.8849251, -1.2183968, 1.3194978
9: -0.3993464, 0.4918939, -0.7576594, 0.8534768, -1.2528231, 1.2495532

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 131

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5060568
time: 1.76 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5783356
time: 2.17 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.5261132, 0.5601768, -0.8631139, 0.7372196, -1.2633328, 1.4232907
1: -0.4705340, 0.5071646, -0.6564021, 0.8717753, -1.3423092, 1.1635667
2: -0.3742779, 0.6899731, -0.6477537, 0.9167081, -1.2909859, 1.3377268
3: -0.3130228, 0.5298095, -0.6959582, 0.6527142, -0.9657370, 1.2257677
4: -0.5564782, 0.5370346, -0.8151168, 0.8005915, -1.3570697, 1.3521514
5: -0.1423194, 1.1591973, -0.6403569, 1.1529043, -1.2952237, 1.7995542
6: -0.3267299, 0.5598539, -0.5604084, 0.7439091, -1.0706390, 1.1202621
7: -0.4267581, 0.6103408, -0.6641110, 0.8826647, -1.3094227, 1.2744517
8: -0.4519157, 0.7141070, -0.7382755, 0.8849251, -1.3368409, 1.4523826
9: -0.5258906, 0.6254055, -0.7576594, 0.8534768, -1.3793674, 1.3830649

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 131

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5060568
time: 2.06 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5784411
time: 2.23 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.5189962, 0.5414410, -0.4026899, 0.4191411, -0.9381374, 0.9441309
1: -0.4519723, 0.6417593, -0.3521138, 0.3794717, -0.8314440, 0.9938731
2: -0.3766927, 0.6576895, -0.2859992, 0.4886915, -0.8653842, 0.9436888
3: -0.3194801, 0.5260263, -0.2267117, 0.4386235, -0.7581036, 0.7527380
4: -0.5400504, 0.5287694, -0.4259376, 0.4128717, -0.9529222, 0.9547070
5: -0.2448135, 1.1612194, 0.1034426, 1.1441865, -1.3889999, 1.0577767
6: -0.3215775, 0.5428849, -0.2389207, 0.4489258, -0.7705033, 0.7818055
7: -0.4300930, 0.5916775, -0.3291392, 0.4452140, -0.8753070, 0.9208167
8: -0.4384841, 0.6982014, -0.3282850, 0.5760045, -1.0144887, 1.0264864
9: -0.5099261, 0.6116219, -0.3940009, 0.4859381, -0.9958642, 1.0056227

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 131

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5702700, upper bound: 1.5060568
time: 2.30 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5702700, upper bound: 1.5060568
time: 2.09 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.7890387, 0.6955979, -0.4001870, 0.4156592, -1.2046980, 1.0957849
1: -0.6102881, 0.8136971, -0.3492570, 0.3760026, -0.9862908, 1.1629541
2: -0.5792634, 0.8663443, -0.2838330, 0.4840446, -1.0633080, 1.1501774
3: -0.6016492, 0.6282682, -0.2245906, 0.4365792, -1.0382283, 0.8528588
4: -0.7547023, 0.7394649, -0.4232208, 0.4098117, -1.1645141, 1.1626856
5: -0.5361838, 1.1703507, 0.1099933, 1.1426980, -1.6788819, 1.0603575
6: -0.4909047, 0.7040354, -0.2367814, 0.4465680, -0.9374728, 0.9408168
7: -0.6173807, 0.8155841, -0.3267075, 0.4412563, -1.0586370, 1.1422915
8: -0.6670679, 0.8455794, -0.3252604, 0.5730474, -1.2401153, 1.1708398
9: -0.7079532, 0.8020560, -0.3909015, 0.4825158, -1.1904690, 1.1929575

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 131

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5702700, upper bound: 1.5060568
time: 1.92 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5702700, upper bound: 1.5060568
time: 2.13 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.5189962, 0.5414410, -0.5346003, 0.5504180, -1.0694141, 1.0760412
1: -0.4519723, 0.6417593, -0.4613726, 0.6573250, -1.1092973, 1.1031319
2: -0.3766927, 0.6576895, -0.3883922, 0.6712835, -1.0479763, 1.0460818
3: -0.3194801, 0.5260263, -0.3360583, 0.5322127, -0.8516927, 0.8620846
4: -0.5400504, 0.5287694, -0.5515785, 0.5412838, -1.0813344, 1.0803480
5: -0.2448135, 1.1612194, -0.2655362, 1.1559851, -1.4007986, 1.4267555
6: -0.3215775, 0.5428849, -0.3297866, 0.5524096, -0.8739872, 0.8726715
7: -0.4300930, 0.5916775, -0.4420976, 0.6037713, -1.0338643, 1.0337751
8: -0.4384841, 0.6982014, -0.4507531, 0.7079566, -1.1464407, 1.1489544
9: -0.5099261, 0.6116219, -0.5216522, 0.6228814, -1.1328075, 1.1332741

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5834703, upper bound: 1.5884389
time: 1.97 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5834703, upper bound: 1.5884389
time: 1.84 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.7890387, 0.6955979, -0.5254278, 0.5450602, -1.3340989, 1.2210257
1: -0.6102881, 0.8136971, -0.4560657, 0.6506591, -1.2609472, 1.2697628
2: -0.5792634, 0.8663443, -0.3816901, 0.6634907, -1.2427542, 1.2480344
3: -0.6016492, 0.6282682, -0.3266456, 0.5285469, -1.1301961, 0.9549139
4: -0.7547023, 0.7394649, -0.5449716, 0.5341736, -1.2888759, 1.2844365
5: -0.5361838, 1.1703507, -0.2550684, 1.1542438, -1.6904275, 1.4254191
6: -0.4909047, 0.7040354, -0.3250381, 0.5469781, -1.0378828, 1.0290735
7: -0.6173807, 0.8155841, -0.4352484, 0.5965722, -1.2139530, 1.2508324
8: -0.6670679, 0.8455794, -0.4437878, 0.7022979, -1.3693658, 1.2893672
9: -0.7079532, 0.8020560, -0.5149752, 0.6164190, -1.3243721, 1.3170311

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5834703, upper bound: 1.5902935
time: 2.44 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5834703, upper bound: 1.5902935
time: 2.52 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.4071087, 0.4252596, -0.5735637, 0.5754527, -0.9825615, 0.9988233
1: -0.3569512, 0.3870762, -0.4841228, 0.6916738, -1.0486250, 0.8711990
2: -0.2898312, 0.4966636, -0.4182384, 0.7036884, -0.9935197, 0.9149020
3: -0.2303881, 0.4423752, -0.3760577, 0.5487165, -0.7791046, 0.8184329
4: -0.4306142, 0.4182070, -0.5823968, 0.5718654, -1.0024797, 1.0006039
5: 0.0903304, 1.1526667, -0.3174203, 1.1639028, -1.0735724, 1.4700871
6: -0.2427156, 0.4529262, -0.3525820, 0.5757295, -0.8184450, 0.8055082
7: -0.3334745, 0.4523583, -0.4724032, 0.6351199, -0.9685944, 0.9247615
8: -0.3334717, 0.5812222, -0.4819312, 0.7318722, -1.0653439, 1.0631535
9: -0.3993464, 0.4918939, -0.5509792, 0.6508268, -1.0501732, 1.0428731

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 8

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5328025
time: 1.83 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5845202
time: 2.49 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.5261132, 0.5601768, -0.5735637, 0.5754527, -1.1015658, 1.1337405
1: -0.4705340, 0.5071646, -0.4841228, 0.6916738, -1.1622078, 0.9912874
2: -0.3742779, 0.6899731, -0.4182384, 0.7036884, -1.0779663, 1.1082115
3: -0.3130228, 0.5298095, -0.3760577, 0.5487165, -0.8617393, 0.9058672
4: -0.5564782, 0.5370346, -0.5823968, 0.5718654, -1.1283436, 1.1194314
5: -0.1423194, 1.1591973, -0.3174203, 1.1639028, -1.3062222, 1.4766176
6: -0.3267299, 0.5598539, -0.3525820, 0.5757295, -0.9024594, 0.9124359
7: -0.4267581, 0.6103408, -0.4724032, 0.6351199, -1.0618780, 1.0827439
8: -0.4519157, 0.7141070, -0.4819312, 0.7318722, -1.1837879, 1.1960382
9: -0.5258906, 0.6254055, -0.5509792, 0.6508268, -1.1767174, 1.1763847

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 8

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5328025
time: 1.98 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5845202
time: 2.07 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.4071087, 0.4252596, -0.9077567, 0.7489418, -1.1560504, 1.3330163
1: -0.3569512, 0.3870762, -0.6828596, 0.8937833, -1.2507346, 1.0699358
2: -0.2898312, 0.4966636, -0.6783161, 0.9376457, -1.2274768, 1.1749797
3: -0.2303881, 0.4423752, -0.7361358, 0.6580015, -0.8883896, 1.1785109
4: -0.4306142, 0.4182070, -0.8468796, 0.8335834, -1.2641976, 1.2650867
5: 0.0903304, 1.1526667, -0.6771894, 1.1738083, -1.0834780, 1.8298562
6: -0.2427156, 0.4529262, -0.5737228, 0.7708660, -1.0135816, 1.0266490
7: -0.3334745, 0.4523583, -0.7008318, 0.8944464, -1.2279209, 1.1531901
8: -0.3334717, 0.5812222, -0.7795265, 0.8966410, -1.2301127, 1.3607488
9: -0.3993464, 0.4918939, -0.7760882, 0.8712727, -1.2706192, 1.2679820

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 8

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5060568
time: 2.25 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5783356
time: 2.00 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.5261132, 0.5601768, -0.9077567, 0.7489418, -1.2750549, 1.4679335
1: -0.4705340, 0.5071646, -0.6828596, 0.8937833, -1.3643173, 1.1900241
2: -0.3742779, 0.6899731, -0.6783161, 0.9376457, -1.3119235, 1.3682891
3: -0.3130228, 0.5298095, -0.7361358, 0.6580015, -0.9710243, 1.2659453
4: -0.5564782, 0.5370346, -0.8468796, 0.8335834, -1.3900616, 1.3839142
5: -0.1423194, 1.1591973, -0.6771894, 1.1738083, -1.3161278, 1.8363867
6: -0.3267299, 0.5598539, -0.5737228, 0.7708660, -1.0975959, 1.1335766
7: -0.4267581, 0.6103408, -0.7008318, 0.8944464, -1.3212044, 1.3111725
8: -0.4519157, 0.7141070, -0.7795265, 0.8966410, -1.3485568, 1.4936335
9: -0.5258906, 0.6254055, -0.7760882, 0.8712727, -1.3971634, 1.4014937

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 8

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5060568
time: 2.00 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5784411
time: 2.12 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.5189962, 0.5414410, -0.4273857, 0.4485906, -0.9675869, 0.9688267
1: -0.4519723, 0.6417593, -0.3760331, 0.4125217, -0.8644940, 1.0177923
2: -0.3766927, 0.6576895, -0.3044457, 0.5294700, -0.9061627, 0.9621353
3: -0.3194801, 0.5260263, -0.2445247, 0.4578295, -0.7773096, 0.7705510
4: -0.5400504, 0.5287694, -0.4519042, 0.4386348, -0.9786853, 0.9806736
5: -0.2448135, 1.1612194, 0.0454088, 1.1700767, -1.4148903, 1.1158105
6: -0.3215775, 0.5428849, -0.2572561, 0.4709196, -0.7924972, 0.8001410
7: -0.4300930, 0.5916775, -0.3497696, 0.4801137, -0.9102067, 0.9414471
8: -0.4384841, 0.6982014, -0.3535936, 0.6039436, -1.0424278, 1.0517950
9: -0.5099261, 0.6116219, -0.4209062, 0.5147840, -1.0247101, 1.0325280

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 131

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5702700, upper bound: 1.5060568
time: 2.35 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5702700, upper bound: 1.5060568
time: 2.14 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.7890387, 0.6955979, -0.4242224, 0.4450007, -1.2340394, 1.1198202
1: -0.6102881, 0.8136971, -0.3730846, 0.4083835, -1.0186716, 1.1867818
2: -0.5792634, 0.8663443, -0.3021763, 0.5244129, -1.1036763, 1.1685206
3: -0.6016492, 0.6282682, -0.2423404, 0.4554326, -1.0570817, 0.8706086
4: -0.7547023, 0.7394649, -0.4486009, 0.4354814, -1.1901838, 1.1880658
5: -0.5361838, 1.1703507, 0.0525133, 1.1684123, -1.7045960, 1.1178374
6: -0.4909047, 0.7040354, -0.2549961, 0.4681417, -0.9590464, 0.9590315
7: -0.6173807, 0.8155841, -0.3472473, 0.4758478, -1.0932286, 1.1628313
8: -0.6670679, 0.8455794, -0.3504899, 0.6003999, -1.2674677, 1.1960694
9: -0.7079532, 0.8020560, -0.4175776, 0.5112474, -1.2192006, 1.2196336

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 131

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5702700, upper bound: 1.5060568
time: 2.13 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5702700, upper bound: 1.5060568
time: 2.09 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.5189962, 0.5414410, -0.5738508, 0.5759040, -1.0949003, 1.1152918
1: -0.4519723, 0.6417593, -0.4839600, 0.6847404, -1.1367127, 1.1257193
2: -0.3766927, 0.6576895, -0.4179204, 0.7037759, -1.0804687, 1.0756099
3: -0.3194801, 0.5260263, -0.3756537, 0.5489379, -0.8684180, 0.9016801
4: -0.5400504, 0.5287694, -0.5821376, 0.5716643, -1.1117148, 1.1109071
5: -0.2448135, 1.1612194, -0.3128855, 1.1789155, -1.4237289, 1.4741049
6: -0.3215775, 0.5428849, -0.3524698, 0.5755888, -0.8971663, 0.8953546
7: -0.4300930, 0.5916775, -0.4721063, 0.6370016, -1.0670946, 1.0637839
8: -0.4384841, 0.6982014, -0.4816706, 0.7319694, -1.1704535, 1.1798719
9: -0.5099261, 0.6116219, -0.5507526, 0.6506543, -1.1605804, 1.1623745

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 8

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5834703, upper bound: 1.5884389
time: 1.84 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5834703, upper bound: 1.5884389
time: 2.25 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.7890387, 0.6955979, -0.5629867, 0.5694859, -1.3585246, 1.2585845
1: -0.6102881, 0.8136971, -0.4774605, 0.6768270, -1.2871151, 1.2911576
2: -0.5792634, 0.8663443, -0.4099371, 0.6952052, -1.2744687, 1.2762815
3: -0.6016492, 0.6282682, -0.3646598, 0.5445760, -1.1462252, 0.9929280
4: -0.7547023, 0.7394649, -0.5734157, 0.5632377, -1.3179400, 1.3128805
5: -0.5361838, 1.1703507, -0.2997380, 1.1770294, -1.7132132, 1.4700887
6: -0.4909047, 0.7040354, -0.3456150, 0.5690962, -1.0600009, 1.0496504
7: -0.6173807, 0.8155841, -0.4643679, 0.6280951, -1.2454759, 1.2799520
8: -0.6670679, 0.8455794, -0.4726092, 0.7258670, -1.3929348, 1.3181887
9: -0.7079532, 0.8020560, -0.5428527, 0.6431112, -1.3510643, 1.3449087

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 8

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5834703, upper bound: 1.5902935
time: 2.27 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5834703, upper bound: 1.5902935
time: 2.11 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 6.53 seconds
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5780317
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5780317
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5780317
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5780317
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5702700
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5702700
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5705009
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5705009
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5835800, upper bound: 1.5816530
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5835800, upper bound: 1.5816530
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5835800, upper bound: 1.5819786
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5835800, upper bound: 1.5819786
NS_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5328025
NS_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5328025
NS_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5328025
NS_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5328025
NS_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5060568
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5783356
NS_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5060568
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5784411
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5702700, upper bound: 1.5060568
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5702700, upper bound: 1.5060568
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5702700, upper bound: 1.5060568
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5702700, upper bound: 1.5060568
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5834703, upper bound: 1.5884389
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5834703, upper bound: 1.5884389
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5834703, upper bound: 1.5902935
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5834703, upper bound: 1.5902935
NS_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5328025
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5845202
NS_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5328025
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5845202
NS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5060568
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5783356
NS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5060568
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5051724, upper bound: 1.5784411
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5702700, upper bound: 1.5060568
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5702700, upper bound: 1.5060568
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5702700, upper bound: 1.5060568
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5702700, upper bound: 1.5060568
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5834703, upper bound: 1.5884389
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5834703, upper bound: 1.5884389
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5834703, upper bound: 1.5902935
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 5, lower bound: -1.5834703, upper bound: 1.5902935

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.3846650, 0.3941466, -0.4897729, 0.5190340, -0.9036990, 0.8839195
1: -0.3319941, 0.3536625, -0.4321097, 0.6187906, -0.9507847, 0.7857722
2: -0.2704943, 0.4554276, -0.3550040, 0.6257194, -0.8962137, 0.8104316
3: -0.2115196, 0.4238317, -0.2907113, 0.5107954, -0.7223150, 0.7145430
4: -0.4065562, 0.3909165, -0.5177408, 0.5034260, -0.9099822, 0.9086573
5: 0.1512546, 1.1255324, -0.2044711, 1.1373084, -0.9860537, 1.3300035
6: -0.2236446, 0.4320862, -0.3058346, 0.5231596, -0.7468042, 0.7379208
7: -0.3116984, 0.4164688, -0.4072961, 0.5635933, -0.8752917, 0.8237649
8: -0.3066485, 0.5550406, -0.4149080, 0.6760141, -0.9826626, 0.9699486
9: -0.3718071, 0.4616198, -0.4871923, 0.5877865, -0.9595937, 0.9488121

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 131

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5182330, upper bound: 1.5416536
time: 2.15 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.4785258, upper bound: 1.5417614
time: 1.81 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.3846650, 0.3941466, -0.5189962, 0.5414410, -0.9261061, 0.9131429
1: -0.3319941, 0.3536625, -0.4519723, 0.6417593, -0.9737535, 0.8056349
2: -0.2704943, 0.4554276, -0.3766927, 0.6576895, -0.9281838, 0.8321204
3: -0.2115196, 0.4238317, -0.3194801, 0.5260263, -0.7375458, 0.7433118
4: -0.4065562, 0.3909165, -0.5400504, 0.5287694, -0.9353257, 0.9309669
5: 0.1512546, 1.1255324, -0.2448135, 1.1612194, -1.0099647, 1.3703458
6: -0.2236446, 0.4320862, -0.3215775, 0.5428849, -0.7665295, 0.7536637
7: -0.3116984, 0.4164688, -0.4300930, 0.5916775, -0.9033759, 0.8465618
8: -0.3066485, 0.5550406, -0.4384841, 0.6982014, -1.0048499, 0.9935247
9: -0.3718071, 0.4616198, -0.5099261, 0.6116219, -0.9834290, 0.9715459

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 131

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5182330, upper bound: 1.5416536
time: 1.92 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.4785258, upper bound: 1.5417615
time: 1.89 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.4976699, 0.5514714, -0.4897729, 0.5190340, -1.0167040, 1.0412443
1: -0.4630894, 0.4839199, -0.4321097, 0.6187906, -1.0818800, 0.9160296
2: -0.3678207, 0.6678746, -0.3550040, 0.6257194, -0.9935400, 1.0228785
3: -0.3079270, 0.5140205, -0.2907113, 0.5107954, -0.8187223, 0.8047317
4: -0.5304418, 0.5295544, -0.5177408, 0.5034260, -1.0338677, 1.0472951
5: -0.1186955, 1.1348166, -0.2044711, 1.1373084, -1.2560039, 1.3392878
6: -0.3196089, 0.5404234, -0.3058346, 0.5231596, -0.8427684, 0.8462579
7: -0.4204883, 0.5938017, -0.4072961, 0.5635933, -0.9840816, 1.0010978
8: -0.4448276, 0.6881066, -0.4149080, 0.6760141, -1.1208417, 1.1030146
9: -0.5130015, 0.6167052, -0.4871923, 0.5877865, -1.1007880, 1.1038976

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 131

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.4965745, upper bound: 1.5386175
time: 2.02 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.4668125, upper bound: 1.5390539
time: 2.07 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.4976699, 0.5514714, -0.5189962, 0.5414410, -1.0391109, 1.0704676
1: -0.4630894, 0.4839199, -0.4519723, 0.6417593, -1.1048486, 0.9358922
2: -0.3678207, 0.6678746, -0.3766927, 0.6576895, -1.0255102, 1.0445673
3: -0.3079270, 0.5140205, -0.3194801, 0.5260263, -0.8339533, 0.8335006
4: -0.5304418, 0.5295544, -0.5400504, 0.5287694, -1.0592113, 1.0696049
5: -0.1186955, 1.1348166, -0.2448135, 1.1612194, -1.2799149, 1.3796301
6: -0.3196089, 0.5404234, -0.3215775, 0.5428849, -0.8624938, 0.8620009
7: -0.4204883, 0.5938017, -0.4300930, 0.5916775, -1.0121658, 1.0238947
8: -0.4448276, 0.6881066, -0.4384841, 0.6982014, -1.1430290, 1.1265907
9: -0.5130015, 0.6167052, -0.5099261, 0.6116219, -1.1246233, 1.1266313

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 131

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.4965745, upper bound: 1.5386175
time: 1.77 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.4668125, upper bound: 1.5390539
time: 2.01 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.3846650, 0.3941466, -0.7955889, 0.6996196, -1.0842847, 1.1897354
1: -0.3319941, 0.3536625, -0.6183987, 0.8186010, -1.1505952, 0.9720613
2: -0.2704943, 0.4554276, -0.5940627, 0.8671799, -1.1376741, 1.0494903
3: -0.2115196, 0.4238317, -0.6223035, 0.6275892, -0.8391088, 1.0461352
4: -0.4065562, 0.3909165, -0.7609484, 0.7508355, -1.1573918, 1.1518648
5: 0.1512546, 1.1255324, -0.5552799, 1.1499333, -0.9986787, 1.6808122
6: -0.2236446, 0.4320862, -0.5114857, 0.7123836, -0.9360282, 0.9435719
7: -0.3116984, 0.4164688, -0.6216893, 0.8242326, -1.1359310, 1.0381582
8: -0.3066485, 0.5550406, -0.6844128, 0.8487247, -1.1553732, 1.2394534
9: -0.3718071, 0.4616198, -0.7094772, 0.8068328, -1.1786399, 1.1710970

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 131

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5154998, upper bound: 1.5324437
time: 1.76 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.4774282, upper bound: 1.5325705
time: 1.81 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.3846650, 0.3941466, -0.7949440, 0.6999147, -1.0845797, 1.1890907
1: -0.3319941, 0.3536625, -0.6173127, 0.8182845, -1.1502787, 0.9709753
2: -0.2704943, 0.4554276, -0.5931125, 0.8664836, -1.1369778, 1.0485401
3: -0.2115196, 0.4238317, -0.6204950, 0.6282682, -0.8397878, 1.0443267
4: -0.4065562, 0.3909165, -0.7597365, 0.7499406, -1.1564969, 1.1506529
5: 0.1512546, 1.1255324, -0.5578788, 1.1703507, -1.0190960, 1.6834111
6: -0.2236446, 0.4320862, -0.5108553, 0.7113250, -0.9349697, 0.9429415
7: -0.3116984, 0.4164688, -0.6211529, 0.8236696, -1.1353681, 1.0376217
8: -0.3066485, 0.5550406, -0.6829801, 0.8487276, -1.1553761, 1.2380207
9: -0.3718071, 0.4616198, -0.7085228, 0.8061590, -1.1779661, 1.1701427

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 131

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5154998, upper bound: 1.5324438
time: 2.21 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.4774282, upper bound: 1.5325706
time: 1.90 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.4976699, 0.5514714, -0.7955889, 0.6996196, -1.1972895, 1.3470602
1: -0.4630894, 0.4839199, -0.6183987, 0.8186010, -1.2816904, 1.1023185
2: -0.3678207, 0.6678746, -0.5940627, 0.8671799, -1.2350006, 1.2619373
3: -0.3079270, 0.5140205, -0.6223035, 0.6275892, -0.9355162, 1.1363240
4: -0.5304418, 0.5295544, -0.7609484, 0.7508355, -1.2812772, 1.2905028
5: -0.1186955, 1.1348166, -0.5552799, 1.1499333, -1.2686288, 1.6900965
6: -0.3196089, 0.5404234, -0.5114857, 0.7123836, -1.0319924, 1.0519091
7: -0.4204883, 0.5938017, -0.6216893, 0.8242326, -1.2447209, 1.2154911
8: -0.4448276, 0.6881066, -0.6844128, 0.8487247, -1.2935523, 1.3725195
9: -0.5130015, 0.6167052, -0.7094772, 0.8068328, -1.3198342, 1.3261824

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 131

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.4965745, upper bound: 1.5325658
time: 1.71 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.4668125, upper bound: 1.5327449
time: 1.87 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.4976699, 0.5514714, -0.7949440, 0.6999147, -1.1975846, 1.3464154
1: -0.4630894, 0.4839199, -0.6173127, 0.8182845, -1.2813740, 1.1012325
2: -0.3678207, 0.6678746, -0.5931125, 0.8664836, -1.2343043, 1.2609870
3: -0.3079270, 0.5140205, -0.6204950, 0.6282682, -0.9361952, 1.1345155
4: -0.5304418, 0.5295544, -0.7597365, 0.7499406, -1.2803824, 1.2892909
5: -0.1186955, 1.1348166, -0.5578788, 1.1703507, -1.2890462, 1.6926954
6: -0.3196089, 0.5404234, -0.5108553, 0.7113250, -1.0309340, 1.0512787
7: -0.4204883, 0.5938017, -0.6211529, 0.8236696, -1.2441579, 1.2149546
8: -0.4448276, 0.6881066, -0.6829801, 0.8487276, -1.2935551, 1.3710867
9: -0.5130015, 0.6167052, -0.7085228, 0.8061590, -1.3191605, 1.3252280

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 131

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.4965745, upper bound: 1.5325658
time: 1.93 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.4668125, upper bound: 1.5327449
time: 1.79 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.4897729, 0.5190340, -0.4897729, 0.5190340, -1.0088069, 1.0088069
1: -0.4321097, 0.6187906, -0.4321097, 0.6187906, -1.0509002, 1.0509002
2: -0.3550040, 0.6257194, -0.3550040, 0.6257194, -0.9807233, 0.9807233
3: -0.2907113, 0.5107954, -0.2907113, 0.5107954, -0.8015066, 0.8015066
4: -0.5177408, 0.5034260, -0.5177408, 0.5034260, -1.0211668, 1.0211668
5: -0.2044711, 1.1373084, -0.2044711, 1.1373084, -1.3417795, 1.3417795
6: -0.3058346, 0.5231596, -0.3058346, 0.5231596, -0.8289942, 0.8289942
7: -0.4072961, 0.5635933, -0.4072961, 0.5635933, -0.9708893, 0.9708893
8: -0.4149080, 0.6760141, -0.4149080, 0.6760141, -1.0909221, 1.0909221
9: -0.4871923, 0.5877865, -0.4871923, 0.5877865, -1.0749788, 1.0749788

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5803197, upper bound: 1.5468169
time: 2.42 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5561153, upper bound: 1.5483101
time: 1.97 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.4897729, 0.5190340, -0.5189962, 0.5414410, -1.0312140, 1.0380303
1: -0.4321097, 0.6187906, -0.4519723, 0.6417593, -1.0738686, 1.0707629
2: -0.3550040, 0.6257194, -0.3766927, 0.6576895, -1.0126934, 1.0024121
3: -0.2907113, 0.5107954, -0.3194801, 0.5260263, -0.8167375, 0.8302754
4: -0.5177408, 0.5034260, -0.5400504, 0.5287694, -1.0465102, 1.0434763
5: -0.2044711, 1.1373084, -0.2448135, 1.1612194, -1.3656905, 1.3821218
6: -0.3058346, 0.5231596, -0.3215775, 0.5428849, -0.8487195, 0.8447371
7: -0.4072961, 0.5635933, -0.4300930, 0.5916775, -0.9989735, 0.9936862
8: -0.4149080, 0.6760141, -0.4384841, 0.6982014, -1.1131094, 1.1144983
9: -0.4871923, 0.5877865, -0.5099261, 0.6116219, -1.0988142, 1.0977125

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5803197, upper bound: 1.5468169
time: 2.36 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5561153, upper bound: 1.5483101
time: 1.87 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.7496231, 0.6859171, -0.4897729, 0.5190340, -1.2686571, 1.1756899
1: -0.5881135, 0.7931735, -0.4321097, 0.6187906, -1.2069042, 1.2252829
2: -0.5522286, 0.8548584, -0.3550040, 0.6257194, -1.1779480, 1.2098624
3: -0.5610748, 0.6149958, -0.2907113, 0.5107954, -1.0718701, 0.9057070
4: -0.7187871, 0.7081248, -0.5177408, 0.5034260, -1.2222130, 1.2258656
5: -0.4895747, 1.1498928, -0.2044711, 1.1373084, -1.6268830, 1.3543639
6: -0.4465384, 0.6813953, -0.3058346, 0.5231596, -0.9696980, 0.9872298
7: -0.6118792, 0.7767581, -0.4072961, 0.5635933, -1.1754725, 1.1840541
8: -0.6199242, 0.8393949, -0.4149080, 0.6760141, -1.2959384, 1.2543030
9: -0.6868015, 0.7742674, -0.4871923, 0.5877865, -1.2745880, 1.2614598

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 131

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5752039, upper bound: 1.5462807
time: 1.91 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5482833, upper bound: 1.5479671
time: 2.21 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.7496231, 0.6859171, -0.5189962, 0.5414410, -1.2910640, 1.2049134
1: -0.5881135, 0.7931735, -0.4519723, 0.6417593, -1.2298728, 1.2451458
2: -0.5522286, 0.8548584, -0.3766927, 0.6576895, -1.2099180, 1.2315512
3: -0.5610748, 0.6149958, -0.3194801, 0.5260263, -1.0871011, 0.9344758
4: -0.7187871, 0.7081248, -0.5400504, 0.5287694, -1.2475566, 1.2481751
5: -0.4895747, 1.1498928, -0.2448135, 1.1612194, -1.6507940, 1.3947062
6: -0.4465384, 0.6813953, -0.3215775, 0.5428849, -0.9894233, 1.0029728
7: -0.6118792, 0.7767581, -0.4300930, 0.5916775, -1.2035567, 1.2068510
8: -0.6199242, 0.8393949, -0.4384841, 0.6982014, -1.3181256, 1.2778790
9: -0.6868015, 0.7742674, -0.5099261, 0.6116219, -1.2984233, 1.2841935

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 131

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5752039, upper bound: 1.5462807
time: 2.03 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5482833, upper bound: 1.5479671
time: 2.03 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.4071087, 0.4252596, -0.7955889, 0.6991391, -1.1062479, 1.2208484
1: -0.3569512, 0.3870762, -0.6165254, 0.8186010, -1.1755521, 1.0036016
2: -0.2898312, 0.4966636, -0.5938188, 0.8671799, -1.1570110, 1.0904825
3: -0.2303881, 0.4423752, -0.6214541, 0.6275892, -0.8579773, 1.0638292
4: -0.4306142, 0.4182070, -0.7609484, 0.7470133, -1.1776276, 1.1791553
5: 0.0903304, 1.1526667, -0.5544360, 1.1499333, -1.0596030, 1.7071028
6: -0.2427156, 0.4529262, -0.5114857, 0.7050282, -0.9477438, 0.9644119
7: -0.3334745, 0.4523583, -0.6188263, 0.8242326, -1.1577071, 1.0711846
8: -0.3334717, 0.5812222, -0.6786488, 0.8487247, -1.1821964, 1.2598710
9: -0.3993464, 0.4918939, -0.7094772, 0.8059303, -1.2052767, 1.2013711

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 131

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5153658, upper bound: 1.5402952
time: 1.91 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.4774282, upper bound: 1.5409315
time: 1.90 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.5261132, 0.5601768, -0.7955889, 0.6991391, -1.2252523, 1.3557657
1: -0.4705340, 0.5071646, -0.6165254, 0.8186010, -1.2891350, 1.1236899
2: -0.3742779, 0.6899731, -0.5938188, 0.8671799, -1.2414577, 1.2837919
3: -0.3130228, 0.5298095, -0.6214541, 0.6275892, -0.9406121, 1.1512636
4: -0.5564782, 0.5370346, -0.7609484, 0.7470133, -1.3034916, 1.2979829
5: -0.1423194, 1.1591973, -0.5544360, 1.1499333, -1.2922528, 1.7136333
6: -0.3267299, 0.5598539, -0.5114857, 0.7050282, -1.0317581, 1.0713396
7: -0.4267581, 0.6103408, -0.6188263, 0.8242326, -1.2509906, 1.2291671
8: -0.4519157, 0.7141070, -0.6786488, 0.8487247, -1.3006405, 1.3927559
9: -0.5258906, 0.6254055, -0.7094772, 0.8059303, -1.3318210, 1.3348827

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 131

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.4965238, upper bound: 1.5399197
time: 1.96 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.4668085, upper bound: 1.5407442
time: 1.89 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.5189962, 0.5414410, -0.3846650, 0.3941466, -0.9131429, 0.9261061
1: -0.4519723, 0.6417593, -0.3319941, 0.3536625, -0.8056349, 0.9737535
2: -0.3766927, 0.6576895, -0.2704943, 0.4554276, -0.8321204, 0.9281838
3: -0.3194801, 0.5260263, -0.2115196, 0.4238317, -0.7433118, 0.7375458
4: -0.5400504, 0.5287694, -0.4065562, 0.3909165, -0.9309669, 0.9353257
5: -0.2448135, 1.1612194, 0.1512546, 1.1255324, -1.3703458, 1.0099647
6: -0.3215775, 0.5428849, -0.2236446, 0.4320862, -0.7536637, 0.7665295
7: -0.4300930, 0.5916775, -0.3116984, 0.4164688, -0.8465618, 0.9033759
8: -0.4384841, 0.6982014, -0.3066485, 0.5550406, -0.9935247, 1.0048499
9: -0.5099261, 0.6116219, -0.3718071, 0.4616198, -0.9715459, 0.9834290

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 8

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5630788, upper bound: 1.4672855
time: 2.26 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5390539, upper bound: 1.4672855
time: 2.32 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.5189962, 0.5414410, -0.4976699, 0.5514714, -1.0704676, 1.0391109
1: -0.4519723, 0.6417593, -0.4630894, 0.4839199, -0.9358922, 1.1048486
2: -0.3766927, 0.6576895, -0.3678207, 0.6678746, -1.0445673, 1.0255102
3: -0.3194801, 0.5260263, -0.3079270, 0.5140205, -0.8335006, 0.8339533
4: -0.5400504, 0.5287694, -0.5304418, 0.5295544, -1.0696049, 1.0592113
5: -0.2448135, 1.1612194, -0.1186955, 1.1348166, -1.3796301, 1.2799149
6: -0.3215775, 0.5428849, -0.3196089, 0.5404234, -0.8620009, 0.8624938
7: -0.4300930, 0.5916775, -0.4204883, 0.5938017, -1.0238947, 1.0121658
8: -0.4384841, 0.6982014, -0.4448276, 0.6881066, -1.1265907, 1.1430290
9: -0.5099261, 0.6116219, -0.5130015, 0.6167052, -1.1266313, 1.1246233

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5630788, upper bound: 1.4672855
time: 1.93 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5390539, upper bound: 1.4672855
time: 2.19 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.7890387, 0.6955979, -0.3846650, 0.3941466, -1.1831853, 1.0802629
1: -0.6102881, 0.8136971, -0.3319941, 0.3536625, -0.9639506, 1.1456913
2: -0.5792634, 0.8663443, -0.2704943, 0.4554276, -1.0346911, 1.1368386
3: -0.6016492, 0.6282682, -0.2115196, 0.4238317, -1.0254809, 0.8397878
4: -0.7547023, 0.7394649, -0.4065562, 0.3909165, -1.1456188, 1.1460211
5: -0.5361838, 1.1703507, 0.1512546, 1.1255324, -1.6617162, 1.0190960
6: -0.4909047, 0.7040354, -0.2236446, 0.4320862, -0.9229909, 0.9276800
7: -0.6173807, 0.8155841, -0.3116984, 0.4164688, -1.0338495, 1.1272825
8: -0.6670679, 0.8455794, -0.3066485, 0.5550406, -1.2221085, 1.1522279
9: -0.7079532, 0.8020560, -0.3718071, 0.4616198, -1.1695729, 1.1738632

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 8

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5536762, upper bound: 1.4672855
time: 2.14 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5324936, upper bound: 1.4672855
time: 2.04 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.7890387, 0.6955979, -0.4976699, 0.5514714, -1.3405101, 1.1932678
1: -0.6102881, 0.8136971, -0.4630894, 0.4839199, -1.0942080, 1.2767866
2: -0.5792634, 0.8663443, -0.3678207, 0.6678746, -1.2471380, 1.2341651
3: -0.6016492, 0.6282682, -0.3079270, 0.5140205, -1.1156697, 0.9361952
4: -0.7547023, 0.7394649, -0.5304418, 0.5295544, -1.2842567, 1.2699066
5: -0.5361838, 1.1703507, -0.1186955, 1.1348166, -1.6710005, 1.2890462
6: -0.4909047, 0.7040354, -0.3196089, 0.5404234, -1.0313282, 1.0236443
7: -0.6173807, 0.8155841, -0.4204883, 0.5938017, -1.2111825, 1.2360723
8: -0.6670679, 0.8455794, -0.4448276, 0.6881066, -1.3551745, 1.2904069
9: -0.7079532, 0.8020560, -0.5130015, 0.6167052, -1.3246584, 1.3150575

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5536762, upper bound: 1.4672855
time: 1.90 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5324936, upper bound: 1.4672855
time: 2.13 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.5189962, 0.5414410, -0.4897729, 0.5190340, -1.0380303, 1.0312140
1: -0.4519723, 0.6417593, -0.4321097, 0.6187906, -1.0707629, 1.0738688
2: -0.3766927, 0.6576895, -0.3550040, 0.6257194, -1.0024121, 1.0126934
3: -0.3194801, 0.5260263, -0.2907113, 0.5107954, -0.8302754, 0.8167375
4: -0.5400504, 0.5287694, -0.5177408, 0.5034260, -1.0434763, 1.0465102
5: -0.2448135, 1.1612194, -0.2044711, 1.1373084, -1.3821218, 1.3656905
6: -0.3215775, 0.5428849, -0.3058346, 0.5231596, -0.8447371, 0.8487195
7: -0.4300930, 0.5916775, -0.4072961, 0.5635933, -0.9936862, 0.9989735
8: -0.4384841, 0.6982014, -0.4149080, 0.6760141, -1.1144983, 1.1131094
9: -0.5099261, 0.6116219, -0.4871923, 0.5877865, -1.0977125, 1.0988142

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5796025, upper bound: 1.5503897
time: 2.24 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5552883, upper bound: 1.5525761
time: 2.34 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.5189962, 0.5414410, -0.7496231, 0.6859171, -1.2049134, 1.2910640
1: -0.4519723, 0.6417593, -0.5881135, 0.7931735, -1.2451458, 1.2298728
2: -0.3766927, 0.6576895, -0.5522286, 0.8548584, -1.2315512, 1.2099180
3: -0.3194801, 0.5260263, -0.5610748, 0.6149958, -0.9344758, 1.0871011
4: -0.5400504, 0.5287694, -0.7187871, 0.7081248, -1.2481751, 1.2475566
5: -0.2448135, 1.1612194, -0.4895747, 1.1498928, -1.3947062, 1.6507940
6: -0.3215775, 0.5428849, -0.4465384, 0.6813953, -1.0029728, 0.9894233
7: -0.4300930, 0.5916775, -0.6118792, 0.7767581, -1.2068510, 1.2035567
8: -0.4384841, 0.6982014, -0.6199242, 0.8393949, -1.2778790, 1.3181256
9: -0.5099261, 0.6116219, -0.6868015, 0.7742674, -1.2841935, 1.2984233

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5796025, upper bound: 1.5503897
time: 2.18 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5552883, upper bound: 1.5525761
time: 1.82 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.7890387, 0.6955979, -0.4897729, 0.5190340, -1.3080728, 1.1853708
1: -0.6102881, 0.8136971, -0.4321097, 0.6187906, -1.2290788, 1.2458068
2: -0.5792634, 0.8663443, -0.3550040, 0.6257194, -1.2049828, 1.2213483
3: -0.6016492, 0.6282682, -0.2907113, 0.5107954, -1.1124445, 0.9189795
4: -0.7547023, 0.7394649, -0.5177408, 0.5034260, -1.2581283, 1.2572057
5: -0.5361838, 1.1703507, -0.2044711, 1.1373084, -1.6734922, 1.3748217
6: -0.4909047, 0.7040354, -0.3058346, 0.5231596, -1.0140643, 1.0098699
7: -0.6173807, 0.8155841, -0.4072961, 0.5635933, -1.1809740, 1.2228801
8: -0.6670679, 0.8455794, -0.4149080, 0.6760141, -1.3430820, 1.2604874
9: -0.7079532, 0.8020560, -0.4871923, 0.5877865, -1.2957397, 1.2892483

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5751841, upper bound: 1.5515038
time: 4.86 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5482833, upper bound: 1.5538225
time: 1.84 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.7890387, 0.6955979, -0.7496231, 0.6720124, -1.4610512, 1.4452209
1: -0.6102881, 0.8136971, -0.5873997, 0.7931735, -1.4034617, 1.4010968
2: -0.5792634, 0.8663443, -0.5442715, 0.8548584, -1.4341218, 1.4106159
3: -0.6016492, 0.6282682, -0.5583533, 0.6149958, -1.2166450, 1.1866215
4: -0.7547023, 0.7394649, -0.7054991, 0.7081248, -1.4628271, 1.4449639
5: -0.5361838, 1.1703507, -0.4849943, 1.1498928, -1.6860766, 1.6553450
6: -0.4909047, 0.7040354, -0.4390086, 0.6813953, -1.1723001, 1.1430440
7: -0.6173807, 0.8155841, -0.6000801, 0.7767581, -1.3941388, 1.4156642
8: -0.6670679, 0.8455794, -0.6145568, 0.8393949, -1.5064628, 1.4601362
9: -0.7079532, 0.8020560, -0.6775312, 0.7742674, -1.4822206, 1.4795873

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5751841, upper bound: 1.5515038
time: 2.07 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5482833, upper bound: 1.5538225
time: 2.16 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.4071087, 0.4252596, -0.5189962, 0.5414410, -0.9485497, 0.9442558
1: -0.3569512, 0.3870762, -0.4519723, 0.6417593, -0.9987105, 0.8390485
2: -0.2898312, 0.4966636, -0.3766927, 0.6576895, -0.9475207, 0.8733563
3: -0.2303881, 0.4423752, -0.3194801, 0.5260263, -0.7564144, 0.7618552
4: -0.4306142, 0.4182070, -0.5400504, 0.5287694, -0.9593837, 0.9582575
5: 0.0903304, 1.1526667, -0.2448135, 1.1612194, -1.0708890, 1.3974802
6: -0.2427156, 0.4529262, -0.3215775, 0.5428849, -0.7856004, 0.7745037
7: -0.3334745, 0.4523583, -0.4300930, 0.5916775, -0.9251521, 0.8824513
8: -0.3334717, 0.5812222, -0.4384841, 0.6982014, -1.0316730, 1.0197064
9: -0.3993464, 0.4918939, -0.5099261, 0.6116219, -1.0109682, 1.0018200

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 131

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5181581, upper bound: 1.5489836
time: 2.01 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4785258, upper bound: 1.5493680
time: 1.92 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.5261132, 0.5601768, -0.5189962, 0.5414410, -1.0675542, 1.0791731
1: -0.4705340, 0.5071646, -0.4519723, 0.6417593, -1.1122932, 0.9591369
2: -0.3742779, 0.6899731, -0.3766927, 0.6576895, -1.0319674, 1.0666658
3: -0.3130228, 0.5298095, -0.3194801, 0.5260263, -0.8390491, 0.8492896
4: -0.5564782, 0.5370346, -0.5400504, 0.5287694, -1.0852476, 1.0770850
5: -0.1423194, 1.1591973, -0.2448135, 1.1612194, -1.3035388, 1.4040108
6: -0.3267299, 0.5598539, -0.3215775, 0.5428849, -0.8696147, 0.8814314
7: -0.4267581, 0.6103408, -0.4300930, 0.5916775, -1.0184356, 1.0404338
8: -0.4519157, 0.7141070, -0.4384841, 0.6982014, -1.1501172, 1.1525912
9: -0.5258906, 0.6254055, -0.5099261, 0.6116219, -1.1375124, 1.1353316

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 131

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.4965238, upper bound: 1.4806897
time: 1.99 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.4668085, upper bound: 1.4806897
time: 1.88 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.4071087, 0.4252596, -0.8245026, 0.7058088, -1.1129174, 1.2497623
1: -0.3569512, 0.3870762, -0.6334322, 0.8302587, -1.1872098, 1.0205084
2: -0.2898312, 0.4966636, -0.6130311, 0.8797157, -1.1695468, 1.1096947
3: -0.2303881, 0.4423752, -0.6461918, 0.6302514, -0.8606395, 1.0885670
4: -0.4306142, 0.4182070, -0.7810256, 0.7680447, -1.1986589, 1.1992327
5: 0.0903304, 1.1526667, -0.5761247, 1.1705613, -1.0802310, 1.7287915
6: -0.2427156, 0.4529262, -0.5185205, 0.7224741, -0.9651896, 0.9714466
7: -0.3334745, 0.4523583, -0.6430010, 0.8297208, -1.1631954, 1.0953593
8: -0.3334717, 0.5812222, -0.7052547, 0.8553404, -1.1888120, 1.2864769
9: -0.3993464, 0.4918939, -0.7199774, 0.8163187, -1.2156651, 1.2118714

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 131

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5153658, upper bound: 1.5402952
time: 1.79 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.4774282, upper bound: 1.5409315
time: 1.96 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.5261132, 0.5601768, -0.8245026, 0.7058088, -1.2319219, 1.3846796
1: -0.4705340, 0.5071646, -0.6334322, 0.8302587, -1.3007927, 1.1405969
2: -0.3742779, 0.6899731, -0.6130311, 0.8797157, -1.2539935, 1.3030043
3: -0.3130228, 0.5298095, -0.6461918, 0.6302514, -0.9432743, 1.1760013
4: -0.5564782, 0.5370346, -0.7810256, 0.7680447, -1.3245229, 1.3180602
5: -0.1423194, 1.1591973, -0.5761247, 1.1705613, -1.3128808, 1.7353220
6: -0.3267299, 0.5598539, -0.5185205, 0.7224741, -1.0492040, 1.0783744
7: -0.4267581, 0.6103408, -0.6430010, 0.8297208, -1.2564789, 1.2533418
8: -0.4519157, 0.7141070, -0.7052547, 0.8553404, -1.3072561, 1.4193617
9: -0.5258906, 0.6254055, -0.7199774, 0.8163187, -1.3422093, 1.3453829

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 131

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.4965238, upper bound: 1.5399197
time: 1.72 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.4668085, upper bound: 1.4672855
time: 4.91 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.5189962, 0.5414410, -0.4071087, 0.4252596, -0.9442558, 0.9485497
1: -0.4519723, 0.6417593, -0.3569512, 0.3870762, -0.8390485, 0.9987105
2: -0.3766927, 0.6576895, -0.2898312, 0.4966636, -0.8733563, 0.9475207
3: -0.3194801, 0.5260263, -0.2303881, 0.4423752, -0.7618552, 0.7564144
4: -0.5400504, 0.5287694, -0.4306142, 0.4182070, -0.9582575, 0.9593837
5: -0.2448135, 1.1612194, 0.0903304, 1.1526667, -1.3974802, 1.0708890
6: -0.3215775, 0.5428849, -0.2427156, 0.4529262, -0.7745037, 0.7856004
7: -0.4300930, 0.5916775, -0.3334745, 0.4523583, -0.8824513, 0.9251521
8: -0.4384841, 0.6982014, -0.3334717, 0.5812222, -1.0197064, 1.0316730
9: -0.5099261, 0.6116219, -0.3993464, 0.4918939, -1.0018200, 1.0109682

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5630788, upper bound: 1.4672855
time: 1.90 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5390539, upper bound: 1.4672855
time: 1.91 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.5189962, 0.5414410, -0.5261132, 0.5601768, -1.0791731, 1.0675542
1: -0.4519723, 0.6417593, -0.4705340, 0.5071646, -0.9591369, 1.1122932
2: -0.3766927, 0.6576895, -0.3742779, 0.6899731, -1.0666658, 1.0319674
3: -0.3194801, 0.5260263, -0.3130228, 0.5298095, -0.8492896, 0.8390491
4: -0.5400504, 0.5287694, -0.5564782, 0.5370346, -1.0770850, 1.0852476
5: -0.2448135, 1.1612194, -0.1423194, 1.1591973, -1.4040108, 1.3035388
6: -0.3215775, 0.5428849, -0.3267299, 0.5598539, -0.8814314, 0.8696147
7: -0.4300930, 0.5916775, -0.4267581, 0.6103408, -1.0404338, 1.0184356
8: -0.4384841, 0.6982014, -0.4519157, 0.7141070, -1.1525912, 1.1501172
9: -0.5099261, 0.6116219, -0.5258906, 0.6254055, -1.1353316, 1.1375124

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 8

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5630788, upper bound: 1.4672855
time: 2.24 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5390539, upper bound: 1.4672855
time: 2.27 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.7890387, 0.6955979, -0.4071087, 0.4252596, -1.2142982, 1.1027067
1: -0.6102881, 0.8136971, -0.3569512, 0.3870762, -0.9973644, 1.1706483
2: -0.5792634, 0.8663443, -0.2898312, 0.4966636, -1.0759270, 1.1561755
3: -0.6016492, 0.6282682, -0.2303881, 0.4423752, -1.0440243, 0.8586563
4: -0.7547023, 0.7394649, -0.4306142, 0.4182070, -1.1729094, 1.1700791
5: -0.5361838, 1.1703507, 0.0903304, 1.1526667, -1.6888504, 1.0800203
6: -0.4909047, 0.7040354, -0.2427156, 0.4529262, -0.9438309, 0.9467509
7: -0.6173807, 0.8155841, -0.3334745, 0.4523583, -1.0697391, 1.1490586
8: -0.6670679, 0.8455794, -0.3334717, 0.5812222, -1.2482901, 1.1790512
9: -0.7079532, 0.8020560, -0.3993464, 0.4918939, -1.1998470, 1.2014024

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5536762, upper bound: 1.4672855
time: 1.85 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5324936, upper bound: 1.4672855
time: 1.88 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.7890387, 0.6955979, -0.5261132, 0.5601768, -1.3492155, 1.2217110
1: -0.6102881, 0.8136971, -0.4705340, 0.5071646, -1.1174527, 1.2842311
2: -0.5792634, 0.8663443, -0.3742779, 0.6899731, -1.2692366, 1.2406222
3: -0.6016492, 0.6282682, -0.3130228, 0.5298095, -1.1314588, 0.9412911
4: -0.7547023, 0.7394649, -0.5564782, 0.5370346, -1.2917368, 1.2959430
5: -0.5361838, 1.1703507, -0.1423194, 1.1591973, -1.6953812, 1.3126701
6: -0.4909047, 0.7040354, -0.3267299, 0.5598539, -1.0507586, 1.0307653
7: -0.6173807, 0.8155841, -0.4267581, 0.6103408, -1.2277215, 1.2423422
8: -0.6670679, 0.8455794, -0.4519157, 0.7141070, -1.3811749, 1.2974951
9: -0.7079532, 0.8020560, -0.5258906, 0.6254055, -1.3333586, 1.3279467

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5536762, upper bound: 1.4672855
time: 1.92 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5324936, upper bound: 1.4672855
time: 2.00 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.5189962, 0.5414410, -0.5189962, 0.5414410, -1.0604372, 1.0604372
1: -0.4519723, 0.6417593, -0.4519723, 0.6417593, -1.0937316, 1.0937316
2: -0.3766927, 0.6576895, -0.3766927, 0.6576895, -1.0343822, 1.0343822
3: -0.3194801, 0.5260263, -0.3194801, 0.5260263, -0.8455064, 0.8455064
4: -0.5400504, 0.5287694, -0.5400504, 0.5287694, -1.0688199, 1.0688199
5: -0.2448135, 1.1612194, -0.2448135, 1.1612194, -1.4060328, 1.4060328
6: -0.3215775, 0.5428849, -0.3215775, 0.5428849, -0.8644624, 0.8644624
7: -0.4300930, 0.5916775, -0.4300930, 0.5916775, -1.0217705, 1.0217705
8: -0.4384841, 0.6982014, -0.4384841, 0.6982014, -1.1366855, 1.1366855
9: -0.5099261, 0.6116219, -0.5099261, 0.6116219, -1.1215479, 1.1215479

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 8

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5796825, upper bound: 1.5503897
time: 2.50 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5553159, upper bound: 1.5525761
time: 1.96 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.5189962, 0.5414410, -0.7890387, 0.6955979, -1.2145941, 1.3304797
1: -0.4519723, 0.6417593, -0.6102881, 0.8136971, -1.2656693, 1.2520474
2: -0.3766927, 0.6576895, -0.5792634, 0.8663443, -1.2430370, 1.2369530
3: -0.3194801, 0.5260263, -0.6016492, 0.6282682, -0.9477483, 1.1276755
4: -0.5400504, 0.5287694, -0.7547023, 0.7394649, -1.2795153, 1.2834718
5: -0.2448135, 1.1612194, -0.5361838, 1.1703507, -1.4151642, 1.6974032
6: -0.3215775, 0.5428849, -0.4909047, 0.7040354, -1.0256130, 1.0337896
7: -0.4300930, 0.5916775, -0.6173807, 0.8155841, -1.2456770, 1.2090583
8: -0.4384841, 0.6982014, -0.6670679, 0.8455794, -1.2840636, 1.3652692
9: -0.5099261, 0.6116219, -0.7079532, 0.8020560, -1.3119822, 1.3195751

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5796825, upper bound: 1.5503897
time: 2.51 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5553159, upper bound: 1.5525761
time: 2.09 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.7890387, 0.6955979, -0.5189962, 0.5414410, -1.3304797, 1.2145941
1: -0.6102881, 0.8136971, -0.4519723, 0.6417593, -1.2520474, 1.2656693
2: -0.5792634, 0.8663443, -0.3766927, 0.6576895, -1.2369530, 1.2430370
3: -0.6016492, 0.6282682, -0.3194801, 0.5260263, -1.1276755, 0.9477483
4: -0.7547023, 0.7394649, -0.5400504, 0.5287694, -1.2834718, 1.2795153
5: -0.5361838, 1.1703507, -0.2448135, 1.1612194, -1.6974032, 1.4151642
6: -0.4909047, 0.7040354, -0.3215775, 0.5428849, -1.0337896, 1.0256130
7: -0.6173807, 0.8155841, -0.4300930, 0.5916775, -1.2090583, 1.2456770
8: -0.6670679, 0.8455794, -0.4384841, 0.6982014, -1.3652692, 1.2840636
9: -0.7079532, 0.8020560, -0.5099261, 0.6116219, -1.3195751, 1.3119822

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 8

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5751841, upper bound: 1.5515038
time: 1.95 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5482833, upper bound: 1.5538225
time: 2.91 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.7890387, 0.6955979, -0.7586306, 0.6945495, -1.4835882, 1.4542285
1: -0.6102881, 0.8136971, -0.6001272, 0.8047414, -1.4150295, 1.4138243
2: -0.5792634, 0.8663443, -0.5648618, 0.8619384, -1.4412018, 1.4312061
3: -0.6016492, 0.6282682, -0.5652003, 0.6271629, -1.2288121, 1.1934686
4: -0.7547023, 0.7394649, -0.7398537, 0.7153342, -1.4700365, 1.4793186
5: -0.5361838, 1.1703507, -0.5183118, 1.1703507, -1.7065344, 1.6886625
6: -0.4909047, 0.7040354, -0.4568561, 0.6978746, -1.1887794, 1.1608915
7: -0.6173807, 0.8155841, -0.6142606, 0.7988310, -1.4162118, 1.4298446
8: -0.6670679, 0.8455794, -0.6380535, 0.8441756, -1.5112436, 1.4836330
9: -0.7079532, 0.8020560, -0.6891103, 0.7900261, -1.4979793, 1.4911664

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5751841, upper bound: 1.5515038
time: 2.20 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5482833, upper bound: 1.5538225
time: 2.22 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 6.52 seconds
NS_A1_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5182330, upper bound: 1.5416536
NS_A1_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.4785258, upper bound: 1.5417614
NS_A1_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5182330, upper bound: 1.5416536
NS_A1_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.4785258, upper bound: 1.5417615
NS_A1_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.4965745, upper bound: 1.5386175
NS_A1_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.4668125, upper bound: 1.5390539
NS_A1_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.4965745, upper bound: 1.5386175
NS_A1_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.4668125, upper bound: 1.5390539
NS_A1_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5154998, upper bound: 1.5324437
NS_A1_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.4774282, upper bound: 1.5325705
NS_A1_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5154998, upper bound: 1.5324438
NS_A1_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.4774282, upper bound: 1.5325706
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.4965745, upper bound: 1.5325658
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.4668125, upper bound: 1.5327449
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.4965745, upper bound: 1.5325658
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.4668125, upper bound: 1.5327449
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5803197, upper bound: 1.5468169
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5561153, upper bound: 1.5483101
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5803197, upper bound: 1.5468169
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5561153, upper bound: 1.5483101
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5752039, upper bound: 1.5462807
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5482833, upper bound: 1.5479671
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5752039, upper bound: 1.5462807
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5482833, upper bound: 1.5479671
NS_A2_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5153658, upper bound: 1.5402952
NS_A2_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.4774282, upper bound: 1.5409315
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.4965238, upper bound: 1.5399197
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.4668085, upper bound: 1.5407442
NS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5630788, upper bound: 1.4672855
NS_A2_B1_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5390539, upper bound: 1.4672855
NS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5630788, upper bound: 1.4672855
NS_A2_B1_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5390539, upper bound: 1.4672855
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5536762, upper bound: 1.4672855
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5324936, upper bound: 1.4672855
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5536762, upper bound: 1.4672855
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5324936, upper bound: 1.4672855
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5796025, upper bound: 1.5503897
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5552883, upper bound: 1.5525761
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5796025, upper bound: 1.5503897
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5552883, upper bound: 1.5525761
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5751841, upper bound: 1.5515038
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5482833, upper bound: 1.5538225
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5751841, upper bound: 1.5515038
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5482833, upper bound: 1.5538225
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5181581, upper bound: 1.5489836
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.4785258, upper bound: 1.5493680
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.4965238, upper bound: 1.4806897
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.4668085, upper bound: 1.4806897
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5153658, upper bound: 1.5402952
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.4774282, upper bound: 1.5409315
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.4965238, upper bound: 1.5399197
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.4668085, upper bound: 1.4672855
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5630788, upper bound: 1.4672855
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5390539, upper bound: 1.4672855
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5630788, upper bound: 1.4672855
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5390539, upper bound: 1.4672855
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5536762, upper bound: 1.4672855
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5324936, upper bound: 1.4672855
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5536762, upper bound: 1.4672855
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5324936, upper bound: 1.4672855
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5796825, upper bound: 1.5503897
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5553159, upper bound: 1.5525761
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5796825, upper bound: 1.5503897
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5553159, upper bound: 1.5525761
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5751841, upper bound: 1.5515038
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5482833, upper bound: 1.5538225
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5751841, upper bound: 1.5515038
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.52
Output dim: 5, lower bound: -1.5482833, upper bound: 1.5538225

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.4754875, 0.5041244, -0.4897729, 0.5190340, -0.9945216, 0.9938973
1: -0.4195700, 0.6029646, -0.4321097, 0.6187906, -1.0383606, 1.0350744
2: -0.3447527, 0.6051040, -0.3550040, 0.6257194, -0.9704720, 0.9601079
3: -0.2807004, 0.5007627, -0.2907113, 0.5107954, -0.7914958, 0.7914740
4: -0.5036144, 0.4905354, -0.5177408, 0.5034260, -1.0070404, 1.0082762
5: -0.1760631, 1.1253035, -0.2044711, 1.1373084, -1.3133714, 1.3297746
6: -0.2962036, 0.5119283, -0.3058346, 0.5231596, -0.8193631, 0.8177629
7: -0.3957849, 0.5460519, -0.4072961, 0.5635933, -0.9593782, 0.9533479
8: -0.4019130, 0.6614447, -0.4149080, 0.6760141, -1.0779271, 1.0763527
9: -0.4736983, 0.5724872, -0.4871923, 0.5877865, -1.0614848, 1.0596795

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5549696, upper bound: 1.5468169
time: 2.05 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5549696, upper bound: 1.5468169
time: 2.01 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.5881246, 0.6215871, -0.4875406, 0.5167068, -1.1048315, 1.1091278
1: -0.5223123, 0.7716802, -0.4301183, 0.6163005, -1.1386127, 1.2017984
2: -0.4286846, 0.7717447, -0.3533907, 0.6224651, -1.0511496, 1.1251354
3: -0.3630258, 0.5796276, -0.2891278, 0.5092427, -0.8722684, 0.8687555
4: -0.6192546, 0.5953373, -0.5155047, 0.5013958, -1.1206504, 1.1108420
5: -0.4290684, 1.1409847, -0.2000751, 1.1359791, -1.5650475, 1.3410598
6: -0.3739221, 0.6037532, -0.3043258, 0.5213772, -0.8952993, 0.9080790
7: -0.4897975, 0.6808434, -0.4054887, 0.5608951, -1.0506926, 1.0863321
8: -0.5081969, 0.7784118, -0.4128508, 0.6737299, -1.1819268, 1.1912626
9: -0.5839511, 0.6970068, -0.4850589, 0.5853753, -1.1693263, 1.1820657

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 8

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 139

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5469578, upper bound: 1.5445673
time: 2.08 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5522201, upper bound: 1.5445673
time: 2.29 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.4754875, 0.5041244, -0.5189962, 0.5414410, -1.0169286, 1.0231206
1: -0.4195700, 0.6029646, -0.4519723, 0.6417593, -1.0613292, 1.0549369
2: -0.3447527, 0.6051040, -0.3766927, 0.6576895, -1.0024421, 0.9817967
3: -0.2807004, 0.5007627, -0.3194801, 0.5260263, -0.8067267, 0.8202428
4: -0.5036144, 0.4905354, -0.5400504, 0.5287694, -1.0323839, 1.0305859
5: -0.1760631, 1.1253035, -0.2448135, 1.1612194, -1.3372824, 1.3701169
6: -0.2962036, 0.5119283, -0.3215775, 0.5428849, -0.8390884, 0.8335059
7: -0.3957849, 0.5460519, -0.4300930, 0.5916775, -0.9874624, 0.9761448
8: -0.4019130, 0.6614447, -0.4384841, 0.6982014, -1.1001143, 1.0999289
9: -0.4736983, 0.5724872, -0.5099261, 0.6116219, -1.0853202, 1.0824132

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 8

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5646074, upper bound: 1.5468169
time: 1.90 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5646074, upper bound: 1.5468169
time: 2.27 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.5881246, 0.6215871, -0.5155014, 0.5391255, -1.1272502, 1.1370885
1: -0.5223123, 0.7716802, -0.4496791, 0.6387928, -1.1611050, 1.2213593
2: -0.4286846, 0.7717447, -0.3741045, 0.6542978, -1.0829824, 1.1458492
3: -0.3630258, 0.5796276, -0.3157113, 0.5243894, -0.8874152, 0.8953389
4: -0.6192546, 0.5953373, -0.5373744, 0.5257761, -1.1450307, 1.1327118
5: -0.4290684, 1.1409847, -0.2401195, 1.1598092, -1.5888777, 1.3811042
6: -0.3739221, 0.6037532, -0.3196136, 0.5406715, -0.9145936, 0.9233668
7: -0.4897975, 0.6808434, -0.4272481, 0.5886873, -1.0784848, 1.1080914
8: -0.5081969, 0.7784118, -0.4355938, 0.6958022, -1.2039992, 1.2140056
9: -0.5839511, 0.6970068, -0.5071619, 0.6090349, -1.1929860, 1.2041688

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 139

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5583565, upper bound: 1.5445673
time: 2.33 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5621454, upper bound: 1.5445673
time: 2.20 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.7218752, 0.6680031, -0.4897729, 0.5190340, -1.2409092, 1.1577761
1: -0.5722384, 0.7719613, -0.4321097, 0.6187906, -1.1910290, 1.2040710
2: -0.5312012, 0.8314193, -0.3550040, 0.6257194, -1.1569207, 1.1864233
3: -0.5326589, 0.6035323, -0.2907113, 0.5107954, -1.0434542, 0.8942435
4: -0.6976064, 0.6867061, -0.5177408, 0.5034260, -1.2010324, 1.2044469
5: -0.4554137, 1.1380396, -0.2044711, 1.1373084, -1.5927221, 1.3425107
6: -0.4313159, 0.6652055, -0.3058346, 0.5231596, -0.9544755, 0.9710400
7: -0.5899240, 0.7537770, -0.4072961, 0.5635933, -1.1535172, 1.1610730
8: -0.5985237, 0.8221496, -0.4149080, 0.6760141, -1.2745378, 1.2370576
9: -0.6657983, 0.7547584, -0.4871923, 0.5877865, -1.2535849, 1.2419508

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5471853, upper bound: 1.5462807
time: 2.06 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5471853, upper bound: 1.5462807
time: 2.36 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.9368192, 0.8050085, -0.4875406, 0.5167068, -1.4535260, 1.2925491
1: -0.6975300, 0.9770324, -0.4301183, 0.6163005, -1.3138305, 1.4071507
2: -0.6971894, 1.0142831, -0.3533907, 0.6224651, -1.3196545, 1.3676738
3: -0.7573170, 0.6909338, -0.2891278, 0.5092427, -1.2665597, 0.9800616
4: -0.8647246, 0.8550386, -0.5155047, 0.5013958, -1.3661205, 1.3705434
5: -0.7412480, 1.1555876, -0.2000751, 1.1359791, -1.8772271, 1.3556627
6: -0.5503955, 0.7928089, -0.3043258, 0.5213772, -1.0717727, 1.0971347
7: -0.7630181, 0.9317169, -0.4054887, 0.5608951, -1.3239132, 1.3372056
8: -0.7672837, 0.9560648, -0.4128508, 0.6737299, -1.4410136, 1.3689157
9: -0.8313136, 0.9080188, -0.4850589, 0.5853753, -1.4166889, 1.3930777

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 8

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 139

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5390101, upper bound: 1.5443293
time: 1.93 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5444165, upper bound: 1.5443293
time: 1.94 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.7218752, 0.6680031, -0.5189962, 0.5414410, -1.2633162, 1.1869993
1: -0.5722384, 0.7719613, -0.4519723, 0.6417593, -1.2139976, 1.2239336
2: -0.5312012, 0.8314193, -0.3766927, 0.6576895, -1.1888907, 1.2081120
3: -0.5326589, 0.6035323, -0.3194801, 0.5260263, -1.0586852, 0.9230123
4: -0.6976064, 0.6867061, -0.5400504, 0.5287694, -1.2263758, 1.2267566
5: -0.4554137, 1.1380396, -0.2448135, 1.1612194, -1.6166331, 1.3828530
6: -0.4313159, 0.6652055, -0.3215775, 0.5428849, -0.9742008, 0.9867830
7: -0.5899240, 0.7537770, -0.4300930, 0.5916775, -1.1816015, 1.1838700
8: -0.5985237, 0.8221496, -0.4384841, 0.6982014, -1.2967250, 1.2606337
9: -0.6657983, 0.7547584, -0.5099261, 0.6116219, -1.2774202, 1.2646844

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5573126, upper bound: 1.5462807
time: 2.20 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5573126, upper bound: 1.5462807
time: 2.15 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.9368192, 0.8050085, -0.5155014, 0.5391255, -1.4759448, 1.3205099
1: -0.6975300, 0.9770324, -0.4496791, 0.6387928, -1.3363228, 1.4267116
2: -0.6971894, 1.0142831, -0.3741045, 0.6542978, -1.3514872, 1.3883876
3: -0.7573170, 0.6909338, -0.3157113, 0.5243894, -1.2817065, 1.0066451
4: -0.8647246, 0.8550386, -0.5373744, 0.5257761, -1.3905008, 1.3924131
5: -0.7412480, 1.1555876, -0.2401195, 1.1598092, -1.9010572, 1.3957071
6: -0.5503955, 0.7928089, -0.3196136, 0.5406715, -1.0910671, 1.1124226
7: -0.7630181, 0.9317169, -0.4272481, 0.5886873, -1.3517054, 1.3589649
8: -0.7672837, 0.9560648, -0.4355938, 0.6958022, -1.4630859, 1.3916587
9: -0.8313136, 0.9080188, -0.5071619, 0.6090349, -1.4403485, 1.4151807

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 8

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 139

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5509116, upper bound: 1.5443293
time: 2.03 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5548332, upper bound: 1.5443293
time: 1.98 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.4982275, 0.5270124, -0.3846650, 0.3941466, -0.8923741, 0.9116775
1: -0.4386029, 0.6235245, -0.3319941, 0.3536625, -0.7922654, 0.9555186
2: -0.3618047, 0.6366280, -0.2704943, 0.4554276, -0.8172323, 0.9071223
3: -0.2982162, 0.5160669, -0.2115196, 0.4238317, -0.7220479, 0.7275865
4: -0.5251009, 0.5113853, -0.4065562, 0.3909165, -0.9160174, 0.9179415
5: -0.2162966, 1.1493881, 0.1512546, 1.1255324, -1.3418289, 0.9981334
6: -0.3109518, 0.5295895, -0.2236446, 0.4320862, -0.7430379, 0.7532341
7: -0.4143371, 0.5732750, -0.3116984, 0.4164688, -0.8308059, 0.8849735
8: -0.4222885, 0.6836212, -0.3066485, 0.5550406, -0.9773291, 0.9902697
9: -0.4942709, 0.5957920, -0.3718071, 0.4616198, -0.9558907, 0.9675992

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 131

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5416283, upper bound: 1.4806897
time: 1.87 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5416283, upper bound: 1.4806897
time: 2.22 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.4982275, 0.5270124, -0.4976699, 0.5514714, -1.0496988, 1.0246823
1: -0.4386029, 0.6235245, -0.4630894, 0.4839199, -0.9225228, 1.0866139
2: -0.3618047, 0.6366280, -0.3678207, 0.6678746, -1.0296793, 1.0044488
3: -0.2982162, 0.5160669, -0.3079270, 0.5140205, -0.8122367, 0.8239939
4: -0.5251009, 0.5113853, -0.5304418, 0.5295544, -1.0546553, 1.0418270
5: -0.2162966, 1.1493881, -0.1186955, 1.1348166, -1.3511132, 1.2680836
6: -0.3109518, 0.5295895, -0.3196089, 0.5404234, -0.8513752, 0.8491983
7: -0.4143371, 0.5732750, -0.4204883, 0.5938017, -1.0081389, 0.9937633
8: -0.4222885, 0.6836212, -0.4448276, 0.6881066, -1.1103951, 1.1284487
9: -0.4942709, 0.5957920, -0.5130015, 0.6167052, -1.1109761, 1.1087935

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 131

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5386175, upper bound: 1.4672855
time: 2.36 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5386175, upper bound: 1.4672855
time: 2.03 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.7571833, 0.6773893, -0.3846650, 0.3941466, -1.1513299, 1.0620544
1: -0.5920590, 0.7907068, -0.3319941, 0.3536625, -0.9457215, 1.1227009
2: -0.5554752, 0.8424433, -0.2704943, 0.4554276, -1.0109029, 1.1129376
3: -0.5688046, 0.6157334, -0.2115196, 0.4238317, -0.9926363, 0.8272530
4: -0.7295642, 0.7147332, -0.4065562, 0.3909165, -1.1204808, 1.1212895
5: -0.4977410, 1.1583612, 0.1512546, 1.1255324, -1.6232734, 1.0071065
6: -0.4703338, 0.6854675, -0.2236446, 0.4320862, -0.9024200, 0.9091121
7: -0.5957260, 0.7884539, -0.3116984, 0.4164688, -1.0121948, 1.1001523
8: -0.6401069, 0.8284048, -0.3066485, 0.5550406, -1.1951475, 1.1350533
9: -0.6849521, 0.7796617, -0.3718071, 0.4616198, -1.1465719, 1.1514689

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 131

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5324308, upper bound: 1.4806897
time: 2.46 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5324308, upper bound: 1.4806897
time: 2.25 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.88 + 595.76 = 600.64 seconds
