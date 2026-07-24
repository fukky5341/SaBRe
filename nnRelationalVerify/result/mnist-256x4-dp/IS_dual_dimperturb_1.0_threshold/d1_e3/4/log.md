## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.01246608


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986)
1: (-0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912)
2: (0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394)
3: (-0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711)
4: (-0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0039527, 0.0039527)
5: (-0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973)
6: (-0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037)
7: (-0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0221395, 0.0221395)
8: (0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723)
9: (-0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0153951, 0.0153951)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.17 + 2.43 = 3.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0138512, upper bound: 0.0138512

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0136219, upper bound: 0.0136279
time: 1.11 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0136279, upper bound: 0.0136279
time: 1.13 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.36 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.36
Output dim: 8, lower bound: -0.0136219, upper bound: 0.0136279
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.36
Output dim: 8, lower bound: -0.0136279, upper bound: 0.0136279

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0079012, 0.0189335, 0.0079677, 0.0189473, -0.0110461, 0.0109658
1: -0.0040253, 0.0012966, -0.0040341, 0.0012991, -0.0053244, 0.0053307
2: 0.0033502, 0.0093588, 0.0033434, 0.0093010, -0.0059507, 0.0060155
3: -0.0010786, 0.0037767, -0.0010101, 0.0038656, -0.0049443, 0.0047867
4: -0.0052851, -0.0011380, -0.0053741, -0.0011833, -0.0037906, 0.0039045
5: -0.0006697, 0.0040982, -0.0006663, 0.0040790, -0.0047487, 0.0047644
6: -0.0066449, 0.0012392, -0.0065820, 0.0014600, -0.0081049, 0.0078213
7: -0.0263639, -0.0024751, -0.0268626, -0.0027325, -0.0212673, 0.0218811
8: 0.9714150, 0.9943640, 0.9710557, 0.9941553, -0.0227404, 0.0233083
9: -0.0062903, 0.0096045, -0.0061227, 0.0099168, -0.0152386, 0.0148294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130278, upper bound: 0.0133153
time: 1.58 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0133603, upper bound: 0.0133753
time: 1.16 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0080345, 0.0189424, 0.0079021, 0.0189513, -0.0109168, 0.0110404
1: -0.0040154, 0.0012792, -0.0040521, 0.0013140, -0.0053294, 0.0053312
2: 0.0033464, 0.0092541, 0.0033415, 0.0093469, -0.0060005, 0.0059126
3: -0.0009840, 0.0038242, -0.0010341, 0.0038943, -0.0048783, 0.0048582
4: -0.0053316, -0.0011887, -0.0053994, -0.0011785, -0.0038160, 0.0039138
5: -0.0006660, 0.0040485, -0.0006672, 0.0041087, -0.0047748, 0.0047156
6: -0.0065789, 0.0013530, -0.0065838, 0.0015480, -0.0081268, 0.0079368
7: -0.0266236, -0.0027659, -0.0270036, -0.0027016, -0.0213953, 0.0219349
8: 0.9712325, 0.9941128, 0.9709520, 0.9941968, -0.0229643, 0.0231608
9: -0.0060991, 0.0097657, -0.0061449, 0.0100060, -0.0152448, 0.0149278

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130232, upper bound: 0.0133153
time: 1.48 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0133753, upper bound: 0.0133753
time: 1.19 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.89 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 3.89
Output dim: 8, lower bound: -0.0130278, upper bound: 0.0133153
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 3.89
Output dim: 8, lower bound: -0.0133603, upper bound: 0.0133753
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 3.89
Output dim: 8, lower bound: -0.0130232, upper bound: 0.0133153
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 3.89
Output dim: 8, lower bound: -0.0133753, upper bound: 0.0133753

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: 0.0077745, 0.0186029, 0.0080866, 0.0188346, -0.0110602, 0.0105163
1: -0.0040496, 0.0011403, -0.0040002, 0.0012243, -0.0052739, 0.0051406
2: 0.0034611, 0.0094540, 0.0033810, 0.0092178, -0.0057567, 0.0060730
3: -0.0011335, 0.0037083, -0.0009570, 0.0038304, -0.0049640, 0.0046653
4: -0.0050138, -0.0010954, -0.0052831, -0.0011938, -0.0034818, 0.0038171
5: -0.0005596, 0.0041504, -0.0006277, 0.0040249, -0.0045844, 0.0047782
6: -0.0066991, 0.0009686, -0.0065777, 0.0013253, -0.0080244, 0.0075463
7: -0.0247642, -0.0022374, -0.0263239, -0.0027963, -0.0195449, 0.0214514
8: 0.9728870, 0.9945763, 0.9715581, 0.9940782, -0.0211912, 0.0230182
9: -0.0064407, 0.0085552, -0.0060780, 0.0095611, -0.0149201, 0.0136615

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_A1_A1

### Relational analysis result of IS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127060, upper bound: 0.0130872
time: 1.12 seconds

## Relational analysis of IS_A1_A1_A2

### Relational analysis result of IS_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127681, upper bound: 0.0130872
time: 1.17 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: 0.0079843, 0.0188288, 0.0079731, 0.0189401, -0.0109558, 0.0108558
1: -0.0040020, 0.0012391, -0.0040326, 0.0012953, -0.0052973, 0.0052717
2: 0.0033841, 0.0093012, 0.0033457, 0.0092972, -0.0059131, 0.0059555
3: -0.0010403, 0.0037526, -0.0010075, 0.0038640, -0.0049043, 0.0047602
4: -0.0052220, -0.0011460, -0.0053697, -0.0011838, -0.0036244, 0.0038917
5: -0.0006211, 0.0040606, -0.0006630, 0.0040766, -0.0046977, 0.0047236
6: -0.0066404, 0.0011438, -0.0065818, 0.0014534, -0.0080939, 0.0077255
7: -0.0260004, -0.0025233, -0.0268375, -0.0027355, -0.0202683, 0.0218149
8: 0.9717661, 0.9943070, 0.9710802, 0.9941519, -0.0223858, 0.0232267
9: -0.0062574, 0.0093624, -0.0061206, 0.0099002, -0.0151865, 0.0142407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_A2_A1

### Relational analysis result of IS_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130572, upper bound: 0.0131682
time: 1.50 seconds

## Relational analysis of IS_A1_A2_A2

### Relational analysis result of IS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0131515, upper bound: 0.0131682
time: 1.18 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: 0.0079412, 0.0186103, 0.0080189, 0.0188386, -0.0108975, 0.0105913
1: -0.0040347, 0.0011173, -0.0040187, 0.0012391, -0.0052738, 0.0051361
2: 0.0034577, 0.0093243, 0.0033792, 0.0092652, -0.0058075, 0.0059451
3: -0.0010170, 0.0037545, -0.0009817, 0.0038591, -0.0048760, 0.0047362
4: -0.0050534, -0.0011594, -0.0053077, -0.0011888, -0.0035017, 0.0038092
5: -0.0005547, 0.0040873, -0.0006286, 0.0040552, -0.0046099, 0.0047159
6: -0.0066225, 0.0010674, -0.0065794, 0.0014131, -0.0080356, 0.0076469
7: -0.0249830, -0.0025955, -0.0264595, -0.0027643, -0.0196334, 0.0213898
8: 0.9727404, 0.9942614, 0.9714587, 0.9941213, -0.0213809, 0.0228027
9: -0.0062097, 0.0086907, -0.0061012, 0.0096468, -0.0148625, 0.0137400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_A1_A1

### Relational analysis result of IS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0130872
time: 1.12 seconds

## Relational analysis of IS_A2_A1_A2

### Relational analysis result of IS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0130872
time: 1.18 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: 0.0081132, 0.0188382, 0.0079073, 0.0189441, -0.0108309, 0.0109309
1: -0.0039931, 0.0012240, -0.0040506, 0.0013101, -0.0053032, 0.0052746
2: 0.0033799, 0.0091990, 0.0033438, 0.0093433, -0.0059634, 0.0058552
3: -0.0009472, 0.0038002, -0.0010316, 0.0038927, -0.0048399, 0.0048318
4: -0.0052701, -0.0011960, -0.0053951, -0.0011790, -0.0036524, 0.0039017
5: -0.0006178, 0.0040126, -0.0006639, 0.0041064, -0.0047241, 0.0046765
6: -0.0065746, 0.0012590, -0.0065835, 0.0015414, -0.0081160, 0.0078425
7: -0.0262704, -0.0028103, -0.0269787, -0.0027046, -0.0204007, 0.0218737
8: 0.9715734, 0.9940606, 0.9709761, 0.9941935, -0.0226201, 0.0230846
9: -0.0060684, 0.0095303, -0.0061429, 0.0099894, -0.0151958, 0.0143528

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_A2_A1

### Relational analysis result of IS_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130558, upper bound: 0.0131682
time: 1.57 seconds

## Relational analysis of IS_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0131682, upper bound: 0.0131682
time: 1.18 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.93 seconds
IS_A1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 3.93
Output dim: 8, lower bound: -0.0127060, upper bound: 0.0130872
IS_A1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 3.93
Output dim: 8, lower bound: -0.0127681, upper bound: 0.0130872
IS_A1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 3.93
Output dim: 8, lower bound: -0.0130572, upper bound: 0.0131682
IS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.93
Output dim: 8, lower bound: -0.0131515, upper bound: 0.0131682
IS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 3.93
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0130872
IS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 3.93
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0130872
IS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 3.93
Output dim: 8, lower bound: -0.0130558, upper bound: 0.0131682
IS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.93
Output dim: 8, lower bound: -0.0131682, upper bound: 0.0131682

## BFS IS instance: IS_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0080929, 0.0185533, 0.0081682, 0.0188214, -0.0107284, 0.0103852
1: -0.0039673, 0.0010577, -0.0039785, 0.0012032, -0.0051705, 0.0050362
2: 0.0034788, 0.0092297, 0.0033858, 0.0091601, -0.0056812, 0.0058439
3: -0.0010035, 0.0036382, -0.0009241, 0.0038074, -0.0048109, 0.0045623
4: -0.0049200, -0.0011302, -0.0052573, -0.0012024, -0.0033730, 0.0035357
5: -0.0005518, 0.0040088, -0.0006255, 0.0039884, -0.0045402, 0.0046344
6: -0.0066797, 0.0007150, -0.0065725, 0.0012516, -0.0079313, 0.0072875
7: -0.0242347, -0.0024411, -0.0261798, -0.0028479, -0.0188910, 0.0186574
8: 0.9733176, 0.9943494, 0.9716747, 0.9940194, -0.0207018, 0.0226746
9: -0.0063017, 0.0082136, -0.0060428, 0.0094684, -0.0138631, 0.0132641

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_A1_B1

### Relational analysis result of IS_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127060, upper bound: 0.0127847
time: 1.10 seconds

## Relational analysis of IS_A1_A1_A1_B2

### Relational analysis result of IS_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127060, upper bound: 0.0130872
time: 1.06 seconds

## BFS IS instance: IS_A1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0068087, 0.0185524, 0.0081954, 0.0188163, -0.0120076, 0.0103570
1: -0.0042306, 0.0012934, -0.0039723, 0.0011965, -0.0054272, 0.0052657
2: 0.0034796, 0.0101545, 0.0033876, 0.0091407, -0.0056611, 0.0067669
3: -0.0015647, 0.0036803, -0.0009138, 0.0038034, -0.0053680, 0.0045941
4: -0.0049252, -0.0009405, -0.0052505, -0.0012050, -0.0033866, 0.0037824
5: -0.0005544, 0.0045500, -0.0006248, 0.0039766, -0.0045310, 0.0051749
6: -0.0068399, 0.0010449, -0.0065712, 0.0012317, -0.0080716, 0.0076161
7: -0.0242427, -0.0013184, -0.0261425, -0.0028644, -0.0189568, 0.0201030
8: 0.9732838, 0.9955425, 0.9717060, 0.9939986, -0.0207149, 0.0238365
9: -0.0070691, 0.0082311, -0.0060310, 0.0094441, -0.0148047, 0.0133063

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_A2_B1

### Relational analysis result of IS_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127681, upper bound: 0.0127847
time: 1.55 seconds

## Relational analysis of IS_A1_A1_A2_B2

### Relational analysis result of IS_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127681, upper bound: 0.0130872
time: 1.53 seconds

## BFS IS instance: IS_A1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0083055, 0.0187768, 0.0080534, 0.0189269, -0.0106214, 0.0107234
1: -0.0039174, 0.0011571, -0.0040112, 0.0012740, -0.0051914, 0.0051683
2: 0.0034030, 0.0090757, 0.0033506, 0.0092404, -0.0058373, 0.0057251
3: -0.0009109, 0.0036682, -0.0009750, 0.0038408, -0.0047517, 0.0046432
4: -0.0051203, -0.0011813, -0.0053435, -0.0011921, -0.0035133, 0.0038255
5: -0.0006129, 0.0039166, -0.0006608, 0.0040404, -0.0046533, 0.0045775
6: -0.0066209, 0.0008647, -0.0065767, 0.0013794, -0.0080003, 0.0074413
7: -0.0254298, -0.0027351, -0.0266913, -0.0027856, -0.0195936, 0.0214284
8: 0.9722265, 0.9940638, 0.9711984, 0.9940944, -0.0218679, 0.0228654
9: -0.0061136, 0.0089945, -0.0060863, 0.0098060, -0.0149279, 0.0138342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_A1_B1

### Relational analysis result of IS_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0127670
time: 1.18 seconds

## Relational analysis of IS_A1_A2_A1_B2

### Relational analysis result of IS_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0129170
time: 1.66 seconds

## BFS IS instance: IS_A1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0069434, 0.0187820, 0.0080811, 0.0189220, -0.0119786, 0.0107009
1: -0.0041974, 0.0013851, -0.0040048, 0.0012679, -0.0054652, 0.0053900
2: 0.0034013, 0.0100584, 0.0033522, 0.0092207, -0.0058194, 0.0067062
3: -0.0015091, 0.0037173, -0.0009647, 0.0038364, -0.0053455, 0.0046820
4: -0.0051387, -0.0009777, -0.0053381, -0.0011948, -0.0035354, 0.0040753
5: -0.0006160, 0.0044893, -0.0006601, 0.0040284, -0.0046444, 0.0051494
6: -0.0067939, 0.0012106, -0.0065753, 0.0013601, -0.0081540, 0.0077860
7: -0.0255066, -0.0015269, -0.0266585, -0.0028023, -0.0197103, 0.0228726
8: 0.9721356, 0.9953537, 0.9712247, 0.9940732, -0.0219376, 0.0241290
9: -0.0069351, 0.0090564, -0.0060743, 0.0097845, -0.0158923, 0.0139119

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_A2_B1

### Relational analysis result of IS_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130682, upper bound: 0.0127670
time: 1.10 seconds

## Relational analysis of IS_A1_A2_A2_B2

### Relational analysis result of IS_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130682, upper bound: 0.0129170
time: 1.70 seconds

## BFS IS instance: IS_A2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0082565, 0.0185611, 0.0080999, 0.0188254, -0.0105689, 0.0104613
1: -0.0039485, 0.0010368, -0.0039972, 0.0012179, -0.0051664, 0.0050340
2: 0.0034755, 0.0091027, 0.0033840, 0.0092079, -0.0057324, 0.0057187
3: -0.0008895, 0.0036775, -0.0009490, 0.0038355, -0.0047251, 0.0046264
4: -0.0049572, -0.0011940, -0.0052818, -0.0011972, -0.0033922, 0.0035409
5: -0.0005470, 0.0039456, -0.0006264, 0.0040190, -0.0045660, 0.0045721
6: -0.0066028, 0.0008008, -0.0065744, 0.0013392, -0.0079420, 0.0073752
7: -0.0244393, -0.0027993, -0.0263133, -0.0028154, -0.0189645, 0.0187495
8: 0.9731785, 0.9940299, 0.9715770, 0.9940630, -0.0208845, 0.0224530
9: -0.0060726, 0.0083402, -0.0060663, 0.0095527, -0.0138559, 0.0133394

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_A1_B1

### Relational analysis result of IS_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0130642
time: 1.55 seconds

## Relational analysis of IS_A2_A1_A1_B2

### Relational analysis result of IS_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0130642
time: 1.56 seconds

## BFS IS instance: IS_A2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0069627, 0.0185597, 0.0081268, 0.0188202, -0.0118574, 0.0104329
1: -0.0042152, 0.0012591, -0.0039911, 0.0012113, -0.0054264, 0.0052502
2: 0.0034765, 0.0100375, 0.0033858, 0.0091888, -0.0057124, 0.0066517
3: -0.0014544, 0.0037225, -0.0009388, 0.0038316, -0.0052860, 0.0046613
4: -0.0049666, -0.0009945, -0.0052755, -0.0011999, -0.0034077, 0.0039792
5: -0.0005493, 0.0044909, -0.0006257, 0.0040074, -0.0045566, 0.0051166
6: -0.0067734, 0.0011315, -0.0065730, 0.0013191, -0.0080926, 0.0077045
7: -0.0244683, -0.0016217, -0.0262805, -0.0028320, -0.0190576, 0.0223614
8: 0.9731277, 0.9952570, 0.9716047, 0.9940425, -0.0209147, 0.0236523
9: -0.0068703, 0.0083713, -0.0060546, 0.0095312, -0.0155224, 0.0133890

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_A2_B1

### Relational analysis result of IS_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0130642
time: 1.38 seconds

## Relational analysis of IS_A2_A1_A2_B2

### Relational analysis result of IS_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0130642
time: 1.78 seconds

## BFS IS instance: IS_A2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0084286, 0.0187860, 0.0079869, 0.0189309, -0.0105023, 0.0107991
1: -0.0039089, 0.0011429, -0.0040293, 0.0012887, -0.0051977, 0.0051722
2: 0.0033987, 0.0089764, 0.0033487, 0.0092869, -0.0058882, 0.0056277
3: -0.0008197, 0.0037132, -0.0009994, 0.0038692, -0.0046888, 0.0047126
4: -0.0051675, -0.0012301, -0.0053690, -0.0011872, -0.0035409, 0.0038367
5: -0.0006096, 0.0038723, -0.0006617, 0.0040704, -0.0046800, 0.0045340
6: -0.0065548, 0.0009793, -0.0065785, 0.0014675, -0.0080223, 0.0075578
7: -0.0256950, -0.0030151, -0.0268321, -0.0027544, -0.0197220, 0.0214942
8: 0.9720360, 0.9938253, 0.9710945, 0.9941365, -0.0221005, 0.0227308
9: -0.0059282, 0.0091599, -0.0061088, 0.0098951, -0.0149424, 0.0139458

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_A1_B1

### Relational analysis result of IS_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0127670
time: 1.31 seconds

## Relational analysis of IS_A2_A2_A1_B2

### Relational analysis result of IS_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0129170
time: 1.53 seconds

## BFS IS instance: IS_A2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0070348, 0.0187912, 0.0080144, 0.0189261, -0.0118913, 0.0107768
1: -0.0041962, 0.0013731, -0.0040231, 0.0012822, -0.0054785, 0.0053962
2: 0.0033970, 0.0099868, 0.0033504, 0.0092675, -0.0058705, 0.0066365
3: -0.0014337, 0.0037640, -0.0009890, 0.0038649, -0.0052986, 0.0047530
4: -0.0051883, -0.0010215, -0.0053632, -0.0011899, -0.0035639, 0.0040902
5: -0.0006126, 0.0044600, -0.0006610, 0.0040585, -0.0046711, 0.0051210
6: -0.0067363, 0.0013294, -0.0065771, 0.0014476, -0.0081839, 0.0079065
7: -0.0257798, -0.0017740, -0.0267990, -0.0027711, -0.0198500, 0.0229459
8: 0.9719387, 0.9951462, 0.9711208, 0.9941155, -0.0221768, 0.0240254
9: -0.0067789, 0.0092272, -0.0060969, 0.0098735, -0.0159291, 0.0140245

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_A2_B1

### Relational analysis result of IS_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130872, upper bound: 0.0127670
time: 1.53 seconds

## Relational analysis of IS_A2_A2_A2_B2

### Relational analysis result of IS_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130872, upper bound: 0.0129170
time: 1.98 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.77 seconds
IS_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 8, lower bound: -0.0127060, upper bound: 0.0127847
IS_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 8, lower bound: -0.0127060, upper bound: 0.0130872
IS_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 8, lower bound: -0.0127681, upper bound: 0.0127847
IS_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 8, lower bound: -0.0127681, upper bound: 0.0130872
IS_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0127670
IS_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0129170
IS_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 8, lower bound: -0.0130682, upper bound: 0.0127670
IS_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 8, lower bound: -0.0130682, upper bound: 0.0129170
IS_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0130642
IS_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0130642
IS_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0130642
IS_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0130642
IS_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0127670
IS_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0129170
IS_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 8, lower bound: -0.0130872, upper bound: 0.0127670
IS_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 8, lower bound: -0.0130872, upper bound: 0.0129170

## BFS IS instance: IS_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0080929, 0.0185533, 0.0079600, 0.0186028, -0.0105098, 0.0105934
1: -0.0039673, 0.0010577, -0.0040308, 0.0011171, -0.0050844, 0.0050885
2: 0.0034788, 0.0092297, 0.0034592, 0.0093104, -0.0058316, 0.0057705
3: -0.0010035, 0.0036382, -0.0010086, 0.0037730, -0.0047765, 0.0046468
4: -0.0049200, -0.0011302, -0.0050746, -0.0011633, -0.0033845, 0.0033319
5: -0.0005518, 0.0040088, -0.0005529, 0.0040793, -0.0046311, 0.0045617
6: -0.0066797, 0.0007150, -0.0066207, 0.0011004, -0.0077801, 0.0073356
7: -0.0242347, -0.0024411, -0.0251014, -0.0026186, -0.0189883, 0.0174783
8: 0.9733176, 0.9943494, 0.9726524, 0.9942375, -0.0209199, 0.0216970
9: -0.0063017, 0.0082136, -0.0061945, 0.0087639, -0.0131001, 0.0133234

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127060, upper bound: 0.0127106
time: 1.14 seconds

## Relational analysis of IS_A1_A1_A1_B1_B2

### Relational analysis result of IS_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127060, upper bound: 0.0127847
time: 1.15 seconds

## BFS IS instance: IS_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0080929, 0.0185533, 0.0081282, 0.0188291, -0.0107361, 0.0104251
1: -0.0039673, 0.0010577, -0.0039900, 0.0012221, -0.0051894, 0.0050477
2: 0.0034788, 0.0092297, 0.0033821, 0.0091880, -0.0057091, 0.0058476
3: -0.0010035, 0.0036382, -0.0009401, 0.0038190, -0.0048225, 0.0045784
4: -0.0049200, -0.0011302, -0.0052846, -0.0011992, -0.0033765, 0.0035971
5: -0.0005518, 0.0040088, -0.0006158, 0.0040064, -0.0045582, 0.0046247
6: -0.0066797, 0.0007150, -0.0065726, 0.0012904, -0.0079701, 0.0072876
7: -0.0242347, -0.0024411, -0.0263490, -0.0028284, -0.0189072, 0.0190791
8: 0.9733176, 0.9943494, 0.9715251, 0.9940445, -0.0207269, 0.0228243
9: -0.0063017, 0.0082136, -0.0060567, 0.0095783, -0.0140799, 0.0132789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127060, upper bound: 0.0130010
time: 1.17 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127060, upper bound: 0.0130872
time: 1.54 seconds

## BFS IS instance: IS_A1_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0068087, 0.0185524, 0.0079797, 0.0185971, -0.0117883, 0.0105727
1: -0.0042306, 0.0012934, -0.0040262, 0.0011100, -0.0053407, 0.0053196
2: 0.0034796, 0.0101545, 0.0034613, 0.0092965, -0.0058169, 0.0066932
3: -0.0015647, 0.0036803, -0.0010008, 0.0037698, -0.0053345, 0.0046811
4: -0.0049252, -0.0009405, -0.0050653, -0.0011652, -0.0033991, 0.0035762
5: -0.0005544, 0.0045500, -0.0005520, 0.0040708, -0.0046251, 0.0051021
6: -0.0068399, 0.0010449, -0.0066193, 0.0010823, -0.0079222, 0.0076642
7: -0.0242427, -0.0013184, -0.0250518, -0.0026300, -0.0190621, 0.0189108
8: 0.9732838, 0.9955425, 0.9726932, 0.9942229, -0.0209391, 0.0228493
9: -0.0070691, 0.0082311, -0.0061862, 0.0087317, -0.0140347, 0.0133702

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_A2_B1_B1

### Relational analysis result of IS_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127681, upper bound: 0.0127709
time: 1.63 seconds

## Relational analysis of IS_A1_A1_A2_B1_B2

### Relational analysis result of IS_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127681, upper bound: 0.0127847
time: 1.56 seconds

## BFS IS instance: IS_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0068087, 0.0185524, 0.0081549, 0.0188253, -0.0120166, 0.0103975
1: -0.0042306, 0.0012934, -0.0039840, 0.0012167, -0.0054473, 0.0052774
2: 0.0034796, 0.0101545, 0.0033836, 0.0091690, -0.0056894, 0.0067709
3: -0.0015647, 0.0036803, -0.0009303, 0.0038150, -0.0053796, 0.0046106
4: -0.0049252, -0.0009405, -0.0052794, -0.0012017, -0.0033902, 0.0038454
5: -0.0005544, 0.0045500, -0.0006151, 0.0039949, -0.0045492, 0.0051652
6: -0.0068399, 0.0010449, -0.0065713, 0.0012717, -0.0081116, 0.0076163
7: -0.0242427, -0.0013184, -0.0263201, -0.0028443, -0.0189734, 0.0205246
8: 0.9732838, 0.9955425, 0.9715488, 0.9940241, -0.0207403, 0.0239937
9: -0.0070691, 0.0082311, -0.0060453, 0.0095597, -0.0150332, 0.0133214

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_A2_B2_B1

### Relational analysis result of IS_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127681, upper bound: 0.0130642
time: 1.18 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2

### Relational analysis result of IS_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127681, upper bound: 0.0130872
time: 1.56 seconds

## BFS IS instance: IS_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0083055, 0.0187768, 0.0079600, 0.0186028, -0.0102973, 0.0108168
1: -0.0039174, 0.0011571, -0.0040308, 0.0011171, -0.0050345, 0.0051879
2: 0.0034030, 0.0090757, 0.0034592, 0.0093104, -0.0059074, 0.0056164
3: -0.0009109, 0.0036682, -0.0010086, 0.0037730, -0.0046839, 0.0046767
4: -0.0051203, -0.0011813, -0.0050746, -0.0011633, -0.0036224, 0.0035262
5: -0.0006129, 0.0039166, -0.0005529, 0.0040793, -0.0046922, 0.0044695
6: -0.0066209, 0.0008647, -0.0066207, 0.0011004, -0.0077213, 0.0074854
7: -0.0254298, -0.0027351, -0.0251014, -0.0026186, -0.0203211, 0.0197478
8: 0.9722265, 0.9940638, 0.9726524, 0.9942375, -0.0220110, 0.0214114
9: -0.0061136, 0.0089945, -0.0061945, 0.0087639, -0.0138049, 0.0142101

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_A2_A1_B1_B1

### Relational analysis result of IS_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0127012
time: 1.23 seconds

## Relational analysis of IS_A1_A2_A1_B1_B2

### Relational analysis result of IS_A1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0127670
time: 1.27 seconds

## BFS IS instance: IS_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0083055, 0.0187768, 0.0081282, 0.0188291, -0.0105236, 0.0106486
1: -0.0039174, 0.0011571, -0.0039900, 0.0012221, -0.0051396, 0.0051471
2: 0.0034030, 0.0090757, 0.0033821, 0.0091880, -0.0057849, 0.0056935
3: -0.0009109, 0.0036682, -0.0009401, 0.0038190, -0.0047299, 0.0046083
4: -0.0051203, -0.0011813, -0.0052846, -0.0011992, -0.0035054, 0.0036693
5: -0.0006129, 0.0039166, -0.0006158, 0.0040064, -0.0046193, 0.0045325
6: -0.0066209, 0.0008647, -0.0065726, 0.0012904, -0.0079113, 0.0074373
7: -0.0254298, -0.0027351, -0.0263490, -0.0028284, -0.0195533, 0.0204826
8: 0.9722265, 0.9940638, 0.9715251, 0.9940445, -0.0218180, 0.0225387
9: -0.0061136, 0.0089945, -0.0060567, 0.0095783, -0.0143778, 0.0138015

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_A2_A1_B2_B1

### Relational analysis result of IS_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0128676
time: 1.78 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2

### Relational analysis result of IS_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0129170
time: 1.76 seconds

## BFS IS instance: IS_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0069434, 0.0187820, 0.0079797, 0.0185971, -0.0116537, 0.0108023
1: -0.0041974, 0.0013851, -0.0040262, 0.0011100, -0.0053074, 0.0054113
2: 0.0034013, 0.0100584, 0.0034613, 0.0092965, -0.0058952, 0.0065971
3: -0.0015091, 0.0037173, -0.0010008, 0.0037698, -0.0052789, 0.0047181
4: -0.0051387, -0.0009777, -0.0050653, -0.0011652, -0.0036480, 0.0037736
5: -0.0006160, 0.0044893, -0.0005520, 0.0040708, -0.0046868, 0.0050414
6: -0.0067939, 0.0012106, -0.0066193, 0.0010823, -0.0078762, 0.0078299
7: -0.0255066, -0.0015269, -0.0250518, -0.0026300, -0.0204555, 0.0211836
8: 0.9721356, 0.9953537, 0.9726932, 0.9942229, -0.0220873, 0.0226605
9: -0.0069351, 0.0090564, -0.0061862, 0.0087317, -0.0147603, 0.0142973

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_A2_B1_B1

### Relational analysis result of IS_A1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130682, upper bound: 0.0127573
time: 1.45 seconds

## Relational analysis of IS_A1_A2_A2_B1_B2

### Relational analysis result of IS_A1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130682, upper bound: 0.0127670
time: 1.79 seconds

## BFS IS instance: IS_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0069434, 0.0187820, 0.0081549, 0.0188253, -0.0118819, 0.0106271
1: -0.0041974, 0.0013851, -0.0039840, 0.0012167, -0.0054140, 0.0053691
2: 0.0034013, 0.0100584, 0.0033836, 0.0091690, -0.0057677, 0.0066748
3: -0.0015091, 0.0037173, -0.0009303, 0.0038150, -0.0053241, 0.0046476
4: -0.0051387, -0.0009777, -0.0052794, -0.0012017, -0.0035276, 0.0039182
5: -0.0006160, 0.0044893, -0.0006151, 0.0039949, -0.0046109, 0.0051045
6: -0.0067939, 0.0012106, -0.0065713, 0.0012717, -0.0080656, 0.0077820
7: -0.0255066, -0.0015269, -0.0263201, -0.0028443, -0.0196707, 0.0219203
8: 0.9721356, 0.9953537, 0.9715488, 0.9940241, -0.0218885, 0.0238049
9: -0.0069351, 0.0090564, -0.0060453, 0.0095597, -0.0153407, 0.0138796

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_A2_B2_B1

### Relational analysis result of IS_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130682, upper bound: 0.0129087
time: 1.58 seconds

## Relational analysis of IS_A1_A2_A2_B2_B2

### Relational analysis result of IS_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130682, upper bound: 0.0129170
time: 1.54 seconds

## BFS IS instance: IS_A2_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0082565, 0.0185611, 0.0081087, 0.0188077, -0.0105512, 0.0104525
1: -0.0039485, 0.0010368, -0.0039686, 0.0011990, -0.0051476, 0.0050054
2: 0.0034755, 0.0091027, 0.0033926, 0.0092140, -0.0057385, 0.0057101
3: -0.0008895, 0.0036775, -0.0009904, 0.0037184, -0.0046079, 0.0046679
4: -0.0049572, -0.0011940, -0.0051689, -0.0011581, -0.0034476, 0.0034076
5: -0.0005470, 0.0039456, -0.0006295, 0.0040046, -0.0045515, 0.0045752
6: -0.0066028, 0.0008008, -0.0066353, 0.0010330, -0.0076357, 0.0074361
7: -0.0244393, -0.0027993, -0.0256880, -0.0025970, -0.0192573, 0.0179922
8: 0.9731785, 0.9940299, 0.9720328, 0.9942193, -0.0210408, 0.0219972
9: -0.0060726, 0.0083402, -0.0062072, 0.0091601, -0.0134015, 0.0135208

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A1_A1_B1_B1

### Relational analysis result of IS_A2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0127709
time: 1.19 seconds

## Relational analysis of IS_A2_A1_A1_B1_B2

### Relational analysis result of IS_A2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0130642
time: 1.24 seconds

## BFS IS instance: IS_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0082565, 0.0185611, 0.0082367, 0.0188166, -0.0105602, 0.0103244
1: -0.0039485, 0.0010368, -0.0039593, 0.0011832, -0.0051317, 0.0049962
2: 0.0034755, 0.0091027, 0.0033888, 0.0091120, -0.0056366, 0.0057139
3: -0.0008895, 0.0036775, -0.0008974, 0.0037650, -0.0046545, 0.0045749
4: -0.0049572, -0.0011940, -0.0052127, -0.0012079, -0.0033786, 0.0034176
5: -0.0005470, 0.0039456, -0.0006253, 0.0039573, -0.0045043, 0.0045709
6: -0.0066028, 0.0008008, -0.0065693, 0.0011448, -0.0077475, 0.0073701
7: -0.0244393, -0.0027993, -0.0259294, -0.0028823, -0.0188976, 0.0179640
8: 0.9731785, 0.9940299, 0.9718627, 0.9939756, -0.0207971, 0.0221673
9: -0.0060726, 0.0083402, -0.0060186, 0.0093100, -0.0134412, 0.0132826

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A1_A1_B2_B1

### Relational analysis result of IS_A2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0127710
time: 1.13 seconds

## Relational analysis of IS_A2_A1_A1_B2_B2

### Relational analysis result of IS_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0130642
time: 1.12 seconds

## BFS IS instance: IS_A2_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0069627, 0.0185597, 0.0081367, 0.0188027, -0.0118400, 0.0104230
1: -0.0042152, 0.0012591, -0.0039622, 0.0011922, -0.0054074, 0.0052212
2: 0.0034765, 0.0100375, 0.0033944, 0.0091942, -0.0057177, 0.0066431
3: -0.0014544, 0.0037225, -0.0009798, 0.0037148, -0.0051692, 0.0047023
4: -0.0049666, -0.0009945, -0.0051618, -0.0011608, -0.0034631, 0.0038527
5: -0.0005493, 0.0044909, -0.0006288, 0.0039926, -0.0045418, 0.0051198
6: -0.0067734, 0.0011315, -0.0066340, 0.0010130, -0.0077864, 0.0077655
7: -0.0244683, -0.0016217, -0.0256506, -0.0026133, -0.0193544, 0.0216840
8: 0.9731277, 0.9952570, 0.9720643, 0.9941990, -0.0210713, 0.0231928
9: -0.0068703, 0.0083713, -0.0061956, 0.0091358, -0.0150920, 0.0135728

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A1_A2_B1_B1

### Relational analysis result of IS_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0127709
time: 1.15 seconds

## Relational analysis of IS_A2_A1_A2_B1_B2

### Relational analysis result of IS_A2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0130642
time: 1.13 seconds

## BFS IS instance: IS_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0069627, 0.0185597, 0.0082633, 0.0188113, -0.0118486, 0.0102964
1: -0.0042152, 0.0012591, -0.0039533, 0.0011765, -0.0053916, 0.0052124
2: 0.0034765, 0.0100375, 0.0033906, 0.0090932, -0.0056167, 0.0066469
3: -0.0014544, 0.0037225, -0.0008873, 0.0037615, -0.0052159, 0.0046098
4: -0.0049666, -0.0009945, -0.0052067, -0.0012105, -0.0033941, 0.0038717
5: -0.0005493, 0.0044909, -0.0006245, 0.0039458, -0.0044950, 0.0051155
6: -0.0067734, 0.0011315, -0.0065678, 0.0011245, -0.0078980, 0.0076993
7: -0.0244683, -0.0016217, -0.0258971, -0.0028986, -0.0189906, 0.0217845
8: 0.9731277, 0.9952570, 0.9718916, 0.9939551, -0.0208274, 0.0233654
9: -0.0068703, 0.0083713, -0.0060069, 0.0092886, -0.0151625, 0.0133323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A1_A2_B2_B1

### Relational analysis result of IS_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0127710
time: 1.46 seconds

## Relational analysis of IS_A2_A1_A2_B2_B2

### Relational analysis result of IS_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0130642
time: 1.14 seconds

## BFS IS instance: IS_A2_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0084286, 0.0187860, 0.0078843, 0.0186063, -0.0101777, 0.0109017
1: -0.0039089, 0.0011429, -0.0040516, 0.0011327, -0.0050416, 0.0051945
2: 0.0033987, 0.0089764, 0.0034575, 0.0093634, -0.0059647, 0.0055189
3: -0.0008197, 0.0037132, -0.0010360, 0.0038003, -0.0046200, 0.0047492
4: -0.0051675, -0.0012301, -0.0050976, -0.0011573, -0.0036537, 0.0035340
5: -0.0006096, 0.0038723, -0.0005538, 0.0041134, -0.0047231, 0.0044261
6: -0.0065548, 0.0009793, -0.0066226, 0.0011875, -0.0077423, 0.0076019
7: -0.0256950, -0.0030151, -0.0252260, -0.0025809, -0.0204921, 0.0197851
8: 0.9720360, 0.9938253, 0.9725629, 0.9942871, -0.0222511, 0.0212624
9: -0.0059282, 0.0091599, -0.0062213, 0.0088428, -0.0138046, 0.0143356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A2_A1_B1_B1

### Relational analysis result of IS_A2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0127573
time: 1.18 seconds

## Relational analysis of IS_A2_A2_A1_B1_B2

### Relational analysis result of IS_A2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0127573
time: 1.14 seconds

## BFS IS instance: IS_A2_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0084286, 0.0187860, 0.0080598, 0.0188332, -0.0104047, 0.0107262
1: -0.0039089, 0.0011429, -0.0040087, 0.0012371, -0.0051460, 0.0051516
2: 0.0033987, 0.0089764, 0.0033801, 0.0092359, -0.0058372, 0.0055963
3: -0.0008197, 0.0037132, -0.0009652, 0.0038474, -0.0046671, 0.0046784
4: -0.0051675, -0.0012301, -0.0053097, -0.0011940, -0.0035332, 0.0036748
5: -0.0006096, 0.0038723, -0.0006167, 0.0040371, -0.0046467, 0.0044890
6: -0.0065548, 0.0009793, -0.0065745, 0.0013794, -0.0079342, 0.0075538
7: -0.0256950, -0.0030151, -0.0264908, -0.0027958, -0.0196827, 0.0204936
8: 0.9720360, 0.9938253, 0.9714186, 0.9940882, -0.0220522, 0.0224067
9: -0.0059282, 0.0091599, -0.0060803, 0.0096681, -0.0143744, 0.0139136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A2_A1_B2_B1

### Relational analysis result of IS_A2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0129087
time: 1.75 seconds

## Relational analysis of IS_A2_A2_A1_B2_B2

### Relational analysis result of IS_A2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0129087
time: 1.79 seconds

## BFS IS instance: IS_A2_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0070348, 0.0187912, 0.0079037, 0.0186006, -0.0115658, 0.0108875
1: -0.0041962, 0.0013731, -0.0040471, 0.0011258, -0.0053220, 0.0054202
2: 0.0033970, 0.0099868, 0.0034596, 0.0093497, -0.0059527, 0.0065272
3: -0.0014337, 0.0037640, -0.0010283, 0.0037971, -0.0052309, 0.0047923
4: -0.0051883, -0.0010215, -0.0050892, -0.0011592, -0.0036783, 0.0037849
5: -0.0006126, 0.0044600, -0.0005530, 0.0041050, -0.0047176, 0.0050129
6: -0.0067363, 0.0013294, -0.0066211, 0.0011697, -0.0079060, 0.0079506
7: -0.0257798, -0.0017740, -0.0251820, -0.0025926, -0.0206222, 0.0212358
8: 0.9719387, 0.9951462, 0.9726003, 0.9942733, -0.0223346, 0.0225459
9: -0.0067789, 0.0092272, -0.0062129, 0.0088141, -0.0147842, 0.0144190

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130872, upper bound: 0.0127573
time: 1.21 seconds

## Relational analysis of IS_A2_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130872, upper bound: 0.0127573
time: 1.09 seconds

## BFS IS instance: IS_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0070348, 0.0187912, 0.0080857, 0.0188294, -0.0117946, 0.0107055
1: -0.0041962, 0.0013731, -0.0040029, 0.0012318, -0.0054280, 0.0053760
2: 0.0033970, 0.0099868, 0.0033815, 0.0092175, -0.0058205, 0.0066053
3: -0.0014337, 0.0037640, -0.0009556, 0.0038436, -0.0052773, 0.0047196
4: -0.0051883, -0.0010215, -0.0053051, -0.0011966, -0.0035563, 0.0039245
5: -0.0006126, 0.0044600, -0.0006160, 0.0040260, -0.0046385, 0.0050760
6: -0.0067363, 0.0013294, -0.0065731, 0.0013606, -0.0080969, 0.0079026
7: -0.0257798, -0.0017740, -0.0264635, -0.0028115, -0.0198116, 0.0219292
8: 0.9719387, 0.9951462, 0.9714417, 0.9940683, -0.0221296, 0.0237044
9: -0.0067789, 0.0092272, -0.0060690, 0.0096505, -0.0153489, 0.0139930

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130872, upper bound: 0.0129087
time: 1.20 seconds

## Relational analysis of IS_A2_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130872, upper bound: 0.0129087
time: 1.11 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.54 seconds
IS_A1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 8, lower bound: -0.0127060, upper bound: 0.0127106
IS_A1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 8, lower bound: -0.0127060, upper bound: 0.0127847
IS_A1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 8, lower bound: -0.0127060, upper bound: 0.0130010
IS_A1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 8, lower bound: -0.0127060, upper bound: 0.0130872
IS_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 8, lower bound: -0.0127681, upper bound: 0.0127709
IS_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 8, lower bound: -0.0127681, upper bound: 0.0127847
IS_A1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 8, lower bound: -0.0127681, upper bound: 0.0130642
IS_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 8, lower bound: -0.0127681, upper bound: 0.0130872
IS_A1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0127012
IS_A1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0127670
IS_A1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0128676
IS_A1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0129170
IS_A1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 8, lower bound: -0.0130682, upper bound: 0.0127573
IS_A1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 8, lower bound: -0.0130682, upper bound: 0.0127670
IS_A1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 8, lower bound: -0.0130682, upper bound: 0.0129087
IS_A1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 8, lower bound: -0.0130682, upper bound: 0.0129170
IS_A2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0127709
IS_A2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0130642
IS_A2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0127710
IS_A2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0130642
IS_A2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0127709
IS_A2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0130642
IS_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0127710
IS_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0130642
IS_A2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0127573
IS_A2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0127573
IS_A2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0129087
IS_A2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0129087
IS_A2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 8, lower bound: -0.0130872, upper bound: 0.0127573
IS_A2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 8, lower bound: -0.0130872, upper bound: 0.0127573
IS_A2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 8, lower bound: -0.0130872, upper bound: 0.0129087
IS_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 8, lower bound: -0.0130872, upper bound: 0.0129087

## BFS IS instance: IS_A1_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0080929, 0.0185533, 0.0082006, 0.0185661, -0.0104732, 0.0103527
1: -0.0039673, 0.0010577, -0.0039645, 0.0010567, -0.0050240, 0.0050222
2: 0.0034788, 0.0092297, 0.0034723, 0.0091412, -0.0056623, 0.0057574
3: -0.0010035, 0.0036382, -0.0009118, 0.0037130, -0.0047165, 0.0045501
4: -0.0049200, -0.0011302, -0.0050060, -0.0011893, -0.0033554, 0.0032583
5: -0.0005518, 0.0040088, -0.0005472, 0.0039708, -0.0045226, 0.0045561
6: -0.0066797, 0.0007150, -0.0066061, 0.0008964, -0.0075761, 0.0073211
7: -0.0242347, -0.0024411, -0.0247159, -0.0027709, -0.0188208, 0.0170620
8: 0.9733176, 0.9943494, 0.9729701, 0.9940659, -0.0207483, 0.0213793
9: -0.0063017, 0.0082136, -0.0060926, 0.0085146, -0.0128385, 0.0132027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_A1_B1_B1_B1

### Relational analysis result of IS_A1_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127109, upper bound: 0.0127007
time: 1.15 seconds

## Relational analysis of IS_A1_A1_A1_B1_B1_B2

### Relational analysis result of IS_A1_A1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127109, upper bound: 0.0127106
time: 1.15 seconds

## BFS IS instance: IS_A1_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0080929, 0.0185533, 0.0068974, 0.0185647, -0.0104717, 0.0116559
1: -0.0039673, 0.0010577, -0.0042330, 0.0012826, -0.0052499, 0.0052907
2: 0.0034788, 0.0092297, 0.0034733, 0.0100833, -0.0066045, 0.0057564
3: -0.0010035, 0.0036382, -0.0014806, 0.0037621, -0.0047656, 0.0051189
4: -0.0049200, -0.0011302, -0.0050100, -0.0009890, -0.0035938, 0.0032897
5: -0.0005518, 0.0040088, -0.0005496, 0.0045201, -0.0050719, 0.0045584
6: -0.0066797, 0.0007150, -0.0067770, 0.0012379, -0.0079176, 0.0074920
7: -0.0242347, -0.0024411, -0.0247173, -0.0015889, -0.0201400, 0.0172641
8: 0.9733176, 0.9943494, 0.9729388, 0.9952987, -0.0219811, 0.0214106
9: -0.0063017, 0.0082136, -0.0068935, 0.0085286, -0.0129340, 0.0141308

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_A1_B1_B2_B1

### Relational analysis result of IS_A1_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127109, upper bound: 0.0127709
time: 1.60 seconds

## Relational analysis of IS_A1_A1_A1_B1_B2_B2

### Relational analysis result of IS_A1_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127109, upper bound: 0.0127847
time: 1.72 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0080929, 0.0185533, 0.0083568, 0.0187905, -0.0106975, 0.0101966
1: -0.0039673, 0.0010577, -0.0039289, 0.0011624, -0.0051297, 0.0049866
2: 0.0034788, 0.0092297, 0.0033961, 0.0090266, -0.0055477, 0.0058336
3: -0.0010035, 0.0036382, -0.0008474, 0.0037561, -0.0047596, 0.0044856
4: -0.0049200, -0.0011302, -0.0052092, -0.0012242, -0.0033486, 0.0035190
5: -0.0005518, 0.0040088, -0.0006098, 0.0039046, -0.0044565, 0.0046187
6: -0.0066797, 0.0007150, -0.0065582, 0.0010844, -0.0077641, 0.0072731
7: -0.0242347, -0.0024411, -0.0259289, -0.0029789, -0.0187479, 0.0186340
8: 0.9733176, 0.9943494, 0.9718639, 0.9938715, -0.0205538, 0.0224854
9: -0.0063017, 0.0082136, -0.0059534, 0.0093075, -0.0137995, 0.0131651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127060, upper bound: 0.0129817
time: 1.19 seconds

## Relational analysis of IS_A1_A1_A1_B2_B1_B2

### Relational analysis result of IS_A1_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127060, upper bound: 0.0130010
time: 1.16 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0080929, 0.0185533, 0.0069692, 0.0187955, -0.0107025, 0.0115841
1: -0.0039673, 0.0010577, -0.0042135, 0.0013943, -0.0053616, 0.0052712
2: 0.0034788, 0.0092297, 0.0033944, 0.0100326, -0.0065538, 0.0058353
3: -0.0010035, 0.0036382, -0.0014595, 0.0038055, -0.0048090, 0.0050977
4: -0.0049200, -0.0011302, -0.0052279, -0.0010157, -0.0036030, 0.0035568
5: -0.0005518, 0.0040088, -0.0006129, 0.0044893, -0.0050411, 0.0046217
6: -0.0066797, 0.0007150, -0.0067396, 0.0014358, -0.0081155, 0.0074546
7: -0.0242347, -0.0024411, -0.0260001, -0.0017402, -0.0201567, 0.0188614
8: 0.9733176, 0.9943494, 0.9717728, 0.9951893, -0.0218717, 0.0225765
9: -0.0063017, 0.0082136, -0.0068028, 0.0093666, -0.0139279, 0.0141579

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127060, upper bound: 0.0130642
time: 1.12 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127060, upper bound: 0.0130872
time: 1.11 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0068087, 0.0185524, 0.0078817, 0.0185843, -0.0117756, 0.0106707
1: -0.0042306, 0.0012934, -0.0040217, 0.0011110, -0.0053416, 0.0053150
2: 0.0034796, 0.0101545, 0.0034678, 0.0093788, -0.0058992, 0.0066867
3: -0.0015647, 0.0036803, -0.0010904, 0.0036834, -0.0052481, 0.0047706
4: -0.0049252, -0.0009405, -0.0049796, -0.0011070, -0.0034325, 0.0034828
5: -0.0005544, 0.0045500, -0.0005567, 0.0041026, -0.0046570, 0.0051068
6: -0.0068399, 0.0010449, -0.0066925, 0.0008779, -0.0077178, 0.0077375
7: -0.0242427, -0.0013184, -0.0245771, -0.0023076, -0.0192571, 0.0183925
8: 0.9732838, 0.9955425, 0.9730402, 0.9945000, -0.0212162, 0.0225022
9: -0.0070691, 0.0082311, -0.0063927, 0.0084343, -0.0137155, 0.0135068

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_A1_A2_B1_B1_B1

### Relational analysis result of IS_A1_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127816, upper bound: 0.0126904
time: 1.13 seconds

## Relational analysis of IS_A1_A1_A2_B1_B1_B2

### Relational analysis result of IS_A1_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127816, upper bound: 0.0126904
time: 1.14 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0068087, 0.0185524, 0.0080430, 0.0185916, -0.0117829, 0.0105094
1: -0.0042306, 0.0012934, -0.0040073, 0.0010892, -0.0053199, 0.0053007
2: 0.0034796, 0.0101545, 0.0034645, 0.0092525, -0.0057729, 0.0066900
3: -0.0015647, 0.0036803, -0.0009760, 0.0037293, -0.0052940, 0.0046562
4: -0.0049252, -0.0009405, -0.0050205, -0.0011705, -0.0033927, 0.0035538
5: -0.0005544, 0.0045500, -0.0005517, 0.0040415, -0.0045958, 0.0051018
6: -0.0068399, 0.0010449, -0.0066158, 0.0009769, -0.0078168, 0.0076607
7: -0.0242427, -0.0013184, -0.0247996, -0.0026624, -0.0190315, 0.0188346
8: 0.9732838, 0.9955425, 0.9728891, 0.9941831, -0.0208994, 0.0226534
9: -0.0070691, 0.0082311, -0.0061640, 0.0085722, -0.0139372, 0.0133441

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_A1_A2_B1_B2_B1

### Relational analysis result of IS_A1_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127816, upper bound: 0.0126971
time: 1.18 seconds

## Relational analysis of IS_A1_A1_A2_B1_B2_B2

### Relational analysis result of IS_A1_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127816, upper bound: 0.0126971
time: 1.12 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0068087, 0.0185524, 0.0080966, 0.0188115, -0.0120027, 0.0104558
1: -0.0042306, 0.0012934, -0.0039736, 0.0012120, -0.0054426, 0.0052669
2: 0.0034796, 0.0101545, 0.0033906, 0.0092222, -0.0057426, 0.0067639
3: -0.0015647, 0.0036803, -0.0009961, 0.0037263, -0.0052909, 0.0046764
4: -0.0049252, -0.0009405, -0.0051905, -0.0011575, -0.0034062, 0.0037439
5: -0.0005544, 0.0045500, -0.0006182, 0.0040111, -0.0045655, 0.0051683
6: -0.0068399, 0.0010449, -0.0066341, 0.0010511, -0.0078910, 0.0076790
7: -0.0242427, -0.0013184, -0.0258253, -0.0025931, -0.0190561, 0.0199558
8: 0.9732838, 0.9955425, 0.9719090, 0.9942254, -0.0209417, 0.0236335
9: -0.0070691, 0.0082311, -0.0062098, 0.0092495, -0.0146849, 0.0133942

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_A1_A2_B2_B1_B1

### Relational analysis result of IS_A1_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127681, upper bound: 0.0129490
time: 1.26 seconds

## Relational analysis of IS_A1_A1_A2_B2_B1_B2

### Relational analysis result of IS_A1_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127681, upper bound: 0.0129490
time: 1.14 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0068087, 0.0185524, 0.0082217, 0.0188207, -0.0120119, 0.0103307
1: -0.0042306, 0.0012934, -0.0039652, 0.0011971, -0.0054278, 0.0052586
2: 0.0034796, 0.0101545, 0.0033862, 0.0091222, -0.0056426, 0.0067683
3: -0.0015647, 0.0036803, -0.0009042, 0.0037733, -0.0053380, 0.0045845
4: -0.0049252, -0.0009405, -0.0052392, -0.0012072, -0.0033835, 0.0038238
5: -0.0005544, 0.0045500, -0.0006149, 0.0039644, -0.0045188, 0.0051649
6: -0.0068399, 0.0010449, -0.0065680, 0.0011661, -0.0080060, 0.0076130
7: -0.0242427, -0.0013184, -0.0260971, -0.0028778, -0.0189385, 0.0204444
8: 0.9732838, 0.9955425, 0.9717151, 0.9939815, -0.0206977, 0.0238274
9: -0.0070691, 0.0082311, -0.0060216, 0.0094185, -0.0149447, 0.0132934

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127681, upper bound: 0.0129607
time: 1.29 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2_B2

### Relational analysis result of IS_A1_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127681, upper bound: 0.0129607
time: 1.16 seconds

## BFS IS instance: IS_A1_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0083055, 0.0187768, 0.0082006, 0.0185661, -0.0102606, 0.0105761
1: -0.0039174, 0.0011571, -0.0039645, 0.0010567, -0.0049741, 0.0051216
2: 0.0034030, 0.0090757, 0.0034723, 0.0091412, -0.0057382, 0.0056033
3: -0.0009109, 0.0036682, -0.0009118, 0.0037130, -0.0046239, 0.0045800
4: -0.0051203, -0.0011813, -0.0050060, -0.0011893, -0.0035934, 0.0034524
5: -0.0006129, 0.0039166, -0.0005472, 0.0039708, -0.0045836, 0.0044639
6: -0.0066209, 0.0008647, -0.0066061, 0.0008964, -0.0075173, 0.0074708
7: -0.0254298, -0.0027351, -0.0247159, -0.0027709, -0.0201535, 0.0192974
8: 0.9722265, 0.9940638, 0.9729701, 0.9940659, -0.0218394, 0.0210937
9: -0.0061136, 0.0089945, -0.0060926, 0.0085146, -0.0135385, 0.0140894

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_A1_B1_B1_B1

### Relational analysis result of IS_A1_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0126915
time: 1.15 seconds

## Relational analysis of IS_A1_A2_A1_B1_B1_B2

### Relational analysis result of IS_A1_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0127012
time: 1.15 seconds

## BFS IS instance: IS_A1_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0083055, 0.0187768, 0.0068974, 0.0185647, -0.0102592, 0.0118794
1: -0.0039174, 0.0011571, -0.0042330, 0.0012826, -0.0052000, 0.0053901
2: 0.0034030, 0.0090757, 0.0034733, 0.0100833, -0.0066803, 0.0056023
3: -0.0009109, 0.0036682, -0.0014806, 0.0037621, -0.0046730, 0.0051488
4: -0.0051203, -0.0011813, -0.0050100, -0.0009890, -0.0038317, 0.0034759
5: -0.0006129, 0.0039166, -0.0005496, 0.0045201, -0.0051330, 0.0044662
6: -0.0066209, 0.0008647, -0.0067770, 0.0012379, -0.0078588, 0.0076417
7: -0.0254298, -0.0027351, -0.0247173, -0.0015889, -0.0214728, 0.0194191
8: 0.9722265, 0.9940638, 0.9729388, 0.9952987, -0.0230722, 0.0211250
9: -0.0061136, 0.0089945, -0.0068935, 0.0085286, -0.0136122, 0.0150176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_A1_B1_B2_B1

### Relational analysis result of IS_A1_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0127573
time: 1.15 seconds

## Relational analysis of IS_A1_A2_A1_B1_B2_B2

### Relational analysis result of IS_A1_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0127670
time: 1.18 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0083055, 0.0187768, 0.0083568, 0.0187905, -0.0104850, 0.0104200
1: -0.0039174, 0.0011571, -0.0039289, 0.0011624, -0.0050798, 0.0050861
2: 0.0034030, 0.0090757, 0.0033961, 0.0090266, -0.0056235, 0.0056795
3: -0.0009109, 0.0036682, -0.0008474, 0.0037561, -0.0046670, 0.0045156
4: -0.0051203, -0.0011813, -0.0052092, -0.0012242, -0.0034787, 0.0035938
5: -0.0006129, 0.0039166, -0.0006098, 0.0039046, -0.0045175, 0.0045265
6: -0.0066209, 0.0008647, -0.0065582, 0.0010844, -0.0077053, 0.0074228
7: -0.0254298, -0.0027351, -0.0259289, -0.0029789, -0.0193964, 0.0200219
8: 0.9722265, 0.9940638, 0.9718639, 0.9938715, -0.0216449, 0.0221999
9: -0.0061136, 0.0089945, -0.0059534, 0.0093075, -0.0141063, 0.0136921

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_A1_B2_B1_B1

### Relational analysis result of IS_A1_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130573, upper bound: 0.0128609
time: 1.16 seconds

## Relational analysis of IS_A1_A2_A1_B2_B1_B2

### Relational analysis result of IS_A1_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130573, upper bound: 0.0128675
time: 1.58 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0083055, 0.0187768, 0.0069692, 0.0187955, -0.0104900, 0.0118076
1: -0.0039174, 0.0011571, -0.0042135, 0.0013943, -0.0053117, 0.0053707
2: 0.0034030, 0.0090757, 0.0033944, 0.0100326, -0.0066296, 0.0056812
3: -0.0009109, 0.0036682, -0.0014595, 0.0038055, -0.0047164, 0.0051277
4: -0.0051203, -0.0011813, -0.0052279, -0.0010157, -0.0037295, 0.0036238
5: -0.0006129, 0.0039166, -0.0006129, 0.0044893, -0.0051022, 0.0045295
6: -0.0066209, 0.0008647, -0.0067396, 0.0014358, -0.0080567, 0.0076043
7: -0.0254298, -0.0027351, -0.0260001, -0.0017402, -0.0207862, 0.0201786
8: 0.9722265, 0.9940638, 0.9717728, 0.9951893, -0.0229628, 0.0222909
9: -0.0061136, 0.0089945, -0.0068028, 0.0093666, -0.0142078, 0.0146734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_A1_B2_B2_B1

### Relational analysis result of IS_A1_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130573, upper bound: 0.0129087
time: 1.15 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2_B2

### Relational analysis result of IS_A1_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130573, upper bound: 0.0129170
time: 1.18 seconds

## BFS IS instance: IS_A1_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0069434, 0.0187820, 0.0078817, 0.0185843, -0.0116409, 0.0109002
1: -0.0041974, 0.0013851, -0.0040217, 0.0011110, -0.0053084, 0.0054068
2: 0.0034013, 0.0100584, 0.0034678, 0.0093788, -0.0059775, 0.0065906
3: -0.0015091, 0.0037173, -0.0010904, 0.0036834, -0.0051925, 0.0048077
4: -0.0051387, -0.0009777, -0.0049796, -0.0011070, -0.0036814, 0.0036831
5: -0.0006160, 0.0044893, -0.0005567, 0.0041026, -0.0047186, 0.0050460
6: -0.0067939, 0.0012106, -0.0066925, 0.0008779, -0.0076718, 0.0079032
7: -0.0255066, -0.0015269, -0.0245771, -0.0023076, -0.0206505, 0.0206904
8: 0.9721356, 0.9953537, 0.9730402, 0.9945000, -0.0223644, 0.0223135
9: -0.0069351, 0.0090564, -0.0063927, 0.0084343, -0.0144480, 0.0144338

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_A2_A2_B1_B1_B1

### Relational analysis result of IS_A1_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130682, upper bound: 0.0126848
time: 1.28 seconds

## Relational analysis of IS_A1_A2_A2_B1_B1_B2

### Relational analysis result of IS_A1_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130682, upper bound: 0.0126848
time: 1.44 seconds

## BFS IS instance: IS_A1_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0069434, 0.0187820, 0.0080430, 0.0185916, -0.0116482, 0.0107390
1: -0.0041974, 0.0013851, -0.0040073, 0.0010892, -0.0052866, 0.0053925
2: 0.0034013, 0.0100584, 0.0034645, 0.0092525, -0.0058512, 0.0065939
3: -0.0015091, 0.0037173, -0.0009760, 0.0037293, -0.0052384, 0.0046933
4: -0.0051387, -0.0009777, -0.0050205, -0.0011705, -0.0036417, 0.0037418
5: -0.0006160, 0.0044893, -0.0005517, 0.0040415, -0.0046575, 0.0050410
6: -0.0067939, 0.0012106, -0.0066158, 0.0009769, -0.0077708, 0.0078264
7: -0.0255066, -0.0015269, -0.0247996, -0.0026624, -0.0204249, 0.0210015
8: 0.9721356, 0.9953537, 0.9728891, 0.9941831, -0.0220475, 0.0224646
9: -0.0069351, 0.0090564, -0.0061640, 0.0085722, -0.0146352, 0.0142711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_A2_A2_B1_B2_B1

### Relational analysis result of IS_A1_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130682, upper bound: 0.0126931
time: 1.27 seconds

## Relational analysis of IS_A1_A2_A2_B1_B2_B2

### Relational analysis result of IS_A1_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130682, upper bound: 0.0126931
time: 1.53 seconds

## BFS IS instance: IS_A1_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0069434, 0.0187820, 0.0080966, 0.0188115, -0.0118681, 0.0106854
1: -0.0041974, 0.0013851, -0.0039736, 0.0012120, -0.0054093, 0.0053587
2: 0.0034013, 0.0100584, 0.0033906, 0.0092222, -0.0058209, 0.0066678
3: -0.0015091, 0.0037173, -0.0009961, 0.0037263, -0.0052354, 0.0047134
4: -0.0051387, -0.0009777, -0.0051905, -0.0011575, -0.0035495, 0.0038227
5: -0.0006160, 0.0044893, -0.0006182, 0.0040111, -0.0046271, 0.0051076
6: -0.0067939, 0.0012106, -0.0066341, 0.0010511, -0.0078450, 0.0078447
7: -0.0255066, -0.0015269, -0.0258253, -0.0025931, -0.0197960, 0.0214088
8: 0.9721356, 0.9953537, 0.9719090, 0.9942254, -0.0220898, 0.0234447
9: -0.0069351, 0.0090564, -0.0062098, 0.0092495, -0.0150114, 0.0139721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_A2_A2_B2_B1_B1

### Relational analysis result of IS_A1_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0131515, upper bound: 0.0128568
time: 1.16 seconds

## Relational analysis of IS_A1_A2_A2_B2_B1_B2

### Relational analysis result of IS_A1_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0131515, upper bound: 0.0128568
time: 1.52 seconds

## BFS IS instance: IS_A1_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0069434, 0.0187820, 0.0082217, 0.0188207, -0.0118773, 0.0105603
1: -0.0041974, 0.0013851, -0.0039652, 0.0011971, -0.0053945, 0.0053503
2: 0.0034013, 0.0100584, 0.0033862, 0.0091222, -0.0057209, 0.0066721
3: -0.0015091, 0.0037173, -0.0009042, 0.0037733, -0.0052824, 0.0046215
4: -0.0051387, -0.0009777, -0.0052392, -0.0012072, -0.0035213, 0.0038903
5: -0.0006160, 0.0044893, -0.0006149, 0.0039644, -0.0045805, 0.0051042
6: -0.0067939, 0.0012106, -0.0065680, 0.0011661, -0.0079600, 0.0077787
7: -0.0255066, -0.0015269, -0.0260971, -0.0028778, -0.0196380, 0.0217605
8: 0.9721356, 0.9953537, 0.9717151, 0.9939815, -0.0218459, 0.0236386
9: -0.0069351, 0.0090564, -0.0060216, 0.0094185, -0.0152323, 0.0138532

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_A2_A2_B2_B2_B1

### Relational analysis result of IS_A1_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0131515, upper bound: 0.0128625
time: 1.13 seconds

## Relational analysis of IS_A1_A2_A2_B2_B2_B2

### Relational analysis result of IS_A1_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0131515, upper bound: 0.0128625
time: 1.13 seconds

## BFS IS instance: IS_A2_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0082565, 0.0185611, 0.0078612, 0.0185899, -0.0103335, 0.0107000
1: -0.0039485, 0.0010368, -0.0040267, 0.0011182, -0.0050668, 0.0050635
2: 0.0034755, 0.0091027, 0.0034657, 0.0093932, -0.0059178, 0.0056369
3: -0.0008895, 0.0036775, -0.0010985, 0.0036865, -0.0045761, 0.0047759
4: -0.0049572, -0.0011940, -0.0049890, -0.0011051, -0.0034766, 0.0032091
5: -0.0005470, 0.0039456, -0.0005575, 0.0041116, -0.0046586, 0.0045031
6: -0.0066028, 0.0008008, -0.0066939, 0.0008959, -0.0074986, 0.0074948
7: -0.0244393, -0.0027993, -0.0246223, -0.0022947, -0.0194695, 0.0168424
8: 0.9731785, 0.9940299, 0.9730005, 0.9945157, -0.0213372, 0.0210294
9: -0.0060726, 0.0083402, -0.0064020, 0.0084639, -0.0126567, 0.0136448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A1_A1_B1_B1_B1

### Relational analysis result of IS_A2_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0127172
time: 1.12 seconds

## Relational analysis of IS_A2_A1_A1_B1_B1_B2

### Relational analysis result of IS_A2_A1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0127816
time: 1.59 seconds

## BFS IS instance: IS_A2_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0082565, 0.0185611, 0.0080695, 0.0188152, -0.0105588, 0.0104916
1: -0.0039485, 0.0010368, -0.0039797, 0.0012173, -0.0051658, 0.0050166
2: 0.0034755, 0.0091027, 0.0033890, 0.0092414, -0.0057659, 0.0057136
3: -0.0008895, 0.0036775, -0.0010061, 0.0037298, -0.0046193, 0.0046836
4: -0.0049572, -0.0011940, -0.0051954, -0.0011550, -0.0034512, 0.0034660
5: -0.0005470, 0.0039456, -0.0006189, 0.0040227, -0.0045697, 0.0045645
6: -0.0066028, 0.0008008, -0.0066353, 0.0010696, -0.0076723, 0.0074362
7: -0.0244393, -0.0027993, -0.0258514, -0.0025773, -0.0192745, 0.0183928
8: 0.9731785, 0.9940299, 0.9718860, 0.9942452, -0.0210667, 0.0221439
9: -0.0060726, 0.0083402, -0.0062211, 0.0092663, -0.0136036, 0.0135363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A1_A1_B1_B2_B1

### Relational analysis result of IS_A2_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0129857
time: 1.10 seconds

## Relational analysis of IS_A2_A1_A1_B1_B2_B2

### Relational analysis result of IS_A2_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0130682
time: 1.09 seconds

## BFS IS instance: IS_A2_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0082565, 0.0185611, 0.0080239, 0.0185974, -0.0103410, 0.0105372
1: -0.0039485, 0.0010368, -0.0040117, 0.0010961, -0.0050446, 0.0050486
2: 0.0034755, 0.0091027, 0.0034624, 0.0092661, -0.0057906, 0.0056403
3: -0.0008895, 0.0036775, -0.0009836, 0.0037321, -0.0046216, 0.0046611
4: -0.0049572, -0.0011940, -0.0050278, -0.0011686, -0.0033928, 0.0032165
5: -0.0005470, 0.0039456, -0.0005526, 0.0040497, -0.0045967, 0.0044982
6: -0.0066028, 0.0008008, -0.0066172, 0.0009942, -0.0075970, 0.0074181
7: -0.0244393, -0.0027993, -0.0248398, -0.0026507, -0.0190180, 0.0168031
8: 0.9731785, 0.9940299, 0.9728556, 0.9941984, -0.0210199, 0.0211744
9: -0.0060726, 0.0083402, -0.0061724, 0.0085985, -0.0126879, 0.0133548

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A1_A1_B2_B1_B1

### Relational analysis result of IS_A2_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0127007
time: 1.74 seconds

## Relational analysis of IS_A2_A1_A1_B2_B1_B2

### Relational analysis result of IS_A2_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0127710
time: 1.78 seconds

## BFS IS instance: IS_A2_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0082565, 0.0185611, 0.0081960, 0.0188245, -0.0105680, 0.0103651
1: -0.0039485, 0.0010368, -0.0039710, 0.0012024, -0.0051509, 0.0050078
2: 0.0034755, 0.0091027, 0.0033848, 0.0091405, -0.0056650, 0.0057179
3: -0.0008895, 0.0036775, -0.0009138, 0.0037767, -0.0046662, 0.0045913
4: -0.0049572, -0.0011940, -0.0052431, -0.0012046, -0.0033822, 0.0034780
5: -0.0005470, 0.0039456, -0.0006156, 0.0039755, -0.0045225, 0.0045612
6: -0.0066028, 0.0008008, -0.0065694, 0.0011851, -0.0077878, 0.0073703
7: -0.0244393, -0.0027993, -0.0261207, -0.0028622, -0.0189150, 0.0183734
8: 0.9731785, 0.9940299, 0.9716934, 0.9940014, -0.0208229, 0.0223365
9: -0.0060726, 0.0083402, -0.0060329, 0.0094339, -0.0136552, 0.0132981

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A1_A1_B2_B2_B1

### Relational analysis result of IS_A2_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0129817
time: 2.14 seconds

## Relational analysis of IS_A2_A1_A1_B2_B2_B2

### Relational analysis result of IS_A2_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0130642
time: 1.30 seconds

## BFS IS instance: IS_A2_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0069627, 0.0185597, 0.0078817, 0.0185843, -0.0116216, 0.0106779
1: -0.0042152, 0.0012591, -0.0040217, 0.0011110, -0.0053262, 0.0052808
2: 0.0034765, 0.0100375, 0.0034678, 0.0093788, -0.0059023, 0.0065698
3: -0.0014544, 0.0037225, -0.0010904, 0.0036834, -0.0051378, 0.0048129
4: -0.0049666, -0.0009945, -0.0049796, -0.0011070, -0.0034931, 0.0036560
5: -0.0005493, 0.0044909, -0.0005567, 0.0041026, -0.0046518, 0.0050477
6: -0.0067734, 0.0011315, -0.0066925, 0.0008779, -0.0076513, 0.0078241
7: -0.0244683, -0.0016217, -0.0245771, -0.0023076, -0.0195731, 0.0205747
8: 0.9731277, 0.9952570, 0.9730402, 0.9945000, -0.0213723, 0.0222168
9: -0.0068703, 0.0083713, -0.0063927, 0.0084343, -0.0143502, 0.0137012

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A1_A2_B1_B1_B1

### Relational analysis result of IS_A2_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0127109
time: 1.22 seconds

## Relational analysis of IS_A2_A1_A2_B1_B1_B2

### Relational analysis result of IS_A2_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0127109
time: 1.19 seconds

## BFS IS instance: IS_A2_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0069627, 0.0185597, 0.0080966, 0.0188115, -0.0118487, 0.0104631
1: -0.0042152, 0.0012591, -0.0039736, 0.0012120, -0.0054271, 0.0052327
2: 0.0034765, 0.0100375, 0.0033906, 0.0092222, -0.0057457, 0.0066470
3: -0.0014544, 0.0037225, -0.0009961, 0.0037263, -0.0051807, 0.0047186
4: -0.0049666, -0.0009945, -0.0051905, -0.0011575, -0.0034668, 0.0039012
5: -0.0005493, 0.0044909, -0.0006182, 0.0040111, -0.0045603, 0.0051092
6: -0.0067734, 0.0011315, -0.0066341, 0.0010511, -0.0078245, 0.0077656
7: -0.0244683, -0.0016217, -0.0258253, -0.0025931, -0.0193720, 0.0219527
8: 0.9731277, 0.9952570, 0.9719090, 0.9942254, -0.0210977, 0.0233480
9: -0.0068703, 0.0083713, -0.0062098, 0.0092495, -0.0152625, 0.0135886

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A1_A2_B1_B2_B1

### Relational analysis result of IS_A2_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0129542
time: 1.59 seconds

## Relational analysis of IS_A2_A1_A2_B1_B2_B2

### Relational analysis result of IS_A2_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0129542
time: 1.20 seconds

## BFS IS instance: IS_A2_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0069627, 0.0185597, 0.0080430, 0.0185916, -0.0116289, 0.0105166
1: -0.0042152, 0.0012591, -0.0040073, 0.0010892, -0.0053044, 0.0052664
2: 0.0034765, 0.0100375, 0.0034645, 0.0092525, -0.0057761, 0.0065730
3: -0.0014544, 0.0037225, -0.0009760, 0.0037293, -0.0051837, 0.0046985
4: -0.0049666, -0.0009945, -0.0050205, -0.0011705, -0.0034093, 0.0036718
5: -0.0005493, 0.0044909, -0.0005517, 0.0040415, -0.0045907, 0.0050427
6: -0.0067734, 0.0011315, -0.0066158, 0.0009769, -0.0077504, 0.0077473
7: -0.0244683, -0.0016217, -0.0247996, -0.0026624, -0.0191183, 0.0206564
8: 0.9731277, 0.9952570, 0.9728891, 0.9941831, -0.0210554, 0.0223680
9: -0.0068703, 0.0083713, -0.0061640, 0.0085722, -0.0144104, 0.0134090

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A1_A2_B2_B1_B1

### Relational analysis result of IS_A2_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0126904
time: 1.22 seconds

## Relational analysis of IS_A2_A1_A2_B2_B1_B2

### Relational analysis result of IS_A2_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0126904
time: 1.82 seconds

## BFS IS instance: IS_A2_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0069627, 0.0185597, 0.0082217, 0.0188207, -0.0118579, 0.0103380
1: -0.0042152, 0.0012591, -0.0039652, 0.0011971, -0.0054123, 0.0052243
2: 0.0034765, 0.0100375, 0.0033862, 0.0091222, -0.0056458, 0.0066513
3: -0.0014544, 0.0037225, -0.0009042, 0.0037733, -0.0052277, 0.0046267
4: -0.0049666, -0.0009945, -0.0052392, -0.0012072, -0.0033979, 0.0039249
5: -0.0005493, 0.0044909, -0.0006149, 0.0039644, -0.0045137, 0.0051058
6: -0.0067734, 0.0011315, -0.0065680, 0.0011661, -0.0079395, 0.0076996
7: -0.0244683, -0.0016217, -0.0260971, -0.0028778, -0.0190086, 0.0220795
8: 0.9731277, 0.9952570, 0.9717151, 0.9939815, -0.0208538, 0.0235419
9: -0.0068703, 0.0083713, -0.0060216, 0.0094185, -0.0153491, 0.0133483

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A1_A2_B2_B2_B1

### Relational analysis result of IS_A2_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0129490
time: 1.56 seconds

## Relational analysis of IS_A2_A1_A2_B2_B2_B2

### Relational analysis result of IS_A2_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0129490
time: 1.26 seconds

## BFS IS instance: IS_A2_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0084286, 0.0187860, 0.0078612, 0.0185899, -0.0101613, 0.0109248
1: -0.0039089, 0.0011429, -0.0040267, 0.0011182, -0.0050271, 0.0051696
2: 0.0033987, 0.0089764, 0.0034657, 0.0093932, -0.0059945, 0.0055106
3: -0.0008197, 0.0037132, -0.0010985, 0.0036865, -0.0045062, 0.0048117
4: -0.0051675, -0.0012301, -0.0049890, -0.0011051, -0.0037292, 0.0034145
5: -0.0006096, 0.0038723, -0.0005575, 0.0041116, -0.0047212, 0.0044298
6: -0.0065548, 0.0009793, -0.0066939, 0.0008959, -0.0074507, 0.0076732
7: -0.0256950, -0.0030151, -0.0246223, -0.0022947, -0.0209279, 0.0191517
8: 0.9720360, 0.9938253, 0.9730005, 0.9945157, -0.0224797, 0.0208248
9: -0.0059282, 0.0091599, -0.0064020, 0.0084639, -0.0133993, 0.0145893

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A2_A1_B1_B1_B1

### Relational analysis result of IS_A2_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0126915
time: 1.59 seconds

## Relational analysis of IS_A2_A2_A1_B1_B1_B2

### Relational analysis result of IS_A2_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0127573
time: 1.23 seconds

## BFS IS instance: IS_A2_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0084286, 0.0187860, 0.0080239, 0.0185974, -0.0101689, 0.0107621
1: -0.0039089, 0.0011429, -0.0040117, 0.0010961, -0.0050050, 0.0051547
2: 0.0033987, 0.0089764, 0.0034624, 0.0092661, -0.0058674, 0.0055140
3: -0.0008197, 0.0037132, -0.0009836, 0.0037321, -0.0045517, 0.0046968
4: -0.0051675, -0.0012301, -0.0050278, -0.0011686, -0.0036401, 0.0034276
5: -0.0006096, 0.0038723, -0.0005526, 0.0040497, -0.0046594, 0.0044249
6: -0.0065548, 0.0009793, -0.0066172, 0.0009942, -0.0075490, 0.0075966
7: -0.0256950, -0.0030151, -0.0248398, -0.0026507, -0.0204254, 0.0192045
8: 0.9720360, 0.9938253, 0.9728556, 0.9941984, -0.0221624, 0.0209697
9: -0.0059282, 0.0091599, -0.0061724, 0.0085985, -0.0134488, 0.0142780

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A2_A1_B1_B2_B1

### Relational analysis result of IS_A2_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0126915
time: 1.58 seconds

## Relational analysis of IS_A2_A2_A1_B1_B2_B2

### Relational analysis result of IS_A2_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0127573
time: 1.12 seconds

## BFS IS instance: IS_A2_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0084286, 0.0187860, 0.0080695, 0.0188152, -0.0103867, 0.0107165
1: -0.0039089, 0.0011429, -0.0039797, 0.0012173, -0.0051262, 0.0051227
2: 0.0033987, 0.0089764, 0.0033890, 0.0092414, -0.0058427, 0.0055873
3: -0.0008197, 0.0037132, -0.0010061, 0.0037298, -0.0045494, 0.0047194
4: -0.0051675, -0.0012301, -0.0051954, -0.0011550, -0.0035950, 0.0035469
5: -0.0006096, 0.0038723, -0.0006189, 0.0040227, -0.0046323, 0.0044912
6: -0.0065548, 0.0009793, -0.0066353, 0.0010696, -0.0076244, 0.0076146
7: -0.0256950, -0.0030151, -0.0258514, -0.0025773, -0.0200320, 0.0198187
8: 0.9720360, 0.9938253, 0.9718860, 0.9942452, -0.0222092, 0.0219393
9: -0.0059282, 0.0091599, -0.0062211, 0.0092663, -0.0139359, 0.0141200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A2_A1_B2_B1_B1

### Relational analysis result of IS_A2_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130558, upper bound: 0.0128609
time: 1.12 seconds

## Relational analysis of IS_A2_A2_A1_B2_B1_B2

### Relational analysis result of IS_A2_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130558, upper bound: 0.0129087
time: 1.13 seconds

## BFS IS instance: IS_A2_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0084286, 0.0187860, 0.0081960, 0.0188245, -0.0103959, 0.0105900
1: -0.0039089, 0.0011429, -0.0039710, 0.0012024, -0.0051113, 0.0051139
2: 0.0033987, 0.0089764, 0.0033848, 0.0091405, -0.0057417, 0.0055916
3: -0.0008197, 0.0037132, -0.0009138, 0.0037767, -0.0045963, 0.0046270
4: -0.0051675, -0.0012301, -0.0052431, -0.0012046, -0.0035204, 0.0035685
5: -0.0006096, 0.0038723, -0.0006156, 0.0039755, -0.0045852, 0.0044879
6: -0.0065548, 0.0009793, -0.0065694, 0.0011851, -0.0077399, 0.0075487
7: -0.0256950, -0.0030151, -0.0261207, -0.0028622, -0.0196184, 0.0199194
8: 0.9720360, 0.9938253, 0.9716934, 0.9940014, -0.0219654, 0.0221319
9: -0.0059282, 0.0091599, -0.0060329, 0.0094339, -0.0140187, 0.0138592

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A2_A1_B2_B2_B1

### Relational analysis result of IS_A2_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130558, upper bound: 0.0128609
time: 1.12 seconds

## Relational analysis of IS_A2_A2_A1_B2_B2_B2

### Relational analysis result of IS_A2_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130558, upper bound: 0.0129087
time: 1.18 seconds

## BFS IS instance: IS_A2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0070348, 0.0187912, 0.0078817, 0.0185843, -0.0115495, 0.0109095
1: -0.0041962, 0.0013731, -0.0040217, 0.0011110, -0.0053072, 0.0053948
2: 0.0033970, 0.0099868, 0.0034678, 0.0093788, -0.0059818, 0.0065191
3: -0.0014337, 0.0037640, -0.0010904, 0.0036834, -0.0051172, 0.0048544
4: -0.0051883, -0.0010215, -0.0049796, -0.0011070, -0.0037549, 0.0036649
5: -0.0006126, 0.0044600, -0.0005567, 0.0041026, -0.0047152, 0.0050167
6: -0.0067363, 0.0013294, -0.0066925, 0.0008779, -0.0076141, 0.0080220
7: -0.0257798, -0.0017740, -0.0245771, -0.0023076, -0.0210689, 0.0205888
8: 0.9719387, 0.9951462, 0.9730402, 0.9945000, -0.0225613, 0.0221059
9: -0.0067789, 0.0092272, -0.0063927, 0.0084343, -0.0143760, 0.0146775

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A2_A2_B1_B1_B1

### Relational analysis result of IS_A2_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130872, upper bound: 0.0126848
time: 1.19 seconds

## Relational analysis of IS_A2_A2_A2_B1_B1_B2

### Relational analysis result of IS_A2_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130872, upper bound: 0.0126848
time: 1.25 seconds

## BFS IS instance: IS_A2_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0070348, 0.0187912, 0.0080430, 0.0185916, -0.0115568, 0.0107482
1: -0.0041962, 0.0013731, -0.0040073, 0.0010892, -0.0052855, 0.0053804
2: 0.0033970, 0.0099868, 0.0034645, 0.0092525, -0.0058556, 0.0065223
3: -0.0014337, 0.0037640, -0.0009760, 0.0037293, -0.0051631, 0.0047400
4: -0.0051883, -0.0010215, -0.0050205, -0.0011705, -0.0036648, 0.0036772
5: -0.0006126, 0.0044600, -0.0005517, 0.0040415, -0.0046540, 0.0050117
6: -0.0067363, 0.0013294, -0.0066158, 0.0009769, -0.0077132, 0.0079452
7: -0.0257798, -0.0017740, -0.0247996, -0.0026624, -0.0205553, 0.0206532
8: 0.9719387, 0.9951462, 0.9728891, 0.9941831, -0.0222444, 0.0222571
9: -0.0067789, 0.0092272, -0.0061640, 0.0085722, -0.0144241, 0.0143614

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A2_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130872, upper bound: 0.0126848
time: 1.17 seconds

## Relational analysis of IS_A2_A2_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130872, upper bound: 0.0126848
time: 1.22 seconds

## BFS IS instance: IS_A2_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0070348, 0.0187912, 0.0080966, 0.0188115, -0.0117767, 0.0106946
1: -0.0041962, 0.0013731, -0.0039736, 0.0012120, -0.0054082, 0.0053467
2: 0.0033970, 0.0099868, 0.0033906, 0.0092222, -0.0058252, 0.0065963
3: -0.0014337, 0.0037640, -0.0009961, 0.0037263, -0.0051600, 0.0047601
4: -0.0051883, -0.0010215, -0.0051905, -0.0011575, -0.0036168, 0.0037970
5: -0.0006126, 0.0044600, -0.0006182, 0.0040111, -0.0046237, 0.0050782
6: -0.0067363, 0.0013294, -0.0066341, 0.0010511, -0.0077874, 0.0079635
7: -0.0257798, -0.0017740, -0.0258253, -0.0025931, -0.0201510, 0.0212512
8: 0.9719387, 0.9951462, 0.9719090, 0.9942254, -0.0222867, 0.0232372
9: -0.0067789, 0.0092272, -0.0062098, 0.0092495, -0.0149122, 0.0141924

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A2_A2_B2_B1_B1

### Relational analysis result of IS_A2_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0131682, upper bound: 0.0128568
time: 1.85 seconds

## Relational analysis of IS_A2_A2_A2_B2_B1_B2

### Relational analysis result of IS_A2_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0131682, upper bound: 0.0128568
time: 2.00 seconds

## BFS IS instance: IS_A2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0070348, 0.0187912, 0.0082217, 0.0188207, -0.0117859, 0.0105696
1: -0.0041962, 0.0013731, -0.0039652, 0.0011971, -0.0053934, 0.0053383
2: 0.0033970, 0.0099868, 0.0033862, 0.0091222, -0.0057253, 0.0066006
3: -0.0014337, 0.0037640, -0.0009042, 0.0037733, -0.0052071, 0.0046682
4: -0.0051883, -0.0010215, -0.0052392, -0.0012072, -0.0035435, 0.0038167
5: -0.0006126, 0.0044600, -0.0006149, 0.0039644, -0.0045770, 0.0050748
6: -0.0067363, 0.0013294, -0.0065680, 0.0011661, -0.0079023, 0.0078975
7: -0.0257798, -0.0017740, -0.0260971, -0.0028778, -0.0197474, 0.0213602
8: 0.9719387, 0.9951462, 0.9717151, 0.9939815, -0.0220428, 0.0234311
9: -0.0067789, 0.0092272, -0.0060216, 0.0094185, -0.0149854, 0.0139386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A2_A2_B2_B2_B1

### Relational analysis result of IS_A2_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0131682, upper bound: 0.0128568
time: 1.24 seconds

## Relational analysis of IS_A2_A2_A2_B2_B2_B2

### Relational analysis result of IS_A2_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0131682, upper bound: 0.0128568
time: 1.55 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.04 seconds
IS_A1_A1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127109, upper bound: 0.0127007
IS_A1_A1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127109, upper bound: 0.0127106
IS_A1_A1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127109, upper bound: 0.0127709
IS_A1_A1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127109, upper bound: 0.0127847
IS_A1_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127060, upper bound: 0.0129817
IS_A1_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127060, upper bound: 0.0130010
IS_A1_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127060, upper bound: 0.0130642
IS_A1_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127060, upper bound: 0.0130872
IS_A1_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127816, upper bound: 0.0126904
IS_A1_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127816, upper bound: 0.0126904
IS_A1_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127816, upper bound: 0.0126971
IS_A1_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127816, upper bound: 0.0126971
IS_A1_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127681, upper bound: 0.0129490
IS_A1_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127681, upper bound: 0.0129490
IS_A1_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127681, upper bound: 0.0129607
IS_A1_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127681, upper bound: 0.0129607
IS_A1_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0126915
IS_A1_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0127012
IS_A1_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0127573
IS_A1_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0127670
IS_A1_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0130573, upper bound: 0.0128609
IS_A1_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0130573, upper bound: 0.0128675
IS_A1_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0130573, upper bound: 0.0129087
IS_A1_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0130573, upper bound: 0.0129170
IS_A1_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0130682, upper bound: 0.0126848
IS_A1_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0130682, upper bound: 0.0126848
IS_A1_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0130682, upper bound: 0.0126931
IS_A1_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0130682, upper bound: 0.0126931
IS_A1_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0131515, upper bound: 0.0128568
IS_A1_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0131515, upper bound: 0.0128568
IS_A1_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0131515, upper bound: 0.0128625
IS_A1_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0131515, upper bound: 0.0128625
IS_A2_A1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0127172
IS_A2_A1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0127816
IS_A2_A1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0129857
IS_A2_A1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0130682
IS_A2_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0127007
IS_A2_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0127710
IS_A2_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0129817
IS_A2_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0130642
IS_A2_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0127109
IS_A2_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0127109
IS_A2_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0129542
IS_A2_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0129542
IS_A2_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0126904
IS_A2_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0126904
IS_A2_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0129490
IS_A2_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0129490
IS_A2_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0126915
IS_A2_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0127573
IS_A2_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0126915
IS_A2_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0127573
IS_A2_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0130558, upper bound: 0.0128609
IS_A2_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0130558, upper bound: 0.0129087
IS_A2_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0130558, upper bound: 0.0128609
IS_A2_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0130558, upper bound: 0.0129087
IS_A2_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0130872, upper bound: 0.0126848
IS_A2_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0130872, upper bound: 0.0126848
IS_A2_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0130872, upper bound: 0.0126848
IS_A2_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0130872, upper bound: 0.0126848
IS_A2_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0131682, upper bound: 0.0128568
IS_A2_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0131682, upper bound: 0.0128568
IS_A2_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0131682, upper bound: 0.0128568
IS_A2_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0131682, upper bound: 0.0128568

## BFS IS instance: IS_A1_A1_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0080929, 0.0185533, 0.0080929, 0.0185533, -0.0104604, 0.0104604
1: -0.0039673, 0.0010577, -0.0039673, 0.0010577, -0.0050250, 0.0050250
2: 0.0034788, 0.0092297, 0.0034788, 0.0092297, -0.0057509, 0.0057509
3: -0.0010035, 0.0036382, -0.0010035, 0.0036382, -0.0046417, 0.0046417
4: -0.0049200, -0.0011302, -0.0049200, -0.0011302, -0.0031663, 0.0031663
5: -0.0005518, 0.0040088, -0.0005518, 0.0040088, -0.0045607, 0.0045607
6: -0.0066797, 0.0007150, -0.0066797, 0.0007150, -0.0073947, 0.0073947
7: -0.0242347, -0.0024411, -0.0242347, -0.0024411, -0.0165468, 0.0165468
8: 0.9733176, 0.9943494, 0.9733176, 0.9943494, -0.0210317, 0.0210317
9: -0.0063017, 0.0082136, -0.0063017, 0.0082136, -0.0125206, 0.0125206

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_A1_A1_B1_B1_B1_A1

### Relational analysis result of IS_A1_A1_A1_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0110390, upper bound: 0.0120674
time: 1.07 seconds

## Relational analysis of IS_A1_A1_A1_B1_B1_B1_A2

### Relational analysis result of IS_A1_A1_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126328, upper bound: 0.0126161
time: 1.64 seconds

## BFS IS instance: IS_A1_A1_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0080929, 0.0185533, 0.0082565, 0.0185611, -0.0104682, 0.0102969
1: -0.0039673, 0.0010577, -0.0039485, 0.0010368, -0.0050041, 0.0050062
2: 0.0034788, 0.0092297, 0.0034755, 0.0091027, -0.0056238, 0.0057543
3: -0.0010035, 0.0036382, -0.0008895, 0.0036775, -0.0046810, 0.0045278
4: -0.0049200, -0.0011302, -0.0049572, -0.0011940, -0.0031361, 0.0032362
5: -0.0005518, 0.0040088, -0.0005470, 0.0039456, -0.0044974, 0.0045558
6: -0.0066797, 0.0007150, -0.0066028, 0.0008008, -0.0074805, 0.0073177
7: -0.0242347, -0.0024411, -0.0244393, -0.0027993, -0.0164269, 0.0169889
8: 0.9733176, 0.9943494, 0.9731785, 0.9940299, -0.0207123, 0.0211709
9: -0.0063017, 0.0082136, -0.0060726, 0.0083402, -0.0127398, 0.0123924

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_A1_A1_B1_B1_B2_A1

### Relational analysis result of IS_A1_A1_A1_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0110390, upper bound: 0.0120789
time: 1.03 seconds

## Relational analysis of IS_A1_A1_A1_B1_B1_B2_A2

### Relational analysis result of IS_A1_A1_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126328, upper bound: 0.0126255
time: 1.26 seconds

## BFS IS instance: IS_A1_A1_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0080929, 0.0185533, 0.0068087, 0.0185524, -0.0104595, 0.0117446
1: -0.0039673, 0.0010577, -0.0042306, 0.0012934, -0.0052607, 0.0052883
2: 0.0034788, 0.0092297, 0.0034796, 0.0101545, -0.0066756, 0.0057501
3: -0.0010035, 0.0036382, -0.0015647, 0.0036803, -0.0046838, 0.0052029
4: -0.0049200, -0.0011302, -0.0049252, -0.0009405, -0.0034122, 0.0031946
5: -0.0005518, 0.0040088, -0.0005544, 0.0045500, -0.0051019, 0.0045632
6: -0.0066797, 0.0007150, -0.0068399, 0.0010449, -0.0077246, 0.0075549
7: -0.0242347, -0.0024411, -0.0242427, -0.0013184, -0.0179779, 0.0167367
8: 0.9733176, 0.9943494, 0.9732838, 0.9955425, -0.0222248, 0.0210656
9: -0.0063017, 0.0082136, -0.0070691, 0.0082311, -0.0126087, 0.0134655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_A1_A1_B1_B2_B1_A1

### Relational analysis result of IS_A1_A1_A1_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0109168, upper bound: 0.0120224
time: 1.13 seconds

## Relational analysis of IS_A1_A1_A1_B1_B2_B1_A2

### Relational analysis result of IS_A1_A1_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125937, upper bound: 0.0126532
time: 1.15 seconds

## BFS IS instance: IS_A1_A1_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0080929, 0.0185533, 0.0069627, 0.0185597, -0.0104667, 0.0115906
1: -0.0039673, 0.0010577, -0.0042152, 0.0012591, -0.0052264, 0.0052729
2: 0.0034788, 0.0092297, 0.0034765, 0.0100375, -0.0065587, 0.0057533
3: -0.0010035, 0.0036382, -0.0014544, 0.0037225, -0.0047260, 0.0050926
4: -0.0049200, -0.0011302, -0.0049666, -0.0009945, -0.0035872, 0.0032657
5: -0.0005518, 0.0040088, -0.0005493, 0.0044909, -0.0050428, 0.0045581
6: -0.0066797, 0.0007150, -0.0067734, 0.0011315, -0.0078112, 0.0074884
7: -0.0242347, -0.0024411, -0.0244683, -0.0016217, -0.0201076, 0.0171754
8: 0.9733176, 0.9943494, 0.9731277, 0.9952570, -0.0219394, 0.0212216
9: -0.0063017, 0.0082136, -0.0068703, 0.0083713, -0.0128313, 0.0141035

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_A1_A1_B1_B2_B2_A1

### Relational analysis result of IS_A1_A1_A1_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0109168, upper bound: 0.0120299
time: 1.53 seconds

## Relational analysis of IS_A1_A1_A1_B1_B2_B2_A2

### Relational analysis result of IS_A1_A1_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125937, upper bound: 0.0126659
time: 1.11 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0080929, 0.0185533, 0.0083055, 0.0187768, -0.0106839, 0.0102479
1: -0.0039673, 0.0010577, -0.0039174, 0.0011571, -0.0051244, 0.0049751
2: 0.0034788, 0.0092297, 0.0034030, 0.0090757, -0.0055968, 0.0058267
3: -0.0010035, 0.0036382, -0.0009109, 0.0036682, -0.0046717, 0.0045491
4: -0.0049200, -0.0011302, -0.0051203, -0.0011813, -0.0033633, 0.0034185
5: -0.0005518, 0.0040088, -0.0006129, 0.0039166, -0.0044685, 0.0046217
6: -0.0066797, 0.0007150, -0.0066209, 0.0008647, -0.0075444, 0.0073358
7: -0.0242347, -0.0024411, -0.0254298, -0.0027351, -0.0188232, 0.0180696
8: 0.9733176, 0.9943494, 0.9722265, 0.9940638, -0.0207462, 0.0221229
9: -0.0063017, 0.0082136, -0.0061136, 0.0089945, -0.0134521, 0.0132312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_A1_A1_B2_B1_B1_A1

### Relational analysis result of IS_A1_A1_A1_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0110390, upper bound: 0.0123976
time: 1.05 seconds

## Relational analysis of IS_A1_A1_A1_B2_B1_B1_A2

### Relational analysis result of IS_A1_A1_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126245, upper bound: 0.0128809
time: 1.17 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0080929, 0.0185533, 0.0084286, 0.0187860, -0.0106931, 0.0101248
1: -0.0039673, 0.0010577, -0.0039089, 0.0011429, -0.0051102, 0.0049666
2: 0.0034788, 0.0092297, 0.0033987, 0.0089764, -0.0054975, 0.0058310
3: -0.0010035, 0.0036382, -0.0008197, 0.0037132, -0.0047167, 0.0044579
4: -0.0049200, -0.0011302, -0.0051675, -0.0012301, -0.0033415, 0.0034966
5: -0.0005518, 0.0040088, -0.0006096, 0.0038723, -0.0044241, 0.0046185
6: -0.0066797, 0.0007150, -0.0065548, 0.0009793, -0.0076590, 0.0072698
7: -0.0242347, -0.0024411, -0.0256950, -0.0030151, -0.0187112, 0.0185493
8: 0.9733176, 0.9943494, 0.9720360, 0.9938253, -0.0205077, 0.0223134
9: -0.0063017, 0.0082136, -0.0059282, 0.0091599, -0.0137060, 0.0131351

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_A1_A1_B2_B1_B2_A1

### Relational analysis result of IS_A1_A1_A1_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0110390, upper bound: 0.0124180
time: 1.29 seconds

## Relational analysis of IS_A1_A1_A1_B2_B1_B2_A2

### Relational analysis result of IS_A1_A1_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126245, upper bound: 0.0128991
time: 1.14 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0080929, 0.0185533, 0.0069434, 0.0187820, -0.0106891, 0.0116099
1: -0.0039673, 0.0010577, -0.0041974, 0.0013851, -0.0053524, 0.0052550
2: 0.0034788, 0.0092297, 0.0034013, 0.0100584, -0.0065795, 0.0058284
3: -0.0010035, 0.0036382, -0.0015091, 0.0037173, -0.0047208, 0.0051473
4: -0.0049200, -0.0011302, -0.0051387, -0.0009777, -0.0036143, 0.0034547
5: -0.0005518, 0.0040088, -0.0006160, 0.0044893, -0.0050411, 0.0046249
6: -0.0066797, 0.0007150, -0.0067939, 0.0012106, -0.0078903, 0.0075089
7: -0.0242347, -0.0024411, -0.0255066, -0.0015269, -0.0202204, 0.0182912
8: 0.9733176, 0.9943494, 0.9721356, 0.9953537, -0.0220361, 0.0222138
9: -0.0063017, 0.0082136, -0.0069351, 0.0090564, -0.0135809, 0.0142013

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_A1_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_A1_A1_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0109168, upper bound: 0.0124020
time: 1.14 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2_B1_A2

### Relational analysis result of IS_A1_A1_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125885, upper bound: 0.0129376
time: 1.18 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0080929, 0.0185533, 0.0070348, 0.0187912, -0.0106983, 0.0115185
1: -0.0039673, 0.0010577, -0.0041962, 0.0013731, -0.0053404, 0.0052539
2: 0.0034788, 0.0092297, 0.0033970, 0.0099868, -0.0065080, 0.0058328
3: -0.0010035, 0.0036382, -0.0014337, 0.0037640, -0.0047675, 0.0050720
4: -0.0049200, -0.0011302, -0.0051883, -0.0010215, -0.0035961, 0.0035372
5: -0.0005518, 0.0040088, -0.0006126, 0.0044600, -0.0050118, 0.0046214
6: -0.0066797, 0.0007150, -0.0067363, 0.0013294, -0.0080091, 0.0074512
7: -0.0242347, -0.0024411, -0.0257798, -0.0017740, -0.0201215, 0.0187952
8: 0.9733176, 0.9943494, 0.9719387, 0.9951462, -0.0218285, 0.0224106
9: -0.0063017, 0.0082136, -0.0067789, 0.0092272, -0.0138439, 0.0141293

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_A1_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_A1_A1_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0109168, upper bound: 0.0124276
time: 1.47 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2_B2_A2

### Relational analysis result of IS_A1_A1_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125885, upper bound: 0.0129595
time: 1.15 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0068087, 0.0185524, 0.0080929, 0.0185533, -0.0117446, 0.0104595
1: -0.0042306, 0.0012934, -0.0039673, 0.0010577, -0.0052883, 0.0052607
2: 0.0034796, 0.0101545, 0.0034788, 0.0092297, -0.0057501, 0.0066756
3: -0.0015647, 0.0036803, -0.0010035, 0.0036382, -0.0052029, 0.0046838
4: -0.0049252, -0.0009405, -0.0049200, -0.0011302, -0.0031946, 0.0034122
5: -0.0005544, 0.0045500, -0.0005518, 0.0040088, -0.0045632, 0.0051019
6: -0.0068399, 0.0010449, -0.0066797, 0.0007150, -0.0075549, 0.0077246
7: -0.0242427, -0.0013184, -0.0242347, -0.0024411, -0.0167367, 0.0179779
8: 0.9732838, 0.9955425, 0.9733176, 0.9943494, -0.0210656, 0.0222248
9: -0.0070691, 0.0082311, -0.0063017, 0.0082136, -0.0134655, 0.0126087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_A1_A2_B1_B1_B1_B1

### Relational analysis result of IS_A1_A1_A2_B1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120261, upper bound: 0.0108956
time: 1.49 seconds

## Relational analysis of IS_A1_A1_A2_B1_B1_B1_B2

### Relational analysis result of IS_A1_A1_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126644, upper bound: 0.0125937
time: 1.21 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0068087, 0.0185524, 0.0068087, 0.0185524, -0.0117437, 0.0117437
1: -0.0042306, 0.0012934, -0.0042306, 0.0012934, -0.0055240, 0.0055240
2: 0.0034796, 0.0101545, 0.0034796, 0.0101545, -0.0066749, 0.0066749
3: -0.0015647, 0.0036803, -0.0015647, 0.0036803, -0.0052450, 0.0052450
4: -0.0049252, -0.0009405, -0.0049252, -0.0009405, -0.0033730, 0.0033730
5: -0.0005544, 0.0045500, -0.0005544, 0.0045500, -0.0051044, 0.0051044
6: -0.0068399, 0.0010449, -0.0068399, 0.0010449, -0.0078848, 0.0078848
7: -0.0242427, -0.0013184, -0.0242427, -0.0013184, -0.0176704, 0.0176704
8: 0.9732838, 0.9955425, 0.9732838, 0.9955425, -0.0222587, 0.0222587
9: -0.0070691, 0.0082311, -0.0070691, 0.0082311, -0.0133348, 0.0133348

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_A1_A2_B1_B1_B2_A1

### Relational analysis result of IS_A1_A1_A2_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0108986, upper bound: 0.0120061
time: 1.09 seconds

## Relational analysis of IS_A1_A1_A2_B1_B1_B2_A2

### Relational analysis result of IS_A1_A1_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126644, upper bound: 0.0125937
time: 1.60 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0068087, 0.0185524, 0.0082565, 0.0185611, -0.0117524, 0.0102959
1: -0.0042306, 0.0012934, -0.0039485, 0.0010368, -0.0052675, 0.0052419
2: 0.0034796, 0.0101545, 0.0034755, 0.0091027, -0.0056231, 0.0066790
3: -0.0015647, 0.0036803, -0.0008895, 0.0036775, -0.0052421, 0.0045698
4: -0.0049252, -0.0009405, -0.0049572, -0.0011940, -0.0031644, 0.0034820
5: -0.0005544, 0.0045500, -0.0005470, 0.0039456, -0.0045000, 0.0050970
6: -0.0068399, 0.0010449, -0.0066028, 0.0008008, -0.0076408, 0.0076477
7: -0.0242427, -0.0013184, -0.0244393, -0.0027993, -0.0166168, 0.0184200
8: 0.9732838, 0.9955425, 0.9731785, 0.9940299, -0.0207462, 0.0223640
9: -0.0070691, 0.0082311, -0.0060726, 0.0083402, -0.0136848, 0.0124806

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_A1_A2_B1_B2_B1_B1

### Relational analysis result of IS_A1_A1_A2_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120267, upper bound: 0.0110270
time: 1.12 seconds

## Relational analysis of IS_A1_A1_A2_B1_B2_B1_B2

### Relational analysis result of IS_A1_A1_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126644, upper bound: 0.0125802
time: 1.19 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0068087, 0.0185524, 0.0069627, 0.0185597, -0.0117509, 0.0115896
1: -0.0042306, 0.0012934, -0.0042152, 0.0012591, -0.0054897, 0.0055085
2: 0.0034796, 0.0101545, 0.0034765, 0.0100375, -0.0065579, 0.0066780
3: -0.0015647, 0.0036803, -0.0014544, 0.0037225, -0.0052872, 0.0051347
4: -0.0049252, -0.0009405, -0.0049666, -0.0009945, -0.0035916, 0.0034453
5: -0.0005544, 0.0045500, -0.0005493, 0.0044909, -0.0050453, 0.0050993
6: -0.0068399, 0.0010449, -0.0067734, 0.0011315, -0.0079714, 0.0078184
7: -0.0242427, -0.0013184, -0.0244683, -0.0016217, -0.0202772, 0.0181135
8: 0.9732838, 0.9955425, 0.9731277, 0.9952570, -0.0219733, 0.0224147
9: -0.0070691, 0.0082311, -0.0068703, 0.0083713, -0.0135576, 0.0141281

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_A1_A2_B1_B2_B2_A1

### Relational analysis result of IS_A1_A1_A2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0108986, upper bound: 0.0120104
time: 1.29 seconds

## Relational analysis of IS_A1_A1_A2_B1_B2_B2_A2

### Relational analysis result of IS_A1_A1_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126644, upper bound: 0.0125802
time: 1.21 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0068087, 0.0185524, 0.0083055, 0.0187768, -0.0119681, 0.0102469
1: -0.0042306, 0.0012934, -0.0039174, 0.0011571, -0.0053878, 0.0052108
2: 0.0034796, 0.0101545, 0.0034030, 0.0090757, -0.0055960, 0.0067514
3: -0.0015647, 0.0036803, -0.0009109, 0.0036682, -0.0052328, 0.0045912
4: -0.0049252, -0.0009405, -0.0051203, -0.0011813, -0.0033838, 0.0036644
5: -0.0005544, 0.0045500, -0.0006129, 0.0039166, -0.0044710, 0.0051629
6: -0.0068399, 0.0010449, -0.0066209, 0.0008647, -0.0077046, 0.0076658
7: -0.0242427, -0.0013184, -0.0254298, -0.0027351, -0.0189205, 0.0195007
8: 0.9732838, 0.9955425, 0.9722265, 0.9940638, -0.0207800, 0.0233160
9: -0.0070691, 0.0082311, -0.0061136, 0.0089945, -0.0143971, 0.0132959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_A1_A2_B2_B1_B1_A1

### Relational analysis result of IS_A1_A1_A2_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0108986, upper bound: 0.0123413
time: 1.07 seconds

## Relational analysis of IS_A1_A1_A2_B2_B1_B1_A2

### Relational analysis result of IS_A1_A1_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126498, upper bound: 0.0128285
time: 1.15 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0068087, 0.0185524, 0.0069434, 0.0187820, -0.0119732, 0.0116090
1: -0.0042306, 0.0012934, -0.0041974, 0.0013851, -0.0056158, 0.0054907
2: 0.0034796, 0.0101545, 0.0034013, 0.0100584, -0.0065787, 0.0067532
3: -0.0015647, 0.0036803, -0.0015091, 0.0037173, -0.0052820, 0.0051894
4: -0.0049252, -0.0009405, -0.0051387, -0.0009777, -0.0036186, 0.0036339
5: -0.0005544, 0.0045500, -0.0006160, 0.0044893, -0.0050437, 0.0051661
6: -0.0068399, 0.0010449, -0.0067939, 0.0012106, -0.0080505, 0.0078388
7: -0.0242427, -0.0013184, -0.0255066, -0.0015269, -0.0203928, 0.0192324
8: 0.9732838, 0.9955425, 0.9721356, 0.9953537, -0.0220699, 0.0234069
9: -0.0070691, 0.0082311, -0.0069351, 0.0090564, -0.0143011, 0.0142254

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_A1_A2_B2_B1_B2_A1

### Relational analysis result of IS_A1_A1_A2_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0108986, upper bound: 0.0123413
time: 1.20 seconds

## Relational analysis of IS_A1_A1_A2_B2_B1_B2_A2

### Relational analysis result of IS_A1_A1_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126498, upper bound: 0.0128285
time: 1.17 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0068087, 0.0185524, 0.0084286, 0.0187860, -0.0119773, 0.0101238
1: -0.0042306, 0.0012934, -0.0039089, 0.0011429, -0.0053735, 0.0052023
2: 0.0034796, 0.0101545, 0.0033987, 0.0089764, -0.0054968, 0.0067557
3: -0.0015647, 0.0036803, -0.0008197, 0.0037132, -0.0052779, 0.0045000
4: -0.0049252, -0.0009405, -0.0051675, -0.0012301, -0.0033621, 0.0037425
5: -0.0005544, 0.0045500, -0.0006096, 0.0038723, -0.0044267, 0.0051597
6: -0.0068399, 0.0010449, -0.0065548, 0.0009793, -0.0078192, 0.0075997
7: -0.0242427, -0.0013184, -0.0256950, -0.0030151, -0.0188085, 0.0199803
8: 0.9732838, 0.9955425, 0.9720360, 0.9938253, -0.0205415, 0.0235065
9: -0.0070691, 0.0082311, -0.0059282, 0.0091599, -0.0146509, 0.0131998

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_A1_A2_B2_B2_B1_A1

### Relational analysis result of IS_A1_A1_A2_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0108986, upper bound: 0.0123586
time: 1.12 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2_B1_A2

### Relational analysis result of IS_A1_A1_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126498, upper bound: 0.0128323
time: 1.16 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0068087, 0.0185524, 0.0070348, 0.0187912, -0.0119825, 0.0115176
1: -0.0042306, 0.0012934, -0.0041962, 0.0013731, -0.0056038, 0.0054896
2: 0.0034796, 0.0101545, 0.0033970, 0.0099868, -0.0065072, 0.0067575
3: -0.0015647, 0.0036803, -0.0014337, 0.0037640, -0.0053287, 0.0051140
4: -0.0049252, -0.0009405, -0.0051883, -0.0010215, -0.0036009, 0.0037144
5: -0.0005544, 0.0045500, -0.0006126, 0.0044600, -0.0050143, 0.0051626
6: -0.0068399, 0.0010449, -0.0067363, 0.0013294, -0.0081693, 0.0077812
7: -0.0242427, -0.0013184, -0.0257798, -0.0017740, -0.0202912, 0.0197146
8: 0.9732838, 0.9955425, 0.9719387, 0.9951462, -0.0218624, 0.0236037
9: -0.0070691, 0.0082311, -0.0067789, 0.0092272, -0.0145634, 0.0141538

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_A1_A2_B2_B2_B2_A1

### Relational analysis result of IS_A1_A1_A2_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0108986, upper bound: 0.0123586
time: 1.36 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2_B2_A2

### Relational analysis result of IS_A1_A1_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126498, upper bound: 0.0128323
time: 1.58 seconds

## BFS IS instance: IS_A1_A2_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0083055, 0.0187768, 0.0080929, 0.0185533, -0.0102479, 0.0106839
1: -0.0039174, 0.0011571, -0.0039673, 0.0010577, -0.0049751, 0.0051244
2: 0.0034030, 0.0090757, 0.0034788, 0.0092297, -0.0058267, 0.0055968
3: -0.0009109, 0.0036682, -0.0010035, 0.0036382, -0.0045491, 0.0046717
4: -0.0051203, -0.0011813, -0.0049200, -0.0011302, -0.0034185, 0.0033633
5: -0.0006129, 0.0039166, -0.0005518, 0.0040088, -0.0046217, 0.0044685
6: -0.0066209, 0.0008647, -0.0066797, 0.0007150, -0.0073358, 0.0075444
7: -0.0254298, -0.0027351, -0.0242347, -0.0024411, -0.0180696, 0.0188232
8: 0.9722265, 0.9940638, 0.9733176, 0.9943494, -0.0221229, 0.0207462
9: -0.0061136, 0.0089945, -0.0063017, 0.0082136, -0.0132312, 0.0134521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_A2_A1_B1_B1_B1_B1

### Relational analysis result of IS_A1_A2_A1_B1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0123984, upper bound: 0.0110390
time: 1.09 seconds

## Relational analysis of IS_A1_A2_A1_B1_B1_B1_B2

### Relational analysis result of IS_A1_A2_A1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128846, upper bound: 0.0126054
time: 1.15 seconds

## BFS IS instance: IS_A1_A2_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0083055, 0.0187768, 0.0082565, 0.0185611, -0.0102557, 0.0105203
1: -0.0039174, 0.0011571, -0.0039485, 0.0010368, -0.0049543, 0.0051057
2: 0.0034030, 0.0090757, 0.0034755, 0.0091027, -0.0056996, 0.0056002
3: -0.0009109, 0.0036682, -0.0008895, 0.0036775, -0.0045883, 0.0045577
4: -0.0051203, -0.0011813, -0.0049572, -0.0011940, -0.0033883, 0.0034220
5: -0.0006129, 0.0039166, -0.0005470, 0.0039456, -0.0045585, 0.0044636
6: -0.0066209, 0.0008647, -0.0066028, 0.0008008, -0.0074217, 0.0074674
7: -0.0254298, -0.0027351, -0.0244393, -0.0027993, -0.0179497, 0.0191088
8: 0.9722265, 0.9940638, 0.9731785, 0.9940299, -0.0218034, 0.0208853
9: -0.0061136, 0.0089945, -0.0060726, 0.0083402, -0.0134161, 0.0133240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_A2_A1_B1_B1_B2_B1

### Relational analysis result of IS_A1_A2_A1_B1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0123984, upper bound: 0.0111862
time: 1.12 seconds

## Relational analysis of IS_A1_A2_A1_B1_B1_B2_B2

### Relational analysis result of IS_A1_A2_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128846, upper bound: 0.0126152
time: 1.14 seconds

## BFS IS instance: IS_A1_A2_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0083055, 0.0187768, 0.0068087, 0.0185524, -0.0102469, 0.0119681
1: -0.0039174, 0.0011571, -0.0042306, 0.0012934, -0.0052108, 0.0053878
2: 0.0034030, 0.0090757, 0.0034796, 0.0101545, -0.0067514, 0.0055960
3: -0.0009109, 0.0036682, -0.0015647, 0.0036803, -0.0045912, 0.0052328
4: -0.0051203, -0.0011813, -0.0049252, -0.0009405, -0.0036644, 0.0033838
5: -0.0006129, 0.0039166, -0.0005544, 0.0045500, -0.0051629, 0.0044710
6: -0.0066209, 0.0008647, -0.0068399, 0.0010449, -0.0076658, 0.0077046
7: -0.0254298, -0.0027351, -0.0242427, -0.0013184, -0.0195007, 0.0189205
8: 0.9722265, 0.9940638, 0.9732838, 0.9955425, -0.0233160, 0.0207800
9: -0.0061136, 0.0089945, -0.0070691, 0.0082311, -0.0132959, 0.0143971

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_A2_A1_B1_B2_B1_B1

### Relational analysis result of IS_A1_A2_A1_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0123413, upper bound: 0.0109371
time: 1.13 seconds

## Relational analysis of IS_A1_A2_A1_B1_B2_B1_B2

### Relational analysis result of IS_A1_A2_A1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128285, upper bound: 0.0126380
time: 1.28 seconds

## BFS IS instance: IS_A1_A2_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0083055, 0.0187768, 0.0069627, 0.0185597, -0.0102542, 0.0118140
1: -0.0039174, 0.0011571, -0.0042152, 0.0012591, -0.0051765, 0.0053723
2: 0.0034030, 0.0090757, 0.0034765, 0.0100375, -0.0066345, 0.0055992
3: -0.0009109, 0.0036682, -0.0014544, 0.0037225, -0.0046334, 0.0051226
4: -0.0051203, -0.0011813, -0.0049666, -0.0009945, -0.0038252, 0.0034451
5: -0.0006129, 0.0039166, -0.0005493, 0.0044909, -0.0051038, 0.0044659
6: -0.0066209, 0.0008647, -0.0067734, 0.0011315, -0.0077524, 0.0076381
7: -0.0254298, -0.0027351, -0.0244683, -0.0016217, -0.0214403, 0.0192364
8: 0.9722265, 0.9940638, 0.9731277, 0.9952570, -0.0230305, 0.0209361
9: -0.0061136, 0.0089945, -0.0068703, 0.0083713, -0.0134910, 0.0149902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_A2_A1_B1_B2_B2_A1

### Relational analysis result of IS_A1_A2_A1_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0118908, upper bound: 0.0120315
time: 1.58 seconds

## Relational analysis of IS_A1_A2_A1_B1_B2_B2_A2

### Relational analysis result of IS_A1_A2_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128285, upper bound: 0.0126465
time: 1.68 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0083055, 0.0187768, 0.0083055, 0.0187768, -0.0104713, 0.0104713
1: -0.0039174, 0.0011571, -0.0039174, 0.0011571, -0.0050746, 0.0050746
2: 0.0034030, 0.0090757, 0.0034030, 0.0090757, -0.0056726, 0.0056726
3: -0.0009109, 0.0036682, -0.0009109, 0.0036682, -0.0045791, 0.0045791
4: -0.0051203, -0.0011813, -0.0051203, -0.0011813, -0.0034988, 0.0034988
5: -0.0006129, 0.0039166, -0.0006129, 0.0039166, -0.0045295, 0.0045295
6: -0.0066209, 0.0008647, -0.0066209, 0.0008647, -0.0074855, 0.0074855
7: -0.0254298, -0.0027351, -0.0254298, -0.0027351, -0.0195166, 0.0195166
8: 0.9722265, 0.9940638, 0.9722265, 0.9940638, -0.0218373, 0.0218373
9: -0.0061136, 0.0089945, -0.0061136, 0.0089945, -0.0137772, 0.0137772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_A2_A1_B2_B1_B1_A1

### Relational analysis result of IS_A1_A2_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123432, upper bound: 0.0125273
time: 1.15 seconds

## Relational analysis of IS_A1_A2_A1_B2_B1_B1_A2

### Relational analysis result of IS_A1_A2_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129646, upper bound: 0.0127449
time: 1.20 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0083055, 0.0187768, 0.0084286, 0.0187860, -0.0104805, 0.0103482
1: -0.0039174, 0.0011571, -0.0039089, 0.0011429, -0.0050603, 0.0050660
2: 0.0034030, 0.0090757, 0.0033987, 0.0089764, -0.0055734, 0.0056769
3: -0.0009109, 0.0036682, -0.0008197, 0.0037132, -0.0046241, 0.0044879
4: -0.0051203, -0.0011813, -0.0051675, -0.0012301, -0.0034719, 0.0035666
5: -0.0006129, 0.0039166, -0.0006096, 0.0038723, -0.0044852, 0.0045263
6: -0.0066209, 0.0008647, -0.0065548, 0.0009793, -0.0076002, 0.0074195
7: -0.0254298, -0.0027351, -0.0256950, -0.0030151, -0.0193615, 0.0198710
8: 0.9722265, 0.9940638, 0.9720360, 0.9938253, -0.0215988, 0.0220278
9: -0.0061136, 0.0089945, -0.0059282, 0.0091599, -0.0140035, 0.0136636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A1

### Relational analysis result of IS_A1_A2_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123432, upper bound: 0.0125314
time: 1.15 seconds

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A2

### Relational analysis result of IS_A1_A2_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129646, upper bound: 0.0127518
time: 1.56 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0083055, 0.0187768, 0.0069434, 0.0187820, -0.0104765, 0.0118334
1: -0.0039174, 0.0011571, -0.0041974, 0.0013851, -0.0053026, 0.0053545
2: 0.0034030, 0.0090757, 0.0034013, 0.0100584, -0.0066553, 0.0056744
3: -0.0009109, 0.0036682, -0.0015091, 0.0037173, -0.0046282, 0.0051773
4: -0.0051203, -0.0011813, -0.0051387, -0.0009777, -0.0037487, 0.0035282
5: -0.0006129, 0.0039166, -0.0006160, 0.0044893, -0.0051022, 0.0045327
6: -0.0066209, 0.0008647, -0.0067939, 0.0012106, -0.0078315, 0.0076586
7: -0.0254298, -0.0027351, -0.0255066, -0.0015269, -0.0209066, 0.0196633
8: 0.9722265, 0.9940638, 0.9721356, 0.9953537, -0.0231272, 0.0219282
9: -0.0061136, 0.0089945, -0.0069351, 0.0090564, -0.0138783, 0.0147443

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_A2_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_A2_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123131, upper bound: 0.0125544
time: 1.15 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2_B1_A2

### Relational analysis result of IS_A1_A2_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129180, upper bound: 0.0127713
time: 1.13 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0083055, 0.0187768, 0.0070348, 0.0187912, -0.0104858, 0.0117420
1: -0.0039174, 0.0011571, -0.0041962, 0.0013731, -0.0052906, 0.0053533
2: 0.0034030, 0.0090757, 0.0033970, 0.0099868, -0.0065838, 0.0056787
3: -0.0009109, 0.0036682, -0.0014337, 0.0037640, -0.0046749, 0.0051019
4: -0.0051203, -0.0011813, -0.0051883, -0.0010215, -0.0037230, 0.0035955
5: -0.0006129, 0.0039166, -0.0006126, 0.0044600, -0.0050729, 0.0045292
6: -0.0066209, 0.0008647, -0.0067363, 0.0013294, -0.0079503, 0.0076009
7: -0.0254298, -0.0027351, -0.0257798, -0.0017740, -0.0207522, 0.0200182
8: 0.9722265, 0.9940638, 0.9719387, 0.9951462, -0.0229197, 0.0221251
9: -0.0061136, 0.0089945, -0.0067789, 0.0092272, -0.0140993, 0.0146451

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_A2_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123131, upper bound: 0.0125577
time: 1.11 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_A2

### Relational analysis result of IS_A1_A2_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129180, upper bound: 0.0127786
time: 1.16 seconds

## BFS IS instance: IS_A1_A2_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0069434, 0.0187820, 0.0080929, 0.0185533, -0.0116099, 0.0106891
1: -0.0041974, 0.0013851, -0.0039673, 0.0010577, -0.0052550, 0.0053524
2: 0.0034013, 0.0100584, 0.0034788, 0.0092297, -0.0058284, 0.0065795
3: -0.0015091, 0.0037173, -0.0010035, 0.0036382, -0.0051473, 0.0047208
4: -0.0051387, -0.0009777, -0.0049200, -0.0011302, -0.0034547, 0.0036143
5: -0.0006160, 0.0044893, -0.0005518, 0.0040088, -0.0046249, 0.0050411
6: -0.0067939, 0.0012106, -0.0066797, 0.0007150, -0.0075089, 0.0078903
7: -0.0255066, -0.0015269, -0.0242347, -0.0024411, -0.0182912, 0.0202204
8: 0.9721356, 0.9953537, 0.9733176, 0.9943494, -0.0222138, 0.0220361
9: -0.0069351, 0.0090564, -0.0063017, 0.0082136, -0.0142013, 0.0135809

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_A2_A2_B1_B1_B1_B1

### Relational analysis result of IS_A1_A2_A2_B1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0124020, upper bound: 0.0108956
time: 1.21 seconds

## Relational analysis of IS_A1_A2_A2_B1_B1_B1_B2

### Relational analysis result of IS_A1_A2_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129421, upper bound: 0.0125885
time: 1.35 seconds

## BFS IS instance: IS_A1_A2_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0069434, 0.0187820, 0.0068087, 0.0185524, -0.0116090, 0.0119732
1: -0.0041974, 0.0013851, -0.0042306, 0.0012934, -0.0054907, 0.0056158
2: 0.0034013, 0.0100584, 0.0034796, 0.0101545, -0.0067532, 0.0065787
3: -0.0015091, 0.0037173, -0.0015647, 0.0036803, -0.0051894, 0.0052820
4: -0.0051387, -0.0009777, -0.0049252, -0.0009405, -0.0036339, 0.0036186
5: -0.0006160, 0.0044893, -0.0005544, 0.0045500, -0.0051661, 0.0050437
6: -0.0067939, 0.0012106, -0.0068399, 0.0010449, -0.0078388, 0.0080505
7: -0.0255066, -0.0015269, -0.0242427, -0.0013184, -0.0192324, 0.0203928
8: 0.9721356, 0.9953537, 0.9732838, 0.9955425, -0.0234069, 0.0220699
9: -0.0069351, 0.0090564, -0.0070691, 0.0082311, -0.0142254, 0.0143011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_A2_A2_B1_B1_B2_B1

### Relational analysis result of IS_A1_A2_A2_B1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0124020, upper bound: 0.0108956
time: 1.15 seconds

## Relational analysis of IS_A1_A2_A2_B1_B1_B2_B2

### Relational analysis result of IS_A1_A2_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129421, upper bound: 0.0125885
time: 1.15 seconds

## BFS IS instance: IS_A1_A2_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0069434, 0.0187820, 0.0082565, 0.0185611, -0.0116177, 0.0105255
1: -0.0041974, 0.0013851, -0.0039485, 0.0010368, -0.0052342, 0.0053337
2: 0.0034013, 0.0100584, 0.0034755, 0.0091027, -0.0057014, 0.0065829
3: -0.0015091, 0.0037173, -0.0008895, 0.0036775, -0.0051866, 0.0046068
4: -0.0051387, -0.0009777, -0.0049572, -0.0011940, -0.0034244, 0.0036730
5: -0.0006160, 0.0044893, -0.0005470, 0.0039456, -0.0045617, 0.0050363
6: -0.0067939, 0.0012106, -0.0066028, 0.0008008, -0.0075947, 0.0078134
7: -0.0255066, -0.0015269, -0.0244393, -0.0027993, -0.0181712, 0.0205060
8: 0.9721356, 0.9953537, 0.9731785, 0.9940299, -0.0218943, 0.0221752
9: -0.0069351, 0.0090564, -0.0060726, 0.0083402, -0.0143862, 0.0134528

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_A2_A2_B1_B2_B1_B1

### Relational analysis result of IS_A1_A2_A2_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0124020, upper bound: 0.0110270
time: 1.14 seconds

## Relational analysis of IS_A1_A2_A2_B1_B2_B1_B2

### Relational analysis result of IS_A1_A2_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129421, upper bound: 0.0125750
time: 1.64 seconds

## BFS IS instance: IS_A1_A2_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0069434, 0.0187820, 0.0069627, 0.0185597, -0.0116163, 0.0118192
1: -0.0041974, 0.0013851, -0.0042152, 0.0012591, -0.0054564, 0.0056003
2: 0.0034013, 0.0100584, 0.0034765, 0.0100375, -0.0066362, 0.0065819
3: -0.0015091, 0.0037173, -0.0014544, 0.0037225, -0.0052316, 0.0051717
4: -0.0051387, -0.0009777, -0.0049666, -0.0009945, -0.0038405, 0.0036792
5: -0.0006160, 0.0044893, -0.0005493, 0.0044909, -0.0051070, 0.0050386
6: -0.0067939, 0.0012106, -0.0067734, 0.0011315, -0.0079254, 0.0079841
7: -0.0255066, -0.0015269, -0.0244683, -0.0016217, -0.0216717, 0.0207092
8: 0.9721356, 0.9953537, 0.9731277, 0.9952570, -0.0231214, 0.0222260
9: -0.0069351, 0.0090564, -0.0068703, 0.0083713, -0.0144198, 0.0150552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_A2_A2_B1_B2_B2_A1

### Relational analysis result of IS_A1_A2_A2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120000, upper bound: 0.0120122
time: 1.26 seconds

## Relational analysis of IS_A1_A2_A2_B1_B2_B2_A2

### Relational analysis result of IS_A1_A2_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129421, upper bound: 0.0125750
time: 1.89 seconds

## BFS IS instance: IS_A1_A2_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0069434, 0.0187820, 0.0083055, 0.0187768, -0.0118334, 0.0104765
1: -0.0041974, 0.0013851, -0.0039174, 0.0011571, -0.0053545, 0.0053026
2: 0.0034013, 0.0100584, 0.0034030, 0.0090757, -0.0056744, 0.0066553
3: -0.0015091, 0.0037173, -0.0009109, 0.0036682, -0.0051773, 0.0046282
4: -0.0051387, -0.0009777, -0.0051203, -0.0011813, -0.0035282, 0.0037487
5: -0.0006160, 0.0044893, -0.0006129, 0.0039166, -0.0045327, 0.0051022
6: -0.0067939, 0.0012106, -0.0066209, 0.0008647, -0.0076586, 0.0078315
7: -0.0255066, -0.0015269, -0.0254298, -0.0027351, -0.0196633, 0.0209066
8: 0.9721356, 0.9953537, 0.9722265, 0.9940638, -0.0219282, 0.0231272
9: -0.0069351, 0.0090564, -0.0061136, 0.0089945, -0.0147443, 0.0138783

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_A2_A2_B2_B1_B1_B1

### Relational analysis result of IS_A1_A2_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127519, upper bound: 0.0120928
time: 1.22 seconds

## Relational analysis of IS_A1_A2_A2_B2_B1_B1_B2

### Relational analysis result of IS_A1_A2_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130127, upper bound: 0.0127363
time: 1.22 seconds

## BFS IS instance: IS_A1_A2_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0069434, 0.0187820, 0.0069434, 0.0187820, -0.0118386, 0.0118386
1: -0.0041974, 0.0013851, -0.0041974, 0.0013851, -0.0055825, 0.0055825
2: 0.0034013, 0.0100584, 0.0034013, 0.0100584, -0.0066571, 0.0066571
3: -0.0015091, 0.0037173, -0.0015091, 0.0037173, -0.0052264, 0.0052264
4: -0.0051387, -0.0009777, -0.0051387, -0.0009777, -0.0037611, 0.0037611
5: -0.0006160, 0.0044893, -0.0006160, 0.0044893, -0.0051054, 0.0051054
6: -0.0067939, 0.0012106, -0.0067939, 0.0012106, -0.0080045, 0.0080045
7: -0.0255066, -0.0015269, -0.0255066, -0.0015269, -0.0211298, 0.0211298
8: 0.9721356, 0.9953537, 0.9721356, 0.9953537, -0.0232181, 0.0232181
9: -0.0069351, 0.0090564, -0.0069351, 0.0090564, -0.0148033, 0.0148033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_A2_A2_B2_B1_B2_A1

### Relational analysis result of IS_A1_A2_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124169, upper bound: 0.0125306
time: 1.47 seconds

## Relational analysis of IS_A1_A2_A2_B2_B1_B2_A2

### Relational analysis result of IS_A1_A2_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130127, upper bound: 0.0127363
time: 1.20 seconds

## BFS IS instance: IS_A1_A2_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0069434, 0.0187820, 0.0084286, 0.0187860, -0.0118426, 0.0103534
1: -0.0041974, 0.0013851, -0.0039089, 0.0011429, -0.0053403, 0.0052940
2: 0.0034013, 0.0100584, 0.0033987, 0.0089764, -0.0055751, 0.0066596
3: -0.0015091, 0.0037173, -0.0008197, 0.0037132, -0.0052223, 0.0045370
4: -0.0051387, -0.0009777, -0.0051675, -0.0012301, -0.0035013, 0.0038165
5: -0.0006160, 0.0044893, -0.0006096, 0.0038723, -0.0044883, 0.0050989
6: -0.0067939, 0.0012106, -0.0065548, 0.0009793, -0.0077732, 0.0077654
7: -0.0255066, -0.0015269, -0.0256950, -0.0030151, -0.0195081, 0.0212610
8: 0.9721356, 0.9953537, 0.9720360, 0.9938253, -0.0216897, 0.0233177
9: -0.0069351, 0.0090564, -0.0059282, 0.0091599, -0.0149707, 0.0137648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_A2_A2_B2_B2_B1_B1

### Relational analysis result of IS_A1_A2_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127519, upper bound: 0.0121065
time: 1.18 seconds

## Relational analysis of IS_A1_A2_A2_B2_B2_B1_B2

### Relational analysis result of IS_A1_A2_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130127, upper bound: 0.0127229
time: 1.17 seconds

## BFS IS instance: IS_A1_A2_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0069434, 0.0187820, 0.0070348, 0.0187912, -0.0118478, 0.0117472
1: -0.0041974, 0.0013851, -0.0041962, 0.0013731, -0.0055705, 0.0055814
2: 0.0034013, 0.0100584, 0.0033970, 0.0099868, -0.0065855, 0.0066614
3: -0.0015091, 0.0037173, -0.0014337, 0.0037640, -0.0052731, 0.0051510
4: -0.0051387, -0.0009777, -0.0051883, -0.0010215, -0.0037361, 0.0038284
5: -0.0006160, 0.0044893, -0.0006126, 0.0044600, -0.0050760, 0.0051019
6: -0.0067939, 0.0012106, -0.0067363, 0.0013294, -0.0081233, 0.0079469
7: -0.0255066, -0.0015269, -0.0257798, -0.0017740, -0.0209722, 0.0214839
8: 0.9721356, 0.9953537, 0.9719387, 0.9951462, -0.0230106, 0.0234150
9: -0.0069351, 0.0090564, -0.0067789, 0.0092272, -0.0150235, 0.0147049

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_A2_A2_B2_B2_B2_A1

### Relational analysis result of IS_A1_A2_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124169, upper bound: 0.0125207
time: 1.53 seconds

## Relational analysis of IS_A1_A2_A2_B2_B2_B2_A2

### Relational analysis result of IS_A1_A2_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130127, upper bound: 0.0127229
time: 1.57 seconds

## BFS IS instance: IS_A2_A1_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0082565, 0.0185611, 0.0080929, 0.0185533, -0.0102969, 0.0104682
1: -0.0039485, 0.0010368, -0.0039673, 0.0010577, -0.0050062, 0.0050041
2: 0.0034755, 0.0091027, 0.0034788, 0.0092297, -0.0057543, 0.0056238
3: -0.0008895, 0.0036775, -0.0010035, 0.0036382, -0.0045278, 0.0046810
4: -0.0049572, -0.0011940, -0.0049200, -0.0011302, -0.0032362, 0.0031361
5: -0.0005470, 0.0039456, -0.0005518, 0.0040088, -0.0045558, 0.0044974
6: -0.0066028, 0.0008008, -0.0066797, 0.0007150, -0.0073177, 0.0074805
7: -0.0244393, -0.0027993, -0.0242347, -0.0024411, -0.0169889, 0.0164269
8: 0.9731785, 0.9940299, 0.9733176, 0.9943494, -0.0211709, 0.0207123
9: -0.0060726, 0.0083402, -0.0063017, 0.0082136, -0.0123924, 0.0127398

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_A1_A1_B1_B1_B1_B1

### Relational analysis result of IS_A2_A1_A1_B1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120123, upper bound: 0.0109371
time: 1.08 seconds

## Relational analysis of IS_A2_A1_A1_B1_B1_B1_B2

### Relational analysis result of IS_A2_A1_A1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125802, upper bound: 0.0126003
time: 1.19 seconds

## BFS IS instance: IS_A2_A1_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0082565, 0.0185611, 0.0068087, 0.0185524, -0.0102959, 0.0117524
1: -0.0039485, 0.0010368, -0.0042306, 0.0012934, -0.0052419, 0.0052675
2: 0.0034755, 0.0091027, 0.0034796, 0.0101545, -0.0066790, 0.0056231
3: -0.0008895, 0.0036775, -0.0015647, 0.0036803, -0.0045698, 0.0052421
4: -0.0049572, -0.0011940, -0.0049252, -0.0009405, -0.0034820, 0.0031644
5: -0.0005470, 0.0039456, -0.0005544, 0.0045500, -0.0050970, 0.0045000
6: -0.0066028, 0.0008008, -0.0068399, 0.0010449, -0.0076477, 0.0076408
7: -0.0244393, -0.0027993, -0.0242427, -0.0013184, -0.0184200, 0.0166168
8: 0.9731785, 0.9940299, 0.9732838, 0.9955425, -0.0223640, 0.0207462
9: -0.0060726, 0.0083402, -0.0070691, 0.0082311, -0.0124806, 0.0136848

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_A1_A1_B1_B1_B2_A1

### Relational analysis result of IS_A2_A1_A1_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0110483, upper bound: 0.0120278
time: 1.40 seconds

## Relational analysis of IS_A2_A1_A1_B1_B1_B2_A2

### Relational analysis result of IS_A2_A1_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125802, upper bound: 0.0126644
time: 1.14 seconds

## BFS IS instance: IS_A2_A1_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0082565, 0.0185611, 0.0083055, 0.0187768, -0.0105203, 0.0102557
1: -0.0039485, 0.0010368, -0.0039174, 0.0011571, -0.0051057, 0.0049543
2: 0.0034755, 0.0091027, 0.0034030, 0.0090757, -0.0056002, 0.0056996
3: -0.0008895, 0.0036775, -0.0009109, 0.0036682, -0.0045577, 0.0045883
4: -0.0049572, -0.0011940, -0.0051203, -0.0011813, -0.0034220, 0.0033883
5: -0.0005470, 0.0039456, -0.0006129, 0.0039166, -0.0044636, 0.0045585
6: -0.0066028, 0.0008008, -0.0066209, 0.0008647, -0.0074674, 0.0074217
7: -0.0244393, -0.0027993, -0.0254298, -0.0027351, -0.0191088, 0.0179497
8: 0.9731785, 0.9940299, 0.9722265, 0.9940638, -0.0208853, 0.0218034
9: -0.0060726, 0.0083402, -0.0061136, 0.0089945, -0.0133240, 0.0134161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_A1_A1_B1_B2_B1_A1

### Relational analysis result of IS_A2_A1_A1_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0110483, upper bound: 0.0123770
time: 1.15 seconds

## Relational analysis of IS_A2_A1_A1_B1_B2_B1_A2

### Relational analysis result of IS_A2_A1_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125750, upper bound: 0.0128582
time: 1.20 seconds

## BFS IS instance: IS_A2_A1_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0082565, 0.0185611, 0.0069434, 0.0187820, -0.0105255, 0.0116177
1: -0.0039485, 0.0010368, -0.0041974, 0.0013851, -0.0053337, 0.0052342
2: 0.0034755, 0.0091027, 0.0034013, 0.0100584, -0.0065829, 0.0057014
3: -0.0008895, 0.0036775, -0.0015091, 0.0037173, -0.0046068, 0.0051866
4: -0.0049572, -0.0011940, -0.0051387, -0.0009777, -0.0036730, 0.0034244
5: -0.0005470, 0.0039456, -0.0006160, 0.0044893, -0.0050363, 0.0045617
6: -0.0066028, 0.0008008, -0.0067939, 0.0012106, -0.0078134, 0.0075947
7: -0.0244393, -0.0027993, -0.0255066, -0.0015269, -0.0205060, 0.0181712
8: 0.9731785, 0.9940299, 0.9721356, 0.9953537, -0.0221752, 0.0218943
9: -0.0060726, 0.0083402, -0.0069351, 0.0090564, -0.0134528, 0.0143862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_A1_A1_B1_B2_B2_A1

### Relational analysis result of IS_A2_A1_A1_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0110483, upper bound: 0.0124020
time: 1.16 seconds

## Relational analysis of IS_A2_A1_A1_B1_B2_B2_A2

### Relational analysis result of IS_A2_A1_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125750, upper bound: 0.0129421
time: 1.21 seconds

## BFS IS instance: IS_A2_A1_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0082565, 0.0185611, 0.0082565, 0.0185611, -0.0103047, 0.0103047
1: -0.0039485, 0.0010368, -0.0039485, 0.0010368, -0.0049854, 0.0049854
2: 0.0034755, 0.0091027, 0.0034755, 0.0091027, -0.0056272, 0.0056272
3: -0.0008895, 0.0036775, -0.0008895, 0.0036775, -0.0045670, 0.0045670
4: -0.0049572, -0.0011940, -0.0049572, -0.0011940, -0.0031432, 0.0031432
5: -0.0005470, 0.0039456, -0.0005470, 0.0039456, -0.0044926, 0.0044926
6: -0.0066028, 0.0008008, -0.0066028, 0.0008008, -0.0074036, 0.0074036
7: -0.0244393, -0.0027993, -0.0244393, -0.0027993, -0.0163868, 0.0163868
8: 0.9731785, 0.9940299, 0.9731785, 0.9940299, -0.0208514, 0.0208514
9: -0.0060726, 0.0083402, -0.0060726, 0.0083402, -0.0124258, 0.0124258

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_A1_A1_B2_B1_B1_A1

### Relational analysis result of IS_A2_A1_A1_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0111738, upper bound: 0.0120655
time: 1.39 seconds

## Relational analysis of IS_A2_A1_A1_B2_B1_B1_A2

### Relational analysis result of IS_A2_A1_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125802, upper bound: 0.0125840
time: 1.19 seconds

## BFS IS instance: IS_A2_A1_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0082565, 0.0185611, 0.0069627, 0.0185597, -0.0103032, 0.0115984
1: -0.0039485, 0.0010368, -0.0042152, 0.0012591, -0.0052076, 0.0052520
2: 0.0034755, 0.0091027, 0.0034765, 0.0100375, -0.0065621, 0.0056262
3: -0.0008895, 0.0036775, -0.0014544, 0.0037225, -0.0046120, 0.0051318
4: -0.0049572, -0.0011940, -0.0049666, -0.0009945, -0.0036021, 0.0031734
5: -0.0005470, 0.0039456, -0.0005493, 0.0044909, -0.0050379, 0.0044949
6: -0.0066028, 0.0008008, -0.0067734, 0.0011315, -0.0077343, 0.0075743
7: -0.0244393, -0.0027993, -0.0244683, -0.0016217, -0.0201676, 0.0165809
8: 0.9731785, 0.9940299, 0.9731277, 0.9952570, -0.0220785, 0.0209022
9: -0.0060726, 0.0083402, -0.0068703, 0.0083713, -0.0125183, 0.0141589

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_A1_A1_B2_B1_B2_A1

### Relational analysis result of IS_A2_A1_A1_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0111738, upper bound: 0.0120668
time: 1.38 seconds

## Relational analysis of IS_A2_A1_A1_B2_B1_B2_A2

### Relational analysis result of IS_A2_A1_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125802, upper bound: 0.0126532
time: 1.21 seconds

## BFS IS instance: IS_A2_A1_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0082565, 0.0185611, 0.0084286, 0.0187860, -0.0105295, 0.0101326
1: -0.0039485, 0.0010368, -0.0039089, 0.0011429, -0.0050914, 0.0049457
2: 0.0034755, 0.0091027, 0.0033987, 0.0089764, -0.0055009, 0.0057039
3: -0.0008895, 0.0036775, -0.0008197, 0.0037132, -0.0046028, 0.0044971
4: -0.0049572, -0.0011940, -0.0051675, -0.0012301, -0.0033539, 0.0034004
5: -0.0005470, 0.0039456, -0.0006096, 0.0038723, -0.0044193, 0.0045552
6: -0.0066028, 0.0008008, -0.0065548, 0.0009793, -0.0075821, 0.0073556
7: -0.0244393, -0.0027993, -0.0256950, -0.0030151, -0.0187534, 0.0179298
8: 0.9731785, 0.9940299, 0.9720360, 0.9938253, -0.0206468, 0.0219939
9: -0.0060726, 0.0083402, -0.0059282, 0.0091599, -0.0133754, 0.0131826

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_A1_A1_B2_B2_B1_A1

### Relational analysis result of IS_A2_A1_A1_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0111738, upper bound: 0.0124059
time: 1.14 seconds

## Relational analysis of IS_A2_A1_A1_B2_B2_B1_A2

### Relational analysis result of IS_A2_A1_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125750, upper bound: 0.0128543
time: 1.69 seconds

## BFS IS instance: IS_A2_A1_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0082565, 0.0185611, 0.0070348, 0.0187912, -0.0105348, 0.0115263
1: -0.0039485, 0.0010368, -0.0041962, 0.0013731, -0.0053217, 0.0052330
2: 0.0034755, 0.0091027, 0.0033970, 0.0099868, -0.0065114, 0.0057057
3: -0.0008895, 0.0036775, -0.0014337, 0.0037640, -0.0046535, 0.0051112
4: -0.0049572, -0.0011940, -0.0051883, -0.0010215, -0.0036074, 0.0034369
5: -0.0005470, 0.0039456, -0.0006126, 0.0044600, -0.0050070, 0.0045582
6: -0.0066028, 0.0008008, -0.0067363, 0.0013294, -0.0079322, 0.0075371
7: -0.0244393, -0.0027993, -0.0257798, -0.0017740, -0.0201635, 0.0181487
8: 0.9731785, 0.9940299, 0.9719387, 0.9951462, -0.0219676, 0.0220912
9: -0.0060726, 0.0083402, -0.0067789, 0.0092272, -0.0135035, 0.0141726

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_A1_A1_B2_B2_B2_A1

### Relational analysis result of IS_A2_A1_A1_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0111738, upper bound: 0.0124342
time: 1.41 seconds

## Relational analysis of IS_A2_A1_A1_B2_B2_B2_A2

### Relational analysis result of IS_A2_A1_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125750, upper bound: 0.0129376
time: 1.67 seconds

## BFS IS instance: IS_A2_A1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0069627, 0.0185597, 0.0080929, 0.0185533, -0.0115906, 0.0104667
1: -0.0042152, 0.0012591, -0.0039673, 0.0010577, -0.0052729, 0.0052264
2: 0.0034765, 0.0100375, 0.0034788, 0.0092297, -0.0057533, 0.0065587
3: -0.0014544, 0.0037225, -0.0010035, 0.0036382, -0.0050926, 0.0047260
4: -0.0049666, -0.0009945, -0.0049200, -0.0011302, -0.0032657, 0.0035872
5: -0.0005493, 0.0044909, -0.0005518, 0.0040088, -0.0045581, 0.0050428
6: -0.0067734, 0.0011315, -0.0066797, 0.0007150, -0.0074884, 0.0078112
7: -0.0244683, -0.0016217, -0.0242347, -0.0024411, -0.0171755, 0.0201076
8: 0.9731277, 0.9952570, 0.9733176, 0.9943494, -0.0212216, 0.0219394
9: -0.0068703, 0.0083713, -0.0063017, 0.0082136, -0.0141035, 0.0128313

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_A1_A2_B1_B1_B1_B1

### Relational analysis result of IS_A2_A1_A2_B1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120273, upper bound: 0.0108956
time: 1.17 seconds

## Relational analysis of IS_A2_A1_A2_B1_B1_B1_B2

### Relational analysis result of IS_A2_A1_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126659, upper bound: 0.0125937
time: 1.22 seconds

## BFS IS instance: IS_A2_A1_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0069627, 0.0185597, 0.0068087, 0.0185524, -0.0115896, 0.0117509
1: -0.0042152, 0.0012591, -0.0042306, 0.0012934, -0.0055085, 0.0054897
2: 0.0034765, 0.0100375, 0.0034796, 0.0101545, -0.0066780, 0.0065579
3: -0.0014544, 0.0037225, -0.0015647, 0.0036803, -0.0051347, 0.0052872
4: -0.0049666, -0.0009945, -0.0049252, -0.0009405, -0.0034453, 0.0035916
5: -0.0005493, 0.0044909, -0.0005544, 0.0045500, -0.0050993, 0.0050453
6: -0.0067734, 0.0011315, -0.0068399, 0.0010449, -0.0078184, 0.0079714
7: -0.0244683, -0.0016217, -0.0242427, -0.0013184, -0.0181135, 0.0202772
8: 0.9731277, 0.9952570, 0.9732838, 0.9955425, -0.0224147, 0.0219733
9: -0.0068703, 0.0083713, -0.0070691, 0.0082311, -0.0141281, 0.0135576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_A1_A2_B1_B1_B2_B1

### Relational analysis result of IS_A2_A1_A2_B1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120273, upper bound: 0.0108956
time: 1.20 seconds

## Relational analysis of IS_A2_A1_A2_B1_B1_B2_B2

### Relational analysis result of IS_A2_A1_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126659, upper bound: 0.0125937
time: 1.22 seconds

## BFS IS instance: IS_A2_A1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0069627, 0.0185597, 0.0083055, 0.0187768, -0.0118140, 0.0102542
1: -0.0042152, 0.0012591, -0.0039174, 0.0011571, -0.0053723, 0.0051765
2: 0.0034765, 0.0100375, 0.0034030, 0.0090757, -0.0055992, 0.0066345
3: -0.0014544, 0.0037225, -0.0009109, 0.0036682, -0.0051226, 0.0046334
4: -0.0049666, -0.0009945, -0.0051203, -0.0011813, -0.0034451, 0.0038252
5: -0.0005493, 0.0044909, -0.0006129, 0.0039166, -0.0044659, 0.0051038
6: -0.0067734, 0.0011315, -0.0066209, 0.0008647, -0.0076381, 0.0077524
7: -0.0244683, -0.0016217, -0.0254298, -0.0027351, -0.0192364, 0.0214403
8: 0.9731277, 0.9952570, 0.9722265, 0.9940638, -0.0209361, 0.0230305
9: -0.0068703, 0.0083713, -0.0061136, 0.0089945, -0.0149902, 0.0134910

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_A1_A2_B1_B2_B1_B1

### Relational analysis result of IS_A2_A1_A2_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120300, upper bound: 0.0118908
time: 1.26 seconds

## Relational analysis of IS_A2_A1_A2_B1_B2_B1_B2

### Relational analysis result of IS_A2_A1_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126465, upper bound: 0.0128285
time: 1.09 seconds

## BFS IS instance: IS_A2_A1_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0069627, 0.0185597, 0.0069434, 0.0187820, -0.0118192, 0.0116163
1: -0.0042152, 0.0012591, -0.0041974, 0.0013851, -0.0056003, 0.0054564
2: 0.0034765, 0.0100375, 0.0034013, 0.0100584, -0.0065819, 0.0066362
3: -0.0014544, 0.0037225, -0.0015091, 0.0037173, -0.0051717, 0.0052316
4: -0.0049666, -0.0009945, -0.0051387, -0.0009777, -0.0036792, 0.0038405
5: -0.0005493, 0.0044909, -0.0006160, 0.0044893, -0.0050386, 0.0051070
6: -0.0067734, 0.0011315, -0.0067939, 0.0012106, -0.0079841, 0.0079254
7: -0.0244683, -0.0016217, -0.0255066, -0.0015269, -0.0207092, 0.0216716
8: 0.9731277, 0.9952570, 0.9721356, 0.9953537, -0.0222260, 0.0231214
9: -0.0068703, 0.0083713, -0.0069351, 0.0090564, -0.0150552, 0.0144199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_A1_A2_B1_B2_B2_B1

### Relational analysis result of IS_A2_A1_A2_B1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120300, upper bound: 0.0118908
time: 1.21 seconds

## Relational analysis of IS_A2_A1_A2_B1_B2_B2_B2

### Relational analysis result of IS_A2_A1_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126465, upper bound: 0.0128285
time: 1.79 seconds

## BFS IS instance: IS_A2_A1_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0069627, 0.0185597, 0.0082565, 0.0185611, -0.0115984, 0.0103032
1: -0.0042152, 0.0012591, -0.0039485, 0.0010368, -0.0052520, 0.0052076
2: 0.0034765, 0.0100375, 0.0034755, 0.0091027, -0.0056262, 0.0065621
3: -0.0014544, 0.0037225, -0.0008895, 0.0036775, -0.0051318, 0.0046120
4: -0.0049666, -0.0009945, -0.0049572, -0.0011940, -0.0031734, 0.0036021
5: -0.0005493, 0.0044909, -0.0005470, 0.0039456, -0.0044949, 0.0050379
6: -0.0067734, 0.0011315, -0.0066028, 0.0008008, -0.0075743, 0.0077343
7: -0.0244683, -0.0016217, -0.0244393, -0.0027993, -0.0165809, 0.0201676
8: 0.9731277, 0.9952570, 0.9731785, 0.9940299, -0.0209022, 0.0220785
9: -0.0068703, 0.0083713, -0.0060726, 0.0083402, -0.0141589, 0.0125183

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_A1_A2_B2_B1_B1_B1

### Relational analysis result of IS_A2_A1_A2_B2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120649, upper bound: 0.0111410
time: 1.20 seconds

## Relational analysis of IS_A2_A1_A2_B2_B1_B1_B2

### Relational analysis result of IS_A2_A1_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126659, upper bound: 0.0125735
time: 1.41 seconds

## BFS IS instance: IS_A2_A1_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0069627, 0.0185597, 0.0069627, 0.0185597, -0.0115969, 0.0115969
1: -0.0042152, 0.0012591, -0.0042152, 0.0012591, -0.0054742, 0.0054742
2: 0.0034765, 0.0100375, 0.0034765, 0.0100375, -0.0065611, 0.0065611
3: -0.0014544, 0.0037225, -0.0014544, 0.0037225, -0.0051769, 0.0051769
4: -0.0049666, -0.0009945, -0.0049666, -0.0009945, -0.0036083, 0.0036083
5: -0.0005493, 0.0044909, -0.0005493, 0.0044909, -0.0050402, 0.0050402
6: -0.0067734, 0.0011315, -0.0067734, 0.0011315, -0.0079050, 0.0079050
7: -0.0244683, -0.0016217, -0.0244683, -0.0016217, -0.0203653, 0.0203653
8: 0.9731277, 0.9952570, 0.9731277, 0.9952570, -0.0221293, 0.0221293
9: -0.0068703, 0.0083713, -0.0068703, 0.0083713, -0.0141905, 0.0141905

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_A1_A2_B2_B1_B2_A1

### Relational analysis result of IS_A2_A1_A2_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0111561, upper bound: 0.0120515
time: 1.43 seconds

## Relational analysis of IS_A2_A1_A2_B2_B1_B2_A2

### Relational analysis result of IS_A2_A1_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126659, upper bound: 0.0125735
time: 1.29 seconds

## BFS IS instance: IS_A2_A1_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0069627, 0.0185597, 0.0084286, 0.0187860, -0.0118232, 0.0101311
1: -0.0042152, 0.0012591, -0.0039089, 0.0011429, -0.0053581, 0.0051680
2: 0.0034765, 0.0100375, 0.0033987, 0.0089764, -0.0054999, 0.0066388
3: -0.0014544, 0.0037225, -0.0008197, 0.0037132, -0.0051676, 0.0045422
4: -0.0049666, -0.0009945, -0.0051675, -0.0012301, -0.0033770, 0.0038494
5: -0.0005493, 0.0044909, -0.0006096, 0.0038723, -0.0044215, 0.0051006
6: -0.0067734, 0.0011315, -0.0065548, 0.0009793, -0.0077528, 0.0076863
7: -0.0244683, -0.0016217, -0.0256950, -0.0030151, -0.0188771, 0.0215750
8: 0.9731277, 0.9952570, 0.9720360, 0.9938253, -0.0206976, 0.0232210
9: -0.0068703, 0.0083713, -0.0059282, 0.0091599, -0.0150821, 0.0132563

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_A1_A2_B2_B2_B1_B1

### Relational analysis result of IS_A2_A1_A2_B2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120649, upper bound: 0.0119636
time: 1.48 seconds

## Relational analysis of IS_A2_A1_A2_B2_B2_B1_B2

### Relational analysis result of IS_A2_A1_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126465, upper bound: 0.0128217
time: 1.68 seconds

## BFS IS instance: IS_A2_A1_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0069627, 0.0185597, 0.0070348, 0.0187912, -0.0118285, 0.0115248
1: -0.0042152, 0.0012591, -0.0041962, 0.0013731, -0.0055883, 0.0054553
2: 0.0034765, 0.0100375, 0.0033970, 0.0099868, -0.0065104, 0.0066406
3: -0.0014544, 0.0037225, -0.0014337, 0.0037640, -0.0052184, 0.0051562
4: -0.0049666, -0.0009945, -0.0051883, -0.0010215, -0.0036135, 0.0038638
5: -0.0005493, 0.0044909, -0.0006126, 0.0044600, -0.0050092, 0.0051035
6: -0.0067734, 0.0011315, -0.0067363, 0.0013294, -0.0081029, 0.0078678
7: -0.0244683, -0.0016217, -0.0257798, -0.0017740, -0.0203621, 0.0218055
8: 0.9731277, 0.9952570, 0.9719387, 0.9951462, -0.0220184, 0.0233183
9: -0.0068703, 0.0083713, -0.0067789, 0.0092272, -0.0151430, 0.0142037

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_A1_A2_B2_B2_B2_B1

### Relational analysis result of IS_A2_A1_A2_B2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120649, upper bound: 0.0119636
time: 1.19 seconds

## Relational analysis of IS_A2_A1_A2_B2_B2_B2_B2

### Relational analysis result of IS_A2_A1_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126465, upper bound: 0.0128217
time: 1.25 seconds

## BFS IS instance: IS_A2_A2_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0084286, 0.0187860, 0.0080929, 0.0185533, -0.0101248, 0.0106931
1: -0.0039089, 0.0011429, -0.0039673, 0.0010577, -0.0049666, 0.0051102
2: 0.0033987, 0.0089764, 0.0034788, 0.0092297, -0.0058310, 0.0054975
3: -0.0008197, 0.0037132, -0.0010035, 0.0036382, -0.0044579, 0.0047167
4: -0.0051675, -0.0012301, -0.0049200, -0.0011302, -0.0034966, 0.0033415
5: -0.0006096, 0.0038723, -0.0005518, 0.0040088, -0.0046185, 0.0044241
6: -0.0065548, 0.0009793, -0.0066797, 0.0007150, -0.0072698, 0.0076590
7: -0.0256950, -0.0030151, -0.0242347, -0.0024411, -0.0185493, 0.0187112
8: 0.9720360, 0.9938253, 0.9733176, 0.9943494, -0.0223134, 0.0205077
9: -0.0059282, 0.0091599, -0.0063017, 0.0082136, -0.0131351, 0.0137060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_A2_A1_B1_B1_B1_B1

### Relational analysis result of IS_A2_A2_A1_B1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0123586, upper bound: 0.0109371
time: 1.15 seconds

## Relational analysis of IS_A2_A2_A1_B1_B1_B1_B2

### Relational analysis result of IS_A2_A2_A1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128323, upper bound: 0.0125926
time: 1.65 seconds

## BFS IS instance: IS_A2_A2_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0084286, 0.0187860, 0.0068087, 0.0185524, -0.0101238, 0.0119773
1: -0.0039089, 0.0011429, -0.0042306, 0.0012934, -0.0052023, 0.0053735
2: 0.0033987, 0.0089764, 0.0034796, 0.0101545, -0.0067557, 0.0054968
3: -0.0008197, 0.0037132, -0.0015647, 0.0036803, -0.0045000, 0.0052779
4: -0.0051675, -0.0012301, -0.0049252, -0.0009405, -0.0037425, 0.0033621
5: -0.0006096, 0.0038723, -0.0005544, 0.0045500, -0.0051597, 0.0044267
6: -0.0065548, 0.0009793, -0.0068399, 0.0010449, -0.0075997, 0.0078192
7: -0.0256950, -0.0030151, -0.0242427, -0.0013184, -0.0199803, 0.0188086
8: 0.9720360, 0.9938253, 0.9732838, 0.9955425, -0.0235065, 0.0205415
9: -0.0059282, 0.0091599, -0.0070691, 0.0082311, -0.0131998, 0.0146509

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_A2_A1_B1_B1_B2_B1

### Relational analysis result of IS_A2_A2_A1_B1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0123586, upper bound: 0.0109371
time: 1.63 seconds

## Relational analysis of IS_A2_A2_A1_B1_B1_B2_B2

### Relational analysis result of IS_A2_A2_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128323, upper bound: 0.0126498
time: 1.65 seconds

## BFS IS instance: IS_A2_A2_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0084286, 0.0187860, 0.0082565, 0.0185611, -0.0101326, 0.0105295
1: -0.0039089, 0.0011429, -0.0039485, 0.0010368, -0.0049457, 0.0050914
2: 0.0033987, 0.0089764, 0.0034755, 0.0091027, -0.0057039, 0.0055009
3: -0.0008197, 0.0037132, -0.0008895, 0.0036775, -0.0044971, 0.0046028
4: -0.0051675, -0.0012301, -0.0049572, -0.0011940, -0.0034004, 0.0033539
5: -0.0006096, 0.0038723, -0.0005470, 0.0039456, -0.0045552, 0.0044193
6: -0.0065548, 0.0009793, -0.0066028, 0.0008008, -0.0073556, 0.0075821
7: -0.0256950, -0.0030151, -0.0244393, -0.0027993, -0.0179298, 0.0187534
8: 0.9720360, 0.9938253, 0.9731785, 0.9940299, -0.0219939, 0.0206468
9: -0.0059282, 0.0091599, -0.0060726, 0.0083402, -0.0131827, 0.0133754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_A2_A1_B1_B2_B1_B1

### Relational analysis result of IS_A2_A2_A1_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0123792, upper bound: 0.0111859
time: 1.61 seconds

## Relational analysis of IS_A2_A2_A1_B1_B2_B1_B2

### Relational analysis result of IS_A2_A2_A1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128323, upper bound: 0.0125738
time: 1.30 seconds

## BFS IS instance: IS_A2_A2_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0084286, 0.0187860, 0.0069627, 0.0185597, -0.0101311, 0.0118232
1: -0.0039089, 0.0011429, -0.0042152, 0.0012591, -0.0051680, 0.0053581
2: 0.0033987, 0.0089764, 0.0034765, 0.0100375, -0.0066388, 0.0054999
3: -0.0008197, 0.0037132, -0.0014544, 0.0037225, -0.0045422, 0.0051676
4: -0.0051675, -0.0012301, -0.0049666, -0.0009945, -0.0038494, 0.0033770
5: -0.0006096, 0.0038723, -0.0005493, 0.0044909, -0.0051006, 0.0044215
6: -0.0065548, 0.0009793, -0.0067734, 0.0011315, -0.0076863, 0.0077528
7: -0.0256950, -0.0030151, -0.0244683, -0.0016217, -0.0215749, 0.0188771
8: 0.9720360, 0.9938253, 0.9731277, 0.9952570, -0.0232210, 0.0206976
9: -0.0059282, 0.0091599, -0.0068703, 0.0083713, -0.0132563, 0.0150821

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_A2_A1_B1_B2_B2_A1

### Relational analysis result of IS_A2_A2_A1_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0119981, upper bound: 0.0120668
time: 1.56 seconds

## Relational analysis of IS_A2_A2_A1_B1_B2_B2_A2

### Relational analysis result of IS_A2_A2_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128323, upper bound: 0.0126380
time: 1.28 seconds

## BFS IS instance: IS_A2_A2_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0084286, 0.0187860, 0.0083055, 0.0187768, -0.0103482, 0.0104805
1: -0.0039089, 0.0011429, -0.0039174, 0.0011571, -0.0050660, 0.0050603
2: 0.0033987, 0.0089764, 0.0034030, 0.0090757, -0.0056769, 0.0055734
3: -0.0008197, 0.0037132, -0.0009109, 0.0036682, -0.0044879, 0.0046241
4: -0.0051675, -0.0012301, -0.0051203, -0.0011813, -0.0035666, 0.0034719
5: -0.0006096, 0.0038723, -0.0006129, 0.0039166, -0.0045263, 0.0044852
6: -0.0065548, 0.0009793, -0.0066209, 0.0008647, -0.0074195, 0.0076002
7: -0.0256950, -0.0030151, -0.0254298, -0.0027351, -0.0198710, 0.0193615
8: 0.9720360, 0.9938253, 0.9722265, 0.9940638, -0.0220278, 0.0215988
9: -0.0059282, 0.0091599, -0.0061136, 0.0089945, -0.0136636, 0.0140035

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_A2_A1_B2_B1_B1_B1

### Relational analysis result of IS_A2_A2_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126746, upper bound: 0.0120975
time: 1.18 seconds

## Relational analysis of IS_A2_A2_A1_B2_B1_B1_B2

### Relational analysis result of IS_A2_A2_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129155, upper bound: 0.0127383
time: 1.25 seconds

## BFS IS instance: IS_A2_A2_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0084286, 0.0187860, 0.0069434, 0.0187820, -0.0103534, 0.0118426
1: -0.0039089, 0.0011429, -0.0041974, 0.0013851, -0.0052940, 0.0053403
2: 0.0033987, 0.0089764, 0.0034013, 0.0100584, -0.0066596, 0.0055751
3: -0.0008197, 0.0037132, -0.0015091, 0.0037173, -0.0045370, 0.0052223
4: -0.0051675, -0.0012301, -0.0051387, -0.0009777, -0.0038165, 0.0035013
5: -0.0006096, 0.0038723, -0.0006160, 0.0044893, -0.0050989, 0.0044883
6: -0.0065548, 0.0009793, -0.0067939, 0.0012106, -0.0077654, 0.0077732
7: -0.0256950, -0.0030151, -0.0255066, -0.0015269, -0.0212610, 0.0195081
8: 0.9720360, 0.9938253, 0.9721356, 0.9953537, -0.0233177, 0.0216897
9: -0.0059282, 0.0091599, -0.0069351, 0.0090564, -0.0137647, 0.0149707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_A2_A1_B2_B1_B2_A1

### Relational analysis result of IS_A2_A2_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123429, upper bound: 0.0125634
time: 1.60 seconds

## Relational analysis of IS_A2_A2_A1_B2_B1_B2_A2

### Relational analysis result of IS_A2_A2_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129155, upper bound: 0.0127828
time: 1.92 seconds

## BFS IS instance: IS_A2_A2_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0084286, 0.0187860, 0.0084286, 0.0187860, -0.0103574, 0.0103574
1: -0.0039089, 0.0011429, -0.0039089, 0.0011429, -0.0050518, 0.0050518
2: 0.0033987, 0.0089764, 0.0033987, 0.0089764, -0.0055776, 0.0055776
3: -0.0008197, 0.0037132, -0.0008197, 0.0037132, -0.0045329, 0.0045329
4: -0.0051675, -0.0012301, -0.0051675, -0.0012301, -0.0034933, 0.0034933
5: -0.0006096, 0.0038723, -0.0006096, 0.0038723, -0.0044819, 0.0044819
6: -0.0065548, 0.0009793, -0.0065548, 0.0009793, -0.0075341, 0.0075341
7: -0.0256950, -0.0030151, -0.0256950, -0.0030151, -0.0194595, 0.0194595
8: 0.9720360, 0.9938253, 0.9720360, 0.9938253, -0.0217893, 0.0217893
9: -0.0059282, 0.0091599, -0.0059282, 0.0091599, -0.0137476, 0.0137476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_A2_A1_B2_B2_B1_A1

### Relational analysis result of IS_A2_A2_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123536, upper bound: 0.0125205
time: 1.42 seconds

## Relational analysis of IS_A2_A2_A1_B2_B2_B1_A2

### Relational analysis result of IS_A2_A2_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129155, upper bound: 0.0127214
time: 1.21 seconds

## BFS IS instance: IS_A2_A2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0084286, 0.0187860, 0.0070348, 0.0187912, -0.0103627, 0.0117512
1: -0.0039089, 0.0011429, -0.0041962, 0.0013731, -0.0052820, 0.0053391
2: 0.0033987, 0.0089764, 0.0033970, 0.0099868, -0.0065881, 0.0055794
3: -0.0008197, 0.0037132, -0.0014337, 0.0037640, -0.0045837, 0.0051470
4: -0.0051675, -0.0012301, -0.0051883, -0.0010215, -0.0037425, 0.0035233
5: -0.0006096, 0.0038723, -0.0006126, 0.0044600, -0.0050696, 0.0044849
6: -0.0065548, 0.0009793, -0.0067363, 0.0013294, -0.0078842, 0.0077156
7: -0.0256950, -0.0030151, -0.0257798, -0.0017740, -0.0208531, 0.0196173
8: 0.9720360, 0.9938253, 0.9719387, 0.9951462, -0.0231102, 0.0218866
9: -0.0059282, 0.0091599, -0.0067789, 0.0092272, -0.0138492, 0.0147187

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_A2_A1_B2_B2_B2_A1

### Relational analysis result of IS_A2_A2_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123536, upper bound: 0.0125550
time: 1.52 seconds

## Relational analysis of IS_A2_A2_A1_B2_B2_B2_A2

### Relational analysis result of IS_A2_A2_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129155, upper bound: 0.0127713
time: 1.24 seconds

## BFS IS instance: IS_A2_A2_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0070348, 0.0187912, 0.0080929, 0.0185533, -0.0115185, 0.0106983
1: -0.0041962, 0.0013731, -0.0039673, 0.0010577, -0.0052539, 0.0053404
2: 0.0033970, 0.0099868, 0.0034788, 0.0092297, -0.0058328, 0.0065080
3: -0.0014337, 0.0037640, -0.0010035, 0.0036382, -0.0050720, 0.0047675
4: -0.0051883, -0.0010215, -0.0049200, -0.0011302, -0.0035372, 0.0035961
5: -0.0006126, 0.0044600, -0.0005518, 0.0040088, -0.0046214, 0.0050118
6: -0.0067363, 0.0013294, -0.0066797, 0.0007150, -0.0074512, 0.0080091
7: -0.0257798, -0.0017740, -0.0242347, -0.0024411, -0.0187952, 0.0201215
8: 0.9719387, 0.9951462, 0.9733176, 0.9943494, -0.0224106, 0.0218285
9: -0.0067789, 0.0092272, -0.0063017, 0.0082136, -0.0141293, 0.0138439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_A2_A2_B1_B1_B1_B1

### Relational analysis result of IS_A2_A2_A2_B1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0124276, upper bound: 0.0108956
time: 1.14 seconds

## Relational analysis of IS_A2_A2_A2_B1_B1_B1_B2

### Relational analysis result of IS_A2_A2_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129595, upper bound: 0.0125885
time: 1.12 seconds

## BFS IS instance: IS_A2_A2_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0070348, 0.0187912, 0.0068087, 0.0185524, -0.0115176, 0.0119825
1: -0.0041962, 0.0013731, -0.0042306, 0.0012934, -0.0054896, 0.0056038
2: 0.0033970, 0.0099868, 0.0034796, 0.0101545, -0.0067575, 0.0065072
3: -0.0014337, 0.0037640, -0.0015647, 0.0036803, -0.0051140, 0.0053287
4: -0.0051883, -0.0010215, -0.0049252, -0.0009405, -0.0037144, 0.0036009
5: -0.0006126, 0.0044600, -0.0005544, 0.0045500, -0.0051626, 0.0050143
6: -0.0067363, 0.0013294, -0.0068399, 0.0010449, -0.0077812, 0.0081693
7: -0.0257798, -0.0017740, -0.0242427, -0.0013184, -0.0197146, 0.0202912
8: 0.9719387, 0.9951462, 0.9732838, 0.9955425, -0.0236037, 0.0218624
9: -0.0067789, 0.0092272, -0.0070691, 0.0082311, -0.0141538, 0.0145634

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_A2_A2_B1_B1_B2_B1

### Relational analysis result of IS_A2_A2_A2_B1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0124276, upper bound: 0.0108956
time: 1.13 seconds

## Relational analysis of IS_A2_A2_A2_B1_B1_B2_B2

### Relational analysis result of IS_A2_A2_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129595, upper bound: 0.0125885
time: 1.15 seconds

## BFS IS instance: IS_A2_A2_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0070348, 0.0187912, 0.0082565, 0.0185611, -0.0115263, 0.0105348
1: -0.0041962, 0.0013731, -0.0039485, 0.0010368, -0.0052330, 0.0053217
2: 0.0033970, 0.0099868, 0.0034755, 0.0091027, -0.0057057, 0.0065114
3: -0.0014337, 0.0037640, -0.0008895, 0.0036775, -0.0051112, 0.0046535
4: -0.0051883, -0.0010215, -0.0049572, -0.0011940, -0.0034369, 0.0036074
5: -0.0006126, 0.0044600, -0.0005470, 0.0039456, -0.0045582, 0.0050070
6: -0.0067363, 0.0013294, -0.0066028, 0.0008008, -0.0075371, 0.0079322
7: -0.0257798, -0.0017740, -0.0244393, -0.0027993, -0.0181487, 0.0201635
8: 0.9719387, 0.9951462, 0.9731785, 0.9940299, -0.0220912, 0.0219676
9: -0.0067789, 0.0092272, -0.0060726, 0.0083402, -0.0141726, 0.0135035

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_A2_A2_B1_B2_B1_B1

### Relational analysis result of IS_A2_A2_A2_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0124485, upper bound: 0.0111412
time: 1.43 seconds

## Relational analysis of IS_A2_A2_A2_B1_B2_B1_B2

### Relational analysis result of IS_A2_A2_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129595, upper bound: 0.0125672
time: 1.19 seconds

## BFS IS instance: IS_A2_A2_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0070348, 0.0187912, 0.0069627, 0.0185597, -0.0115248, 0.0118285
1: -0.0041962, 0.0013731, -0.0042152, 0.0012591, -0.0054553, 0.0055883
2: 0.0033970, 0.0099868, 0.0034765, 0.0100375, -0.0066406, 0.0065104
3: -0.0014337, 0.0037640, -0.0014544, 0.0037225, -0.0051562, 0.0052184
4: -0.0051883, -0.0010215, -0.0049666, -0.0009945, -0.0038638, 0.0036135
5: -0.0006126, 0.0044600, -0.0005493, 0.0044909, -0.0051035, 0.0050092
6: -0.0067363, 0.0013294, -0.0067734, 0.0011315, -0.0078678, 0.0081029
7: -0.0257798, -0.0017740, -0.0244683, -0.0016217, -0.0218055, 0.0203621
8: 0.9719387, 0.9951462, 0.9731277, 0.9952570, -0.0233183, 0.0220184
9: -0.0067789, 0.0092272, -0.0068703, 0.0083713, -0.0142037, 0.0151430

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_A2_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_A2_A2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0121055, upper bound: 0.0120515
time: 1.44 seconds

## Relational analysis of IS_A2_A2_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_A2_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129595, upper bound: 0.0125672
time: 1.15 seconds

## BFS IS instance: IS_A2_A2_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0070348, 0.0187912, 0.0083055, 0.0187768, -0.0117420, 0.0104858
1: -0.0041962, 0.0013731, -0.0039174, 0.0011571, -0.0053533, 0.0052906
2: 0.0033970, 0.0099868, 0.0034030, 0.0090757, -0.0056787, 0.0065838
3: -0.0014337, 0.0037640, -0.0009109, 0.0036682, -0.0051019, 0.0046749
4: -0.0051883, -0.0010215, -0.0051203, -0.0011813, -0.0035955, 0.0037230
5: -0.0006126, 0.0044600, -0.0006129, 0.0039166, -0.0045292, 0.0050729
6: -0.0067363, 0.0013294, -0.0066209, 0.0008647, -0.0076009, 0.0079503
7: -0.0257798, -0.0017740, -0.0254298, -0.0027351, -0.0200182, 0.0207522
8: 0.9719387, 0.9951462, 0.9722265, 0.9940638, -0.0221251, 0.0229197
9: -0.0067789, 0.0092272, -0.0061136, 0.0089945, -0.0146451, 0.0140993

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_A2_A2_B2_B1_B1_B1

### Relational analysis result of IS_A2_A2_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127620, upper bound: 0.0120928
time: 1.20 seconds

## Relational analysis of IS_A2_A2_A2_B2_B1_B1_B2

### Relational analysis result of IS_A2_A2_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130274, upper bound: 0.0127363
time: 1.24 seconds

## BFS IS instance: IS_A2_A2_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0070348, 0.0187912, 0.0069434, 0.0187820, -0.0117472, 0.0118478
1: -0.0041962, 0.0013731, -0.0041974, 0.0013851, -0.0055814, 0.0055705
2: 0.0033970, 0.0099868, 0.0034013, 0.0100584, -0.0066614, 0.0065855
3: -0.0014337, 0.0037640, -0.0015091, 0.0037173, -0.0051510, 0.0052731
4: -0.0051883, -0.0010215, -0.0051387, -0.0009777, -0.0038284, 0.0037361
5: -0.0006126, 0.0044600, -0.0006160, 0.0044893, -0.0051019, 0.0050760
6: -0.0067363, 0.0013294, -0.0067939, 0.0012106, -0.0079469, 0.0081233
7: -0.0257798, -0.0017740, -0.0255066, -0.0015269, -0.0214839, 0.0209722
8: 0.9719387, 0.9951462, 0.9721356, 0.9953537, -0.0234150, 0.0230106
9: -0.0067789, 0.0092272, -0.0069351, 0.0090564, -0.0147049, 0.0150235

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_A2_A2_B2_B1_B2_B1

### Relational analysis result of IS_A2_A2_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127620, upper bound: 0.0120928
time: 1.17 seconds

## Relational analysis of IS_A2_A2_A2_B2_B1_B2_B2

### Relational analysis result of IS_A2_A2_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130274, upper bound: 0.0127363
time: 1.67 seconds

## BFS IS instance: IS_A2_A2_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0070348, 0.0187912, 0.0084286, 0.0187860, -0.0117512, 0.0103627
1: -0.0041962, 0.0013731, -0.0039089, 0.0011429, -0.0053391, 0.0052820
2: 0.0033970, 0.0099868, 0.0033987, 0.0089764, -0.0055794, 0.0065881
3: -0.0014337, 0.0037640, -0.0008197, 0.0037132, -0.0051470, 0.0045837
4: -0.0051883, -0.0010215, -0.0051675, -0.0012301, -0.0035233, 0.0037425
5: -0.0006126, 0.0044600, -0.0006096, 0.0038723, -0.0044849, 0.0050696
6: -0.0067363, 0.0013294, -0.0065548, 0.0009793, -0.0077156, 0.0078842
7: -0.0257798, -0.0017740, -0.0256950, -0.0030151, -0.0196173, 0.0208531
8: 0.9719387, 0.9951462, 0.9720360, 0.9938253, -0.0218866, 0.0231102
9: -0.0067789, 0.0092272, -0.0059282, 0.0091599, -0.0147187, 0.0138492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_A2_A2_B2_B2_B1_B1

### Relational analysis result of IS_A2_A2_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127625, upper bound: 0.0121078
time: 1.29 seconds

## Relational analysis of IS_A2_A2_A2_B2_B2_B1_B2

### Relational analysis result of IS_A2_A2_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130274, upper bound: 0.0127174
time: 1.16 seconds

## BFS IS instance: IS_A2_A2_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0070348, 0.0187912, 0.0070348, 0.0187912, -0.0117564, 0.0117564
1: -0.0041962, 0.0013731, -0.0041962, 0.0013731, -0.0055693, 0.0055693
2: 0.0033970, 0.0099868, 0.0033970, 0.0099868, -0.0065898, 0.0065898
3: -0.0014337, 0.0037640, -0.0014337, 0.0037640, -0.0051978, 0.0051978
4: -0.0051883, -0.0010215, -0.0051883, -0.0010215, -0.0037562, 0.0037562
5: -0.0006126, 0.0044600, -0.0006126, 0.0044600, -0.0050726, 0.0050726
6: -0.0067363, 0.0013294, -0.0067363, 0.0013294, -0.0080657, 0.0080657
7: -0.0257798, -0.0017740, -0.0257798, -0.0017740, -0.0210878, 0.0210878
8: 0.9719387, 0.9951462, 0.9719387, 0.9951462, -0.0232074, 0.0232074
9: -0.0067789, 0.0092272, -0.0067789, 0.0092272, -0.0147793, 0.0147793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_A2_A2_B2_B2_B2_A1

### Relational analysis result of IS_A2_A2_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124688, upper bound: 0.0125175
time: 1.52 seconds

## Relational analysis of IS_A2_A2_A2_B2_B2_B2_A2

### Relational analysis result of IS_A2_A2_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130274, upper bound: 0.0127174
time: 1.18 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.04 seconds
IS_A1_A1_A1_B1_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0110390, upper bound: 0.0120674
IS_A1_A1_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126328, upper bound: 0.0126161
IS_A1_A1_A1_B1_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0110390, upper bound: 0.0120789
IS_A1_A1_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126328, upper bound: 0.0126255
IS_A1_A1_A1_B1_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0109168, upper bound: 0.0120224
IS_A1_A1_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0125937, upper bound: 0.0126532
IS_A1_A1_A1_B1_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0109168, upper bound: 0.0120299
IS_A1_A1_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0125937, upper bound: 0.0126659
IS_A1_A1_A1_B2_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0110390, upper bound: 0.0123976
IS_A1_A1_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126245, upper bound: 0.0128809
IS_A1_A1_A1_B2_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0110390, upper bound: 0.0124180
IS_A1_A1_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126245, upper bound: 0.0128991
IS_A1_A1_A1_B2_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0109168, upper bound: 0.0124020
IS_A1_A1_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0125885, upper bound: 0.0129376
IS_A1_A1_A1_B2_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0109168, upper bound: 0.0124276
IS_A1_A1_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0125885, upper bound: 0.0129595
IS_A1_A1_A2_B1_B1_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0120261, upper bound: 0.0108956
IS_A1_A1_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126644, upper bound: 0.0125937
IS_A1_A1_A2_B1_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0108986, upper bound: 0.0120061
IS_A1_A1_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126644, upper bound: 0.0125937
IS_A1_A1_A2_B1_B2_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0120267, upper bound: 0.0110270
IS_A1_A1_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126644, upper bound: 0.0125802
IS_A1_A1_A2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0108986, upper bound: 0.0120104
IS_A1_A1_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126644, upper bound: 0.0125802
IS_A1_A1_A2_B2_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0108986, upper bound: 0.0123413
IS_A1_A1_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126498, upper bound: 0.0128285
IS_A1_A1_A2_B2_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0108986, upper bound: 0.0123413
IS_A1_A1_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126498, upper bound: 0.0128285
IS_A1_A1_A2_B2_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0108986, upper bound: 0.0123586
IS_A1_A1_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126498, upper bound: 0.0128323
IS_A1_A1_A2_B2_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0108986, upper bound: 0.0123586
IS_A1_A1_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126498, upper bound: 0.0128323
IS_A1_A2_A1_B1_B1_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0123984, upper bound: 0.0110390
IS_A1_A2_A1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0128846, upper bound: 0.0126054
IS_A1_A2_A1_B1_B1_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0123984, upper bound: 0.0111862
IS_A1_A2_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0128846, upper bound: 0.0126152
IS_A1_A2_A1_B1_B2_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0123413, upper bound: 0.0109371
IS_A1_A2_A1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0128285, upper bound: 0.0126380
IS_A1_A2_A1_B1_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0118908, upper bound: 0.0120315
IS_A1_A2_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0128285, upper bound: 0.0126465
IS_A1_A2_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0123432, upper bound: 0.0125273
IS_A1_A2_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0129646, upper bound: 0.0127449
IS_A1_A2_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0123432, upper bound: 0.0125314
IS_A1_A2_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0129646, upper bound: 0.0127518
IS_A1_A2_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0123131, upper bound: 0.0125544
IS_A1_A2_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0129180, upper bound: 0.0127713
IS_A1_A2_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0123131, upper bound: 0.0125577
IS_A1_A2_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0129180, upper bound: 0.0127786
IS_A1_A2_A2_B1_B1_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0124020, upper bound: 0.0108956
IS_A1_A2_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0129421, upper bound: 0.0125885
IS_A1_A2_A2_B1_B1_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0124020, upper bound: 0.0108956
IS_A1_A2_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0129421, upper bound: 0.0125885
IS_A1_A2_A2_B1_B2_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0124020, upper bound: 0.0110270
IS_A1_A2_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0129421, upper bound: 0.0125750
IS_A1_A2_A2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0120000, upper bound: 0.0120122
IS_A1_A2_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0129421, upper bound: 0.0125750
IS_A1_A2_A2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0127519, upper bound: 0.0120928
IS_A1_A2_A2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0130127, upper bound: 0.0127363
IS_A1_A2_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0124169, upper bound: 0.0125306
IS_A1_A2_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0130127, upper bound: 0.0127363
IS_A1_A2_A2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0127519, upper bound: 0.0121065
IS_A1_A2_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0130127, upper bound: 0.0127229
IS_A1_A2_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0124169, upper bound: 0.0125207
IS_A1_A2_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0130127, upper bound: 0.0127229
IS_A2_A1_A1_B1_B1_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0120123, upper bound: 0.0109371
IS_A2_A1_A1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0125802, upper bound: 0.0126003
IS_A2_A1_A1_B1_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0110483, upper bound: 0.0120278
IS_A2_A1_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0125802, upper bound: 0.0126644
IS_A2_A1_A1_B1_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0110483, upper bound: 0.0123770
IS_A2_A1_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0125750, upper bound: 0.0128582
IS_A2_A1_A1_B1_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0110483, upper bound: 0.0124020
IS_A2_A1_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0125750, upper bound: 0.0129421
IS_A2_A1_A1_B2_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0111738, upper bound: 0.0120655
IS_A2_A1_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0125802, upper bound: 0.0125840
IS_A2_A1_A1_B2_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0111738, upper bound: 0.0120668
IS_A2_A1_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0125802, upper bound: 0.0126532
IS_A2_A1_A1_B2_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0111738, upper bound: 0.0124059
IS_A2_A1_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0125750, upper bound: 0.0128543
IS_A2_A1_A1_B2_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0111738, upper bound: 0.0124342
IS_A2_A1_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0125750, upper bound: 0.0129376
IS_A2_A1_A2_B1_B1_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0120273, upper bound: 0.0108956
IS_A2_A1_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126659, upper bound: 0.0125937
IS_A2_A1_A2_B1_B1_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0120273, upper bound: 0.0108956
IS_A2_A1_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126659, upper bound: 0.0125937
IS_A2_A1_A2_B1_B2_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0120300, upper bound: 0.0118908
IS_A2_A1_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126465, upper bound: 0.0128285
IS_A2_A1_A2_B1_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0120300, upper bound: 0.0118908
IS_A2_A1_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126465, upper bound: 0.0128285
IS_A2_A1_A2_B2_B1_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0120649, upper bound: 0.0111410
IS_A2_A1_A2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126659, upper bound: 0.0125735
IS_A2_A1_A2_B2_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0111561, upper bound: 0.0120515
IS_A2_A1_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126659, upper bound: 0.0125735
IS_A2_A1_A2_B2_B2_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0120649, upper bound: 0.0119636
IS_A2_A1_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126465, upper bound: 0.0128217
IS_A2_A1_A2_B2_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0120649, upper bound: 0.0119636
IS_A2_A1_A2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126465, upper bound: 0.0128217
IS_A2_A2_A1_B1_B1_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0123586, upper bound: 0.0109371
IS_A2_A2_A1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0128323, upper bound: 0.0125926
IS_A2_A2_A1_B1_B1_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0123586, upper bound: 0.0109371
IS_A2_A2_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0128323, upper bound: 0.0126498
IS_A2_A2_A1_B1_B2_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0123792, upper bound: 0.0111859
IS_A2_A2_A1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0128323, upper bound: 0.0125738
IS_A2_A2_A1_B1_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0119981, upper bound: 0.0120668
IS_A2_A2_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0128323, upper bound: 0.0126380
IS_A2_A2_A1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126746, upper bound: 0.0120975
IS_A2_A2_A1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0129155, upper bound: 0.0127383
IS_A2_A2_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0123429, upper bound: 0.0125634
IS_A2_A2_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0129155, upper bound: 0.0127828
IS_A2_A2_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0123536, upper bound: 0.0125205
IS_A2_A2_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0129155, upper bound: 0.0127214
IS_A2_A2_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0123536, upper bound: 0.0125550
IS_A2_A2_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0129155, upper bound: 0.0127713
IS_A2_A2_A2_B1_B1_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0124276, upper bound: 0.0108956
IS_A2_A2_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0129595, upper bound: 0.0125885
IS_A2_A2_A2_B1_B1_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0124276, upper bound: 0.0108956
IS_A2_A2_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0129595, upper bound: 0.0125885
IS_A2_A2_A2_B1_B2_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0124485, upper bound: 0.0111412
IS_A2_A2_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0129595, upper bound: 0.0125672
IS_A2_A2_A2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0121055, upper bound: 0.0120515
IS_A2_A2_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0129595, upper bound: 0.0125672
IS_A2_A2_A2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0127620, upper bound: 0.0120928
IS_A2_A2_A2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0130274, upper bound: 0.0127363
IS_A2_A2_A2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0127620, upper bound: 0.0120928
IS_A2_A2_A2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0130274, upper bound: 0.0127363
IS_A2_A2_A2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0127625, upper bound: 0.0121078
IS_A2_A2_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0130274, upper bound: 0.0127174
IS_A2_A2_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0124688, upper bound: 0.0125175
IS_A2_A2_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0130274, upper bound: 0.0127174

## BFS IS instance: IS_A1_A1_A1_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0081551, 0.0185231, 0.0080929, 0.0185533, -0.0103982, 0.0104302
1: -0.0039517, 0.0010343, -0.0039673, 0.0010577, -0.0050094, 0.0050016
2: 0.0034888, 0.0091861, 0.0034788, 0.0092297, -0.0057409, 0.0057073
3: -0.0009779, 0.0036276, -0.0010035, 0.0036382, -0.0046161, 0.0046311
4: -0.0048988, -0.0011364, -0.0049200, -0.0011302, -0.0030353, 0.0031593
5: -0.0005395, 0.0039815, -0.0005518, 0.0040088, -0.0045484, 0.0045333
6: -0.0066769, 0.0006627, -0.0066797, 0.0007150, -0.0073918, 0.0073424
7: -0.0241124, -0.0024790, -0.0242347, -0.0024411, -0.0155837, 0.0165085
8: 0.9734337, 0.9943037, 0.9733176, 0.9943494, -0.0209156, 0.0209861
9: -0.0062750, 0.0081323, -0.0063017, 0.0082136, -0.0124911, 0.0120602

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_A1_A1_B1_B1_B1_A2_B1

### Relational analysis result of IS_A1_A1_A1_B1_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120751, upper bound: 0.0110390
time: 1.41 seconds

## Relational analysis of IS_A1_A1_A1_B1_B1_B1_A2_B2

### Relational analysis result of IS_A1_A1_A1_B1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120751, upper bound: 0.0126328
time: 1.01 seconds

## BFS IS instance: IS_A1_A1_A1_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0081551, 0.0185231, 0.0082565, 0.0185611, -0.0104060, 0.0102667
1: -0.0039517, 0.0010343, -0.0039485, 0.0010368, -0.0049886, 0.0049829
2: 0.0034888, 0.0091861, 0.0034755, 0.0091027, -0.0056138, 0.0057106
3: -0.0009779, 0.0036276, -0.0008895, 0.0036775, -0.0046553, 0.0045171
4: -0.0048988, -0.0011364, -0.0049572, -0.0011940, -0.0030180, 0.0032292
5: -0.0005395, 0.0039815, -0.0005470, 0.0039456, -0.0044851, 0.0045284
6: -0.0066769, 0.0006627, -0.0066028, 0.0008008, -0.0074777, 0.0072655
7: -0.0241124, -0.0024790, -0.0244393, -0.0027993, -0.0155572, 0.0169506
8: 0.9734337, 0.9943037, 0.9731785, 0.9940299, -0.0205962, 0.0211252
9: -0.0062750, 0.0081323, -0.0060726, 0.0083402, -0.0127104, 0.0119757

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_A1_A1_B1_B1_B2_A2_B1

### Relational analysis result of IS_A1_A1_A1_B1_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120751, upper bound: 0.0111862
time: 1.39 seconds

## Relational analysis of IS_A1_A1_A1_B1_B1_B2_A2_B2

### Relational analysis result of IS_A1_A1_A1_B1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120751, upper bound: 0.0126255
time: 1.02 seconds

## BFS IS instance: IS_A1_A1_A1_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0081551, 0.0185231, 0.0068087, 0.0185524, -0.0103973, 0.0117144
1: -0.0039517, 0.0010343, -0.0042306, 0.0012934, -0.0052451, 0.0052649
2: 0.0034888, 0.0091861, 0.0034796, 0.0101545, -0.0066656, 0.0057065
3: -0.0009779, 0.0036276, -0.0015647, 0.0036803, -0.0046581, 0.0051922
4: -0.0048988, -0.0011364, -0.0049252, -0.0009405, -0.0032933, 0.0031876
5: -0.0005395, 0.0039815, -0.0005544, 0.0045500, -0.0050896, 0.0045358
6: -0.0066769, 0.0006627, -0.0068399, 0.0010449, -0.0077218, 0.0075027
7: -0.0241124, -0.0024790, -0.0242427, -0.0013184, -0.0171099, 0.0166984
8: 0.9734337, 0.9943037, 0.9732838, 0.9955425, -0.0221087, 0.0210199
9: -0.0062750, 0.0081323, -0.0070691, 0.0082311, -0.0125793, 0.0130499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_A1_A1_B1_B2_B1_A2_B1

### Relational analysis result of IS_A1_A1_A1_B1_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120084, upper bound: 0.0109371
time: 1.11 seconds

## Relational analysis of IS_A1_A1_A1_B1_B2_B1_A2_B2

### Relational analysis result of IS_A1_A1_A1_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120084, upper bound: 0.0126644
time: 1.66 seconds

## BFS IS instance: IS_A1_A1_A1_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0081551, 0.0185231, 0.0069627, 0.0185597, -0.0104045, 0.0115604
1: -0.0039517, 0.0010343, -0.0042152, 0.0012591, -0.0052108, 0.0052495
2: 0.0034888, 0.0091861, 0.0034765, 0.0100375, -0.0065487, 0.0057096
3: -0.0009779, 0.0036276, -0.0014544, 0.0037225, -0.0047004, 0.0050820
4: -0.0048988, -0.0011364, -0.0049666, -0.0009945, -0.0035227, 0.0032587
5: -0.0005395, 0.0039815, -0.0005493, 0.0044909, -0.0050305, 0.0045307
6: -0.0066769, 0.0006627, -0.0067734, 0.0011315, -0.0078084, 0.0074362
7: -0.0241124, -0.0024790, -0.0244683, -0.0016217, -0.0198915, 0.0171371
8: 0.9734337, 0.9943037, 0.9731277, 0.9952570, -0.0218233, 0.0211760
9: -0.0062750, 0.0081323, -0.0068703, 0.0083713, -0.0128018, 0.0138887

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_A1_A1_B1_B2_B2_A2_B1

### Relational analysis result of IS_A1_A1_A1_B1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120084, upper bound: 0.0110722
time: 1.69 seconds

## Relational analysis of IS_A1_A1_A1_B1_B2_B2_A2_B2

### Relational analysis result of IS_A1_A1_A1_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120084, upper bound: 0.0126659
time: 1.32 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0081551, 0.0185231, 0.0083055, 0.0187768, -0.0106217, 0.0102177
1: -0.0039517, 0.0010343, -0.0039174, 0.0011571, -0.0051089, 0.0049517
2: 0.0034888, 0.0091861, 0.0034030, 0.0090757, -0.0055868, 0.0057831
3: -0.0009779, 0.0036276, -0.0009109, 0.0036682, -0.0046460, 0.0045385
4: -0.0048988, -0.0011364, -0.0051203, -0.0011813, -0.0032973, 0.0034115
5: -0.0005395, 0.0039815, -0.0006129, 0.0039166, -0.0044562, 0.0045943
6: -0.0066769, 0.0006627, -0.0066209, 0.0008647, -0.0075415, 0.0072836
7: -0.0241124, -0.0024790, -0.0254298, -0.0027351, -0.0186095, 0.0180313
8: 0.9734337, 0.9943037, 0.9722265, 0.9940638, -0.0206301, 0.0220772
9: -0.0062750, 0.0081323, -0.0061136, 0.0089945, -0.0134227, 0.0130137

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_A1_A1_B2_B1_B1_A2_B1

### Relational analysis result of IS_A1_A1_A1_B2_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120751, upper bound: 0.0119762
time: 1.12 seconds

## Relational analysis of IS_A1_A1_A1_B2_B1_B1_A2_B2

### Relational analysis result of IS_A1_A1_A1_B2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120751, upper bound: 0.0128846
time: 1.35 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0081551, 0.0185231, 0.0084286, 0.0187860, -0.0106309, 0.0100946
1: -0.0039517, 0.0010343, -0.0039089, 0.0011429, -0.0050946, 0.0049432
2: 0.0034888, 0.0091861, 0.0033987, 0.0089764, -0.0054875, 0.0057874
3: -0.0009779, 0.0036276, -0.0008197, 0.0037132, -0.0046911, 0.0044472
4: -0.0048988, -0.0011364, -0.0051675, -0.0012301, -0.0032781, 0.0034896
5: -0.0005395, 0.0039815, -0.0006096, 0.0038723, -0.0044118, 0.0045911
6: -0.0066769, 0.0006627, -0.0065548, 0.0009793, -0.0076562, 0.0072175
7: -0.0241124, -0.0024790, -0.0256950, -0.0030151, -0.0185029, 0.0185110
8: 0.9734337, 0.9943037, 0.9720360, 0.9938253, -0.0203916, 0.0222677
9: -0.0062750, 0.0081323, -0.0059282, 0.0091599, -0.0136765, 0.0129264

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_A1_A1_B2_B1_B2_A2_B1

### Relational analysis result of IS_A1_A1_A1_B2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120751, upper bound: 0.0120418
time: 1.16 seconds

## Relational analysis of IS_A1_A1_A1_B2_B1_B2_A2_B2

### Relational analysis result of IS_A1_A1_A1_B2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120751, upper bound: 0.0128991
time: 1.12 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0081551, 0.0185231, 0.0069434, 0.0187820, -0.0106268, 0.0115797
1: -0.0039517, 0.0010343, -0.0041974, 0.0013851, -0.0053369, 0.0052317
2: 0.0034888, 0.0091861, 0.0034013, 0.0100584, -0.0065695, 0.0057848
3: -0.0009779, 0.0036276, -0.0015091, 0.0037173, -0.0046951, 0.0051367
4: -0.0048988, -0.0011364, -0.0051387, -0.0009777, -0.0035532, 0.0034477
5: -0.0005395, 0.0039815, -0.0006160, 0.0044893, -0.0050288, 0.0045975
6: -0.0066769, 0.0006627, -0.0067939, 0.0012106, -0.0078875, 0.0074566
7: -0.0241124, -0.0024790, -0.0255066, -0.0015269, -0.0200112, 0.0182528
8: 0.9734337, 0.9943037, 0.9721356, 0.9953537, -0.0219200, 0.0221681
9: -0.0062750, 0.0081323, -0.0069351, 0.0090564, -0.0135515, 0.0140002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_A1_A1_B2_B2_B1_A2_B1

### Relational analysis result of IS_A1_A1_A1_B2_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120117, upper bound: 0.0120003
time: 1.21 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2_B1_A2_B2

### Relational analysis result of IS_A1_A1_A1_B2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120117, upper bound: 0.0129421
time: 1.15 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0081551, 0.0185231, 0.0070348, 0.0187912, -0.0106361, 0.0114883
1: -0.0039517, 0.0010343, -0.0041962, 0.0013731, -0.0053249, 0.0052305
2: 0.0034888, 0.0091861, 0.0033970, 0.0099868, -0.0064980, 0.0057891
3: -0.0009779, 0.0036276, -0.0014337, 0.0037640, -0.0047419, 0.0050613
4: -0.0048988, -0.0011364, -0.0051883, -0.0010215, -0.0035373, 0.0035302
5: -0.0005395, 0.0039815, -0.0006126, 0.0044600, -0.0049995, 0.0045940
6: -0.0066769, 0.0006627, -0.0067363, 0.0013294, -0.0080063, 0.0073990
7: -0.0241124, -0.0024790, -0.0257798, -0.0017740, -0.0199195, 0.0187569
8: 0.9734337, 0.9943037, 0.9719387, 0.9951462, -0.0217124, 0.0223650
9: -0.0062750, 0.0081323, -0.0067789, 0.0092272, -0.0138144, 0.0139329

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_A1_A1_B2_B2_B2_A2_B1

### Relational analysis result of IS_A1_A1_A1_B2_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120117, upper bound: 0.0120718
time: 1.20 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2_B2_A2_B2

### Relational analysis result of IS_A1_A1_A1_B2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120117, upper bound: 0.0129595
time: 1.18 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0068087, 0.0185524, 0.0081551, 0.0185231, -0.0117144, 0.0103973
1: -0.0042306, 0.0012934, -0.0039517, 0.0010343, -0.0052649, 0.0052451
2: 0.0034796, 0.0101545, 0.0034888, 0.0091861, -0.0057065, 0.0066656
3: -0.0015647, 0.0036803, -0.0009779, 0.0036276, -0.0051922, 0.0046581
4: -0.0049252, -0.0009405, -0.0048988, -0.0011364, -0.0031876, 0.0032933
5: -0.0005544, 0.0045500, -0.0005395, 0.0039815, -0.0045358, 0.0050896
6: -0.0068399, 0.0010449, -0.0066769, 0.0006627, -0.0075027, 0.0077218
7: -0.0242427, -0.0013184, -0.0241124, -0.0024790, -0.0166984, 0.0171099
8: 0.9732838, 0.9955425, 0.9734337, 0.9943037, -0.0210199, 0.0221087
9: -0.0070691, 0.0082311, -0.0062750, 0.0081323, -0.0130499, 0.0125793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_A1_A2_B1_B1_B1_B2_A1

### Relational analysis result of IS_A1_A1_A2_B1_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0109371, upper bound: 0.0120084
time: 1.09 seconds

## Relational analysis of IS_A1_A1_A2_B1_B1_B1_B2_A2

### Relational analysis result of IS_A1_A1_A2_B1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109371, upper bound: 0.0125937
time: 1.49 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0068723, 0.0185218, 0.0068087, 0.0185524, -0.0116801, 0.0117131
1: -0.0042149, 0.0012676, -0.0042306, 0.0012934, -0.0055082, 0.0054983
2: 0.0034898, 0.0101097, 0.0034796, 0.0101545, -0.0066647, 0.0066301
3: -0.0015385, 0.0036695, -0.0015647, 0.0036803, -0.0052188, 0.0052341
4: -0.0049032, -0.0009466, -0.0049252, -0.0009405, -0.0032547, 0.0033660
5: -0.0005420, 0.0045222, -0.0005544, 0.0045500, -0.0050920, 0.0050766
6: -0.0068371, 0.0009945, -0.0068399, 0.0010449, -0.0078820, 0.0078344
7: -0.0241178, -0.0013553, -0.0242427, -0.0013184, -0.0167873, 0.0176324
8: 0.9734010, 0.9954968, 0.9732838, 0.9955425, -0.0221415, 0.0222130
9: -0.0070430, 0.0081484, -0.0070691, 0.0082311, -0.0133055, 0.0129189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_A1_A2_B1_B1_B2_A2_B1

### Relational analysis result of IS_A1_A1_A2_B1_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120261, upper bound: 0.0108956
time: 1.12 seconds

## Relational analysis of IS_A1_A1_A2_B1_B1_B2_A2_B2

### Relational analysis result of IS_A1_A1_A2_B1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120261, upper bound: 0.0125937
time: 1.60 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0068087, 0.0185524, 0.0083241, 0.0185306, -0.0117218, 0.0102283
1: -0.0042306, 0.0012934, -0.0039318, 0.0010127, -0.0052433, 0.0052252
2: 0.0034796, 0.0101545, 0.0034854, 0.0090551, -0.0055755, 0.0066691
3: -0.0015647, 0.0036803, -0.0008622, 0.0036666, -0.0052312, 0.0045425
4: -0.0049252, -0.0009405, -0.0049351, -0.0012007, -0.0031566, 0.0033730
5: -0.0005544, 0.0045500, -0.0005348, 0.0039160, -0.0044704, 0.0050849
6: -0.0068399, 0.0010449, -0.0065998, 0.0007472, -0.0075871, 0.0076448
7: -0.0242427, -0.0013184, -0.0243155, -0.0028397, -0.0165732, 0.0176385
8: 0.9732838, 0.9955425, 0.9732951, 0.9939808, -0.0206970, 0.0222474
9: -0.0070691, 0.0082311, -0.0060440, 0.0082579, -0.0132996, 0.0124482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_A1_A2_B1_B2_B1_B2_A1

### Relational analysis result of IS_A1_A1_A2_B1_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0109371, upper bound: 0.0120123
time: 1.39 seconds

## Relational analysis of IS_A1_A1_A2_B1_B2_B1_B2_A2

### Relational analysis result of IS_A1_A1_A2_B1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109371, upper bound: 0.0125802
time: 1.58 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0068723, 0.0185218, 0.0069627, 0.0185597, -0.0116874, 0.0115591
1: -0.0042149, 0.0012676, -0.0042152, 0.0012591, -0.0054739, 0.0054828
2: 0.0034898, 0.0101097, 0.0034765, 0.0100375, -0.0065477, 0.0066332
3: -0.0015385, 0.0036695, -0.0014544, 0.0037225, -0.0052610, 0.0051239
4: -0.0049032, -0.0009466, -0.0049666, -0.0009945, -0.0035280, 0.0034383
5: -0.0005420, 0.0045222, -0.0005493, 0.0044909, -0.0050329, 0.0050714
6: -0.0068371, 0.0009945, -0.0067734, 0.0011315, -0.0079686, 0.0077679
7: -0.0241178, -0.0013553, -0.0244683, -0.0016217, -0.0200605, 0.0180755
8: 0.9734010, 0.9954968, 0.9731277, 0.9952570, -0.0218560, 0.0223691
9: -0.0070430, 0.0081484, -0.0068703, 0.0083713, -0.0135283, 0.0139154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_A1_A2_B1_B2_B2_A2_B1

### Relational analysis result of IS_A1_A1_A2_B1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120267, upper bound: 0.0110270
time: 1.15 seconds

## Relational analysis of IS_A1_A1_A2_B1_B2_B2_A2_B2

### Relational analysis result of IS_A1_A1_A2_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120267, upper bound: 0.0125802
time: 1.08 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0068723, 0.0185218, 0.0083055, 0.0187768, -0.0119045, 0.0102163
1: -0.0042149, 0.0012676, -0.0039174, 0.0011571, -0.0053720, 0.0051851
2: 0.0034898, 0.0101097, 0.0034030, 0.0090757, -0.0055859, 0.0067067
3: -0.0015385, 0.0036695, -0.0009109, 0.0036682, -0.0052067, 0.0045804
4: -0.0049032, -0.0009466, -0.0051203, -0.0011813, -0.0033195, 0.0036573
5: -0.0005420, 0.0045222, -0.0006129, 0.0039166, -0.0044586, 0.0051351
6: -0.0068371, 0.0009945, -0.0066209, 0.0008647, -0.0077018, 0.0076154
7: -0.0241178, -0.0013553, -0.0254298, -0.0027351, -0.0187049, 0.0194617
8: 0.9734010, 0.9954968, 0.9722265, 0.9940638, -0.0206628, 0.0232703
9: -0.0070430, 0.0081484, -0.0061136, 0.0089945, -0.0143677, 0.0130813

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_A1_A2_B2_B1_B1_A2_B1

### Relational analysis result of IS_A1_A1_A2_B2_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120298, upper bound: 0.0118908
time: 1.18 seconds

## Relational analysis of IS_A1_A1_A2_B2_B1_B1_A2_B2

### Relational analysis result of IS_A1_A1_A2_B2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120298, upper bound: 0.0128285
time: 1.10 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0068723, 0.0185218, 0.0069434, 0.0187820, -0.0119097, 0.0115784
1: -0.0042149, 0.0012676, -0.0041974, 0.0013851, -0.0056000, 0.0054650
2: 0.0034898, 0.0101097, 0.0034013, 0.0100584, -0.0065686, 0.0067084
3: -0.0015385, 0.0036695, -0.0015091, 0.0037173, -0.0052558, 0.0051786
4: -0.0049032, -0.0009466, -0.0051387, -0.0009777, -0.0035584, 0.0036270
5: -0.0005420, 0.0045222, -0.0006160, 0.0044893, -0.0050313, 0.0051382
6: -0.0068371, 0.0009945, -0.0067939, 0.0012106, -0.0080478, 0.0077884
7: -0.0241178, -0.0013553, -0.0255066, -0.0015269, -0.0201823, 0.0191944
8: 0.9734010, 0.9954968, 0.9721356, 0.9953537, -0.0219527, 0.0233612
9: -0.0070430, 0.0081484, -0.0069351, 0.0090564, -0.0142718, 0.0140261

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_A1_A2_B2_B1_B2_A2_B1

### Relational analysis result of IS_A1_A1_A2_B2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120298, upper bound: 0.0118908
time: 1.20 seconds

## Relational analysis of IS_A1_A1_A2_B2_B1_B2_A2_B2

### Relational analysis result of IS_A1_A1_A2_B2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120298, upper bound: 0.0128285
time: 1.78 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0068723, 0.0185218, 0.0084286, 0.0187860, -0.0119137, 0.0100932
1: -0.0042149, 0.0012676, -0.0039089, 0.0011429, -0.0053578, 0.0051765
2: 0.0034898, 0.0101097, 0.0033987, 0.0089764, -0.0054866, 0.0067109
3: -0.0015385, 0.0036695, -0.0008197, 0.0037132, -0.0052518, 0.0044892
4: -0.0049032, -0.0009466, -0.0051675, -0.0012301, -0.0033003, 0.0037354
5: -0.0005420, 0.0045222, -0.0006096, 0.0038723, -0.0044142, 0.0051318
6: -0.0068371, 0.0009945, -0.0065548, 0.0009793, -0.0078164, 0.0075493
7: -0.0241178, -0.0013553, -0.0256950, -0.0030151, -0.0185984, 0.0199413
8: 0.9734010, 0.9954968, 0.9720360, 0.9938253, -0.0204243, 0.0234608
9: -0.0070430, 0.0081484, -0.0059282, 0.0091599, -0.0146215, 0.0129940

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_A1_A2_B2_B2_B1_A2_B1

### Relational analysis result of IS_A1_A1_A2_B2_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120299, upper bound: 0.0119569
time: 1.30 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2_B1_A2_B2

### Relational analysis result of IS_A1_A1_A2_B2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120299, upper bound: 0.0128323
time: 1.24 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0068723, 0.0185218, 0.0070348, 0.0187912, -0.0119189, 0.0114870
1: -0.0042149, 0.0012676, -0.0041962, 0.0013731, -0.0055880, 0.0054639
2: 0.0034898, 0.0101097, 0.0033970, 0.0099868, -0.0064970, 0.0067127
3: -0.0015385, 0.0036695, -0.0014337, 0.0037640, -0.0053025, 0.0051032
4: -0.0049032, -0.0009466, -0.0051883, -0.0010215, -0.0035427, 0.0037075
5: -0.0005420, 0.0045222, -0.0006126, 0.0044600, -0.0050019, 0.0051348
6: -0.0068371, 0.0009945, -0.0067363, 0.0013294, -0.0081666, 0.0077307
7: -0.0241178, -0.0013553, -0.0257798, -0.0017740, -0.0200872, 0.0196766
8: 0.9734010, 0.9954968, 0.9719387, 0.9951462, -0.0217451, 0.0235581
9: -0.0070430, 0.0081484, -0.0067789, 0.0092272, -0.0145341, 0.0139592

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_A1_A2_B2_B2_B2_A2_B1

### Relational analysis result of IS_A1_A1_A2_B2_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120298, upper bound: 0.0119569
time: 1.14 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2_B2_A2_B2

### Relational analysis result of IS_A1_A1_A2_B2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120298, upper bound: 0.0128323
time: 1.83 seconds

## BFS IS instance: IS_A1_A2_A1_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0083055, 0.0187768, 0.0081551, 0.0185231, -0.0102177, 0.0106217
1: -0.0039174, 0.0011571, -0.0039517, 0.0010343, -0.0049517, 0.0051089
2: 0.0034030, 0.0090757, 0.0034888, 0.0091861, -0.0057831, 0.0055868
3: -0.0009109, 0.0036682, -0.0009779, 0.0036276, -0.0045385, 0.0046460
4: -0.0051203, -0.0011813, -0.0048988, -0.0011364, -0.0034115, 0.0032973
5: -0.0006129, 0.0039166, -0.0005395, 0.0039815, -0.0045943, 0.0044562
6: -0.0066209, 0.0008647, -0.0066769, 0.0006627, -0.0072836, 0.0075415
7: -0.0254298, -0.0027351, -0.0241124, -0.0024790, -0.0180313, 0.0186095
8: 0.9722265, 0.9940638, 0.9734337, 0.9943037, -0.0220772, 0.0206301
9: -0.0061136, 0.0089945, -0.0062750, 0.0081323, -0.0130137, 0.0134227

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_A2_A1_B1_B1_B1_B2_A1

### Relational analysis result of IS_A1_A2_A1_B1_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0119762, upper bound: 0.0120751
time: 1.19 seconds

## Relational analysis of IS_A1_A2_A1_B1_B1_B1_B2_A2

### Relational analysis result of IS_A1_A2_A1_B1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119762, upper bound: 0.0126245
time: 1.11 seconds

## BFS IS instance: IS_A1_A2_A1_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0083055, 0.0187768, 0.0083241, 0.0185306, -0.0102251, 0.0104527
1: -0.0039174, 0.0011571, -0.0039318, 0.0010127, -0.0049301, 0.0050889
2: 0.0034030, 0.0090757, 0.0034854, 0.0090551, -0.0056521, 0.0055902
3: -0.0009109, 0.0036682, -0.0008622, 0.0036666, -0.0045774, 0.0045304
4: -0.0051203, -0.0011813, -0.0049351, -0.0012007, -0.0033805, 0.0033581
5: -0.0006129, 0.0039166, -0.0005348, 0.0039160, -0.0045289, 0.0044515
6: -0.0066209, 0.0008647, -0.0065998, 0.0007472, -0.0073681, 0.0074645
7: -0.0254298, -0.0027351, -0.0243155, -0.0028397, -0.0179061, 0.0188934
8: 0.9722265, 0.9940638, 0.9732951, 0.9939808, -0.0217543, 0.0207687
9: -0.0061136, 0.0089945, -0.0060440, 0.0082579, -0.0132026, 0.0132916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_A2_A1_B1_B1_B2_B2_A1

### Relational analysis result of IS_A1_A2_A1_B1_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0119762, upper bound: 0.0120789
time: 1.25 seconds

## Relational analysis of IS_A1_A2_A1_B1_B1_B2_B2_A2

### Relational analysis result of IS_A1_A2_A1_B1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119762, upper bound: 0.0126152
time: 1.14 seconds

## BFS IS instance: IS_A1_A2_A1_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0083055, 0.0187768, 0.0068723, 0.0185218, -0.0102163, 0.0119045
1: -0.0039174, 0.0011571, -0.0042149, 0.0012676, -0.0051851, 0.0053720
2: 0.0034030, 0.0090757, 0.0034898, 0.0101097, -0.0067067, 0.0055859
3: -0.0009109, 0.0036682, -0.0015385, 0.0036695, -0.0045804, 0.0052067
4: -0.0051203, -0.0011813, -0.0049032, -0.0009466, -0.0036573, 0.0033195
5: -0.0006129, 0.0039166, -0.0005420, 0.0045222, -0.0051351, 0.0044586
6: -0.0066209, 0.0008647, -0.0068371, 0.0009945, -0.0076154, 0.0077018
7: -0.0254298, -0.0027351, -0.0241178, -0.0013553, -0.0194617, 0.0187049
8: 0.9722265, 0.9940638, 0.9734010, 0.9954968, -0.0232703, 0.0206628
9: -0.0061136, 0.0089945, -0.0070430, 0.0081484, -0.0130813, 0.0143677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_A2_A1_B1_B2_B1_B2_A1

### Relational analysis result of IS_A1_A2_A1_B1_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0118908, upper bound: 0.0120298
time: 1.50 seconds

## Relational analysis of IS_A1_A2_A1_B1_B2_B1_B2_A2

### Relational analysis result of IS_A1_A2_A1_B1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118908, upper bound: 0.0126498
time: 1.12 seconds

## BFS IS instance: IS_A1_A2_A1_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0083731, 0.0187404, 0.0069627, 0.0185597, -0.0101865, 0.0117777
1: -0.0039008, 0.0011301, -0.0042152, 0.0012591, -0.0051599, 0.0053452
2: 0.0034148, 0.0090280, 0.0034765, 0.0100375, -0.0066227, 0.0055516
3: -0.0008830, 0.0036571, -0.0014544, 0.0037225, -0.0046055, 0.0051115
4: -0.0050972, -0.0011876, -0.0049666, -0.0009945, -0.0037643, 0.0032416
5: -0.0005991, 0.0038871, -0.0005493, 0.0044909, -0.0050900, 0.0044363
6: -0.0066180, 0.0008089, -0.0067734, 0.0011315, -0.0077495, 0.0075824
7: -0.0252976, -0.0027738, -0.0244683, -0.0016217, -0.0212162, 0.0170409
8: 0.9723550, 0.9940153, 0.9731277, 0.9952570, -0.0229020, 0.0208876
9: -0.0060862, 0.0089061, -0.0068703, 0.0083713, -0.0127196, 0.0147822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1_B1_B2_B2_A2_B1

### Relational analysis result of IS_A1_A2_A1_B1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0124633, upper bound: 0.0124381
time: 1.76 seconds

## Relational analysis of IS_A1_A2_A1_B1_B2_B2_A2_B2

### Relational analysis result of IS_A1_A2_A1_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125585, upper bound: 0.0123267
time: 1.65 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0074956, 0.0183737, 0.0084838, 0.0186830, -0.0111874, 0.0098899
1: -0.0040560, 0.0010893, -0.0038724, 0.0010866, -0.0051426, 0.0049617
2: 0.0035334, 0.0096780, 0.0034334, 0.0089503, -0.0054169, 0.0062446
3: -0.0012796, 0.0035773, -0.0008371, 0.0036347, -0.0049143, 0.0044144
4: -0.0048447, -0.0010209, -0.0050543, -0.0011984, -0.0029397, 0.0033420
5: -0.0004459, 0.0042453, -0.0005741, 0.0038384, -0.0042843, 0.0048194
6: -0.0067630, 0.0006696, -0.0066152, 0.0007239, -0.0074869, 0.0072848
7: -0.0238134, -0.0017983, -0.0250515, -0.0028381, -0.0150960, 0.0173559
8: 0.9737458, 0.9949877, 0.9725947, 0.9939387, -0.0201930, 0.0223930
9: -0.0067346, 0.0079335, -0.0060412, 0.0087427, -0.0132282, 0.0116699

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1_B2_B1_B1_A1_A1

### Relational analysis result of IS_A1_A2_A1_B2_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0118003, upper bound: 0.0120265
time: 1.37 seconds

## Relational analysis of IS_A1_A2_A1_B2_B1_B1_A1_A2

### Relational analysis result of IS_A1_A2_A1_B2_B1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0111648, upper bound: 0.0120289
time: 1.12 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0083731, 0.0187404, 0.0083055, 0.0187768, -0.0104036, 0.0104350
1: -0.0039008, 0.0011301, -0.0039174, 0.0011571, -0.0050579, 0.0050475
2: 0.0034148, 0.0090280, 0.0034030, 0.0090757, -0.0056609, 0.0056250
3: -0.0008830, 0.0036571, -0.0009109, 0.0036682, -0.0045512, 0.0045680
4: -0.0050972, -0.0011876, -0.0051203, -0.0011813, -0.0034308, 0.0032643
5: -0.0005991, 0.0038871, -0.0006129, 0.0039166, -0.0045157, 0.0045000
6: -0.0066180, 0.0008089, -0.0066209, 0.0008647, -0.0074827, 0.0074298
7: -0.0252976, -0.0027738, -0.0254298, -0.0027351, -0.0192849, 0.0169595
8: 0.9723550, 0.9940153, 0.9722265, 0.9940638, -0.0217088, 0.0217888
9: -0.0060862, 0.0089061, -0.0061136, 0.0089945, -0.0129096, 0.0135535

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_A2_A1_B2_B1_B1_A2_B1

### Relational analysis result of IS_A1_A2_A1_B2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127002, upper bound: 0.0120983
time: 1.47 seconds

## Relational analysis of IS_A1_A2_A1_B2_B1_B1_A2_B2

### Relational analysis result of IS_A1_A2_A1_B2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127002, upper bound: 0.0127631
time: 1.06 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0074956, 0.0183737, 0.0086288, 0.0186925, -0.0111969, 0.0097448
1: -0.0040560, 0.0010893, -0.0038581, 0.0010692, -0.0051253, 0.0049474
2: 0.0035334, 0.0096780, 0.0034291, 0.0088346, -0.0053012, 0.0062489
3: -0.0012796, 0.0035773, -0.0007387, 0.0036749, -0.0049545, 0.0043159
4: -0.0048447, -0.0010209, -0.0051039, -0.0012486, -0.0029164, 0.0034222
5: -0.0004459, 0.0042453, -0.0005707, 0.0037841, -0.0042300, 0.0048160
6: -0.0067630, 0.0006696, -0.0065488, 0.0008289, -0.0075919, 0.0072184
7: -0.0238134, -0.0017983, -0.0253290, -0.0031261, -0.0150310, 0.0178595
8: 0.9737458, 0.9949877, 0.9723949, 0.9936864, -0.0199406, 0.0225928
9: -0.0067346, 0.0079335, -0.0058510, 0.0089153, -0.0134862, 0.0115632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A1_A1

### Relational analysis result of IS_A1_A2_A1_B2_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0118011, upper bound: 0.0120439
time: 1.18 seconds

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A1_A2

### Relational analysis result of IS_A1_A2_A1_B2_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0111648, upper bound: 0.0120435
time: 1.31 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0083731, 0.0187404, 0.0084286, 0.0187860, -0.0104128, 0.0103119
1: -0.0039008, 0.0011301, -0.0039089, 0.0011429, -0.0050437, 0.0050390
2: 0.0034148, 0.0090280, 0.0033987, 0.0089764, -0.0055616, 0.0056293
3: -0.0008830, 0.0036571, -0.0008197, 0.0037132, -0.0045962, 0.0044768
4: -0.0050972, -0.0011876, -0.0051675, -0.0012301, -0.0034057, 0.0033414
5: -0.0005991, 0.0038871, -0.0006096, 0.0038723, -0.0044713, 0.0044967
6: -0.0066180, 0.0008089, -0.0065548, 0.0009793, -0.0075973, 0.0073637
7: -0.0252976, -0.0027738, -0.0256950, -0.0030151, -0.0191353, 0.0174354
8: 0.9723550, 0.9940153, 0.9720360, 0.9938253, -0.0214703, 0.0219793
9: -0.0060862, 0.0089061, -0.0059282, 0.0091599, -0.0131620, 0.0134448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A2_B1

### Relational analysis result of IS_A1_A2_A1_B2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127002, upper bound: 0.0121138
time: 1.19 seconds

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A2_B2

### Relational analysis result of IS_A1_A2_A1_B2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127002, upper bound: 0.0127518
time: 2.20 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0074956, 0.0183737, 0.0071405, 0.0186908, -0.0111952, 0.0112332
1: -0.0040560, 0.0010893, -0.0041470, 0.0013070, -0.0053631, 0.0052363
2: 0.0035334, 0.0096780, 0.0034306, 0.0099186, -0.0063852, 0.0062474
3: -0.0012796, 0.0035773, -0.0014277, 0.0036767, -0.0049563, 0.0050050
4: -0.0048447, -0.0010209, -0.0050735, -0.0009960, -0.0031975, 0.0033794
5: -0.0004459, 0.0042453, -0.0005773, 0.0044020, -0.0048479, 0.0048226
6: -0.0067630, 0.0006696, -0.0067881, 0.0010529, -0.0078160, 0.0074577
7: -0.0238134, -0.0017983, -0.0251347, -0.0016369, -0.0165808, 0.0175893
8: 0.9737458, 0.9949877, 0.9724925, 0.9952161, -0.0214703, 0.0224952
9: -0.0067346, 0.0079335, -0.0068574, 0.0088088, -0.0133528, 0.0126595

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1_B2_B2_B1_A1_A1

### Relational analysis result of IS_A1_A2_A1_B2_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0117182, upper bound: 0.0119752
time: 1.16 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2_B1_A1_A2

### Relational analysis result of IS_A1_A2_A1_B2_B2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0110673, upper bound: 0.0119732
time: 1.38 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0083731, 0.0187404, 0.0069434, 0.0187820, -0.0104088, 0.0117971
1: -0.0039008, 0.0011301, -0.0041974, 0.0013851, -0.0052859, 0.0053274
2: 0.0034148, 0.0090280, 0.0034013, 0.0100584, -0.0066436, 0.0056267
3: -0.0008830, 0.0036571, -0.0015091, 0.0037173, -0.0046003, 0.0051662
4: -0.0050972, -0.0011876, -0.0051387, -0.0009777, -0.0036839, 0.0032984
5: -0.0005991, 0.0038871, -0.0006160, 0.0044893, -0.0050884, 0.0045031
6: -0.0066180, 0.0008089, -0.0067939, 0.0012106, -0.0078287, 0.0076028
7: -0.0252976, -0.0027738, -0.0255066, -0.0015269, -0.0206811, 0.0171736
8: 0.9723550, 0.9940153, 0.9721356, 0.9953537, -0.0229987, 0.0218797
9: -0.0060862, 0.0089061, -0.0069351, 0.0090564, -0.0130242, 0.0145310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_A2_A1_B2_B2_B1_A2_B1

### Relational analysis result of IS_A1_A2_A1_B2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126766, upper bound: 0.0121447
time: 1.23 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2_B1_A2_B2

### Relational analysis result of IS_A1_A2_A1_B2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126766, upper bound: 0.0127828
time: 1.67 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 4.24 seconds
IS_A1_A1_A1_B1_B1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0120751, upper bound: 0.0110390
IS_A1_A1_A1_B1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0120751, upper bound: 0.0126328
IS_A1_A1_A1_B1_B1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0120751, upper bound: 0.0111862
IS_A1_A1_A1_B1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0120751, upper bound: 0.0126255
IS_A1_A1_A1_B1_B2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0120084, upper bound: 0.0109371
IS_A1_A1_A1_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0120084, upper bound: 0.0126644
IS_A1_A1_A1_B1_B2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0120084, upper bound: 0.0110722
IS_A1_A1_A1_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0120084, upper bound: 0.0126659
IS_A1_A1_A1_B2_B1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0120751, upper bound: 0.0119762
IS_A1_A1_A1_B2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0120751, upper bound: 0.0128846
IS_A1_A1_A1_B2_B1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0120751, upper bound: 0.0120418
IS_A1_A1_A1_B2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0120751, upper bound: 0.0128991
IS_A1_A1_A1_B2_B2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0120117, upper bound: 0.0120003
IS_A1_A1_A1_B2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0120117, upper bound: 0.0129421
IS_A1_A1_A1_B2_B2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0120117, upper bound: 0.0120718
IS_A1_A1_A1_B2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0120117, upper bound: 0.0129595
IS_A1_A1_A2_B1_B1_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0109371, upper bound: 0.0120084
IS_A1_A1_A2_B1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0109371, upper bound: 0.0125937
IS_A1_A1_A2_B1_B1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0120261, upper bound: 0.0108956
IS_A1_A1_A2_B1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0120261, upper bound: 0.0125937
IS_A1_A1_A2_B1_B2_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0109371, upper bound: 0.0120123
IS_A1_A1_A2_B1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0109371, upper bound: 0.0125802
IS_A1_A1_A2_B1_B2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0120267, upper bound: 0.0110270
IS_A1_A1_A2_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0120267, upper bound: 0.0125802
IS_A1_A1_A2_B2_B1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0120298, upper bound: 0.0118908
IS_A1_A1_A2_B2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0120298, upper bound: 0.0128285
IS_A1_A1_A2_B2_B1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0120298, upper bound: 0.0118908
IS_A1_A1_A2_B2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0120298, upper bound: 0.0128285
IS_A1_A1_A2_B2_B2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0120299, upper bound: 0.0119569
IS_A1_A1_A2_B2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0120299, upper bound: 0.0128323
IS_A1_A1_A2_B2_B2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0120298, upper bound: 0.0119569
IS_A1_A1_A2_B2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0120298, upper bound: 0.0128323
IS_A1_A2_A1_B1_B1_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0119762, upper bound: 0.0120751
IS_A1_A2_A1_B1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0119762, upper bound: 0.0126245
IS_A1_A2_A1_B1_B1_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0119762, upper bound: 0.0120789
IS_A1_A2_A1_B1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0119762, upper bound: 0.0126152
IS_A1_A2_A1_B1_B2_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0118908, upper bound: 0.0120298
IS_A1_A2_A1_B1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0118908, upper bound: 0.0126498
IS_A1_A2_A1_B1_B2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0124633, upper bound: 0.0124381
IS_A1_A2_A1_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0125585, upper bound: 0.0123267
IS_A1_A2_A1_B2_B1_B1_A1_A1, status: Status.VERIFIED, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0118003, upper bound: 0.0120265
IS_A1_A2_A1_B2_B1_B1_A1_A2, status: Status.VERIFIED, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0111648, upper bound: 0.0120289
IS_A1_A2_A1_B2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0127002, upper bound: 0.0120983
IS_A1_A2_A1_B2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0127002, upper bound: 0.0127631
IS_A1_A2_A1_B2_B1_B2_A1_A1, status: Status.VERIFIED, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0118011, upper bound: 0.0120439
IS_A1_A2_A1_B2_B1_B2_A1_A2, status: Status.VERIFIED, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0111648, upper bound: 0.0120435
IS_A1_A2_A1_B2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0127002, upper bound: 0.0121138
IS_A1_A2_A1_B2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0127002, upper bound: 0.0127518
IS_A1_A2_A1_B2_B2_B1_A1_A1, status: Status.VERIFIED, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0117182, upper bound: 0.0119752
IS_A1_A2_A1_B2_B2_B1_A1_A2, status: Status.VERIFIED, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0110673, upper bound: 0.0119732
IS_A1_A2_A1_B2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0126766, upper bound: 0.0121447
IS_A1_A2_A1_B2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.24
Output dim: 8, lower bound: -0.0126766, upper bound: 0.0127828
IS_A1_A2_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0123131, upper bound: 0.0125577
IS_A1_A2_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0129180, upper bound: 0.0127786
IS_A1_A2_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0129421, upper bound: 0.0125885
IS_A1_A2_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0129421, upper bound: 0.0125885
IS_A1_A2_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0129421, upper bound: 0.0125750
IS_A1_A2_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0129421, upper bound: 0.0125750
IS_A1_A2_A2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0127519, upper bound: 0.0120928
IS_A1_A2_A2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0130127, upper bound: 0.0127363
IS_A1_A2_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0124169, upper bound: 0.0125306
IS_A1_A2_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0130127, upper bound: 0.0127363
IS_A1_A2_A2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0127519, upper bound: 0.0121065
IS_A1_A2_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0130127, upper bound: 0.0127229
IS_A1_A2_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0124169, upper bound: 0.0125207
IS_A1_A2_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0130127, upper bound: 0.0127229
IS_A2_A1_A1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0125802, upper bound: 0.0126003
IS_A2_A1_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0125802, upper bound: 0.0126644
IS_A2_A1_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0125750, upper bound: 0.0128582
IS_A2_A1_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0125750, upper bound: 0.0129421
IS_A2_A1_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0125802, upper bound: 0.0125840
IS_A2_A1_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0125802, upper bound: 0.0126532
IS_A2_A1_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0125750, upper bound: 0.0128543
IS_A2_A1_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0125750, upper bound: 0.0129376
IS_A2_A1_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0126659, upper bound: 0.0125937
IS_A2_A1_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0126659, upper bound: 0.0125937
IS_A2_A1_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0126465, upper bound: 0.0128285
IS_A2_A1_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0126465, upper bound: 0.0128285
IS_A2_A1_A2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0126659, upper bound: 0.0125735
IS_A2_A1_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0126659, upper bound: 0.0125735
IS_A2_A1_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0126465, upper bound: 0.0128217
IS_A2_A1_A2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0126465, upper bound: 0.0128217
IS_A2_A2_A1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0128323, upper bound: 0.0125926
IS_A2_A2_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0128323, upper bound: 0.0126498
IS_A2_A2_A1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0128323, upper bound: 0.0125738
IS_A2_A2_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0128323, upper bound: 0.0126380
IS_A2_A2_A1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0126746, upper bound: 0.0120975
IS_A2_A2_A1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0129155, upper bound: 0.0127383
IS_A2_A2_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0123429, upper bound: 0.0125634
IS_A2_A2_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0129155, upper bound: 0.0127828
IS_A2_A2_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0123536, upper bound: 0.0125205
IS_A2_A2_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0129155, upper bound: 0.0127214
IS_A2_A2_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0123536, upper bound: 0.0125550
IS_A2_A2_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0129155, upper bound: 0.0127713
IS_A2_A2_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0129595, upper bound: 0.0125885
IS_A2_A2_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0129595, upper bound: 0.0125885
IS_A2_A2_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0129595, upper bound: 0.0125672
IS_A2_A2_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0129595, upper bound: 0.0125672
IS_A2_A2_A2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0127620, upper bound: 0.0120928
IS_A2_A2_A2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0130274, upper bound: 0.0127363
IS_A2_A2_A2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0127620, upper bound: 0.0120928
IS_A2_A2_A2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0130274, upper bound: 0.0127363
IS_A2_A2_A2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0127625, upper bound: 0.0121078
IS_A2_A2_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0130274, upper bound: 0.0127174
IS_A2_A2_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0124688, upper bound: 0.0125175
IS_A2_A2_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 8, lower bound: -0.0130274, upper bound: 0.0127174

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.60 + 598.01 = 601.62 seconds
