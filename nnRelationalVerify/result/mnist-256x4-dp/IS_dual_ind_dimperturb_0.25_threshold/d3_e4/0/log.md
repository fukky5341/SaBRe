## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00357444


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041028, -0.0029423, -0.0041028, -0.0029423, -0.0011605, 0.0011605)
1: (-0.0063066, -0.0035117, -0.0063066, -0.0035117, -0.0027950, 0.0027950)
2: (0.9663123, 0.9713900, 0.9663123, 0.9713900, -0.0050777, 0.0050777)
3: (0.0168824, 0.0352833, 0.0168824, 0.0352833, -0.0134085, 0.0134085)
4: (-0.0033765, -0.0010992, -0.0033765, -0.0010992, -0.0022773, 0.0022773)
5: (0.0129680, 0.0152722, 0.0129680, 0.0152722, -0.0023042, 0.0023042)
6: (0.0032027, 0.0051664, 0.0032027, 0.0051664, -0.0019636, 0.0019636)
7: (-0.0169222, -0.0121535, -0.0169222, -0.0121535, -0.0047688, 0.0047688)
8: (0.0033039, 0.0070872, 0.0033039, 0.0070872, -0.0037833, 0.0037833)
9: (0.0028822, 0.0104716, 0.0028822, 0.0104716, -0.0075184, 0.0075184)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.76 + 1.47 = 3.23 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0045354, upper bound: 0.0045354

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045354, upper bound: 0.0045354
time: 0.56 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045354, upper bound: 0.0045354
time: 0.61 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.38 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.38
Output dim: 2, lower bound: -0.0045354, upper bound: 0.0045354
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.38
Output dim: 2, lower bound: -0.0045354, upper bound: 0.0045354

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0040991, -0.0029570, -0.0041012, -0.0029779, -0.0011212, 0.0011442
1: -0.0061656, -0.0035266, -0.0062454, -0.0035480, -0.0026177, 0.0027188
2: 0.9663723, 0.9713833, 0.9664580, 0.9713738, -0.0050015, 0.0049254
3: 0.0181304, 0.0352341, 0.0174247, 0.0351639, -0.0118728, 0.0127205
4: -0.0033728, -0.0011254, -0.0033675, -0.0011627, -0.0022101, 0.0022421
5: 0.0129834, 0.0151763, 0.0130055, 0.0152305, -0.0022471, 0.0021708
6: 0.0032267, 0.0051645, 0.0032608, 0.0051619, -0.0019352, 0.0019037
7: -0.0169095, -0.0124769, -0.0168913, -0.0122940, -0.0046155, 0.0044144
8: 0.0033140, 0.0068306, 0.0033284, 0.0069757, -0.0036617, 0.0035022
9: 0.0029107, 0.0100100, 0.0029514, 0.0102710, -0.0067719, 0.0069319

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045354, upper bound: 0.0045297
time: 0.64 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045354, upper bound: 0.0045354
time: 0.60 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0041015, -0.0029806, -0.0041024, -0.0029543, -0.0011472, 0.0011218
1: -0.0062564, -0.0035508, -0.0062909, -0.0035239, -0.0027325, 0.0027401
2: 0.9664692, 0.9713726, 0.9663613, 0.9713845, -0.0049153, 0.0050113
3: 0.0173274, 0.0351546, 0.0170219, 0.0352431, -0.0122299, 0.0132257
4: -0.0033667, -0.0011676, -0.0033735, -0.0011206, -0.0022462, 0.0022058
5: 0.0130084, 0.0152380, 0.0129806, 0.0152615, -0.0022531, 0.0022574
6: 0.0032654, 0.0051616, 0.0032223, 0.0051649, -0.0018995, 0.0019392
7: -0.0168889, -0.0122688, -0.0169118, -0.0121896, -0.0046993, 0.0046430
8: 0.0033303, 0.0069957, 0.0033121, 0.0070585, -0.0037282, 0.0036835
9: 0.0029568, 0.0103070, 0.0029055, 0.0104200, -0.0073858, 0.0069650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045354, upper bound: 0.0045297
time: 0.57 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045354, upper bound: 0.0045354
time: 0.62 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.97 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.97
Output dim: 2, lower bound: -0.0045354, upper bound: 0.0045297
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.97
Output dim: 2, lower bound: -0.0045354, upper bound: 0.0045354
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.97
Output dim: 2, lower bound: -0.0045354, upper bound: 0.0045297
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.97
Output dim: 2, lower bound: -0.0045354, upper bound: 0.0045354

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040965, -0.0029708, -0.0040942, -0.0028771, -0.0012194, 0.0011234
1: -0.0060700, -0.0035407, -0.0059834, -0.0034451, -0.0026249, 0.0024427
2: 0.9664288, 0.9713771, 0.9660456, 0.9714196, -0.0049908, 0.0053315
3: 0.0189769, 0.0351878, 0.0197438, 0.0355019, -0.0113295, 0.0101588
4: -0.0033693, -0.0011500, -0.0033932, -0.0009829, -0.0023863, 0.0022432
5: 0.0129980, 0.0151112, 0.0128992, 0.0150522, -0.0020542, 0.0022120
6: 0.0032492, 0.0051628, 0.0030963, 0.0051745, -0.0019253, 0.0020665
7: -0.0168975, -0.0126963, -0.0169789, -0.0128950, -0.0040025, 0.0042826
8: 0.0033235, 0.0066565, 0.0032589, 0.0064988, -0.0031753, 0.0033976
9: 0.0029376, 0.0096970, 0.0027556, 0.0094134, -0.0059982, 0.0062736

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045287, upper bound: 0.0045297
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045287, upper bound: 0.0045297
time: 0.63 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040991, -0.0029570, -0.0041007, -0.0029828, -0.0011162, 0.0011437
1: -0.0061656, -0.0035266, -0.0062256, -0.0035530, -0.0026126, 0.0026990
2: 0.9663723, 0.9713833, 0.9664780, 0.9713717, -0.0049993, 0.0049053
3: 0.0181304, 0.0352341, 0.0175994, 0.0351473, -0.0118592, 0.0108975
4: -0.0033728, -0.0011254, -0.0033662, -0.0011715, -0.0022013, 0.0022408
5: 0.0129834, 0.0151763, 0.0130107, 0.0152171, -0.0022336, 0.0021655
6: 0.0032267, 0.0051645, 0.0032689, 0.0051613, -0.0019346, 0.0018956
7: -0.0169095, -0.0124769, -0.0168870, -0.0123392, -0.0045702, 0.0044101
8: 0.0033140, 0.0068306, 0.0033318, 0.0069397, -0.0036258, 0.0034988
9: 0.0029107, 0.0100100, 0.0029610, 0.0102064, -0.0062382, 0.0069077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045287, upper bound: 0.0045354
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045287, upper bound: 0.0045354
time: 0.62 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040989, -0.0029947, -0.0040955, -0.0028528, -0.0012461, 0.0011008
1: -0.0061615, -0.0035651, -0.0060318, -0.0034203, -0.0027412, 0.0024666
2: 0.9665268, 0.9713662, 0.9659461, 0.9714307, -0.0049039, 0.0054201
3: 0.0181670, 0.0351074, 0.0193153, 0.0355836, -0.0113042, 0.0109699
4: -0.0033632, -0.0011927, -0.0033994, -0.0009395, -0.0024236, 0.0022066
5: 0.0130233, 0.0151735, 0.0128736, 0.0150852, -0.0020619, 0.0022999
6: 0.0032883, 0.0051598, 0.0030566, 0.0051776, -0.0018893, 0.0021032
7: -0.0168767, -0.0124864, -0.0170000, -0.0127840, -0.0040927, 0.0045137
8: 0.0033400, 0.0068230, 0.0032421, 0.0065869, -0.0032469, 0.0035809
9: 0.0029841, 0.0099965, 0.0027083, 0.0095718, -0.0063847, 0.0066169

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045297, upper bound: 0.0045297
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045297, upper bound: 0.0045297
time: 0.66 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041015, -0.0029806, -0.0041019, -0.0029591, -0.0011424, 0.0011212
1: -0.0062564, -0.0035508, -0.0062711, -0.0035288, -0.0027276, 0.0027203
2: 0.9664692, 0.9713726, 0.9663811, 0.9713824, -0.0049132, 0.0049915
3: 0.0173274, 0.0351546, 0.0171968, 0.0352269, -0.0122167, 0.0115467
4: -0.0033667, -0.0011676, -0.0033722, -0.0011292, -0.0022375, 0.0022046
5: 0.0130084, 0.0152380, 0.0129857, 0.0152480, -0.0022396, 0.0022523
6: 0.0032654, 0.0051616, 0.0032302, 0.0051643, -0.0018989, 0.0019314
7: -0.0168889, -0.0122688, -0.0169076, -0.0122349, -0.0046540, 0.0046388
8: 0.0033303, 0.0069957, 0.0033155, 0.0070225, -0.0036922, 0.0036802
9: 0.0029568, 0.0103070, 0.0029149, 0.0103553, -0.0069972, 0.0069358

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045297, upper bound: 0.0045354
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045297, upper bound: 0.0045354
time: 0.63 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.02 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 2, lower bound: -0.0045287, upper bound: 0.0045297
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 2, lower bound: -0.0045287, upper bound: 0.0045297
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 2, lower bound: -0.0045287, upper bound: 0.0045354
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 2, lower bound: -0.0045287, upper bound: 0.0045354
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 2, lower bound: -0.0045297, upper bound: 0.0045297
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 2, lower bound: -0.0045297, upper bound: 0.0045297
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 2, lower bound: -0.0045297, upper bound: 0.0045354
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 2, lower bound: -0.0045297, upper bound: 0.0045354

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040920, -0.0028809, -0.0040942, -0.0028771, -0.0012148, 0.0012133
1: -0.0058998, -0.0034490, -0.0059834, -0.0034451, -0.0024546, 0.0025344
2: 0.9660609, 0.9714180, 0.9660456, 0.9714196, -0.0053588, 0.0053724
3: 0.0204838, 0.0354894, 0.0197438, 0.0355019, -0.0096652, 0.0101577
4: -0.0033922, -0.0009896, -0.0033932, -0.0009829, -0.0024093, 0.0024036
5: 0.0129032, 0.0149954, 0.0128992, 0.0150522, -0.0021491, 0.0020961
6: 0.0031024, 0.0051741, 0.0030963, 0.0051745, -0.0020721, 0.0020778
7: -0.0169756, -0.0130868, -0.0169789, -0.0128950, -0.0040806, 0.0038921
8: 0.0032615, 0.0063467, 0.0032589, 0.0064988, -0.0032374, 0.0030878
9: 0.0027628, 0.0091397, 0.0027556, 0.0094134, -0.0059574, 0.0059681

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045287, upper bound: 0.0045287
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045287, upper bound: 0.0045297
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040985, -0.0029607, -0.0040942, -0.0028771, -0.0012214, 0.0011335
1: -0.0061461, -0.0035304, -0.0059834, -0.0034451, -0.0027010, 0.0024529
2: 0.9663876, 0.9713816, 0.9660456, 0.9714196, -0.0050320, 0.0053360
3: 0.0183031, 0.0352216, 0.0197438, 0.0355019, -0.0122146, 0.0101850
4: -0.0033718, -0.0011320, -0.0033932, -0.0009829, -0.0023889, 0.0022611
5: 0.0129874, 0.0151630, 0.0128992, 0.0150522, -0.0020649, 0.0022638
6: 0.0032328, 0.0051641, 0.0030963, 0.0051745, -0.0019418, 0.0020678
7: -0.0169062, -0.0125216, -0.0169789, -0.0128950, -0.0040112, 0.0044573
8: 0.0033165, 0.0067951, 0.0032589, 0.0064988, -0.0031823, 0.0035362
9: 0.0029180, 0.0099462, 0.0027556, 0.0094134, -0.0060368, 0.0065653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045287, upper bound: 0.0045286
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045287, upper bound: 0.0045297
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040920, -0.0028809, -0.0041007, -0.0029828, -0.0011091, 0.0012198
1: -0.0058998, -0.0034490, -0.0062256, -0.0035530, -0.0023468, 0.0027767
2: 0.9660609, 0.9714180, 0.9664780, 0.9713717, -0.0053108, 0.0049399
3: 0.0204838, 0.0354894, 0.0175994, 0.0351473, -0.0092561, 0.0125838
4: -0.0033922, -0.0009896, -0.0033662, -0.0011715, -0.0022207, 0.0023766
5: 0.0129032, 0.0149954, 0.0130107, 0.0152171, -0.0023139, 0.0019846
6: 0.0031024, 0.0051741, 0.0032689, 0.0051613, -0.0020589, 0.0019052
7: -0.0169756, -0.0130868, -0.0168870, -0.0123392, -0.0046364, 0.0038002
8: 0.0032615, 0.0063467, 0.0033318, 0.0069397, -0.0036783, 0.0030149
9: 0.0027628, 0.0091397, 0.0029610, 0.0102064, -0.0065605, 0.0061787

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045287, upper bound: 0.0045354
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045287, upper bound: 0.0045354
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040985, -0.0029606, -0.0041007, -0.0029828, -0.0011157, 0.0011401
1: -0.0061461, -0.0035303, -0.0062256, -0.0035530, -0.0025931, 0.0026953
2: 0.9663871, 0.9713817, 0.9664780, 0.9713717, -0.0049846, 0.0049036
3: 0.0183031, 0.0352219, 0.0175994, 0.0351473, -0.0100547, 0.0108862
4: -0.0033719, -0.0011318, -0.0033662, -0.0011715, -0.0022003, 0.0022344
5: 0.0129873, 0.0151630, 0.0130107, 0.0152171, -0.0022298, 0.0021523
6: 0.0032326, 0.0051641, 0.0032689, 0.0051613, -0.0019287, 0.0018952
7: -0.0169063, -0.0125216, -0.0168870, -0.0123392, -0.0045671, 0.0043653
8: 0.0033165, 0.0067950, 0.0033318, 0.0069397, -0.0036233, 0.0034632
9: 0.0029178, 0.0099462, 0.0029610, 0.0102064, -0.0062138, 0.0065765

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045287, upper bound: 0.0045354
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045287, upper bound: 0.0045354
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040946, -0.0028789, -0.0040955, -0.0028528, -0.0012418, 0.0012166
1: -0.0059988, -0.0034469, -0.0060318, -0.0034203, -0.0025784, 0.0025848
2: 0.9660528, 0.9714189, 0.9659461, 0.9714307, -0.0053779, 0.0054728
3: 0.0196074, 0.0354961, 0.0193153, 0.0355836, -0.0097623, 0.0111530
4: -0.0033927, -0.0009860, -0.0033994, -0.0009395, -0.0024532, 0.0024133
5: 0.0129011, 0.0150627, 0.0128736, 0.0150852, -0.0021841, 0.0021892
6: 0.0030992, 0.0051743, 0.0030566, 0.0051776, -0.0020784, 0.0021177
7: -0.0169774, -0.0128597, -0.0170000, -0.0127840, -0.0041934, 0.0041404
8: 0.0032601, 0.0065269, 0.0032421, 0.0065869, -0.0033268, 0.0032848
9: 0.0027590, 0.0094638, 0.0027083, 0.0095718, -0.0064461, 0.0063282

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045297, upper bound: 0.0045287
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045297, upper bound: 0.0045297
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041010, -0.0029852, -0.0040955, -0.0028528, -0.0012481, 0.0011103
1: -0.0062366, -0.0035555, -0.0060318, -0.0034203, -0.0028163, 0.0024763
2: 0.9664881, 0.9713705, 0.9659461, 0.9714307, -0.0049425, 0.0054244
3: 0.0175022, 0.0351393, 0.0193153, 0.0355836, -0.0121898, 0.0109925
4: -0.0033656, -0.0011758, -0.0033994, -0.0009395, -0.0024261, 0.0022236
5: 0.0130132, 0.0152246, 0.0128736, 0.0150852, -0.0020719, 0.0023510
6: 0.0032728, 0.0051610, 0.0030566, 0.0051776, -0.0019048, 0.0021044
7: -0.0168849, -0.0123141, -0.0170000, -0.0127840, -0.0041009, 0.0046860
8: 0.0033335, 0.0069597, 0.0032421, 0.0065869, -0.0032535, 0.0037176
9: 0.0029657, 0.0102423, 0.0027083, 0.0095718, -0.0064045, 0.0068184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045297, upper bound: 0.0045287
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045297, upper bound: 0.0045297
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040946, -0.0028789, -0.0041019, -0.0029591, -0.0011355, 0.0012230
1: -0.0059988, -0.0034469, -0.0062711, -0.0035288, -0.0024700, 0.0028242
2: 0.9660528, 0.9714189, 0.9663811, 0.9713824, -0.0053296, 0.0050378
3: 0.0196074, 0.0354961, 0.0171968, 0.0352269, -0.0097146, 0.0134705
4: -0.0033927, -0.0009860, -0.0033722, -0.0011292, -0.0022635, 0.0023862
5: 0.0129011, 0.0150627, 0.0129857, 0.0152480, -0.0023470, 0.0020770
6: 0.0030992, 0.0051743, 0.0032302, 0.0051643, -0.0020651, 0.0019441
7: -0.0169774, -0.0128597, -0.0169076, -0.0122349, -0.0047425, 0.0040479
8: 0.0032601, 0.0065269, 0.0033155, 0.0070225, -0.0037624, 0.0032114
9: 0.0027590, 0.0094638, 0.0029149, 0.0103553, -0.0070513, 0.0063116

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045297, upper bound: 0.0045354
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045297, upper bound: 0.0045354
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041010, -0.0029852, -0.0041019, -0.0029591, -0.0011418, 0.0011167
1: -0.0062366, -0.0035554, -0.0062711, -0.0035288, -0.0027078, 0.0027157
2: 0.9664876, 0.9713705, 0.9663811, 0.9713824, -0.0048947, 0.0049894
3: 0.0175022, 0.0351395, 0.0171968, 0.0352269, -0.0104097, 0.0117871
4: -0.0033656, -0.0011757, -0.0033722, -0.0011292, -0.0022364, 0.0021966
5: 0.0130132, 0.0152246, 0.0129857, 0.0152480, -0.0022349, 0.0022388
6: 0.0032727, 0.0051610, 0.0032302, 0.0051643, -0.0018915, 0.0019308
7: -0.0168850, -0.0123141, -0.0169076, -0.0122349, -0.0046500, 0.0045935
8: 0.0033334, 0.0069597, 0.0033155, 0.0070225, -0.0036891, 0.0036443
9: 0.0029655, 0.0102423, 0.0029149, 0.0103553, -0.0066475, 0.0065619

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045297, upper bound: 0.0045354
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045297, upper bound: 0.0045354
time: 0.65 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.16 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0045287, upper bound: 0.0045287
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0045287, upper bound: 0.0045297
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0045287, upper bound: 0.0045286
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0045287, upper bound: 0.0045297
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0045287, upper bound: 0.0045354
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0045287, upper bound: 0.0045354
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0045287, upper bound: 0.0045354
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0045287, upper bound: 0.0045354
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0045297, upper bound: 0.0045287
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0045297, upper bound: 0.0045297
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0045297, upper bound: 0.0045287
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0045297, upper bound: 0.0045297
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0045297, upper bound: 0.0045354
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0045297, upper bound: 0.0045354
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0045297, upper bound: 0.0045354
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0045297, upper bound: 0.0045354

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040920, -0.0028809, -0.0040919, -0.0028798, -0.0012121, 0.0012110
1: -0.0058998, -0.0034490, -0.0058981, -0.0034479, -0.0024519, 0.0024492
2: 0.9660609, 0.9714180, 0.9660566, 0.9714184, -0.0053575, 0.0053614
3: 0.0204838, 0.0354894, 0.0204984, 0.0354929, -0.0093042, 0.0092880
4: -0.0033922, -0.0009896, -0.0033925, -0.0009877, -0.0024045, 0.0024029
5: 0.0129032, 0.0149954, 0.0129021, 0.0149942, -0.0020911, 0.0020933
6: 0.0031024, 0.0051741, 0.0031007, 0.0051742, -0.0020718, 0.0020734
7: -0.0169756, -0.0130868, -0.0169765, -0.0130906, -0.0038851, 0.0038898
8: 0.0032615, 0.0063467, 0.0032608, 0.0063437, -0.0030822, 0.0030859
9: 0.0027628, 0.0091397, 0.0027608, 0.0091344, -0.0056859, 0.0056970

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045277, upper bound: 0.0045124
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045124
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040920, -0.0028809, -0.0040946, -0.0028768, -0.0012152, 0.0012137
1: -0.0058998, -0.0034490, -0.0059988, -0.0034447, -0.0024550, 0.0025498
2: 0.9660609, 0.9714180, 0.9660440, 0.9714199, -0.0053590, 0.0053740
3: 0.0204838, 0.0354894, 0.0196074, 0.0355033, -0.0096627, 0.0106541
4: -0.0033922, -0.0009896, -0.0033933, -0.0009822, -0.0024100, 0.0024037
5: 0.0129032, 0.0149954, 0.0128988, 0.0150627, -0.0021596, 0.0020966
6: 0.0031024, 0.0051741, 0.0030957, 0.0051746, -0.0020722, 0.0020784
7: -0.0169756, -0.0130868, -0.0169792, -0.0128596, -0.0041160, 0.0038925
8: 0.0032615, 0.0063467, 0.0032586, 0.0065269, -0.0032654, 0.0030881
9: 0.0027628, 0.0091397, 0.0027548, 0.0094638, -0.0060591, 0.0059758

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045277, upper bound: 0.0045154
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045154
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040985, -0.0029607, -0.0040919, -0.0028798, -0.0012187, 0.0011312
1: -0.0061461, -0.0035304, -0.0058981, -0.0034479, -0.0026982, 0.0023677
2: 0.9663876, 0.9713816, 0.9660566, 0.9714184, -0.0050308, 0.0053250
3: 0.0183031, 0.0352216, 0.0204984, 0.0354929, -0.0118536, 0.0093153
4: -0.0033718, -0.0011320, -0.0033925, -0.0009877, -0.0023841, 0.0022604
5: 0.0129874, 0.0151630, 0.0129021, 0.0149942, -0.0020069, 0.0022609
6: 0.0032328, 0.0051641, 0.0031007, 0.0051742, -0.0019414, 0.0020634
7: -0.0169062, -0.0125216, -0.0169765, -0.0130906, -0.0038157, 0.0044549
8: 0.0033165, 0.0067951, 0.0032608, 0.0063437, -0.0030272, 0.0035343
9: 0.0029180, 0.0099462, 0.0027608, 0.0091344, -0.0057653, 0.0062942

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045322, upper bound: 0.0045124
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045186, upper bound: 0.0045124
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040985, -0.0029607, -0.0040946, -0.0028768, -0.0012218, 0.0011339
1: -0.0061461, -0.0035304, -0.0059988, -0.0034447, -0.0027014, 0.0024684
2: 0.9663876, 0.9713816, 0.9660440, 0.9714199, -0.0050322, 0.0053376
3: 0.0183031, 0.0352216, 0.0196074, 0.0355033, -0.0122120, 0.0106814
4: -0.0033718, -0.0011320, -0.0033933, -0.0009822, -0.0023896, 0.0022612
5: 0.0129874, 0.0151630, 0.0128988, 0.0150627, -0.0020754, 0.0022642
6: 0.0032328, 0.0051641, 0.0030957, 0.0051746, -0.0019418, 0.0020684
7: -0.0169062, -0.0125216, -0.0169792, -0.0128596, -0.0040466, 0.0044576
8: 0.0033165, 0.0067951, 0.0032586, 0.0065269, -0.0032103, 0.0035364
9: 0.0029180, 0.0099462, 0.0027548, 0.0094638, -0.0061385, 0.0065729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045322, upper bound: 0.0045154
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045186, upper bound: 0.0045154
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040920, -0.0028809, -0.0040985, -0.0029606, -0.0011314, 0.0012176
1: -0.0058998, -0.0034490, -0.0061461, -0.0035303, -0.0023694, 0.0026972
2: 0.9660609, 0.9714180, 0.9663871, 0.9713817, -0.0053208, 0.0050309
3: 0.0204838, 0.0354894, 0.0183031, 0.0352219, -0.0093349, 0.0118505
4: -0.0033922, -0.0009896, -0.0033719, -0.0011318, -0.0022604, 0.0023823
5: 0.0129032, 0.0149954, 0.0129873, 0.0151630, -0.0022598, 0.0020081
6: 0.0031024, 0.0051741, 0.0032326, 0.0051641, -0.0020617, 0.0019415
7: -0.0169756, -0.0130868, -0.0169063, -0.0125216, -0.0044540, 0.0038195
8: 0.0032615, 0.0063467, 0.0033165, 0.0067950, -0.0035336, 0.0030302
9: 0.0027628, 0.0091397, 0.0029178, 0.0099462, -0.0062951, 0.0057887

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045285, upper bound: 0.0045186
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045186
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040920, -0.0028809, -0.0041010, -0.0029852, -0.0011068, 0.0012201
1: -0.0058998, -0.0034490, -0.0062366, -0.0035554, -0.0023444, 0.0027877
2: 0.9660609, 0.9714180, 0.9664876, 0.9713705, -0.0053096, 0.0049303
3: 0.0204838, 0.0354894, 0.0175022, 0.0351395, -0.0095031, 0.0129188
4: -0.0033922, -0.0009896, -0.0033656, -0.0011757, -0.0022165, 0.0023760
5: 0.0129032, 0.0149954, 0.0130132, 0.0152246, -0.0023214, 0.0019822
6: 0.0031024, 0.0051741, 0.0032727, 0.0051610, -0.0020586, 0.0019013
7: -0.0169756, -0.0130868, -0.0168850, -0.0123141, -0.0046616, 0.0037982
8: 0.0032615, 0.0063467, 0.0033334, 0.0069597, -0.0036983, 0.0030133
9: 0.0027628, 0.0091397, 0.0029655, 0.0102423, -0.0066791, 0.0059495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045285, upper bound: 0.0045192
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045192
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040985, -0.0029606, -0.0040985, -0.0029606, -0.0011380, 0.0011380
1: -0.0061461, -0.0035303, -0.0061461, -0.0035303, -0.0026158, 0.0026158
2: 0.9663871, 0.9713817, 0.9663871, 0.9713817, -0.0049946, 0.0049946
3: 0.0183031, 0.0352219, 0.0183031, 0.0352219, -0.0100823, 0.0100823
4: -0.0033719, -0.0011318, -0.0033719, -0.0011318, -0.0022400, 0.0022400
5: 0.0129873, 0.0151630, 0.0129873, 0.0151630, -0.0021757, 0.0021757
6: 0.0032326, 0.0051641, 0.0032326, 0.0051641, -0.0019315, 0.0019315
7: -0.0169063, -0.0125216, -0.0169063, -0.0125216, -0.0043847, 0.0043847
8: 0.0033165, 0.0067950, 0.0033165, 0.0067950, -0.0034786, 0.0034786
9: 0.0029178, 0.0099462, 0.0029178, 0.0099462, -0.0059648, 0.0059648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045349, upper bound: 0.0045186
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045186, upper bound: 0.0045186
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040985, -0.0029606, -0.0041010, -0.0029852, -0.0011134, 0.0011404
1: -0.0061461, -0.0035303, -0.0062366, -0.0035554, -0.0025907, 0.0027063
2: 0.9663871, 0.9713817, 0.9664876, 0.9713705, -0.0049834, 0.0048940
3: 0.0183031, 0.0352219, 0.0175022, 0.0351395, -0.0103596, 0.0113612
4: -0.0033719, -0.0011318, -0.0033656, -0.0011757, -0.0021962, 0.0022338
5: 0.0129873, 0.0151630, 0.0130132, 0.0152246, -0.0022373, 0.0021498
6: 0.0032326, 0.0051641, 0.0032727, 0.0051610, -0.0019284, 0.0018913
7: -0.0169063, -0.0125216, -0.0168850, -0.0123141, -0.0045922, 0.0043633
8: 0.0033165, 0.0067950, 0.0033334, 0.0069597, -0.0036433, 0.0034616
9: 0.0029178, 0.0099462, 0.0029655, 0.0102423, -0.0062822, 0.0062011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045349, upper bound: 0.0045192
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045186, upper bound: 0.0045192
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040946, -0.0028789, -0.0040919, -0.0028798, -0.0012148, 0.0012130
1: -0.0059988, -0.0034469, -0.0058981, -0.0034479, -0.0025509, 0.0024512
2: 0.9660528, 0.9714189, 0.9660566, 0.9714184, -0.0053656, 0.0053623
3: 0.0196074, 0.0354961, 0.0204984, 0.0354929, -0.0106572, 0.0096428
4: -0.0033927, -0.0009860, -0.0033925, -0.0009877, -0.0024050, 0.0024064
5: 0.0129011, 0.0150627, 0.0129021, 0.0149942, -0.0020932, 0.0021607
6: 0.0030992, 0.0051743, 0.0031007, 0.0051742, -0.0020750, 0.0020736
7: -0.0169774, -0.0128597, -0.0169765, -0.0130906, -0.0038868, 0.0041169
8: 0.0032601, 0.0065269, 0.0032608, 0.0063437, -0.0030836, 0.0032661
9: 0.0027590, 0.0094638, 0.0027608, 0.0091344, -0.0059604, 0.0060569

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045280, upper bound: 0.0045124
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045154, upper bound: 0.0045124
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040946, -0.0028789, -0.0040946, -0.0028768, -0.0012179, 0.0012157
1: -0.0059988, -0.0034469, -0.0059988, -0.0034447, -0.0025540, 0.0025518
2: 0.9660528, 0.9714189, 0.9660440, 0.9714199, -0.0053671, 0.0053748
3: 0.0196074, 0.0354961, 0.0196074, 0.0355033, -0.0098629, 0.0098559
4: -0.0033927, -0.0009860, -0.0033933, -0.0009822, -0.0024105, 0.0024072
5: 0.0129011, 0.0150627, 0.0128988, 0.0150627, -0.0021617, 0.0021639
6: 0.0030992, 0.0051743, 0.0030957, 0.0051746, -0.0020754, 0.0020787
7: -0.0169774, -0.0128597, -0.0169792, -0.0128596, -0.0041177, 0.0041196
8: 0.0032601, 0.0065269, 0.0032586, 0.0065269, -0.0032668, 0.0032683
9: 0.0027590, 0.0094638, 0.0027548, 0.0094638, -0.0058472, 0.0058490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045280, upper bound: 0.0045154
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045154, upper bound: 0.0045154
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041010, -0.0029852, -0.0040919, -0.0028798, -0.0012211, 0.0011067
1: -0.0062366, -0.0035555, -0.0058981, -0.0034479, -0.0027887, 0.0023427
2: 0.9664881, 0.9713705, 0.9660566, 0.9714184, -0.0049303, 0.0053139
3: 0.0175022, 0.0351393, 0.0204984, 0.0354929, -0.0129219, 0.0094823
4: -0.0033656, -0.0011758, -0.0033925, -0.0009877, -0.0023778, 0.0022167
5: 0.0130132, 0.0152246, 0.0129021, 0.0149942, -0.0019810, 0.0023225
6: 0.0032728, 0.0051610, 0.0031007, 0.0051742, -0.0019014, 0.0020603
7: -0.0168849, -0.0123141, -0.0169765, -0.0130906, -0.0037943, 0.0046625
8: 0.0033335, 0.0069597, 0.0032608, 0.0063437, -0.0030102, 0.0036990
9: 0.0029657, 0.0102423, 0.0027608, 0.0091344, -0.0059188, 0.0066783

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045322, upper bound: 0.0045124
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045192, upper bound: 0.0045124
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041010, -0.0029852, -0.0040946, -0.0028768, -0.0012242, 0.0011094
1: -0.0062366, -0.0035555, -0.0059988, -0.0034447, -0.0027919, 0.0024433
2: 0.9664881, 0.9713705, 0.9660440, 0.9714199, -0.0049317, 0.0053265
3: 0.0175022, 0.0351393, 0.0196074, 0.0355033, -0.0122983, 0.0097739
4: -0.0033656, -0.0011758, -0.0033933, -0.0009822, -0.0023834, 0.0022175
5: 0.0130132, 0.0152246, 0.0128988, 0.0150627, -0.0020495, 0.0023258
6: 0.0032728, 0.0051610, 0.0030957, 0.0051746, -0.0019018, 0.0020653
7: -0.0168849, -0.0123141, -0.0169792, -0.0128596, -0.0040252, 0.0046652
8: 0.0033335, 0.0069597, 0.0032586, 0.0065269, -0.0031934, 0.0037011
9: 0.0029657, 0.0102423, 0.0027548, 0.0094638, -0.0058958, 0.0064485

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045322, upper bound: 0.0045154
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045192, upper bound: 0.0045154
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040946, -0.0028789, -0.0040985, -0.0029606, -0.0011340, 0.0012196
1: -0.0059988, -0.0034469, -0.0061461, -0.0035303, -0.0024685, 0.0026992
2: 0.9660528, 0.9714189, 0.9663871, 0.9713817, -0.0053289, 0.0050318
3: 0.0196074, 0.0354961, 0.0183031, 0.0352219, -0.0106833, 0.0122052
4: -0.0033927, -0.0009860, -0.0033719, -0.0011318, -0.0022609, 0.0023858
5: 0.0129011, 0.0150627, 0.0129873, 0.0151630, -0.0022619, 0.0020755
6: 0.0030992, 0.0051743, 0.0032326, 0.0051641, -0.0020649, 0.0019417
7: -0.0169774, -0.0128597, -0.0169063, -0.0125216, -0.0044557, 0.0040467
8: 0.0032601, 0.0065269, 0.0033165, 0.0067950, -0.0035349, 0.0032104
9: 0.0027590, 0.0094638, 0.0029178, 0.0099462, -0.0065695, 0.0061543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045296, upper bound: 0.0045186
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045154, upper bound: 0.0045186
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040946, -0.0028789, -0.0041010, -0.0029852, -0.0011094, 0.0012221
1: -0.0059988, -0.0034469, -0.0062366, -0.0035554, -0.0024434, 0.0027897
2: 0.9660528, 0.9714189, 0.9664876, 0.9713705, -0.0053177, 0.0049312
3: 0.0196074, 0.0354961, 0.0175022, 0.0351395, -0.0097775, 0.0122913
4: -0.0033927, -0.0009860, -0.0033656, -0.0011757, -0.0022170, 0.0023796
5: 0.0129011, 0.0150627, 0.0130132, 0.0152246, -0.0023235, 0.0020496
6: 0.0030992, 0.0051743, 0.0032727, 0.0051610, -0.0020618, 0.0019016
7: -0.0169774, -0.0128597, -0.0168850, -0.0123141, -0.0046633, 0.0040253
8: 0.0032601, 0.0065269, 0.0033334, 0.0069597, -0.0036996, 0.0031935
9: 0.0027590, 0.0094638, 0.0029655, 0.0102423, -0.0064453, 0.0059056

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045296, upper bound: 0.0045192
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045154, upper bound: 0.0045192
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041010, -0.0029852, -0.0040985, -0.0029606, -0.0011404, 0.0011134
1: -0.0062366, -0.0035554, -0.0061461, -0.0035303, -0.0027063, 0.0025907
2: 0.9664876, 0.9713705, 0.9663871, 0.9713817, -0.0048940, 0.0049834
3: 0.0175022, 0.0351395, 0.0183031, 0.0352219, -0.0113612, 0.0103596
4: -0.0033656, -0.0011757, -0.0033719, -0.0011318, -0.0022338, 0.0021962
5: 0.0130132, 0.0152246, 0.0129873, 0.0151630, -0.0021498, 0.0022373
6: 0.0032727, 0.0051610, 0.0032326, 0.0051641, -0.0018913, 0.0019284
7: -0.0168850, -0.0123141, -0.0169063, -0.0125216, -0.0043633, 0.0045922
8: 0.0033334, 0.0069597, 0.0033165, 0.0067950, -0.0034616, 0.0036433
9: 0.0029655, 0.0102423, 0.0029178, 0.0099462, -0.0062011, 0.0062822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045350, upper bound: 0.0045186
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045192, upper bound: 0.0045186
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041010, -0.0029852, -0.0041010, -0.0029852, -0.0011158, 0.0011158
1: -0.0062366, -0.0035554, -0.0062366, -0.0035554, -0.0026812, 0.0026812
2: 0.9664876, 0.9713705, 0.9664876, 0.9713705, -0.0048829, 0.0048829
3: 0.0175022, 0.0351395, 0.0175022, 0.0351395, -0.0105072, 0.0105072
4: -0.0033656, -0.0011757, -0.0033656, -0.0011757, -0.0021899, 0.0021899
5: 0.0130132, 0.0152246, 0.0130132, 0.0152246, -0.0022114, 0.0022114
6: 0.0032727, 0.0051610, 0.0032727, 0.0051610, -0.0018883, 0.0018883
7: -0.0168850, -0.0123141, -0.0168850, -0.0123141, -0.0045709, 0.0045709
8: 0.0033334, 0.0069597, 0.0033334, 0.0069597, -0.0036263, 0.0036263
9: 0.0029655, 0.0102423, 0.0029655, 0.0102423, -0.0060902, 0.0060902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045350, upper bound: 0.0045192
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045192, upper bound: 0.0045192
time: 0.68 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.24 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 2, lower bound: -0.0045277, upper bound: 0.0045124
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045124
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 2, lower bound: -0.0045277, upper bound: 0.0045154
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045154
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 2, lower bound: -0.0045322, upper bound: 0.0045124
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 2, lower bound: -0.0045186, upper bound: 0.0045124
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 2, lower bound: -0.0045322, upper bound: 0.0045154
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 2, lower bound: -0.0045186, upper bound: 0.0045154
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 2, lower bound: -0.0045285, upper bound: 0.0045186
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045186
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 2, lower bound: -0.0045285, upper bound: 0.0045192
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045192
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 2, lower bound: -0.0045349, upper bound: 0.0045186
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 2, lower bound: -0.0045186, upper bound: 0.0045186
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 2, lower bound: -0.0045349, upper bound: 0.0045192
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 2, lower bound: -0.0045186, upper bound: 0.0045192
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 2, lower bound: -0.0045280, upper bound: 0.0045124
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 2, lower bound: -0.0045154, upper bound: 0.0045124
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 2, lower bound: -0.0045280, upper bound: 0.0045154
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 2, lower bound: -0.0045154, upper bound: 0.0045154
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 2, lower bound: -0.0045322, upper bound: 0.0045124
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 2, lower bound: -0.0045192, upper bound: 0.0045124
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 2, lower bound: -0.0045322, upper bound: 0.0045154
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 2, lower bound: -0.0045192, upper bound: 0.0045154
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 2, lower bound: -0.0045296, upper bound: 0.0045186
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 2, lower bound: -0.0045154, upper bound: 0.0045186
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 2, lower bound: -0.0045296, upper bound: 0.0045192
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 2, lower bound: -0.0045154, upper bound: 0.0045192
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 2, lower bound: -0.0045350, upper bound: 0.0045186
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 2, lower bound: -0.0045192, upper bound: 0.0045186
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 2, lower bound: -0.0045350, upper bound: 0.0045192
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 2, lower bound: -0.0045192, upper bound: 0.0045192

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040901, -0.0029595, -0.0040912, -0.0028903, -0.0011998, 0.0011317
1: -0.0058297, -0.0035292, -0.0058701, -0.0034586, -0.0023712, 0.0023409
2: 0.9663826, 0.9713822, 0.9660994, 0.9714136, -0.0050310, 0.0052828
3: 0.0211036, 0.0352255, 0.0207463, 0.0354578, -0.0086154, 0.0086387
4: -0.0033721, -0.0011299, -0.0033898, -0.0010064, -0.0023658, 0.0022599
5: 0.0129861, 0.0149477, 0.0129131, 0.0149752, -0.0019891, 0.0020346
6: 0.0032308, 0.0051642, 0.0031178, 0.0051729, -0.0019421, 0.0020464
7: -0.0169073, -0.0132474, -0.0169675, -0.0131548, -0.0037524, 0.0037201
8: 0.0033157, 0.0062193, 0.0032680, 0.0062927, -0.0029770, 0.0029513
9: 0.0029157, 0.0089106, 0.0027811, 0.0090427, -0.0052085, 0.0053034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045124
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045124
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040910, -0.0028999, -0.0040915, -0.0028872, -0.0012037, 0.0011917
1: -0.0058632, -0.0034683, -0.0058842, -0.0034555, -0.0024077, 0.0024159
2: 0.9661385, 0.9714093, 0.9660869, 0.9714150, -0.0052766, 0.0053224
3: 0.0208075, 0.0354257, 0.0206217, 0.0354681, -0.0085007, 0.0090971
4: -0.0033874, -0.0010235, -0.0033906, -0.0010009, -0.0023864, 0.0023671
5: 0.0129232, 0.0149705, 0.0129099, 0.0149848, -0.0020616, 0.0020606
6: 0.0031334, 0.0051717, 0.0031128, 0.0051733, -0.0020399, 0.0020589
7: -0.0169591, -0.0131707, -0.0169701, -0.0131225, -0.0038366, 0.0037994
8: 0.0032746, 0.0062801, 0.0032659, 0.0063183, -0.0030438, 0.0030143
9: 0.0027997, 0.0090201, 0.0027752, 0.0090888, -0.0053434, 0.0053188

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045124
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045124
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040901, -0.0029595, -0.0040938, -0.0028871, -0.0012029, 0.0011343
1: -0.0058297, -0.0035292, -0.0059698, -0.0034554, -0.0023744, 0.0024406
2: 0.9663826, 0.9713822, 0.9660866, 0.9714151, -0.0050325, 0.0052956
3: 0.0211036, 0.0352255, 0.0198639, 0.0354684, -0.0089787, 0.0099877
4: -0.0033721, -0.0011299, -0.0033906, -0.0010008, -0.0023714, 0.0022607
5: 0.0129861, 0.0149477, 0.0129098, 0.0150430, -0.0020569, 0.0020379
6: 0.0032308, 0.0051642, 0.0031126, 0.0051733, -0.0019424, 0.0020516
7: -0.0169073, -0.0132474, -0.0169702, -0.0129261, -0.0039811, 0.0037228
8: 0.0033157, 0.0062193, 0.0032658, 0.0064741, -0.0031584, 0.0029535
9: 0.0029157, 0.0089106, 0.0027750, 0.0093690, -0.0055561, 0.0055822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045154
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045154
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040910, -0.0028999, -0.0040943, -0.0028847, -0.0012063, 0.0011944
1: -0.0058632, -0.0034683, -0.0059864, -0.0034528, -0.0024104, 0.0025180
2: 0.9661385, 0.9714093, 0.9660763, 0.9714163, -0.0052778, 0.0053330
3: 0.0208075, 0.0354257, 0.0197173, 0.0354768, -0.0087923, 0.0104699
4: -0.0033874, -0.0010235, -0.0033912, -0.0009963, -0.0023911, 0.0023678
5: 0.0129232, 0.0149705, 0.0129072, 0.0150543, -0.0021311, 0.0020633
6: 0.0031334, 0.0051717, 0.0031086, 0.0051736, -0.0020402, 0.0020631
7: -0.0169591, -0.0131707, -0.0169724, -0.0128881, -0.0040710, 0.0038017
8: 0.0032746, 0.0062801, 0.0032641, 0.0065043, -0.0032297, 0.0030161
9: 0.0027997, 0.0090201, 0.0027702, 0.0094232, -0.0057314, 0.0055613

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045154
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045154
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040965, -0.0030587, -0.0040912, -0.0028903, -0.0012062, 0.0010325
1: -0.0060710, -0.0036305, -0.0058701, -0.0034586, -0.0026124, 0.0022396
2: 0.9667888, 0.9713370, 0.9660994, 0.9714136, -0.0046248, 0.0052376
3: 0.0189683, 0.0348926, 0.0207463, 0.0354578, -0.0111357, 0.0087280
4: -0.0033468, -0.0013070, -0.0033898, -0.0010064, -0.0023404, 0.0020828
5: 0.0130908, 0.0151119, 0.0129131, 0.0149752, -0.0018844, 0.0021988
6: 0.0033929, 0.0051518, 0.0031178, 0.0051729, -0.0017800, 0.0020340
7: -0.0168210, -0.0126940, -0.0169675, -0.0131548, -0.0036661, 0.0042734
8: 0.0033842, 0.0066583, 0.0032680, 0.0062927, -0.0029086, 0.0033903
9: 0.0031085, 0.0097002, 0.0027811, 0.0090427, -0.0052824, 0.0058799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040038, upper bound: 0.0040305
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045186, upper bound: 0.0045124
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045186, upper bound: 0.0045124
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0029840, -0.0040915, -0.0028872, -0.0012104, 0.0011075
1: -0.0061130, -0.0035542, -0.0058842, -0.0034555, -0.0026575, 0.0023299
2: 0.9664831, 0.9713709, 0.9660869, 0.9714150, -0.0049320, 0.0052840
3: 0.0185965, 0.0351432, 0.0206217, 0.0354681, -0.0110221, 0.0091246
4: -0.0033659, -0.0011737, -0.0033906, -0.0010009, -0.0023649, 0.0022169
5: 0.0130120, 0.0151404, 0.0129099, 0.0149848, -0.0019728, 0.0022306
6: 0.0032709, 0.0051611, 0.0031128, 0.0051733, -0.0019024, 0.0020483
7: -0.0168859, -0.0125977, -0.0169701, -0.0131225, -0.0037634, 0.0043724
8: 0.0033327, 0.0067347, 0.0032659, 0.0063183, -0.0029857, 0.0034689
9: 0.0029634, 0.0098377, 0.0027752, 0.0090888, -0.0054255, 0.0058171

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040014, upper bound: 0.0040305
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045186, upper bound: 0.0045124
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045186, upper bound: 0.0045124
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040965, -0.0030587, -0.0040938, -0.0028871, -0.0012094, 0.0010351
1: -0.0060710, -0.0036305, -0.0059698, -0.0034554, -0.0026156, 0.0023393
2: 0.9667888, 0.9713370, 0.9660866, 0.9714151, -0.0046263, 0.0052505
3: 0.0189683, 0.0348926, 0.0198639, 0.0354684, -0.0114990, 0.0100771
4: -0.0033468, -0.0013070, -0.0033906, -0.0010008, -0.0023460, 0.0020836
5: 0.0130908, 0.0151119, 0.0129098, 0.0150430, -0.0019522, 0.0022021
6: 0.0033929, 0.0051518, 0.0031126, 0.0051733, -0.0017804, 0.0020391
7: -0.0168210, -0.0126940, -0.0169702, -0.0129261, -0.0038949, 0.0042761
8: 0.0033842, 0.0066583, 0.0032658, 0.0064741, -0.0030900, 0.0033925
9: 0.0031085, 0.0097002, 0.0027750, 0.0093690, -0.0056300, 0.0061587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040069, upper bound: 0.0040333
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037730, upper bound: 0.0037721
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0029840, -0.0040943, -0.0028847, -0.0012130, 0.0011102
1: -0.0061130, -0.0035542, -0.0059864, -0.0034528, -0.0026602, 0.0024321
2: 0.9664831, 0.9713709, 0.9660763, 0.9714163, -0.0049332, 0.0052946
3: 0.0185965, 0.0351432, 0.0197173, 0.0354768, -0.0113137, 0.0104974
4: -0.0033659, -0.0011737, -0.0033912, -0.0009963, -0.0023696, 0.0022176
5: 0.0130120, 0.0151404, 0.0129072, 0.0150543, -0.0020423, 0.0022333
6: 0.0032709, 0.0051611, 0.0031086, 0.0051736, -0.0019027, 0.0020526
7: -0.0168859, -0.0125977, -0.0169724, -0.0128881, -0.0039978, 0.0043747
8: 0.0033327, 0.0067347, 0.0032641, 0.0065043, -0.0031716, 0.0034707
9: 0.0029634, 0.0098377, 0.0027702, 0.0094232, -0.0058135, 0.0060597

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040037, upper bound: 0.0040328
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037728, upper bound: 0.0037718
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040901, -0.0029595, -0.0040977, -0.0029724, -0.0011177, 0.0011382
1: -0.0058297, -0.0035292, -0.0061136, -0.0035423, -0.0022874, 0.0025844
2: 0.9663826, 0.9713822, 0.9664354, 0.9713764, -0.0049938, 0.0049468
3: 0.0211036, 0.0352255, 0.0185911, 0.0351824, -0.0086355, 0.0111867
4: -0.0033721, -0.0011299, -0.0033689, -0.0011528, -0.0022193, 0.0022389
5: 0.0129861, 0.0149477, 0.0129997, 0.0151409, -0.0021547, 0.0019480
6: 0.0032308, 0.0051642, 0.0032518, 0.0051626, -0.0019318, 0.0019124
7: -0.0169073, -0.0132474, -0.0168961, -0.0125963, -0.0043110, 0.0036487
8: 0.0033157, 0.0062193, 0.0033246, 0.0067358, -0.0034201, 0.0028947
9: 0.0029157, 0.0089106, 0.0029407, 0.0098397, -0.0057624, 0.0053786

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045186
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045186
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040910, -0.0028999, -0.0040982, -0.0029698, -0.0011212, 0.0011983
1: -0.0058632, -0.0034683, -0.0061332, -0.0035397, -0.0023235, 0.0026648
2: 0.9661385, 0.9714093, 0.9664247, 0.9713775, -0.0052390, 0.0049846
3: 0.0208075, 0.0354257, 0.0184179, 0.0351911, -0.0085868, 0.0116326
4: -0.0033874, -0.0010235, -0.0033695, -0.0011482, -0.0022392, 0.0023461
5: 0.0129232, 0.0149705, 0.0129969, 0.0151542, -0.0022310, 0.0019735
6: 0.0031334, 0.0051717, 0.0032476, 0.0051629, -0.0020295, 0.0019241
7: -0.0169591, -0.0131707, -0.0168983, -0.0125514, -0.0044078, 0.0037277
8: 0.0032746, 0.0062801, 0.0033228, 0.0067714, -0.0034969, 0.0029573
9: 0.0027997, 0.0090201, 0.0029356, 0.0099037, -0.0059613, 0.0053837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045186
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045186
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040901, -0.0029595, -0.0041001, -0.0029966, -0.0010935, 0.0011406
1: -0.0058297, -0.0035292, -0.0062059, -0.0035671, -0.0022626, 0.0026767
2: 0.9663826, 0.9713822, 0.9665347, 0.9713653, -0.0049827, 0.0048475
3: 0.0211036, 0.0352255, 0.0177738, 0.0351010, -0.0088078, 0.0122498
4: -0.0033721, -0.0011299, -0.0033627, -0.0011962, -0.0021760, 0.0022328
5: 0.0129861, 0.0149477, 0.0130253, 0.0152037, -0.0022176, 0.0019224
6: 0.0032308, 0.0051642, 0.0032915, 0.0051596, -0.0019287, 0.0018727
7: -0.0169073, -0.0132474, -0.0168750, -0.0123845, -0.0045228, 0.0036276
8: 0.0033157, 0.0062193, 0.0033413, 0.0069039, -0.0035882, 0.0028779
9: 0.0029157, 0.0089106, 0.0029878, 0.0101419, -0.0061554, 0.0055444

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045192
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045192
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040910, -0.0028999, -0.0041006, -0.0029944, -0.0010965, 0.0012008
1: -0.0058632, -0.0034683, -0.0062239, -0.0035649, -0.0022983, 0.0027556
2: 0.9661385, 0.9714093, 0.9665258, 0.9713662, -0.0052277, 0.0048835
3: 0.0208075, 0.0354257, 0.0176148, 0.0351082, -0.0086193, 0.0127168
4: -0.0033874, -0.0010235, -0.0033632, -0.0011923, -0.0021951, 0.0023398
5: 0.0129232, 0.0149705, 0.0130230, 0.0152159, -0.0022927, 0.0019475
6: 0.0031334, 0.0051717, 0.0032879, 0.0051598, -0.0020264, 0.0018838
7: -0.0169591, -0.0131707, -0.0168769, -0.0123432, -0.0046159, 0.0037062
8: 0.0032746, 0.0062801, 0.0033398, 0.0069366, -0.0036620, 0.0029403
9: 0.0027997, 0.0090201, 0.0029836, 0.0102007, -0.0063524, 0.0054972

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045192
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045192
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040965, -0.0030587, -0.0040977, -0.0029724, -0.0011242, 0.0010390
1: -0.0060710, -0.0036305, -0.0061136, -0.0035423, -0.0025286, 0.0024831
2: 0.9667888, 0.9713370, 0.9664354, 0.9713764, -0.0045876, 0.0049016
3: 0.0189683, 0.0348926, 0.0185911, 0.0351824, -0.0093583, 0.0094009
4: -0.0033468, -0.0013070, -0.0033689, -0.0011528, -0.0021940, 0.0020619
5: 0.0130908, 0.0151119, 0.0129997, 0.0151409, -0.0020501, 0.0021122
6: 0.0033929, 0.0051518, 0.0032518, 0.0051626, -0.0017697, 0.0018999
7: -0.0168210, -0.0126940, -0.0168961, -0.0125963, -0.0042247, 0.0042020
8: 0.0033842, 0.0066583, 0.0033246, 0.0067358, -0.0033517, 0.0033337
9: 0.0031085, 0.0097002, 0.0029407, 0.0098397, -0.0054727, 0.0055563

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040520, upper bound: 0.0041637
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040495, upper bound: 0.0040465
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0029839, -0.0040982, -0.0029698, -0.0011279, 0.0011143
1: -0.0061130, -0.0035541, -0.0061332, -0.0035397, -0.0025733, 0.0025790
2: 0.9664828, 0.9713710, 0.9664247, 0.9713775, -0.0048947, 0.0049463
3: 0.0185965, 0.0351436, 0.0184179, 0.0351911, -0.0092627, 0.0098629
4: -0.0033659, -0.0011735, -0.0033695, -0.0011482, -0.0022177, 0.0021960
5: 0.0130119, 0.0151404, 0.0129969, 0.0151542, -0.0021423, 0.0021435
6: 0.0032707, 0.0051611, 0.0032476, 0.0051629, -0.0018922, 0.0019135
7: -0.0168860, -0.0125977, -0.0168983, -0.0125514, -0.0043346, 0.0043007
8: 0.0033326, 0.0067347, 0.0033228, 0.0067714, -0.0034389, 0.0034119
9: 0.0029632, 0.0098377, 0.0029356, 0.0099037, -0.0055942, 0.0056015

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040465, upper bound: 0.0041623
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040465, upper bound: 0.0040465
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040965, -0.0030587, -0.0041001, -0.0029966, -0.0010999, 0.0010414
1: -0.0060710, -0.0036305, -0.0062059, -0.0035671, -0.0025039, 0.0025755
2: 0.9667888, 0.9713370, 0.9665347, 0.9713653, -0.0045764, 0.0048023
3: 0.0189683, 0.0348926, 0.0177738, 0.0351010, -0.0096402, 0.0106781
4: -0.0033468, -0.0013070, -0.0033627, -0.0011962, -0.0021507, 0.0020557
5: 0.0130908, 0.0151119, 0.0130253, 0.0152037, -0.0021129, 0.0020866
6: 0.0033929, 0.0051518, 0.0032915, 0.0051596, -0.0017667, 0.0018603
7: -0.0168210, -0.0126940, -0.0168750, -0.0123845, -0.0044365, 0.0041809
8: 0.0033842, 0.0066583, 0.0033413, 0.0069039, -0.0035197, 0.0033169
9: 0.0031085, 0.0097002, 0.0029878, 0.0101419, -0.0057828, 0.0057945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040520, upper bound: 0.0041641
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040508, upper bound: 0.0040527
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0029839, -0.0041006, -0.0029944, -0.0011032, 0.0011167
1: -0.0061130, -0.0035541, -0.0062239, -0.0035649, -0.0025481, 0.0026698
2: 0.9664828, 0.9713710, 0.9665258, 0.9713662, -0.0048834, 0.0048452
3: 0.0185965, 0.0351436, 0.0176148, 0.0351082, -0.0094489, 0.0111629
4: -0.0033659, -0.0011735, -0.0033632, -0.0011923, -0.0021736, 0.0021897
5: 0.0130119, 0.0151404, 0.0130230, 0.0152159, -0.0022040, 0.0021174
6: 0.0032707, 0.0051611, 0.0032879, 0.0051598, -0.0018891, 0.0018732
7: -0.0168860, -0.0125977, -0.0168769, -0.0123432, -0.0045428, 0.0042792
8: 0.0033326, 0.0067347, 0.0033398, 0.0069366, -0.0036040, 0.0033949
9: 0.0029632, 0.0098377, 0.0029836, 0.0102007, -0.0059255, 0.0057794

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040465, upper bound: 0.0041624
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040465, upper bound: 0.0040521
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040927, -0.0029701, -0.0040912, -0.0028903, -0.0012024, 0.0011211
1: -0.0059286, -0.0035400, -0.0058701, -0.0034586, -0.0024700, 0.0023301
2: 0.9664259, 0.9713773, 0.9660994, 0.9714136, -0.0049877, 0.0052779
3: 0.0202286, 0.0351901, 0.0207463, 0.0354578, -0.0099136, 0.0089308
4: -0.0033694, -0.0011488, -0.0033898, -0.0010064, -0.0023631, 0.0022410
5: 0.0129973, 0.0150150, 0.0129131, 0.0149752, -0.0019779, 0.0021019
6: 0.0032481, 0.0051629, 0.0031178, 0.0051729, -0.0019248, 0.0020451
7: -0.0168981, -0.0130206, -0.0169675, -0.0131548, -0.0037432, 0.0039468
8: 0.0033230, 0.0063992, 0.0032680, 0.0062927, -0.0029697, 0.0031312
9: 0.0029362, 0.0092341, 0.0027811, 0.0090427, -0.0054856, 0.0056640

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038760, upper bound: 0.0038804
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045154, upper bound: 0.0045124
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045154, upper bound: 0.0045124
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040937, -0.0028991, -0.0040915, -0.0028872, -0.0012065, 0.0011924
1: -0.0059668, -0.0034676, -0.0058842, -0.0034555, -0.0025114, 0.0024166
2: 0.9661356, 0.9714097, 0.9660869, 0.9714150, -0.0052795, 0.0053228
3: 0.0198902, 0.0354283, 0.0206217, 0.0354681, -0.0097967, 0.0094523
4: -0.0033876, -0.0010221, -0.0033906, -0.0010009, -0.0023866, 0.0023685
5: 0.0129224, 0.0150410, 0.0129099, 0.0149848, -0.0020624, 0.0021311
6: 0.0031322, 0.0051718, 0.0031128, 0.0051733, -0.0020411, 0.0020590
7: -0.0169598, -0.0129330, -0.0169701, -0.0131225, -0.0038373, 0.0040372
8: 0.0032740, 0.0064687, 0.0032659, 0.0063183, -0.0030443, 0.0032029
9: 0.0027983, 0.0093593, 0.0027752, 0.0090888, -0.0056250, 0.0056266

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038782, upper bound: 0.0038828
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045154, upper bound: 0.0045124
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045154, upper bound: 0.0045124
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040927, -0.0029701, -0.0040938, -0.0028871, -0.0012056, 0.0011238
1: -0.0059286, -0.0035400, -0.0059698, -0.0034554, -0.0024732, 0.0024298
2: 0.9664259, 0.9713773, 0.9660866, 0.9714151, -0.0049892, 0.0052907
3: 0.0202286, 0.0351901, 0.0198639, 0.0354684, -0.0091396, 0.0091502
4: -0.0033694, -0.0011488, -0.0033906, -0.0010008, -0.0023687, 0.0022418
5: 0.0129973, 0.0150150, 0.0129098, 0.0150430, -0.0020458, 0.0021052
6: 0.0032481, 0.0051629, 0.0031126, 0.0051733, -0.0019252, 0.0020502
7: -0.0168981, -0.0130206, -0.0169702, -0.0129261, -0.0039719, 0.0039496
8: 0.0033230, 0.0063992, 0.0032658, 0.0064741, -0.0031511, 0.0031334
9: 0.0029362, 0.0092341, 0.0027750, 0.0093690, -0.0053673, 0.0054692

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0039050, upper bound: 0.0039119
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037748, upper bound: 0.0037748
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040937, -0.0028991, -0.0040943, -0.0028847, -0.0012091, 0.0011952
1: -0.0059668, -0.0034676, -0.0059864, -0.0034528, -0.0025140, 0.0025188
2: 0.9661356, 0.9714097, 0.9660763, 0.9714163, -0.0052807, 0.0053334
3: 0.0198902, 0.0354283, 0.0197173, 0.0354768, -0.0089777, 0.0096740
4: -0.0033876, -0.0010221, -0.0033912, -0.0009963, -0.0023912, 0.0023691
5: 0.0129224, 0.0150410, 0.0129072, 0.0150543, -0.0021319, 0.0021338
6: 0.0031322, 0.0051718, 0.0031086, 0.0051736, -0.0020414, 0.0020632
7: -0.0169598, -0.0129330, -0.0169724, -0.0128881, -0.0040717, 0.0040394
8: 0.0032740, 0.0064687, 0.0032641, 0.0065043, -0.0032302, 0.0032047
9: 0.0027983, 0.0093593, 0.0027702, 0.0094232, -0.0055014, 0.0054858

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0039050, upper bound: 0.0039119
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037748, upper bound: 0.0037748
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040990, -0.0030860, -0.0040912, -0.0028903, -0.0012087, 0.0010051
1: -0.0061636, -0.0036584, -0.0058701, -0.0034586, -0.0027050, 0.0022117
2: 0.9669008, 0.9713246, 0.9660994, 0.9714136, -0.0045128, 0.0052252
3: 0.0181483, 0.0348008, 0.0207463, 0.0354578, -0.0121683, 0.0087735
4: -0.0033398, -0.0013558, -0.0033898, -0.0010064, -0.0023335, 0.0020340
5: 0.0131196, 0.0151749, 0.0129131, 0.0149752, -0.0018556, 0.0022618
6: 0.0034376, 0.0051483, 0.0031178, 0.0051729, -0.0017353, 0.0020305
7: -0.0167972, -0.0124815, -0.0169675, -0.0131548, -0.0036424, 0.0044859
8: 0.0034030, 0.0068269, 0.0032680, 0.0062927, -0.0028897, 0.0035589
9: 0.0031617, 0.0100034, 0.0027811, 0.0090427, -0.0054120, 0.0062759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040073, upper bound: 0.0040390
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045192, upper bound: 0.0045124
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045192, upper bound: 0.0045124
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041001, -0.0030090, -0.0040915, -0.0028872, -0.0012129, 0.0010825
1: -0.0062061, -0.0035798, -0.0058842, -0.0034555, -0.0027507, 0.0023044
2: 0.9665853, 0.9713596, 0.9660869, 0.9714150, -0.0048297, 0.0052727
3: 0.0177721, 0.0350594, 0.0206217, 0.0354681, -0.0120510, 0.0092923
4: -0.0033595, -0.0012183, -0.0033906, -0.0010009, -0.0023586, 0.0021723
5: 0.0130384, 0.0152038, 0.0129099, 0.0149848, -0.0019464, 0.0022939
6: 0.0033117, 0.0051580, 0.0031128, 0.0051733, -0.0018615, 0.0020452
7: -0.0168642, -0.0123840, -0.0169701, -0.0131225, -0.0037417, 0.0045861
8: 0.0033499, 0.0069042, 0.0032659, 0.0063183, -0.0029684, 0.0036384
9: 0.0030119, 0.0101425, 0.0027752, 0.0090888, -0.0055776, 0.0062017

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040070, upper bound: 0.0040389
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045192, upper bound: 0.0045124
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045192, upper bound: 0.0045124
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040990, -0.0030860, -0.0040938, -0.0028871, -0.0012119, 0.0010078
1: -0.0061636, -0.0036584, -0.0059698, -0.0034554, -0.0027083, 0.0023114
2: 0.9669008, 0.9713246, 0.9660866, 0.9714151, -0.0045143, 0.0052381
3: 0.0181483, 0.0348008, 0.0198639, 0.0354684, -0.0115659, 0.0091156
4: -0.0033398, -0.0013558, -0.0033906, -0.0010008, -0.0023391, 0.0020348
5: 0.0131196, 0.0151749, 0.0129098, 0.0150430, -0.0019234, 0.0022651
6: 0.0034376, 0.0051483, 0.0031126, 0.0051733, -0.0017357, 0.0020357
7: -0.0167972, -0.0124815, -0.0169702, -0.0129261, -0.0038711, 0.0044887
8: 0.0034030, 0.0068269, 0.0032658, 0.0064741, -0.0030711, 0.0035611
9: 0.0031617, 0.0100034, 0.0027750, 0.0093690, -0.0054209, 0.0060508

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040250, upper bound: 0.0040799
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038107, upper bound: 0.0038093
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041001, -0.0030090, -0.0040943, -0.0028847, -0.0012155, 0.0010852
1: -0.0062061, -0.0035798, -0.0059864, -0.0034528, -0.0027533, 0.0024066
2: 0.9665853, 0.9713596, 0.9660763, 0.9714163, -0.0048310, 0.0052833
3: 0.0177721, 0.0350594, 0.0197173, 0.0354768, -0.0114040, 0.0095910
4: -0.0033595, -0.0012183, -0.0033912, -0.0009963, -0.0023632, 0.0021730
5: 0.0130384, 0.0152038, 0.0129072, 0.0150543, -0.0020159, 0.0022967
6: 0.0033117, 0.0051580, 0.0031086, 0.0051736, -0.0018619, 0.0020494
7: -0.0168642, -0.0123840, -0.0169724, -0.0128881, -0.0039761, 0.0045883
8: 0.0033499, 0.0069042, 0.0032641, 0.0065043, -0.0031544, 0.0036401
9: 0.0030119, 0.0101425, 0.0027702, 0.0094232, -0.0055528, 0.0059885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040250, upper bound: 0.0040806
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038103, upper bound: 0.0038089
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040927, -0.0029701, -0.0040977, -0.0029724, -0.0011204, 0.0011276
1: -0.0059286, -0.0035400, -0.0061136, -0.0035423, -0.0023863, 0.0025736
2: 0.9664259, 0.9713773, 0.9664354, 0.9713764, -0.0049505, 0.0049419
3: 0.0202286, 0.0351901, 0.0185911, 0.0351824, -0.0099351, 0.0114788
4: -0.0033694, -0.0011488, -0.0033689, -0.0011528, -0.0022166, 0.0022201
5: 0.0129973, 0.0150150, 0.0129997, 0.0151409, -0.0021436, 0.0020153
6: 0.0032481, 0.0051629, 0.0032518, 0.0051626, -0.0019145, 0.0019111
7: -0.0168981, -0.0130206, -0.0168961, -0.0125963, -0.0043018, 0.0038754
8: 0.0033230, 0.0063992, 0.0033246, 0.0067358, -0.0034128, 0.0030746
9: 0.0029362, 0.0092341, 0.0029407, 0.0098397, -0.0060394, 0.0057440

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038821, upper bound: 0.0038866
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037671, upper bound: 0.0037682
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040937, -0.0028991, -0.0040982, -0.0029698, -0.0011240, 0.0011991
1: -0.0059668, -0.0034676, -0.0061332, -0.0035397, -0.0024272, 0.0026656
2: 0.9661356, 0.9714097, 0.9664247, 0.9713775, -0.0052419, 0.0049850
3: 0.0198902, 0.0354283, 0.0184179, 0.0351911, -0.0098813, 0.0119878
4: -0.0033876, -0.0010221, -0.0033695, -0.0011482, -0.0022394, 0.0023474
5: 0.0129224, 0.0150410, 0.0129969, 0.0151542, -0.0022318, 0.0020441
6: 0.0031322, 0.0051718, 0.0032476, 0.0051629, -0.0020307, 0.0019242
7: -0.0169598, -0.0129330, -0.0168983, -0.0125514, -0.0044084, 0.0039654
8: 0.0032740, 0.0064687, 0.0033228, 0.0067714, -0.0034974, 0.0031459
9: 0.0027983, 0.0093593, 0.0029356, 0.0099037, -0.0062429, 0.0056930

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038842, upper bound: 0.0038889
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037718, upper bound: 0.0037728
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040927, -0.0029701, -0.0041001, -0.0029966, -0.0010961, 0.0011301
1: -0.0059286, -0.0035400, -0.0062059, -0.0035671, -0.0023615, 0.0026659
2: 0.9664259, 0.9713773, 0.9665347, 0.9713653, -0.0049394, 0.0048426
3: 0.0202286, 0.0351901, 0.0177738, 0.0351010, -0.0090462, 0.0115836
4: -0.0033694, -0.0011488, -0.0033627, -0.0011962, -0.0021733, 0.0022139
5: 0.0129973, 0.0150150, 0.0130253, 0.0152037, -0.0022064, 0.0019897
6: 0.0032481, 0.0051629, 0.0032915, 0.0051596, -0.0019114, 0.0018714
7: -0.0168981, -0.0130206, -0.0168750, -0.0123845, -0.0045136, 0.0038543
8: 0.0033230, 0.0063992, 0.0033413, 0.0069039, -0.0035809, 0.0030578
9: 0.0029362, 0.0092341, 0.0029878, 0.0101419, -0.0059175, 0.0055115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0039053, upper bound: 0.0039120
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038087, upper bound: 0.0038100
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040937, -0.0028991, -0.0041006, -0.0029944, -0.0010993, 0.0012015
1: -0.0059668, -0.0034676, -0.0062239, -0.0035649, -0.0024019, 0.0027563
2: 0.9661356, 0.9714097, 0.9665258, 0.9713662, -0.0052307, 0.0048839
3: 0.0198902, 0.0354283, 0.0176148, 0.0351082, -0.0089341, 0.0120914
4: -0.0033876, -0.0010221, -0.0033632, -0.0011923, -0.0021953, 0.0023411
5: 0.0129224, 0.0150410, 0.0130230, 0.0152159, -0.0022935, 0.0020180
6: 0.0031322, 0.0051718, 0.0032879, 0.0051598, -0.0020276, 0.0018839
7: -0.0169598, -0.0129330, -0.0168769, -0.0123432, -0.0046166, 0.0039439
8: 0.0032740, 0.0064687, 0.0033398, 0.0069366, -0.0036625, 0.0031289
9: 0.0027983, 0.0093593, 0.0029836, 0.0102007, -0.0061117, 0.0055244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0039053, upper bound: 0.0039120
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038087, upper bound: 0.0038100
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040990, -0.0030860, -0.0040977, -0.0029724, -0.0011266, 0.0010116
1: -0.0061636, -0.0036584, -0.0061136, -0.0035423, -0.0026213, 0.0024552
2: 0.9669008, 0.9713246, 0.9664354, 0.9713764, -0.0044757, 0.0048892
3: 0.0181483, 0.0348008, 0.0185911, 0.0351824, -0.0105999, 0.0095995
4: -0.0033398, -0.0013558, -0.0033689, -0.0011528, -0.0021870, 0.0020131
5: 0.0131196, 0.0151749, 0.0129997, 0.0151409, -0.0020212, 0.0021752
6: 0.0034376, 0.0051483, 0.0032518, 0.0051626, -0.0017250, 0.0018965
7: -0.0167972, -0.0124815, -0.0168961, -0.0125963, -0.0042009, 0.0044146
8: 0.0034030, 0.0068269, 0.0033246, 0.0067358, -0.0033328, 0.0035023
9: 0.0031617, 0.0100034, 0.0029407, 0.0098397, -0.0056930, 0.0058836

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040574, upper bound: 0.0041673
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040519, upper bound: 0.0040465
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041001, -0.0030089, -0.0040982, -0.0029698, -0.0011304, 0.0010892
1: -0.0062061, -0.0035797, -0.0061332, -0.0035397, -0.0026664, 0.0025535
2: 0.9665850, 0.9713597, 0.9664247, 0.9713775, -0.0047925, 0.0049350
3: 0.0177721, 0.0350596, 0.0184179, 0.0351911, -0.0104995, 0.0101408
4: -0.0033595, -0.0012182, -0.0033695, -0.0011482, -0.0022113, 0.0021514
5: 0.0130383, 0.0152038, 0.0129969, 0.0151542, -0.0021159, 0.0022069
6: 0.0033116, 0.0051580, 0.0032476, 0.0051629, -0.0018513, 0.0019104
7: -0.0168642, -0.0123840, -0.0168983, -0.0125514, -0.0043129, 0.0045143
8: 0.0033498, 0.0069042, 0.0033228, 0.0067714, -0.0034216, 0.0035814
9: 0.0030118, 0.0101425, 0.0029356, 0.0099037, -0.0058344, 0.0059012

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040568, upper bound: 0.0041672
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040516, upper bound: 0.0040465
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040990, -0.0030860, -0.0041001, -0.0029966, -0.0011024, 0.0010141
1: -0.0061636, -0.0036584, -0.0062059, -0.0035671, -0.0025965, 0.0025476
2: 0.9669008, 0.9713246, 0.9665347, 0.9713653, -0.0044645, 0.0047899
3: 0.0181483, 0.0348008, 0.0177738, 0.0351010, -0.0097561, 0.0097948
4: -0.0033398, -0.0013558, -0.0033627, -0.0011962, -0.0021437, 0.0020069
5: 0.0131196, 0.0151749, 0.0130253, 0.0152037, -0.0020840, 0.0021496
6: 0.0034376, 0.0051483, 0.0032915, 0.0051596, -0.0017220, 0.0018569
7: -0.0167972, -0.0124815, -0.0168750, -0.0123845, -0.0044127, 0.0043935
8: 0.0034030, 0.0068269, 0.0033413, 0.0069039, -0.0035008, 0.0034855
9: 0.0031617, 0.0100034, 0.0029878, 0.0101419, -0.0056238, 0.0056984

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040574, upper bound: 0.0041797
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040574, upper bound: 0.0040568
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041001, -0.0030089, -0.0041006, -0.0029944, -0.0011057, 0.0010917
1: -0.0062061, -0.0035797, -0.0062239, -0.0035649, -0.0026412, 0.0026442
2: 0.9665850, 0.9713597, 0.9665258, 0.9713662, -0.0047812, 0.0048339
3: 0.0177721, 0.0350596, 0.0176148, 0.0351082, -0.0096386, 0.0103104
4: -0.0033595, -0.0012182, -0.0033632, -0.0011923, -0.0021672, 0.0021451
5: 0.0130383, 0.0152038, 0.0130230, 0.0152159, -0.0021776, 0.0021808
6: 0.0033116, 0.0051580, 0.0032879, 0.0051598, -0.0018482, 0.0018701
7: -0.0168642, -0.0123840, -0.0168769, -0.0123432, -0.0045210, 0.0044928
8: 0.0033498, 0.0069042, 0.0033398, 0.0069366, -0.0035867, 0.0035644
9: 0.0030118, 0.0101425, 0.0029836, 0.0102007, -0.0057236, 0.0057656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040568, upper bound: 0.0041791
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040568, upper bound: 0.0040568
time: 0.57 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.20 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045124
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045124
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045124
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045124
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045154
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045154
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045154
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045154
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0045186, upper bound: 0.0045124
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0045186, upper bound: 0.0045124
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0045186, upper bound: 0.0045124
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0045186, upper bound: 0.0045124
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0040069, upper bound: 0.0040333
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0037730, upper bound: 0.0037721
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0040037, upper bound: 0.0040328
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0037728, upper bound: 0.0037718
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045186
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045186
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045186
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045186
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045192
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045192
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045192
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045192
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0040520, upper bound: 0.0041637
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0040495, upper bound: 0.0040465
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0040465, upper bound: 0.0041623
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0040465, upper bound: 0.0040465
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0040520, upper bound: 0.0041641
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0040508, upper bound: 0.0040527
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0040465, upper bound: 0.0041624
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0040465, upper bound: 0.0040521
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0045154, upper bound: 0.0045124
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0045154, upper bound: 0.0045124
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0045154, upper bound: 0.0045124
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0045154, upper bound: 0.0045124
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0039050, upper bound: 0.0039119
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0037748, upper bound: 0.0037748
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0039050, upper bound: 0.0039119
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0037748, upper bound: 0.0037748
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0045192, upper bound: 0.0045124
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0045192, upper bound: 0.0045124
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0045192, upper bound: 0.0045124
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0045192, upper bound: 0.0045124
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0040250, upper bound: 0.0040799
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0038107, upper bound: 0.0038093
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0040250, upper bound: 0.0040806
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0038103, upper bound: 0.0038089
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0038821, upper bound: 0.0038866
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0037671, upper bound: 0.0037682
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0038842, upper bound: 0.0038889
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0037718, upper bound: 0.0037728
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0039053, upper bound: 0.0039120
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0038087, upper bound: 0.0038100
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0039053, upper bound: 0.0039120
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0038087, upper bound: 0.0038100
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0040574, upper bound: 0.0041673
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0040519, upper bound: 0.0040465
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0040568, upper bound: 0.0041672
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0040516, upper bound: 0.0040465
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0040574, upper bound: 0.0041797
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0040574, upper bound: 0.0040568
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0040568, upper bound: 0.0041791
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 2, lower bound: -0.0040568, upper bound: 0.0040568

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040901, -0.0029595, -0.0040901, -0.0029595, -0.0011306, 0.0011306
1: -0.0058297, -0.0035292, -0.0058297, -0.0035292, -0.0023005, 0.0023005
2: 0.9663826, 0.9713822, 0.9663826, 0.9713822, -0.0049996, 0.0049996
3: 0.0211036, 0.0352255, 0.0211036, 0.0352255, -0.0082878, 0.0082878
4: -0.0033721, -0.0011299, -0.0033721, -0.0011299, -0.0022422, 0.0022422
5: 0.0129861, 0.0149477, 0.0129861, 0.0149477, -0.0019616, 0.0019616
6: 0.0032308, 0.0051642, 0.0032308, 0.0051642, -0.0019334, 0.0019334
7: -0.0169073, -0.0132474, -0.0169073, -0.0132474, -0.0036599, 0.0036599
8: 0.0033157, 0.0062193, 0.0033157, 0.0062193, -0.0029035, 0.0029035
9: 0.0029157, 0.0089106, 0.0029157, 0.0089106, -0.0050084, 0.0050084

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045197, upper bound: 0.0045092
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045277, upper bound: 0.0045124
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040901, -0.0029595, -0.0040910, -0.0028988, -0.0011913, 0.0011315
1: -0.0058297, -0.0035292, -0.0058632, -0.0034673, -0.0023625, 0.0023340
2: 0.9663826, 0.9713822, 0.9661344, 0.9714097, -0.0050271, 0.0052478
3: 0.0211036, 0.0352255, 0.0208075, 0.0354292, -0.0085583, 0.0087316
4: -0.0033721, -0.0011299, -0.0033876, -0.0010216, -0.0023505, 0.0022577
5: 0.0129861, 0.0149477, 0.0129221, 0.0149705, -0.0019844, 0.0020256
6: 0.0032308, 0.0051642, 0.0031317, 0.0051718, -0.0019410, 0.0020325
7: -0.0169073, -0.0132474, -0.0169600, -0.0131707, -0.0037366, 0.0037126
8: 0.0033157, 0.0062193, 0.0032739, 0.0062802, -0.0029644, 0.0029454
9: 0.0029157, 0.0089106, 0.0027977, 0.0090201, -0.0051001, 0.0051224

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045197, upper bound: 0.0045092
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045277, upper bound: 0.0045124
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040910, -0.0028999, -0.0040901, -0.0029595, -0.0011315, 0.0011902
1: -0.0058632, -0.0034683, -0.0058297, -0.0035292, -0.0023340, 0.0023614
2: 0.9661385, 0.9714093, 0.9663826, 0.9713822, -0.0052437, 0.0050267
3: 0.0208075, 0.0354257, 0.0211036, 0.0352255, -0.0087315, 0.0085550
4: -0.0033874, -0.0010235, -0.0033721, -0.0011299, -0.0022575, 0.0023487
5: 0.0129232, 0.0149705, 0.0129861, 0.0149477, -0.0020245, 0.0019844
6: 0.0031334, 0.0051717, 0.0032308, 0.0051642, -0.0020308, 0.0019409
7: -0.0169591, -0.0131707, -0.0169073, -0.0132474, -0.0037117, 0.0037366
8: 0.0032746, 0.0062801, 0.0033157, 0.0062193, -0.0029447, 0.0029644
9: 0.0027997, 0.0090201, 0.0029157, 0.0089106, -0.0051205, 0.0050994

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0044886, upper bound: 0.0045072
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045124
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040910, -0.0028999, -0.0040910, -0.0028988, -0.0011921, 0.0011911
1: -0.0058632, -0.0034683, -0.0058632, -0.0034673, -0.0023959, 0.0023949
2: 0.9661385, 0.9714093, 0.9661344, 0.9714097, -0.0052713, 0.0052750
3: 0.0208075, 0.0354257, 0.0208075, 0.0354292, -0.0084408, 0.0084375
4: -0.0033874, -0.0010235, -0.0033876, -0.0010216, -0.0023657, 0.0023642
5: 0.0129232, 0.0149705, 0.0129221, 0.0149705, -0.0020473, 0.0020484
6: 0.0031334, 0.0051717, 0.0031317, 0.0051718, -0.0020384, 0.0020400
7: -0.0169591, -0.0131707, -0.0169600, -0.0131707, -0.0037885, 0.0037894
8: 0.0032746, 0.0062801, 0.0032739, 0.0062802, -0.0030056, 0.0030063
9: 0.0027997, 0.0090201, 0.0027977, 0.0090201, -0.0052373, 0.0052380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0044886, upper bound: 0.0045072
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045124
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040901, -0.0029595, -0.0040927, -0.0029688, -0.0011213, 0.0011332
1: -0.0058297, -0.0035292, -0.0059286, -0.0035387, -0.0022910, 0.0023994
2: 0.9663826, 0.9713822, 0.9664209, 0.9713779, -0.0049953, 0.0049613
3: 0.0211036, 0.0352255, 0.0202286, 0.0351943, -0.0085841, 0.0095860
4: -0.0033721, -0.0011299, -0.0033698, -0.0011465, -0.0022256, 0.0022399
5: 0.0129861, 0.0149477, 0.0129959, 0.0150150, -0.0020289, 0.0019518
6: 0.0032308, 0.0051642, 0.0032460, 0.0051630, -0.0019322, 0.0019182
7: -0.0169073, -0.0132474, -0.0168992, -0.0130206, -0.0038866, 0.0036518
8: 0.0033157, 0.0062193, 0.0033221, 0.0063992, -0.0030834, 0.0028971
9: 0.0029157, 0.0089106, 0.0029338, 0.0092341, -0.0053698, 0.0052875

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038886, upper bound: 0.0038842
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045197, upper bound: 0.0045110
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045284, upper bound: 0.0045154
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040901, -0.0029595, -0.0040937, -0.0028970, -0.0011931, 0.0011342
1: -0.0058297, -0.0035292, -0.0059668, -0.0034654, -0.0023643, 0.0024376
2: 0.9663826, 0.9713822, 0.9661268, 0.9714106, -0.0050280, 0.0052554
3: 0.0211036, 0.0352255, 0.0198902, 0.0354354, -0.0089174, 0.0100868
4: -0.0033721, -0.0011299, -0.0033881, -0.0010183, -0.0023538, 0.0022582
5: 0.0129861, 0.0149477, 0.0129202, 0.0150410, -0.0020549, 0.0020276
6: 0.0032308, 0.0051642, 0.0031287, 0.0051721, -0.0019412, 0.0020355
7: -0.0169073, -0.0132474, -0.0169616, -0.0129329, -0.0039743, 0.0037142
8: 0.0033157, 0.0062193, 0.0032726, 0.0064687, -0.0031530, 0.0029467
9: 0.0029157, 0.0089106, 0.0027941, 0.0093593, -0.0054960, 0.0054058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038886, upper bound: 0.0038842
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045197, upper bound: 0.0045111
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045284, upper bound: 0.0045154
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040910, -0.0028999, -0.0040927, -0.0029688, -0.0011222, 0.0011929
1: -0.0058632, -0.0034683, -0.0059286, -0.0035387, -0.0023245, 0.0024603
2: 0.9661385, 0.9714093, 0.9664209, 0.9713779, -0.0052394, 0.0049884
3: 0.0208075, 0.0354257, 0.0202286, 0.0351943, -0.0090278, 0.0098533
4: -0.0033874, -0.0010235, -0.0033698, -0.0011465, -0.0022408, 0.0023463
5: 0.0129232, 0.0149705, 0.0129959, 0.0150150, -0.0020918, 0.0019745
6: 0.0031334, 0.0051717, 0.0032460, 0.0051630, -0.0020296, 0.0019256
7: -0.0169591, -0.0131707, -0.0168992, -0.0130206, -0.0039385, 0.0037285
8: 0.0032746, 0.0062801, 0.0033221, 0.0063992, -0.0031246, 0.0029580
9: 0.0027997, 0.0090201, 0.0029338, 0.0092341, -0.0054819, 0.0053785

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038804, upper bound: 0.0038760
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0044886, upper bound: 0.0045089
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045154
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040910, -0.0028999, -0.0040937, -0.0028970, -0.0011940, 0.0011939
1: -0.0058632, -0.0034683, -0.0059668, -0.0034654, -0.0023978, 0.0024985
2: 0.9661385, 0.9714093, 0.9661268, 0.9714106, -0.0052722, 0.0052825
3: 0.0208075, 0.0354257, 0.0198902, 0.0354354, -0.0087340, 0.0097336
4: -0.0033874, -0.0010235, -0.0033881, -0.0010183, -0.0023690, 0.0023646
5: 0.0129232, 0.0149705, 0.0129202, 0.0150410, -0.0021178, 0.0020503
6: 0.0031334, 0.0051717, 0.0031287, 0.0051721, -0.0020386, 0.0020430
7: -0.0169591, -0.0131707, -0.0169616, -0.0129329, -0.0040262, 0.0037910
8: 0.0032746, 0.0062801, 0.0032726, 0.0064687, -0.0031942, 0.0030076
9: 0.0027997, 0.0090201, 0.0027941, 0.0093593, -0.0055459, 0.0054572

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038804, upper bound: 0.0038760
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0044886, upper bound: 0.0045088
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045154
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040965, -0.0030587, -0.0040901, -0.0029595, -0.0011370, 0.0010314
1: -0.0060710, -0.0036305, -0.0058297, -0.0035292, -0.0025418, 0.0021993
2: 0.9667888, 0.9713370, 0.9663826, 0.9713822, -0.0045934, 0.0049544
3: 0.0189683, 0.0348926, 0.0211036, 0.0352255, -0.0108080, 0.0083771
4: -0.0033468, -0.0013070, -0.0033721, -0.0011299, -0.0022169, 0.0020652
5: 0.0130908, 0.0151119, 0.0129861, 0.0149477, -0.0018569, 0.0021257
6: 0.0033929, 0.0051518, 0.0032308, 0.0051642, -0.0017713, 0.0019209
7: -0.0168210, -0.0126940, -0.0169073, -0.0132474, -0.0035736, 0.0042132
8: 0.0033842, 0.0066583, 0.0033157, 0.0062193, -0.0028351, 0.0033426
9: 0.0031085, 0.0097002, 0.0029157, 0.0089106, -0.0050822, 0.0055848

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045238, upper bound: 0.0045117
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045322, upper bound: 0.0045124
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040965, -0.0030587, -0.0040910, -0.0028988, -0.0011977, 0.0010323
1: -0.0060710, -0.0036305, -0.0058632, -0.0034673, -0.0026037, 0.0022327
2: 0.9667888, 0.9713370, 0.9661344, 0.9714097, -0.0046209, 0.0052027
3: 0.0189683, 0.0348926, 0.0208075, 0.0354292, -0.0110786, 0.0088209
4: -0.0033468, -0.0013070, -0.0033876, -0.0010216, -0.0023252, 0.0020807
5: 0.0130908, 0.0151119, 0.0129221, 0.0149705, -0.0018797, 0.0021898
6: 0.0033929, 0.0051518, 0.0031317, 0.0051718, -0.0017789, 0.0020200
7: -0.0168210, -0.0126940, -0.0169600, -0.0131707, -0.0036503, 0.0042660
8: 0.0033842, 0.0066583, 0.0032739, 0.0062802, -0.0028960, 0.0033844
9: 0.0031085, 0.0097002, 0.0027977, 0.0090201, -0.0051740, 0.0056988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045238, upper bound: 0.0045117
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045322, upper bound: 0.0045124
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0029840, -0.0040901, -0.0029595, -0.0011382, 0.0011061
1: -0.0061130, -0.0035542, -0.0058297, -0.0035292, -0.0025838, 0.0022755
2: 0.9664831, 0.9713709, 0.9663826, 0.9713822, -0.0048991, 0.0049883
3: 0.0185965, 0.0351432, 0.0211036, 0.0352255, -0.0112008, 0.0085825
4: -0.0033659, -0.0011737, -0.0033721, -0.0011299, -0.0022360, 0.0021985
5: 0.0130120, 0.0151404, 0.0129861, 0.0149477, -0.0019357, 0.0021543
6: 0.0032709, 0.0051611, 0.0032308, 0.0051642, -0.0018933, 0.0019303
7: -0.0168859, -0.0125977, -0.0169073, -0.0132474, -0.0036385, 0.0043096
8: 0.0033327, 0.0067347, 0.0033157, 0.0062193, -0.0028866, 0.0034190
9: 0.0029634, 0.0098377, 0.0029157, 0.0089106, -0.0052026, 0.0057080

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045078, upper bound: 0.0045114
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045186, upper bound: 0.0045124
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0029840, -0.0040910, -0.0028988, -0.0011988, 0.0011069
1: -0.0061130, -0.0035542, -0.0058632, -0.0034673, -0.0026457, 0.0023089
2: 0.9664831, 0.9713709, 0.9661344, 0.9714097, -0.0049267, 0.0052366
3: 0.0185965, 0.0351432, 0.0208075, 0.0354292, -0.0109621, 0.0085300
4: -0.0033659, -0.0011737, -0.0033876, -0.0010216, -0.0023443, 0.0022139
5: 0.0130120, 0.0151404, 0.0129221, 0.0149705, -0.0019585, 0.0022183
6: 0.0032709, 0.0051611, 0.0031317, 0.0051718, -0.0019009, 0.0020294
7: -0.0168859, -0.0125977, -0.0169600, -0.0131707, -0.0037153, 0.0043623
8: 0.0033327, 0.0067347, 0.0032739, 0.0062802, -0.0029475, 0.0034609
9: 0.0029634, 0.0098377, 0.0027977, 0.0090201, -0.0052626, 0.0057364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045078, upper bound: 0.0045114
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045186, upper bound: 0.0045124
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040965, -0.0030587, -0.0040932, -0.0028937, -0.0012029, 0.0010345
1: -0.0060710, -0.0036305, -0.0059466, -0.0034620, -0.0026090, 0.0023162
2: 0.9667888, 0.9713370, 0.9661132, 0.9714121, -0.0046233, 0.0052238
3: 0.0189683, 0.0348926, 0.0200690, 0.0354465, -0.0114778, 0.0098746
4: -0.0033468, -0.0013070, -0.0033889, -0.0010124, -0.0023344, 0.0020820
5: 0.0130908, 0.0151119, 0.0129167, 0.0150273, -0.0019365, 0.0021952
6: 0.0033929, 0.0051518, 0.0031233, 0.0051725, -0.0017796, 0.0020285
7: -0.0168210, -0.0126940, -0.0169645, -0.0129793, -0.0038417, 0.0042705
8: 0.0033842, 0.0066583, 0.0032703, 0.0064320, -0.0030478, 0.0033880
9: 0.0031085, 0.0097002, 0.0027877, 0.0092932, -0.0055364, 0.0061349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037730, upper bound: 0.0037721
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037730, upper bound: 0.0037721
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040954, -0.0030713, -0.0040895, -0.0019948, -0.0021006, 0.0010182
1: -0.0060296, -0.0036433, -0.0058070, -0.0025446, -0.0034850, 0.0021636
2: 0.9668403, 0.9713314, 0.9624336, 0.9718212, -0.0049809, 0.0088977
3: 0.0193350, 0.0348504, 0.0213053, 0.0384633, -0.0143852, 0.0096638
4: -0.0033436, -0.0013294, -0.0036184, 0.0005919, -0.0039355, 0.0022890
5: 0.0131041, 0.0150837, 0.0119683, 0.0149322, -0.0018282, 0.0031154
6: 0.0034135, 0.0051502, 0.0016549, 0.0052853, -0.0018718, 0.0034952
7: -0.0168100, -0.0127891, -0.0177464, -0.0114198, -0.0053902, 0.0049573
8: 0.0033929, 0.0065829, 0.0026500, 0.0078636, -0.0044708, 0.0039329
9: 0.0031330, 0.0095646, 0.0010403, 0.0088360, -0.0057030, 0.0077283

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037730, upper bound: 0.0037721
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037730, upper bound: 0.0037721
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0029840, -0.0040936, -0.0028912, -0.0012065, 0.0011096
1: -0.0061130, -0.0035542, -0.0059631, -0.0034595, -0.0026535, 0.0024088
2: 0.9664831, 0.9713709, 0.9661030, 0.9714132, -0.0049301, 0.0052680
3: 0.0185965, 0.0351432, 0.0199232, 0.0354549, -0.0112930, 0.0102912
4: -0.0033659, -0.0011737, -0.0033896, -0.0010080, -0.0023579, 0.0022159
5: 0.0130120, 0.0151404, 0.0129140, 0.0150385, -0.0020265, 0.0022264
6: 0.0032709, 0.0051611, 0.0031192, 0.0051728, -0.0019019, 0.0020419
7: -0.0168859, -0.0125977, -0.0169667, -0.0129415, -0.0039444, 0.0043690
8: 0.0033327, 0.0067347, 0.0032686, 0.0064619, -0.0031293, 0.0034662
9: 0.0029634, 0.0098377, 0.0027829, 0.0093470, -0.0057232, 0.0060360

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037728, upper bound: 0.0037718
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037728, upper bound: 0.0037718
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040965, -0.0029970, -0.0040900, -0.0019924, -0.0021041, 0.0010931
1: -0.0060700, -0.0035674, -0.0058266, -0.0025422, -0.0035279, 0.0022592
2: 0.9665360, 0.9713650, 0.9624239, 0.9718221, -0.0052861, 0.0089411
3: 0.0189768, 0.0350998, 0.0211311, 0.0384712, -0.0141810, 0.0100726
4: -0.0033626, -0.0011967, -0.0036190, 0.0005961, -0.0039587, 0.0024222
5: 0.0130256, 0.0151112, 0.0119658, 0.0149456, -0.0019200, 0.0031454
6: 0.0032920, 0.0051595, 0.0016511, 0.0052856, -0.0019935, 0.0035084
7: -0.0168747, -0.0126962, -0.0177484, -0.0114146, -0.0054601, 0.0050522
8: 0.0033416, 0.0066565, 0.0026484, 0.0078680, -0.0045264, 0.0040081
9: 0.0029885, 0.0096970, 0.0010357, 0.0089004, -0.0059119, 0.0075953

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037728, upper bound: 0.0037718
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037728, upper bound: 0.0037718
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040901, -0.0029595, -0.0040965, -0.0030587, -0.0010314, 0.0011370
1: -0.0058297, -0.0035292, -0.0060710, -0.0036305, -0.0021993, 0.0025418
2: 0.9663826, 0.9713822, 0.9667888, 0.9713370, -0.0049544, 0.0045934
3: 0.0211036, 0.0352255, 0.0189683, 0.0348926, -0.0083771, 0.0108080
4: -0.0033721, -0.0011299, -0.0033468, -0.0013070, -0.0020652, 0.0022169
5: 0.0129861, 0.0149477, 0.0130908, 0.0151119, -0.0021257, 0.0018569
6: 0.0032308, 0.0051642, 0.0033929, 0.0051518, -0.0019209, 0.0017713
7: -0.0169073, -0.0132474, -0.0168210, -0.0126940, -0.0042132, 0.0035736
8: 0.0033157, 0.0062193, 0.0033842, 0.0066583, -0.0033426, 0.0028351
9: 0.0029157, 0.0089106, 0.0031085, 0.0097002, -0.0055848, 0.0050822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040422, upper bound: 0.0040049
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045197, upper bound: 0.0045135
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045286, upper bound: 0.0045186
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040901, -0.0029595, -0.0040977, -0.0029839, -0.0011062, 0.0011382
1: -0.0058297, -0.0035292, -0.0061130, -0.0035541, -0.0022756, 0.0025838
2: 0.9663826, 0.9713822, 0.9664828, 0.9713710, -0.0049884, 0.0048994
3: 0.0211036, 0.0352255, 0.0185965, 0.0351436, -0.0085750, 0.0112008
4: -0.0033721, -0.0011299, -0.0033659, -0.0011735, -0.0021986, 0.0022360
5: 0.0129861, 0.0149477, 0.0130119, 0.0151404, -0.0021543, 0.0019358
6: 0.0032308, 0.0051642, 0.0032707, 0.0051611, -0.0019303, 0.0018935
7: -0.0169073, -0.0132474, -0.0168860, -0.0125977, -0.0043096, 0.0036386
8: 0.0033157, 0.0062193, 0.0033326, 0.0067347, -0.0034190, 0.0028867
9: 0.0029157, 0.0089106, 0.0029632, 0.0098377, -0.0057081, 0.0052064

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040422, upper bound: 0.0040049
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045197, upper bound: 0.0045135
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045286, upper bound: 0.0045186
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040910, -0.0028999, -0.0040965, -0.0030587, -0.0010323, 0.0011967
1: -0.0058632, -0.0034683, -0.0060710, -0.0036305, -0.0022327, 0.0026026
2: 0.9661385, 0.9714093, 0.9667888, 0.9713370, -0.0051985, 0.0046205
3: 0.0208075, 0.0354257, 0.0189683, 0.0348926, -0.0088209, 0.0110753
4: -0.0033874, -0.0010235, -0.0033468, -0.0013070, -0.0020804, 0.0023234
5: 0.0129232, 0.0149705, 0.0130908, 0.0151119, -0.0021887, 0.0018797
6: 0.0031334, 0.0051717, 0.0033929, 0.0051518, -0.0020184, 0.0017788
7: -0.0169591, -0.0131707, -0.0168210, -0.0126940, -0.0042651, 0.0036503
8: 0.0032746, 0.0062801, 0.0033842, 0.0066583, -0.0033837, 0.0028960
9: 0.0027997, 0.0090201, 0.0031085, 0.0097002, -0.0056969, 0.0051733

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040305, upper bound: 0.0040014
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0044886, upper bound: 0.0045117
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045186
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040910, -0.0028999, -0.0040977, -0.0029839, -0.0011071, 0.0011978
1: -0.0058632, -0.0034683, -0.0061130, -0.0035541, -0.0023090, 0.0026447
2: 0.9661385, 0.9714093, 0.9664828, 0.9713710, -0.0052325, 0.0049265
3: 0.0208075, 0.0354257, 0.0185965, 0.0351436, -0.0085213, 0.0109588
4: -0.0033874, -0.0010235, -0.0033659, -0.0011735, -0.0022139, 0.0023424
5: 0.0129232, 0.0149705, 0.0130119, 0.0151404, -0.0022173, 0.0019586
6: 0.0031334, 0.0051717, 0.0032707, 0.0051611, -0.0020277, 0.0019010
7: -0.0169591, -0.0131707, -0.0168860, -0.0125977, -0.0043615, 0.0037153
8: 0.0032746, 0.0062801, 0.0033326, 0.0067347, -0.0034602, 0.0029476
9: 0.0027997, 0.0090201, 0.0029632, 0.0098377, -0.0057350, 0.0052745

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040305, upper bound: 0.0040014
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0044886, upper bound: 0.0045117
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045186
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040901, -0.0029595, -0.0040990, -0.0030860, -0.0010040, 0.0011395
1: -0.0058297, -0.0035292, -0.0061636, -0.0036584, -0.0021714, 0.0026344
2: 0.9663826, 0.9713822, 0.9669008, 0.9713246, -0.0049420, 0.0044814
3: 0.0211036, 0.0352255, 0.0181483, 0.0348008, -0.0084226, 0.0118407
4: -0.0033721, -0.0011299, -0.0033398, -0.0013558, -0.0020164, 0.0022099
5: 0.0129861, 0.0149477, 0.0131196, 0.0151749, -0.0021888, 0.0018281
6: 0.0032308, 0.0051642, 0.0034376, 0.0051483, -0.0019175, 0.0017266
7: -0.0169073, -0.0132474, -0.0167972, -0.0124815, -0.0044257, 0.0035498
8: 0.0033157, 0.0062193, 0.0034030, 0.0068269, -0.0035111, 0.0028162
9: 0.0029157, 0.0089106, 0.0031617, 0.0100034, -0.0059808, 0.0052118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040561, upper bound: 0.0040147
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045197, upper bound: 0.0045139
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045286, upper bound: 0.0045192
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040901, -0.0029595, -0.0041001, -0.0030089, -0.0010811, 0.0011406
1: -0.0058297, -0.0035292, -0.0062061, -0.0035797, -0.0022501, 0.0026769
2: 0.9663826, 0.9713822, 0.9665850, 0.9713597, -0.0049770, 0.0047972
3: 0.0211036, 0.0352255, 0.0177721, 0.0350596, -0.0087430, 0.0123308
4: -0.0033721, -0.0011299, -0.0033595, -0.0012182, -0.0021540, 0.0022296
5: 0.0129861, 0.0149477, 0.0130383, 0.0152038, -0.0022177, 0.0019094
6: 0.0032308, 0.0051642, 0.0033116, 0.0051580, -0.0019272, 0.0018526
7: -0.0169073, -0.0132474, -0.0168642, -0.0123840, -0.0045232, 0.0036169
8: 0.0033157, 0.0062193, 0.0033498, 0.0069042, -0.0035885, 0.0028694
9: 0.0029157, 0.0089106, 0.0030118, 0.0101425, -0.0061271, 0.0053645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040561, upper bound: 0.0040147
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045197, upper bound: 0.0045139
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045286, upper bound: 0.0045192
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040910, -0.0028999, -0.0040990, -0.0030860, -0.0010049, 0.0011991
1: -0.0058632, -0.0034683, -0.0061636, -0.0036584, -0.0022048, 0.0026953
2: 0.9661385, 0.9714093, 0.9669008, 0.9713246, -0.0051861, 0.0045086
3: 0.0208075, 0.0354257, 0.0181483, 0.0348008, -0.0088664, 0.0121079
4: -0.0033874, -0.0010235, -0.0033398, -0.0013558, -0.0020316, 0.0023164
5: 0.0129232, 0.0149705, 0.0131196, 0.0151749, -0.0022517, 0.0018509
6: 0.0031334, 0.0051717, 0.0034376, 0.0051483, -0.0020149, 0.0017341
7: -0.0169591, -0.0131707, -0.0167972, -0.0124815, -0.0044776, 0.0036265
8: 0.0032746, 0.0062801, 0.0034030, 0.0068269, -0.0035523, 0.0028771
9: 0.0027997, 0.0090201, 0.0031617, 0.0100034, -0.0060930, 0.0053029

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040389, upper bound: 0.0040070
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0044886, upper bound: 0.0045119
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045192
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040910, -0.0028999, -0.0041001, -0.0030089, -0.0010820, 0.0012003
1: -0.0058632, -0.0034683, -0.0062061, -0.0035797, -0.0022835, 0.0027378
2: 0.9661385, 0.9714093, 0.9665850, 0.9713597, -0.0052212, 0.0048243
3: 0.0208075, 0.0354257, 0.0177721, 0.0350596, -0.0085545, 0.0119878
4: -0.0033874, -0.0010235, -0.0033595, -0.0012182, -0.0021692, 0.0023361
5: 0.0129232, 0.0149705, 0.0130383, 0.0152038, -0.0022806, 0.0019322
6: 0.0031334, 0.0051717, 0.0033116, 0.0051580, -0.0020246, 0.0018601
7: -0.0169591, -0.0131707, -0.0168642, -0.0123840, -0.0045751, 0.0036936
8: 0.0032746, 0.0062801, 0.0033498, 0.0069042, -0.0036297, 0.0029303
9: 0.0027997, 0.0090201, 0.0030118, 0.0101425, -0.0061196, 0.0053782

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040389, upper bound: 0.0040070
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0044886, upper bound: 0.0045119
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045192
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040965, -0.0030587, -0.0040971, -0.0029790, -0.0011175, 0.0010384
1: -0.0060710, -0.0036305, -0.0060906, -0.0035491, -0.0025219, 0.0024602
2: 0.9667888, 0.9713370, 0.9664625, 0.9713733, -0.0045844, 0.0048745
3: 0.0189683, 0.0348926, 0.0187942, 0.0351602, -0.0093371, 0.0092123
4: -0.0033468, -0.0013070, -0.0033672, -0.0011647, -0.0021821, 0.0020602
5: 0.0130908, 0.0151119, 0.0130067, 0.0151252, -0.0020345, 0.0021052
6: 0.0033929, 0.0051518, 0.0032627, 0.0051618, -0.0017689, 0.0018891
7: -0.0168210, -0.0126940, -0.0168903, -0.0126489, -0.0041721, 0.0041963
8: 0.0033842, 0.0066583, 0.0033292, 0.0066941, -0.0033099, 0.0033291
9: 0.0031085, 0.0097002, 0.0029535, 0.0097646, -0.0053854, 0.0055308

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040495, upper bound: 0.0040465
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040495, upper bound: 0.0040465
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040954, -0.0030713, -0.0040930, -0.0021511, -0.0019444, 0.0010217
1: -0.0060296, -0.0036433, -0.0059374, -0.0027041, -0.0033255, 0.0022941
2: 0.9668403, 0.9713314, 0.9630734, 0.9717500, -0.0049097, 0.0082580
3: 0.0193350, 0.0348504, 0.0201505, 0.0379388, -0.0121960, 0.0088623
4: -0.0033436, -0.0013294, -0.0035785, 0.0003130, -0.0036566, 0.0022491
5: 0.0131041, 0.0150837, 0.0121332, 0.0150210, -0.0019169, 0.0029505
6: 0.0034135, 0.0051502, 0.0019102, 0.0052657, -0.0018522, 0.0032400
7: -0.0168100, -0.0127891, -0.0176104, -0.0117629, -0.0050471, 0.0048214
8: 0.0033929, 0.0065829, 0.0027579, 0.0075767, -0.0041838, 0.0038250
9: 0.0031330, 0.0095646, 0.0013441, 0.0092630, -0.0057673, 0.0071020

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040495, upper bound: 0.0040465
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040495, upper bound: 0.0040465
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0029839, -0.0040976, -0.0029764, -0.0011213, 0.0011136
1: -0.0061130, -0.0035541, -0.0061099, -0.0035465, -0.0025665, 0.0025558
2: 0.9664828, 0.9713710, 0.9664519, 0.9713745, -0.0048917, 0.0049191
3: 0.0185965, 0.0351436, 0.0186235, 0.0351689, -0.0092419, 0.0096739
4: -0.0033659, -0.0011735, -0.0033678, -0.0011601, -0.0022058, 0.0021943
5: 0.0130119, 0.0151404, 0.0130039, 0.0151384, -0.0021265, 0.0021365
6: 0.0032707, 0.0051611, 0.0032584, 0.0051621, -0.0018913, 0.0019027
7: -0.0168860, -0.0125977, -0.0168926, -0.0126047, -0.0042813, 0.0042949
8: 0.0033326, 0.0067347, 0.0033274, 0.0067292, -0.0033966, 0.0034074
9: 0.0029632, 0.0098377, 0.0029485, 0.0098277, -0.0055042, 0.0055746

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040465, upper bound: 0.0040465
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040465, upper bound: 0.0040465
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040965, -0.0029970, -0.0040934, -0.0021491, -0.0019474, 0.0010964
1: -0.0060700, -0.0035674, -0.0059537, -0.0027021, -0.0033680, 0.0023862
2: 0.9665360, 0.9713650, 0.9630653, 0.9717509, -0.0052148, 0.0082997
3: 0.0189768, 0.0350998, 0.0200065, 0.0379454, -0.0121005, 0.0093100
4: -0.0033626, -0.0011967, -0.0035790, 0.0003165, -0.0036791, 0.0023823
5: 0.0130256, 0.0151112, 0.0121311, 0.0150321, -0.0020064, 0.0029801
6: 0.0032920, 0.0051595, 0.0019070, 0.0052659, -0.0019739, 0.0032525
7: -0.0168747, -0.0126962, -0.0176121, -0.0117586, -0.0051161, 0.0049159
8: 0.0033416, 0.0066565, 0.0027565, 0.0075803, -0.0042387, 0.0039000
9: 0.0029885, 0.0096970, 0.0013402, 0.0093163, -0.0058850, 0.0071876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040465, upper bound: 0.0040465
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040465, upper bound: 0.0040465
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040965, -0.0030587, -0.0040995, -0.0030034, -0.0010932, 0.0010408
1: -0.0060710, -0.0036305, -0.0061831, -0.0035740, -0.0024970, 0.0025527
2: 0.9667888, 0.9713370, 0.9665622, 0.9713622, -0.0045733, 0.0047748
3: 0.0189683, 0.0348926, 0.0179755, 0.0350783, -0.0096178, 0.0104797
4: -0.0033468, -0.0013070, -0.0033609, -0.0012082, -0.0021386, 0.0020540
5: 0.0130908, 0.0151119, 0.0130324, 0.0151882, -0.0020974, 0.0020795
6: 0.0033929, 0.0051518, 0.0033025, 0.0051587, -0.0017658, 0.0018493
7: -0.0168210, -0.0126940, -0.0168691, -0.0124367, -0.0043842, 0.0041751
8: 0.0033842, 0.0066583, 0.0033460, 0.0068624, -0.0034782, 0.0033123
9: 0.0031085, 0.0097002, 0.0030009, 0.0100673, -0.0056920, 0.0057690

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040508, upper bound: 0.0040527
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040508, upper bound: 0.0040527
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040954, -0.0030713, -0.0040954, -0.0021771, -0.0019183, 0.0010241
1: -0.0060296, -0.0036433, -0.0060274, -0.0027307, -0.0032989, 0.0023841
2: 0.9668403, 0.9713314, 0.9631801, 0.9717382, -0.0048978, 0.0081513
3: 0.0193350, 0.0348504, 0.0193537, 0.0378513, -0.0123550, 0.0101863
4: -0.0033436, -0.0013294, -0.0035718, 0.0002665, -0.0036101, 0.0022424
5: 0.0131041, 0.0150837, 0.0121607, 0.0150822, -0.0019782, 0.0029230
6: 0.0034135, 0.0051502, 0.0019528, 0.0052624, -0.0018489, 0.0031974
7: -0.0168100, -0.0127891, -0.0175878, -0.0118202, -0.0049899, 0.0047987
8: 0.0033929, 0.0065829, 0.0027759, 0.0075288, -0.0041360, 0.0038071
9: 0.0031330, 0.0095646, 0.0013947, 0.0095576, -0.0060656, 0.0072312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040508, upper bound: 0.0040527
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040508, upper bound: 0.0040527
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0029839, -0.0041000, -0.0030012, -0.0010965, 0.0011161
1: -0.0061130, -0.0035541, -0.0062008, -0.0035717, -0.0025412, 0.0026467
2: 0.9664828, 0.9713710, 0.9665533, 0.9713632, -0.0048804, 0.0048177
3: 0.0185965, 0.0351436, 0.0178193, 0.0350857, -0.0094269, 0.0109778
4: -0.0033659, -0.0011735, -0.0033615, -0.0012043, -0.0021616, 0.0021880
5: 0.0130119, 0.0151404, 0.0130301, 0.0152002, -0.0021883, 0.0021104
6: 0.0032707, 0.0051611, 0.0032989, 0.0051590, -0.0018882, 0.0018622
7: -0.0168860, -0.0125977, -0.0168710, -0.0123963, -0.0044898, 0.0042734
8: 0.0033326, 0.0067347, 0.0033445, 0.0068945, -0.0035620, 0.0033903
9: 0.0029632, 0.0098377, 0.0029967, 0.0101251, -0.0058336, 0.0057536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040465, upper bound: 0.0040521
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040465, upper bound: 0.0040521
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040965, -0.0029970, -0.0040959, -0.0021753, -0.0019212, 0.0010990
1: -0.0060700, -0.0035674, -0.0060475, -0.0027288, -0.0033412, 0.0024801
2: 0.9665360, 0.9713650, 0.9631726, 0.9717391, -0.0052030, 0.0081925
3: 0.0189768, 0.0350998, 0.0191759, 0.0378575, -0.0121808, 0.0106539
4: -0.0033626, -0.0011967, -0.0035723, 0.0002698, -0.0036323, 0.0023756
5: 0.0130256, 0.0151112, 0.0121587, 0.0150959, -0.0020703, 0.0029525
6: 0.0032920, 0.0051595, 0.0019498, 0.0052626, -0.0019706, 0.0032097
7: -0.0168747, -0.0126962, -0.0175894, -0.0118161, -0.0050585, 0.0048931
8: 0.0033416, 0.0066565, 0.0027746, 0.0075322, -0.0041906, 0.0038820
9: 0.0029885, 0.0096970, 0.0013912, 0.0096234, -0.0062006, 0.0072151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040465, upper bound: 0.0040521
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040465, upper bound: 0.0040521
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040927, -0.0029701, -0.0040901, -0.0029595, -0.0011332, 0.0011200
1: -0.0059286, -0.0035400, -0.0058297, -0.0035292, -0.0023994, 0.0022897
2: 0.9664259, 0.9713773, 0.9663826, 0.9713822, -0.0049563, 0.0049947
3: 0.0202286, 0.0351901, 0.0211036, 0.0352255, -0.0095860, 0.0085799
4: -0.0033694, -0.0011488, -0.0033721, -0.0011299, -0.0022395, 0.0022234
5: 0.0129973, 0.0150150, 0.0129861, 0.0149477, -0.0019505, 0.0020289
6: 0.0032481, 0.0051629, 0.0032308, 0.0051642, -0.0019161, 0.0019320
7: -0.0168981, -0.0130206, -0.0169073, -0.0132474, -0.0036507, 0.0038866
8: 0.0033230, 0.0063992, 0.0033157, 0.0062193, -0.0028962, 0.0030834
9: 0.0029362, 0.0092341, 0.0029157, 0.0089106, -0.0052854, 0.0053690

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045197, upper bound: 0.0045093
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045279, upper bound: 0.0045124
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040927, -0.0029701, -0.0040910, -0.0028988, -0.0011939, 0.0011209
1: -0.0059286, -0.0035400, -0.0058632, -0.0034673, -0.0024613, 0.0023232
2: 0.9664259, 0.9713773, 0.9661344, 0.9714097, -0.0049838, 0.0052429
3: 0.0202286, 0.0351901, 0.0208075, 0.0354292, -0.0098565, 0.0090237
4: -0.0033694, -0.0011488, -0.0033876, -0.0010216, -0.0023478, 0.0022389
5: 0.0129973, 0.0150150, 0.0129221, 0.0149705, -0.0019732, 0.0020929
6: 0.0032481, 0.0051629, 0.0031317, 0.0051718, -0.0019237, 0.0020312
7: -0.0168981, -0.0130206, -0.0169600, -0.0131707, -0.0037274, 0.0039394
8: 0.0033230, 0.0063992, 0.0032739, 0.0062802, -0.0029571, 0.0031253
9: 0.0029362, 0.0092341, 0.0027977, 0.0090201, -0.0053771, 0.0054830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045197, upper bound: 0.0045093
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045279, upper bound: 0.0045124
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040937, -0.0028991, -0.0040901, -0.0029595, -0.0011342, 0.0011910
1: -0.0059668, -0.0034676, -0.0058297, -0.0035292, -0.0024376, 0.0023622
2: 0.9661356, 0.9714097, 0.9663826, 0.9713822, -0.0052466, 0.0050271
3: 0.0198902, 0.0354283, 0.0211036, 0.0352255, -0.0100868, 0.0089102
4: -0.0033876, -0.0010221, -0.0033721, -0.0011299, -0.0022576, 0.0023500
5: 0.0129224, 0.0150410, 0.0129861, 0.0149477, -0.0020253, 0.0020549
6: 0.0031322, 0.0051718, 0.0032308, 0.0051642, -0.0020320, 0.0019409
7: -0.0169598, -0.0129330, -0.0169073, -0.0132474, -0.0037124, 0.0039743
8: 0.0032740, 0.0064687, 0.0033157, 0.0062193, -0.0029452, 0.0031530
9: 0.0027983, 0.0093593, 0.0029157, 0.0089106, -0.0054020, 0.0054947

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0044917, upper bound: 0.0045075
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045154, upper bound: 0.0045124
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040937, -0.0028991, -0.0040910, -0.0028988, -0.0011949, 0.0011919
1: -0.0059668, -0.0034676, -0.0058632, -0.0034673, -0.0024995, 0.0023956
2: 0.9661356, 0.9714097, 0.9661344, 0.9714097, -0.0052742, 0.0052753
3: 0.0198902, 0.0354283, 0.0208075, 0.0354292, -0.0097367, 0.0087269
4: -0.0033876, -0.0010221, -0.0033876, -0.0010216, -0.0023659, 0.0023655
5: 0.0129224, 0.0150410, 0.0129221, 0.0149705, -0.0020481, 0.0021189
6: 0.0031322, 0.0051718, 0.0031317, 0.0051718, -0.0020396, 0.0020401
7: -0.0169598, -0.0129330, -0.0169600, -0.0131707, -0.0037891, 0.0040271
8: 0.0032740, 0.0064687, 0.0032739, 0.0062802, -0.0030061, 0.0031949
9: 0.0027983, 0.0093593, 0.0027977, 0.0090201, -0.0054546, 0.0055458

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0044917, upper bound: 0.0045075
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045154, upper bound: 0.0045124
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040927, -0.0029701, -0.0040932, -0.0028937, -0.0011991, 0.0011232
1: -0.0059286, -0.0035400, -0.0059466, -0.0034620, -0.0024666, 0.0024066
2: 0.9664259, 0.9713773, 0.9661132, 0.9714121, -0.0049862, 0.0052641
3: 0.0202286, 0.0351901, 0.0200690, 0.0354465, -0.0091184, 0.0089504
4: -0.0033694, -0.0011488, -0.0033889, -0.0010124, -0.0023570, 0.0022402
5: 0.0129973, 0.0150150, 0.0129167, 0.0150273, -0.0020300, 0.0020983
6: 0.0032481, 0.0051629, 0.0031233, 0.0051725, -0.0019244, 0.0020396
7: -0.0168981, -0.0130206, -0.0169645, -0.0129793, -0.0039188, 0.0039439
8: 0.0033230, 0.0063992, 0.0032703, 0.0064320, -0.0031090, 0.0031289
9: 0.0029362, 0.0092341, 0.0027877, 0.0092932, -0.0052817, 0.0054450

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037748, upper bound: 0.0037748
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037748, upper bound: 0.0037748
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040916, -0.0029821, -0.0040895, -0.0019948, -0.0020968, 0.0011074
1: -0.0058869, -0.0035523, -0.0058070, -0.0025446, -0.0033423, 0.0022546
2: 0.9664752, 0.9713719, 0.9624336, 0.9718212, -0.0053459, 0.0089383
3: 0.0205978, 0.0351496, 0.0213053, 0.0384633, -0.0122069, 0.0087153
4: -0.0033664, -0.0011703, -0.0036184, 0.0005919, -0.0039583, 0.0024481
5: 0.0130100, 0.0149866, 0.0119683, 0.0149322, -0.0019222, 0.0030183
6: 0.0032678, 0.0051614, 0.0016549, 0.0052853, -0.0020175, 0.0035064
7: -0.0168876, -0.0131163, -0.0177464, -0.0114198, -0.0054678, 0.0046300
8: 0.0033313, 0.0063233, 0.0026500, 0.0078636, -0.0045323, 0.0036732
9: 0.0029597, 0.0090976, 0.0010403, 0.0088360, -0.0057211, 0.0070977

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037748, upper bound: 0.0037748
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037748, upper bound: 0.0037748
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040937, -0.0028991, -0.0040936, -0.0028912, -0.0012026, 0.0011945
1: -0.0059668, -0.0034676, -0.0059631, -0.0034595, -0.0025074, 0.0024955
2: 0.9661356, 0.9714097, 0.9661030, 0.9714132, -0.0052776, 0.0053067
3: 0.0198902, 0.0354283, 0.0199232, 0.0354549, -0.0089574, 0.0094724
4: -0.0033876, -0.0010221, -0.0033896, -0.0010080, -0.0023796, 0.0023675
5: 0.0129224, 0.0150410, 0.0129140, 0.0150385, -0.0021161, 0.0021270
6: 0.0031322, 0.0051718, 0.0031192, 0.0051728, -0.0020406, 0.0020526
7: -0.0169598, -0.0129330, -0.0169667, -0.0129415, -0.0040183, 0.0040337
8: 0.0032740, 0.0064687, 0.0032686, 0.0064619, -0.0031879, 0.0032002
9: 0.0027983, 0.0093593, 0.0027829, 0.0093470, -0.0054169, 0.0054597

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037748, upper bound: 0.0037748
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037748, upper bound: 0.0037748
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040927, -0.0029112, -0.0040900, -0.0019924, -0.0021002, 0.0011788
1: -0.0059262, -0.0034799, -0.0058266, -0.0025422, -0.0033841, 0.0023467
2: 0.9661851, 0.9714041, 0.9624239, 0.9718221, -0.0056370, 0.0089802
3: 0.0202494, 0.0353876, 0.0211311, 0.0384712, -0.0120729, 0.0092427
4: -0.0033845, -0.0010437, -0.0036190, 0.0005961, -0.0039806, 0.0025752
5: 0.0129352, 0.0150134, 0.0119658, 0.0149456, -0.0020104, 0.0030476
6: 0.0031520, 0.0051703, 0.0016511, 0.0052856, -0.0021336, 0.0035192
7: -0.0169493, -0.0130260, -0.0177484, -0.0114146, -0.0055346, 0.0047224
8: 0.0032824, 0.0063949, 0.0026484, 0.0078680, -0.0045855, 0.0037465
9: 0.0028218, 0.0092264, 0.0010357, 0.0089004, -0.0058579, 0.0071685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037748, upper bound: 0.0037748
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037748, upper bound: 0.0037748
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040990, -0.0030860, -0.0040901, -0.0029595, -0.0011395, 0.0010040
1: -0.0061636, -0.0036584, -0.0058297, -0.0035292, -0.0026344, 0.0021714
2: 0.9669008, 0.9713246, 0.9663826, 0.9713822, -0.0044814, 0.0049420
3: 0.0181483, 0.0348008, 0.0211036, 0.0352255, -0.0118407, 0.0084226
4: -0.0033398, -0.0013558, -0.0033721, -0.0011299, -0.0022099, 0.0020164
5: 0.0131196, 0.0151749, 0.0129861, 0.0149477, -0.0018281, 0.0021888
6: 0.0034376, 0.0051483, 0.0032308, 0.0051642, -0.0017266, 0.0019175
7: -0.0167972, -0.0124815, -0.0169073, -0.0132474, -0.0035498, 0.0044257
8: 0.0034030, 0.0068269, 0.0033157, 0.0062193, -0.0028162, 0.0035111
9: 0.0031617, 0.0100034, 0.0029157, 0.0089106, -0.0052118, 0.0059808

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045238, upper bound: 0.0045117
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045322, upper bound: 0.0045124
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040990, -0.0030860, -0.0040910, -0.0028988, -0.0012002, 0.0010049
1: -0.0061636, -0.0036584, -0.0058632, -0.0034673, -0.0026963, 0.0022048
2: 0.9669008, 0.9713246, 0.9661344, 0.9714097, -0.0045090, 0.0051903
3: 0.0181483, 0.0348008, 0.0208075, 0.0354292, -0.0121112, 0.0088664
4: -0.0033398, -0.0013558, -0.0033876, -0.0010216, -0.0023182, 0.0020319
5: 0.0131196, 0.0151749, 0.0129221, 0.0149705, -0.0018509, 0.0022528
6: 0.0034376, 0.0051483, 0.0031317, 0.0051718, -0.0017343, 0.0020166
7: -0.0167972, -0.0124815, -0.0169600, -0.0131707, -0.0036265, 0.0044785
8: 0.0034030, 0.0068269, 0.0032739, 0.0062802, -0.0028771, 0.0035530
9: 0.0031617, 0.0100034, 0.0027977, 0.0090201, -0.0053035, 0.0060948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045238, upper bound: 0.0045117
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045322, upper bound: 0.0045124
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041001, -0.0030090, -0.0040901, -0.0029595, -0.0011406, 0.0010811
1: -0.0062061, -0.0035798, -0.0058297, -0.0035292, -0.0026769, 0.0022500
2: 0.9665853, 0.9713596, 0.9663826, 0.9713822, -0.0047969, 0.0049770
3: 0.0177721, 0.0350594, 0.0211036, 0.0352255, -0.0123308, 0.0087502
4: -0.0033595, -0.0012183, -0.0033721, -0.0011299, -0.0022296, 0.0021538
5: 0.0130384, 0.0152038, 0.0129861, 0.0149477, -0.0019094, 0.0022177
6: 0.0033117, 0.0051580, 0.0032308, 0.0051642, -0.0018525, 0.0019272
7: -0.0168642, -0.0123840, -0.0169073, -0.0132474, -0.0036168, 0.0045232
8: 0.0033499, 0.0069042, 0.0033157, 0.0062193, -0.0028694, 0.0035885
9: 0.0030119, 0.0101425, 0.0029157, 0.0089106, -0.0053546, 0.0061271

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045078, upper bound: 0.0045115
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045192, upper bound: 0.0045124
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041001, -0.0030090, -0.0040910, -0.0028988, -0.0012013, 0.0010820
1: -0.0062061, -0.0035798, -0.0058632, -0.0034673, -0.0027388, 0.0022834
2: 0.9665853, 0.9713596, 0.9661344, 0.9714097, -0.0048244, 0.0052252
3: 0.0177721, 0.0350594, 0.0208075, 0.0354292, -0.0119910, 0.0085643
4: -0.0033595, -0.0012183, -0.0033876, -0.0010216, -0.0023379, 0.0021693
5: 0.0130384, 0.0152038, 0.0129221, 0.0149705, -0.0019321, 0.0022817
6: 0.0033117, 0.0051580, 0.0031317, 0.0051718, -0.0018601, 0.0020263
7: -0.0168642, -0.0123840, -0.0169600, -0.0131707, -0.0036935, 0.0045760
8: 0.0033499, 0.0069042, 0.0032739, 0.0062802, -0.0029303, 0.0036304
9: 0.0030119, 0.0101425, 0.0027977, 0.0090201, -0.0053658, 0.0061210

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045078, upper bound: 0.0045115
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045192, upper bound: 0.0045124
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040990, -0.0030860, -0.0040932, -0.0028937, -0.0012053, 0.0010072
1: -0.0061636, -0.0036584, -0.0059466, -0.0034620, -0.0027016, 0.0022883
2: 0.9669008, 0.9713246, 0.9661132, 0.9714121, -0.0045114, 0.0052114
3: 0.0181483, 0.0348008, 0.0200690, 0.0354465, -0.0115447, 0.0089158
4: -0.0033398, -0.0013558, -0.0033889, -0.0010124, -0.0023274, 0.0020332
5: 0.0131196, 0.0151749, 0.0129167, 0.0150273, -0.0019076, 0.0022582
6: 0.0034376, 0.0051483, 0.0031233, 0.0051725, -0.0017349, 0.0020250
7: -0.0167972, -0.0124815, -0.0169645, -0.0129793, -0.0038179, 0.0044830
8: 0.0034030, 0.0068269, 0.0032703, 0.0064320, -0.0030289, 0.0035566
9: 0.0031617, 0.0100034, 0.0027877, 0.0092932, -0.0053353, 0.0060265

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038107, upper bound: 0.0038093
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038107, upper bound: 0.0038093
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040979, -0.0030993, -0.0040895, -0.0019948, -0.0021031, 0.0009902
1: -0.0061212, -0.0036719, -0.0058070, -0.0025446, -0.0035767, 0.0021351
2: 0.9669549, 0.9713186, 0.9624336, 0.9718212, -0.0048662, 0.0088850
3: 0.0185236, 0.0347564, 0.0213053, 0.0384633, -0.0146007, 0.0086809
4: -0.0033365, -0.0013794, -0.0036184, 0.0005919, -0.0039284, 0.0022390
5: 0.0131336, 0.0151460, 0.0119683, 0.0149322, -0.0017986, 0.0031778
6: 0.0034592, 0.0051467, 0.0016549, 0.0052853, -0.0018261, 0.0034917
7: -0.0167857, -0.0125788, -0.0177464, -0.0114198, -0.0053659, 0.0051676
8: 0.0034122, 0.0067497, 0.0026500, 0.0078636, -0.0044514, 0.0040997
9: 0.0031874, 0.0098646, 0.0010403, 0.0088360, -0.0056486, 0.0076914

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038107, upper bound: 0.0038093
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038107, upper bound: 0.0038093
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041001, -0.0030090, -0.0040936, -0.0028912, -0.0012090, 0.0010846
1: -0.0062061, -0.0035798, -0.0059631, -0.0034595, -0.0027467, 0.0023833
2: 0.9665853, 0.9713596, 0.9661030, 0.9714132, -0.0048279, 0.0052567
3: 0.0177721, 0.0350594, 0.0199232, 0.0354549, -0.0113837, 0.0093894
4: -0.0033595, -0.0012183, -0.0033896, -0.0010080, -0.0023515, 0.0021713
5: 0.0130384, 0.0152038, 0.0129140, 0.0150385, -0.0020001, 0.0022898
6: 0.0033117, 0.0051580, 0.0031192, 0.0051728, -0.0018610, 0.0020388
7: -0.0168642, -0.0123840, -0.0169667, -0.0129415, -0.0039227, 0.0045827
8: 0.0033499, 0.0069042, 0.0032686, 0.0064619, -0.0031121, 0.0036356
9: 0.0030119, 0.0101425, 0.0027829, 0.0093470, -0.0054683, 0.0059625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038103, upper bound: 0.0038089
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038103, upper bound: 0.0038089
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040990, -0.0030223, -0.0040900, -0.0019924, -0.0021066, 0.0010677
1: -0.0061645, -0.0035933, -0.0058266, -0.0025422, -0.0036223, 0.0022333
2: 0.9666398, 0.9713536, 0.9624239, 0.9718221, -0.0051823, 0.0089297
3: 0.0181406, 0.0350147, 0.0211311, 0.0384712, -0.0144581, 0.0091532
4: -0.0033561, -0.0012420, -0.0036190, 0.0005961, -0.0039522, 0.0023770
5: 0.0130524, 0.0151755, 0.0119658, 0.0149456, -0.0018932, 0.0032097
6: 0.0033335, 0.0051563, 0.0016511, 0.0052856, -0.0019521, 0.0035052
7: -0.0168526, -0.0124795, -0.0177484, -0.0114146, -0.0054380, 0.0052689
8: 0.0033591, 0.0068285, 0.0026484, 0.0078680, -0.0045089, 0.0041801
9: 0.0030378, 0.0100063, 0.0010357, 0.0089004, -0.0058626, 0.0076707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038103, upper bound: 0.0038089
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038103, upper bound: 0.0038089
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040927, -0.0029701, -0.0040971, -0.0029790, -0.0011137, 0.0011270
1: -0.0059286, -0.0035400, -0.0060906, -0.0035491, -0.0023795, 0.0025506
2: 0.9664259, 0.9713773, 0.9664625, 0.9713733, -0.0049474, 0.0049148
3: 0.0202286, 0.0351901, 0.0187942, 0.0351602, -0.0099130, 0.0112886
4: -0.0033694, -0.0011488, -0.0033672, -0.0011647, -0.0022048, 0.0022184
5: 0.0129973, 0.0150150, 0.0130067, 0.0151252, -0.0021280, 0.0020083
6: 0.0032481, 0.0051629, 0.0032627, 0.0051618, -0.0019137, 0.0019002
7: -0.0168981, -0.0130206, -0.0168903, -0.0126489, -0.0042492, 0.0038697
8: 0.0033230, 0.0063992, 0.0033292, 0.0066941, -0.0033711, 0.0030700
9: 0.0029362, 0.0092341, 0.0029535, 0.0097646, -0.0059490, 0.0057184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037671, upper bound: 0.0037682
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037671, upper bound: 0.0037682
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040916, -0.0029821, -0.0040930, -0.0021511, -0.0019406, 0.0011108
1: -0.0058869, -0.0035523, -0.0059374, -0.0027041, -0.0031828, 0.0023851
2: 0.9664752, 0.9713719, 0.9630734, 0.9717500, -0.0052748, 0.0082986
3: 0.0205978, 0.0351496, 0.0201505, 0.0379388, -0.0124811, 0.0109571
4: -0.0033664, -0.0011703, -0.0035785, 0.0003130, -0.0036794, 0.0024082
5: 0.0130100, 0.0149866, 0.0121332, 0.0150210, -0.0020110, 0.0028534
6: 0.0032678, 0.0051614, 0.0019102, 0.0052657, -0.0019979, 0.0032512
7: -0.0168876, -0.0131163, -0.0176104, -0.0117629, -0.0051247, 0.0044941
8: 0.0033313, 0.0063233, 0.0027579, 0.0075767, -0.0042454, 0.0035654
9: 0.0029597, 0.0090976, 0.0013441, 0.0092630, -0.0063033, 0.0070827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037671, upper bound: 0.0037682
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037671, upper bound: 0.0037682
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040937, -0.0028991, -0.0040976, -0.0029764, -0.0011174, 0.0011985
1: -0.0059668, -0.0034676, -0.0061099, -0.0035465, -0.0024204, 0.0026424
2: 0.9661356, 0.9714097, 0.9664519, 0.9713745, -0.0052390, 0.0049577
3: 0.0198902, 0.0354283, 0.0186235, 0.0351689, -0.0098595, 0.0118008
4: -0.0033876, -0.0010221, -0.0033678, -0.0011601, -0.0022275, 0.0023457
5: 0.0129224, 0.0150410, 0.0130039, 0.0151384, -0.0022160, 0.0020371
6: 0.0031322, 0.0051718, 0.0032584, 0.0051621, -0.0020299, 0.0019133
7: -0.0169598, -0.0129330, -0.0168926, -0.0126047, -0.0043551, 0.0039596
8: 0.0032740, 0.0064687, 0.0033274, 0.0067292, -0.0034551, 0.0031414
9: 0.0027983, 0.0093593, 0.0029485, 0.0098277, -0.0061459, 0.0056683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037718, upper bound: 0.0037728
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037718, upper bound: 0.0037728
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040927, -0.0029112, -0.0040934, -0.0021491, -0.0019436, 0.0011822
1: -0.0059262, -0.0034799, -0.0059537, -0.0027021, -0.0032242, 0.0024738
2: 0.9661851, 0.9714041, 0.9630653, 0.9717509, -0.0055658, 0.0083388
3: 0.0202494, 0.0353876, 0.0200065, 0.0379454, -0.0124496, 0.0114437
4: -0.0033845, -0.0010437, -0.0035790, 0.0003165, -0.0037009, 0.0025353
5: 0.0129352, 0.0150134, 0.0121311, 0.0150321, -0.0020969, 0.0028823
6: 0.0031520, 0.0051703, 0.0019070, 0.0052659, -0.0021139, 0.0032633
7: -0.0169493, -0.0130260, -0.0176121, -0.0117586, -0.0051906, 0.0045861
8: 0.0032824, 0.0063949, 0.0027565, 0.0075803, -0.0042979, 0.0036384
9: 0.0028218, 0.0092264, 0.0013402, 0.0093163, -0.0064944, 0.0070443

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037718, upper bound: 0.0037728
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037718, upper bound: 0.0037728
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040927, -0.0029701, -0.0040995, -0.0030034, -0.0010894, 0.0011295
1: -0.0059286, -0.0035400, -0.0061831, -0.0035740, -0.0023546, 0.0026431
2: 0.9664259, 0.9713773, 0.9665622, 0.9713622, -0.0049363, 0.0048151
3: 0.0202286, 0.0351901, 0.0179755, 0.0350783, -0.0090225, 0.0113979
4: -0.0033694, -0.0011488, -0.0033609, -0.0012082, -0.0021612, 0.0022122
5: 0.0129973, 0.0150150, 0.0130324, 0.0151882, -0.0021909, 0.0019826
6: 0.0032481, 0.0051629, 0.0033025, 0.0051587, -0.0019106, 0.0018604
7: -0.0168981, -0.0130206, -0.0168691, -0.0124367, -0.0044613, 0.0038485
8: 0.0033230, 0.0063992, 0.0033460, 0.0068624, -0.0035394, 0.0030532
9: 0.0029362, 0.0092341, 0.0030009, 0.0100673, -0.0058282, 0.0054846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038087, upper bound: 0.0038100
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038087, upper bound: 0.0038100
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040916, -0.0029821, -0.0040954, -0.0021771, -0.0019145, 0.0011132
1: -0.0058869, -0.0035523, -0.0060274, -0.0027307, -0.0031562, 0.0024751
2: 0.9664752, 0.9713719, 0.9631801, 0.9717382, -0.0052629, 0.0081919
3: 0.0205978, 0.0351496, 0.0193537, 0.0378513, -0.0115928, 0.0110715
4: -0.0033664, -0.0011703, -0.0035718, 0.0002665, -0.0036328, 0.0024016
5: 0.0130100, 0.0149866, 0.0121607, 0.0150822, -0.0020722, 0.0028259
6: 0.0032678, 0.0051614, 0.0019528, 0.0052624, -0.0019946, 0.0032086
7: -0.0168876, -0.0131163, -0.0175878, -0.0118202, -0.0050674, 0.0044714
8: 0.0033313, 0.0063233, 0.0027759, 0.0075288, -0.0041975, 0.0035474
9: 0.0029597, 0.0090976, 0.0013947, 0.0095576, -0.0062069, 0.0068150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038087, upper bound: 0.0038100
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038087, upper bound: 0.0038100
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040937, -0.0028991, -0.0041000, -0.0030012, -0.0010926, 0.0012009
1: -0.0059668, -0.0034676, -0.0062008, -0.0035717, -0.0023951, 0.0027332
2: 0.9661356, 0.9714097, 0.9665533, 0.9713632, -0.0052276, 0.0048563
3: 0.0198902, 0.0354283, 0.0178193, 0.0350857, -0.0089115, 0.0119093
4: -0.0033876, -0.0010221, -0.0033615, -0.0012043, -0.0021833, 0.0023394
5: 0.0129224, 0.0150410, 0.0130301, 0.0152002, -0.0022778, 0.0020109
6: 0.0031322, 0.0051718, 0.0032989, 0.0051590, -0.0020268, 0.0018729
7: -0.0169598, -0.0129330, -0.0168710, -0.0123963, -0.0045635, 0.0039381
8: 0.0032740, 0.0064687, 0.0033445, 0.0068945, -0.0036205, 0.0031243
9: 0.0027983, 0.0093593, 0.0029967, 0.0101251, -0.0060172, 0.0054977

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038087, upper bound: 0.0038100
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038087, upper bound: 0.0038100
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040927, -0.0029112, -0.0040959, -0.0021753, -0.0019174, 0.0011847
1: -0.0059262, -0.0034799, -0.0060475, -0.0027288, -0.0031975, 0.0025676
2: 0.9661851, 0.9714041, 0.9631726, 0.9717391, -0.0055540, 0.0082316
3: 0.0202494, 0.0353876, 0.0191759, 0.0378575, -0.0115291, 0.0115639
4: -0.0033845, -0.0010437, -0.0035723, 0.0002698, -0.0036542, 0.0025286
5: 0.0129352, 0.0150134, 0.0121587, 0.0150959, -0.0021607, 0.0028547
6: 0.0031520, 0.0051703, 0.0019498, 0.0052626, -0.0021106, 0.0032205
7: -0.0169493, -0.0130260, -0.0175894, -0.0118161, -0.0051331, 0.0045633
8: 0.0032824, 0.0063949, 0.0027746, 0.0075322, -0.0042498, 0.0036203
9: 0.0028218, 0.0092264, 0.0013912, 0.0096234, -0.0063870, 0.0068248

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038087, upper bound: 0.0038100
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038087, upper bound: 0.0038100
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040990, -0.0030860, -0.0040971, -0.0029790, -0.0011200, 0.0010110
1: -0.0061636, -0.0036584, -0.0060906, -0.0035491, -0.0026145, 0.0024323
2: 0.9669008, 0.9713246, 0.9664625, 0.9713733, -0.0044725, 0.0048621
3: 0.0181483, 0.0348008, 0.0187942, 0.0351602, -0.0105787, 0.0094109
4: -0.0033398, -0.0013558, -0.0033672, -0.0011647, -0.0021752, 0.0020114
5: 0.0131196, 0.0151749, 0.0130067, 0.0151252, -0.0020056, 0.0021682
6: 0.0034376, 0.0051483, 0.0032627, 0.0051618, -0.0017242, 0.0018857
7: -0.0167972, -0.0124815, -0.0168903, -0.0126489, -0.0041483, 0.0044088
8: 0.0034030, 0.0068269, 0.0033292, 0.0066941, -0.0032910, 0.0034977
9: 0.0031617, 0.0100034, 0.0029535, 0.0097646, -0.0056056, 0.0058580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040519, upper bound: 0.0040465
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040519, upper bound: 0.0040465
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040979, -0.0030993, -0.0040930, -0.0021511, -0.0019468, 0.0009937
1: -0.0061212, -0.0036719, -0.0059374, -0.0027041, -0.0034171, 0.0022655
2: 0.9669549, 0.9713186, 0.9630734, 0.9717500, -0.0047951, 0.0082452
3: 0.0185236, 0.0347564, 0.0201505, 0.0379388, -0.0134486, 0.0090627
4: -0.0033365, -0.0013794, -0.0035785, 0.0003130, -0.0036494, 0.0021991
5: 0.0131336, 0.0151460, 0.0121332, 0.0150210, -0.0018874, 0.0030129
6: 0.0034592, 0.0051467, 0.0019102, 0.0052657, -0.0018065, 0.0032365
7: -0.0167857, -0.0125788, -0.0176104, -0.0117629, -0.0050228, 0.0050317
8: 0.0034122, 0.0067497, 0.0027579, 0.0075767, -0.0041645, 0.0039919
9: 0.0031874, 0.0098646, 0.0013441, 0.0092630, -0.0059902, 0.0074372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040519, upper bound: 0.0040465
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040519, upper bound: 0.0040465
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041001, -0.0030089, -0.0040976, -0.0029764, -0.0011237, 0.0010886
1: -0.0062061, -0.0035797, -0.0061099, -0.0035465, -0.0026597, 0.0025303
2: 0.9665850, 0.9713597, 0.9664519, 0.9713745, -0.0047895, 0.0049077
3: 0.0177721, 0.0350596, 0.0186235, 0.0351689, -0.0104786, 0.0099517
4: -0.0033595, -0.0012182, -0.0033678, -0.0011601, -0.0021995, 0.0021497
5: 0.0130383, 0.0152038, 0.0130039, 0.0151384, -0.0021001, 0.0021999
6: 0.0033116, 0.0051580, 0.0032584, 0.0051621, -0.0018505, 0.0018996
7: -0.0168642, -0.0123840, -0.0168926, -0.0126047, -0.0042596, 0.0045085
8: 0.0033498, 0.0069042, 0.0033274, 0.0067292, -0.0033793, 0.0035768
9: 0.0030118, 0.0101425, 0.0029485, 0.0098277, -0.0057444, 0.0058743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040516, upper bound: 0.0040465
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040516, upper bound: 0.0040465
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040990, -0.0030223, -0.0040934, -0.0021491, -0.0019499, 0.0010711
1: -0.0061645, -0.0035933, -0.0059537, -0.0027021, -0.0034624, 0.0023604
2: 0.9666398, 0.9713536, 0.9630653, 0.9717509, -0.0051110, 0.0082883
3: 0.0181406, 0.0350147, 0.0200065, 0.0379454, -0.0133472, 0.0095891
4: -0.0033561, -0.0012420, -0.0035790, 0.0003165, -0.0036726, 0.0023370
5: 0.0130524, 0.0151755, 0.0121311, 0.0150321, -0.0019797, 0.0030444
6: 0.0033335, 0.0051563, 0.0019070, 0.0052659, -0.0019324, 0.0032493
7: -0.0168526, -0.0124795, -0.0176121, -0.0117586, -0.0050940, 0.0051326
8: 0.0033591, 0.0068285, 0.0027565, 0.0075803, -0.0042212, 0.0040719
9: 0.0030378, 0.0100063, 0.0013402, 0.0093163, -0.0061196, 0.0074911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040516, upper bound: 0.0040465
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040516, upper bound: 0.0040465
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040990, -0.0030860, -0.0040995, -0.0030034, -0.0010956, 0.0010135
1: -0.0061636, -0.0036584, -0.0061831, -0.0035740, -0.0025896, 0.0025248
2: 0.9669008, 0.9713246, 0.9665622, 0.9713622, -0.0044614, 0.0047624
3: 0.0181483, 0.0348008, 0.0179755, 0.0350783, -0.0097336, 0.0096043
4: -0.0033398, -0.0013558, -0.0033609, -0.0012082, -0.0021316, 0.0020052
5: 0.0131196, 0.0151749, 0.0130324, 0.0151882, -0.0020685, 0.0021425
6: 0.0034376, 0.0051483, 0.0033025, 0.0051587, -0.0017211, 0.0018458
7: -0.0167972, -0.0124815, -0.0168691, -0.0124367, -0.0043605, 0.0043876
8: 0.0034030, 0.0068269, 0.0033460, 0.0068624, -0.0034594, 0.0034809
9: 0.0031617, 0.0100034, 0.0030009, 0.0100673, -0.0055381, 0.0056721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040574, upper bound: 0.0040568
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040574, upper bound: 0.0040568
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040979, -0.0030993, -0.0040954, -0.0021771, -0.0019207, 0.0009961
1: -0.0061212, -0.0036719, -0.0060274, -0.0027307, -0.0033905, 0.0023556
2: 0.9669549, 0.9713186, 0.9631801, 0.9717382, -0.0047832, 0.0081385
3: 0.0185236, 0.0347564, 0.0193537, 0.0378513, -0.0125810, 0.0092741
4: -0.0033365, -0.0013794, -0.0035718, 0.0002665, -0.0036029, 0.0021925
5: 0.0131336, 0.0151460, 0.0121607, 0.0150822, -0.0019486, 0.0029854
6: 0.0034592, 0.0051467, 0.0019528, 0.0052624, -0.0018032, 0.0031939
7: -0.0167857, -0.0125788, -0.0175878, -0.0118202, -0.0049655, 0.0050090
8: 0.0034122, 0.0067497, 0.0027759, 0.0075288, -0.0041166, 0.0039739
9: 0.0031874, 0.0098646, 0.0013947, 0.0095576, -0.0059076, 0.0072158

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040574, upper bound: 0.0040568
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040574, upper bound: 0.0040568
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041001, -0.0030089, -0.0041000, -0.0030012, -0.0010990, 0.0010911
1: -0.0062061, -0.0035797, -0.0062008, -0.0035717, -0.0026344, 0.0026211
2: 0.9665850, 0.9713597, 0.9665533, 0.9713632, -0.0047781, 0.0048063
3: 0.0177721, 0.0350596, 0.0178193, 0.0350857, -0.0096172, 0.0101210
4: -0.0033595, -0.0012182, -0.0033615, -0.0012043, -0.0021552, 0.0021433
5: 0.0130383, 0.0152038, 0.0130301, 0.0152002, -0.0021619, 0.0021737
6: 0.0033116, 0.0051580, 0.0032989, 0.0051590, -0.0018474, 0.0018591
7: -0.0168642, -0.0123840, -0.0168710, -0.0123963, -0.0044680, 0.0044870
8: 0.0033498, 0.0069042, 0.0033445, 0.0068945, -0.0035447, 0.0035597
9: 0.0030118, 0.0101425, 0.0029967, 0.0101251, -0.0056372, 0.0057380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040568, upper bound: 0.0040568
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040568, upper bound: 0.0040568
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040990, -0.0030223, -0.0040959, -0.0021753, -0.0019237, 0.0010736
1: -0.0061645, -0.0035933, -0.0060475, -0.0027288, -0.0034357, 0.0024542
2: 0.9666398, 0.9713536, 0.9631726, 0.9717391, -0.0050992, 0.0081810
3: 0.0181406, 0.0350147, 0.0191759, 0.0378575, -0.0124974, 0.0097728
4: -0.0033561, -0.0012420, -0.0035723, 0.0002698, -0.0036259, 0.0023303
5: 0.0130524, 0.0151755, 0.0121587, 0.0150959, -0.0020435, 0.0030168
6: 0.0033335, 0.0051563, 0.0019498, 0.0052626, -0.0019292, 0.0032066
7: -0.0168526, -0.0124795, -0.0175894, -0.0118161, -0.0050365, 0.0051098
8: 0.0033591, 0.0068285, 0.0027746, 0.0075322, -0.0041731, 0.0040539
9: 0.0030378, 0.0100063, 0.0013912, 0.0096234, -0.0060153, 0.0073133

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 58

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040568, upper bound: 0.0040568
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040568, upper bound: 0.0040568
time: 0.61 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.30 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045197, upper bound: 0.0045092
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045277, upper bound: 0.0045124
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045197, upper bound: 0.0045092
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045277, upper bound: 0.0045124
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0044886, upper bound: 0.0045072
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045124
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0044886, upper bound: 0.0045072
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045124
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045197, upper bound: 0.0045110
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045284, upper bound: 0.0045154
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045197, upper bound: 0.0045111
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045284, upper bound: 0.0045154
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0044886, upper bound: 0.0045089
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045154
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0044886, upper bound: 0.0045088
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045154
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045238, upper bound: 0.0045117
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045322, upper bound: 0.0045124
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045238, upper bound: 0.0045117
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045322, upper bound: 0.0045124
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045078, upper bound: 0.0045114
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045186, upper bound: 0.0045124
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045078, upper bound: 0.0045114
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045186, upper bound: 0.0045124
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0037730, upper bound: 0.0037721
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0037730, upper bound: 0.0037721
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0037730, upper bound: 0.0037721
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0037730, upper bound: 0.0037721
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0037728, upper bound: 0.0037718
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0037728, upper bound: 0.0037718
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0037728, upper bound: 0.0037718
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0037728, upper bound: 0.0037718
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045197, upper bound: 0.0045135
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045286, upper bound: 0.0045186
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045197, upper bound: 0.0045135
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045286, upper bound: 0.0045186
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0044886, upper bound: 0.0045117
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045186
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0044886, upper bound: 0.0045117
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045186
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045197, upper bound: 0.0045139
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045286, upper bound: 0.0045192
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045197, upper bound: 0.0045139
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045286, upper bound: 0.0045192
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0044886, upper bound: 0.0045119
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045192
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0044886, upper bound: 0.0045119
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045124, upper bound: 0.0045192
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0040495, upper bound: 0.0040465
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0040495, upper bound: 0.0040465
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0040495, upper bound: 0.0040465
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0040495, upper bound: 0.0040465
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0040465, upper bound: 0.0040465
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0040465, upper bound: 0.0040465
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0040465, upper bound: 0.0040465
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0040465, upper bound: 0.0040465
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0040508, upper bound: 0.0040527
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0040508, upper bound: 0.0040527
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0040508, upper bound: 0.0040527
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0040508, upper bound: 0.0040527
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0040465, upper bound: 0.0040521
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0040465, upper bound: 0.0040521
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0040465, upper bound: 0.0040521
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0040465, upper bound: 0.0040521
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045197, upper bound: 0.0045093
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045279, upper bound: 0.0045124
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045197, upper bound: 0.0045093
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045279, upper bound: 0.0045124
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0044917, upper bound: 0.0045075
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045154, upper bound: 0.0045124
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0044917, upper bound: 0.0045075
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045154, upper bound: 0.0045124
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0037748, upper bound: 0.0037748
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0037748, upper bound: 0.0037748
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0037748, upper bound: 0.0037748
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0037748, upper bound: 0.0037748
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0037748, upper bound: 0.0037748
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0037748, upper bound: 0.0037748
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0037748, upper bound: 0.0037748
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0037748, upper bound: 0.0037748
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045238, upper bound: 0.0045117
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045322, upper bound: 0.0045124
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045238, upper bound: 0.0045117
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045322, upper bound: 0.0045124
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045078, upper bound: 0.0045115
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045192, upper bound: 0.0045124
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045078, upper bound: 0.0045115
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0045192, upper bound: 0.0045124
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0038107, upper bound: 0.0038093
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0038107, upper bound: 0.0038093
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0038107, upper bound: 0.0038093
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0038107, upper bound: 0.0038093
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0038103, upper bound: 0.0038089
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0038103, upper bound: 0.0038089
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0038103, upper bound: 0.0038089
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0038103, upper bound: 0.0038089
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0037671, upper bound: 0.0037682
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0037671, upper bound: 0.0037682
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0037671, upper bound: 0.0037682
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0037671, upper bound: 0.0037682
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0037718, upper bound: 0.0037728
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0037718, upper bound: 0.0037728
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0037718, upper bound: 0.0037728
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0037718, upper bound: 0.0037728
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0038087, upper bound: 0.0038100
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0038087, upper bound: 0.0038100
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0038087, upper bound: 0.0038100
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0038087, upper bound: 0.0038100
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0038087, upper bound: 0.0038100
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0038087, upper bound: 0.0038100
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0038087, upper bound: 0.0038100
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0038087, upper bound: 0.0038100
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0040519, upper bound: 0.0040465
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0040519, upper bound: 0.0040465
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0040519, upper bound: 0.0040465
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0040519, upper bound: 0.0040465
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0040516, upper bound: 0.0040465
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0040516, upper bound: 0.0040465
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0040516, upper bound: 0.0040465
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0040516, upper bound: 0.0040465
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0040574, upper bound: 0.0040568
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0040574, upper bound: 0.0040568
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0040574, upper bound: 0.0040568
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0040574, upper bound: 0.0040568
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0040568, upper bound: 0.0040568
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0040568, upper bound: 0.0040568
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0040568, upper bound: 0.0040568
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 2, lower bound: -0.0040568, upper bound: 0.0040568

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040879, -0.0029459, -0.0040893, -0.0029767, -0.0011112, 0.0011433
1: -0.0057462, -0.0035153, -0.0057988, -0.0035468, -0.0021995, 0.0022835
2: 0.9663272, 0.9713883, 0.9664532, 0.9713744, -0.0050472, 0.0049351
3: 0.0218426, 0.0352711, 0.0213773, 0.0351678, -0.0075219, 0.0080597
4: -0.0033756, -0.0011057, -0.0033677, -0.0011606, -0.0022150, 0.0022621
5: 0.0129718, 0.0148909, 0.0130043, 0.0149267, -0.0019549, 0.0018866
6: 0.0032086, 0.0051659, 0.0032590, 0.0051621, -0.0019534, 0.0019069
7: -0.0169191, -0.0134389, -0.0168923, -0.0133183, -0.0036008, 0.0034534
8: 0.0033064, 0.0061172, 0.0033276, 0.0061630, -0.0028566, 0.0027896
9: 0.0028893, 0.0086373, 0.0029491, 0.0088093, -0.0049939, 0.0047738

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045197, upper bound: 0.0045197
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045197, upper bound: 0.0045276
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040899, -0.0029675, -0.0040901, -0.0029595, -0.0011304, 0.0011226
1: -0.0058215, -0.0035373, -0.0058297, -0.0035292, -0.0022923, 0.0022924
2: 0.9664153, 0.9713786, 0.9663826, 0.9713822, -0.0049669, 0.0049959
3: 0.0211767, 0.0351988, 0.0211036, 0.0352255, -0.0078822, 0.0082610
4: -0.0033701, -0.0011441, -0.0033721, -0.0011299, -0.0022402, 0.0022280
5: 0.0129945, 0.0149421, 0.0129861, 0.0149477, -0.0019532, 0.0019560
6: 0.0032438, 0.0051632, 0.0032308, 0.0051642, -0.0019204, 0.0019324
7: -0.0169003, -0.0132664, -0.0169073, -0.0132474, -0.0036529, 0.0036409
8: 0.0033212, 0.0062042, 0.0033157, 0.0062193, -0.0028980, 0.0028885
9: 0.0029312, 0.0088835, 0.0029157, 0.0089106, -0.0049723, 0.0050299

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045276, upper bound: 0.0045197
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045276, upper bound: 0.0045286
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040879, -0.0029459, -0.0040902, -0.0029160, -0.0011718, 0.0011443
1: -0.0057462, -0.0035153, -0.0058327, -0.0034849, -0.0022614, 0.0023173
2: 0.9663272, 0.9713883, 0.9662048, 0.9714020, -0.0050748, 0.0051835
3: 0.0218426, 0.0352711, 0.0210776, 0.0353714, -0.0077914, 0.0084717
4: -0.0033756, -0.0011057, -0.0033832, -0.0010524, -0.0023233, 0.0022776
5: 0.0129718, 0.0148909, 0.0129403, 0.0149497, -0.0019779, 0.0019507
6: 0.0032086, 0.0051659, 0.0031599, 0.0051697, -0.0019610, 0.0020061
7: -0.0169191, -0.0134389, -0.0169451, -0.0132407, -0.0036784, 0.0035061
8: 0.0033064, 0.0061172, 0.0032857, 0.0062246, -0.0029182, 0.0028315
9: 0.0028893, 0.0086373, 0.0028312, 0.0089201, -0.0050540, 0.0048914

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045150, upper bound: 0.0044886
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045150, upper bound: 0.0045092
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040899, -0.0029675, -0.0040910, -0.0028988, -0.0011910, 0.0011235
1: -0.0058215, -0.0035373, -0.0058632, -0.0034673, -0.0023542, 0.0023259
2: 0.9664153, 0.9713786, 0.9661344, 0.9714097, -0.0049944, 0.0052442
3: 0.0211767, 0.0351988, 0.0208075, 0.0354292, -0.0081492, 0.0087048
4: -0.0033701, -0.0011441, -0.0033876, -0.0010216, -0.0023485, 0.0022435
5: 0.0129945, 0.0149421, 0.0129221, 0.0149705, -0.0019760, 0.0020200
6: 0.0032438, 0.0051632, 0.0031317, 0.0051718, -0.0019280, 0.0020315
7: -0.0169003, -0.0132664, -0.0169600, -0.0131707, -0.0037297, 0.0036937
8: 0.0033212, 0.0062042, 0.0032739, 0.0062802, -0.0029589, 0.0029304
9: 0.0029312, 0.0088835, 0.0027977, 0.0090201, -0.0050640, 0.0051351

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045185, upper bound: 0.0044886
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045185, upper bound: 0.0045124
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040888, -0.0028844, -0.0040893, -0.0029767, -0.0011121, 0.0012049
1: -0.0057820, -0.0034525, -0.0057988, -0.0035468, -0.0022353, 0.0023463
2: 0.9660752, 0.9714164, 0.9664532, 0.9713744, -0.0052992, 0.0049632
3: 0.0215258, 0.0354777, 0.0213773, 0.0351678, -0.0079418, 0.0083153
4: -0.0033913, -0.0009958, -0.0033677, -0.0011606, -0.0022307, 0.0023719
5: 0.0129068, 0.0149153, 0.0130043, 0.0149267, -0.0020198, 0.0019110
6: 0.0031081, 0.0051736, 0.0032590, 0.0051621, -0.0020540, 0.0019147
7: -0.0169726, -0.0133568, -0.0168923, -0.0133183, -0.0036543, 0.0035355
8: 0.0032639, 0.0062302, 0.0033276, 0.0061630, -0.0028991, 0.0029026
9: 0.0027696, 0.0087544, 0.0029491, 0.0088093, -0.0050987, 0.0048463

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0044886, upper bound: 0.0045150
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0044886, upper bound: 0.0045185
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040908, -0.0029076, -0.0040901, -0.0029595, -0.0011312, 0.0011825
1: -0.0058546, -0.0034762, -0.0058297, -0.0035292, -0.0023253, 0.0023535
2: 0.9661704, 0.9714058, 0.9663826, 0.9713822, -0.0052118, 0.0050232
3: 0.0208839, 0.0353997, 0.0211036, 0.0352255, -0.0082808, 0.0085289
4: -0.0033854, -0.0010373, -0.0033721, -0.0011299, -0.0022555, 0.0023348
5: 0.0129314, 0.0149646, 0.0129861, 0.0149477, -0.0020164, 0.0019785
6: 0.0031461, 0.0051707, 0.0032308, 0.0051642, -0.0020181, 0.0019399
7: -0.0169524, -0.0131905, -0.0169073, -0.0132474, -0.0037050, 0.0037168
8: 0.0032799, 0.0062644, 0.0033157, 0.0062193, -0.0029393, 0.0029487
9: 0.0028148, 0.0089918, 0.0029157, 0.0089106, -0.0050846, 0.0051090

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045091, upper bound: 0.0045197
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045091, upper bound: 0.0045277
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040888, -0.0028844, -0.0040902, -0.0029160, -0.0011728, 0.0012058
1: -0.0057820, -0.0034525, -0.0058327, -0.0034849, -0.0022972, 0.0023801
2: 0.9660752, 0.9714164, 0.9662048, 0.9714020, -0.0053267, 0.0052115
3: 0.0215258, 0.0354777, 0.0210776, 0.0353714, -0.0076747, 0.0081732
4: -0.0033913, -0.0009958, -0.0033832, -0.0010524, -0.0023390, 0.0023874
5: 0.0129068, 0.0149153, 0.0129403, 0.0149497, -0.0020429, 0.0019750
6: 0.0031081, 0.0051736, 0.0031599, 0.0051697, -0.0020616, 0.0020138
7: -0.0169726, -0.0133568, -0.0169451, -0.0132407, -0.0037319, 0.0035882
8: 0.0032639, 0.0062302, 0.0032857, 0.0062246, -0.0029607, 0.0029445
9: 0.0027696, 0.0087544, 0.0028312, 0.0089201, -0.0052126, 0.0049768

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0044886, upper bound: 0.0044886
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0044886, upper bound: 0.0045072
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040908, -0.0029076, -0.0040910, -0.0028988, -0.0011919, 0.0011834
1: -0.0058546, -0.0034762, -0.0058632, -0.0034673, -0.0023873, 0.0023870
2: 0.9661704, 0.9714058, 0.9661344, 0.9714097, -0.0052394, 0.0052714
3: 0.0208839, 0.0353997, 0.0208075, 0.0354292, -0.0079880, 0.0084114
4: -0.0033854, -0.0010373, -0.0033876, -0.0010216, -0.0023638, 0.0023503
5: 0.0129314, 0.0149646, 0.0129221, 0.0149705, -0.0020391, 0.0020425
6: 0.0031461, 0.0051707, 0.0031317, 0.0051718, -0.0020258, 0.0020390
7: -0.0169524, -0.0131905, -0.0169600, -0.0131707, -0.0037817, 0.0037696
8: 0.0032799, 0.0062644, 0.0032739, 0.0062802, -0.0030002, 0.0029906
9: 0.0028148, 0.0089918, 0.0027977, 0.0090201, -0.0052019, 0.0052900

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045072, upper bound: 0.0044886
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045072, upper bound: 0.0045124
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040879, -0.0029459, -0.0040919, -0.0029853, -0.0011026, 0.0011460
1: -0.0057462, -0.0035153, -0.0058979, -0.0035555, -0.0021907, 0.0023825
2: 0.9663272, 0.9713883, 0.9664884, 0.9713704, -0.0050432, 0.0048999
3: 0.0218426, 0.0352711, 0.0205004, 0.0351390, -0.0078265, 0.0093772
4: -0.0033756, -0.0011057, -0.0033656, -0.0011759, -0.0021997, 0.0022599
5: 0.0129718, 0.0148909, 0.0130133, 0.0149941, -0.0020223, 0.0018776
6: 0.0032086, 0.0051659, 0.0032730, 0.0051610, -0.0019523, 0.0018929
7: -0.0169191, -0.0134389, -0.0168848, -0.0130911, -0.0038280, 0.0034459
8: 0.0033064, 0.0061172, 0.0033335, 0.0063433, -0.0030369, 0.0027837
9: 0.0028893, 0.0086373, 0.0029658, 0.0091336, -0.0053468, 0.0050564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045197, upper bound: 0.0045197
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045197, upper bound: 0.0045285
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040899, -0.0029675, -0.0040927, -0.0029688, -0.0011211, 0.0011253
1: -0.0058215, -0.0035373, -0.0059286, -0.0035387, -0.0022828, 0.0023913
2: 0.9664153, 0.9713786, 0.9664209, 0.9713779, -0.0049626, 0.0049577
3: 0.0211767, 0.0351988, 0.0202286, 0.0351943, -0.0081942, 0.0095592
4: -0.0033701, -0.0011441, -0.0033698, -0.0011465, -0.0022236, 0.0022256
5: 0.0129945, 0.0149421, 0.0129959, 0.0150150, -0.0020205, 0.0019462
6: 0.0032438, 0.0051632, 0.0032460, 0.0051630, -0.0019192, 0.0019172
7: -0.0169003, -0.0132664, -0.0168992, -0.0130206, -0.0038797, 0.0036328
8: 0.0033212, 0.0062042, 0.0033221, 0.0063992, -0.0030779, 0.0028821
9: 0.0029312, 0.0088835, 0.0029338, 0.0092341, -0.0053337, 0.0052771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045276, upper bound: 0.0045197
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045276, upper bound: 0.0045297
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040879, -0.0029459, -0.0040929, -0.0029134, -0.0011745, 0.0011470
1: -0.0057462, -0.0035153, -0.0059354, -0.0034821, -0.0022641, 0.0024201
2: 0.9663272, 0.9713883, 0.9661940, 0.9714031, -0.0050759, 0.0051943
3: 0.0218426, 0.0352711, 0.0201683, 0.0353803, -0.0081552, 0.0098511
4: -0.0033756, -0.0011057, -0.0033839, -0.0010476, -0.0023280, 0.0022782
5: 0.0129718, 0.0148909, 0.0129375, 0.0150196, -0.0020478, 0.0019534
6: 0.0032086, 0.0051659, 0.0031555, 0.0051700, -0.0019613, 0.0020104
7: -0.0169191, -0.0134389, -0.0169474, -0.0130050, -0.0039140, 0.0035084
8: 0.0033064, 0.0061172, 0.0032839, 0.0064116, -0.0031052, 0.0028333
9: 0.0028893, 0.0086373, 0.0028261, 0.0092564, -0.0054457, 0.0051748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045170, upper bound: 0.0044917
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045170, upper bound: 0.0045110
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040899, -0.0029675, -0.0040937, -0.0028970, -0.0011929, 0.0011263
1: -0.0058215, -0.0035373, -0.0059668, -0.0034654, -0.0023561, 0.0024295
2: 0.9664153, 0.9713786, 0.9661268, 0.9714106, -0.0049953, 0.0052518
3: 0.0211767, 0.0351988, 0.0198902, 0.0354354, -0.0085335, 0.0100601
4: -0.0033701, -0.0011441, -0.0033881, -0.0010183, -0.0023518, 0.0022440
5: 0.0129945, 0.0149421, 0.0129202, 0.0150410, -0.0020465, 0.0020219
6: 0.0032438, 0.0051632, 0.0031287, 0.0051721, -0.0019282, 0.0020345
7: -0.0169003, -0.0132664, -0.0169616, -0.0129329, -0.0039674, 0.0036953
8: 0.0033212, 0.0062042, 0.0032726, 0.0064687, -0.0031475, 0.0029316
9: 0.0029312, 0.0088835, 0.0027941, 0.0093593, -0.0054599, 0.0054052

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045202, upper bound: 0.0044917
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045202, upper bound: 0.0045154
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040888, -0.0028844, -0.0040919, -0.0029853, -0.0011035, 0.0012075
1: -0.0057820, -0.0034525, -0.0058979, -0.0035555, -0.0022265, 0.0024454
2: 0.9660752, 0.9714164, 0.9664884, 0.9713704, -0.0052952, 0.0049280
3: 0.0215258, 0.0354777, 0.0205004, 0.0351390, -0.0082464, 0.0096328
4: -0.0033913, -0.0009958, -0.0033656, -0.0011759, -0.0022154, 0.0023698
5: 0.0129068, 0.0149153, 0.0130133, 0.0149941, -0.0020872, 0.0019019
6: 0.0031081, 0.0051736, 0.0032730, 0.0051610, -0.0020529, 0.0019007
7: -0.0169726, -0.0133568, -0.0168848, -0.0130911, -0.0038815, 0.0035280
8: 0.0032639, 0.0062302, 0.0033335, 0.0063433, -0.0030794, 0.0028967
9: 0.0027696, 0.0087544, 0.0029658, 0.0091336, -0.0054515, 0.0051289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0044886, upper bound: 0.0045150
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0044886, upper bound: 0.0045185
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040908, -0.0029076, -0.0040927, -0.0029688, -0.0011220, 0.0011851
1: -0.0058546, -0.0034762, -0.0059286, -0.0035387, -0.0023159, 0.0024524
2: 0.9661704, 0.9714058, 0.9664209, 0.9713779, -0.0052075, 0.0049849
3: 0.0208839, 0.0353997, 0.0202286, 0.0351943, -0.0085961, 0.0098271
4: -0.0033854, -0.0010373, -0.0033698, -0.0011465, -0.0022389, 0.0023325
5: 0.0129314, 0.0149646, 0.0129959, 0.0150150, -0.0020836, 0.0019687
6: 0.0031461, 0.0051707, 0.0032460, 0.0051630, -0.0020170, 0.0019247
7: -0.0169524, -0.0131905, -0.0168992, -0.0130206, -0.0039317, 0.0037087
8: 0.0032799, 0.0062644, 0.0033221, 0.0063992, -0.0031192, 0.0029423
9: 0.0028148, 0.0089918, 0.0029338, 0.0092341, -0.0054461, 0.0053526

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045093, upper bound: 0.0045197
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045093, upper bound: 0.0045280
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040888, -0.0028844, -0.0040929, -0.0029134, -0.0011754, 0.0012085
1: -0.0057820, -0.0034525, -0.0059354, -0.0034821, -0.0022999, 0.0024829
2: 0.9660752, 0.9714164, 0.9661940, 0.9714031, -0.0053279, 0.0052224
3: 0.0215258, 0.0354777, 0.0201683, 0.0353803, -0.0079753, 0.0094907
4: -0.0033913, -0.0009958, -0.0033839, -0.0010476, -0.0023437, 0.0023881
5: 0.0129068, 0.0149153, 0.0129375, 0.0150196, -0.0021128, 0.0019778
6: 0.0031081, 0.0051736, 0.0031555, 0.0051700, -0.0020619, 0.0020181
7: -0.0169726, -0.0133568, -0.0169474, -0.0130050, -0.0039676, 0.0035905
8: 0.0032639, 0.0062302, 0.0032839, 0.0064116, -0.0031477, 0.0029463
9: 0.0027696, 0.0087544, 0.0028261, 0.0092564, -0.0055044, 0.0052128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0044886, upper bound: 0.0044917
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0044886, upper bound: 0.0045089
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040908, -0.0029076, -0.0040937, -0.0028970, -0.0011938, 0.0011861
1: -0.0058546, -0.0034762, -0.0059668, -0.0034654, -0.0023892, 0.0024906
2: 0.9661704, 0.9714058, 0.9661268, 0.9714106, -0.0052403, 0.0052790
3: 0.0208839, 0.0353997, 0.0198902, 0.0354354, -0.0082974, 0.0097074
4: -0.0033854, -0.0010373, -0.0033881, -0.0010183, -0.0023671, 0.0023508
5: 0.0129314, 0.0149646, 0.0129202, 0.0150410, -0.0021096, 0.0020445
6: 0.0031461, 0.0051707, 0.0031287, 0.0051721, -0.0020260, 0.0020420
7: -0.0169524, -0.0131905, -0.0169616, -0.0129329, -0.0040194, 0.0037712
8: 0.0032799, 0.0062644, 0.0032726, 0.0064687, -0.0031888, 0.0029919
9: 0.0028148, 0.0089918, 0.0027941, 0.0093593, -0.0055104, 0.0054742

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045075, upper bound: 0.0044917
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045075, upper bound: 0.0045154
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040944, -0.0030316, -0.0040893, -0.0029767, -0.0011177, 0.0010577
1: -0.0059912, -0.0036028, -0.0057988, -0.0035468, -0.0024444, 0.0021960
2: 0.9666778, 0.9713494, 0.9664532, 0.9713744, -0.0046966, 0.0048962
3: 0.0196747, 0.0349836, 0.0213773, 0.0351678, -0.0100735, 0.0081684
4: -0.0033537, -0.0012586, -0.0033677, -0.0011606, -0.0021931, 0.0021092
5: 0.0130622, 0.0150576, 0.0130043, 0.0149267, -0.0018645, 0.0020533
6: 0.0033486, 0.0051552, 0.0032590, 0.0051621, -0.0018134, 0.0018962
7: -0.0168446, -0.0128771, -0.0168923, -0.0133183, -0.0035262, 0.0040152
8: 0.0033655, 0.0065131, 0.0033276, 0.0061630, -0.0027975, 0.0031855
9: 0.0030558, 0.0094389, 0.0029491, 0.0088093, -0.0050618, 0.0053461

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045272, upper bound: 0.0045197
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0045272, upper bound: 0.0045286
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040963, -0.0030679, -0.0040901, -0.0029595, -0.0011368, 0.0010222
1: -0.0060622, -0.0036398, -0.0058297, -0.0035292, -0.0025330, 0.0021899
2: 0.9668264, 0.9713329, 0.9663826, 0.9713822, -0.0045558, 0.0049503
3: 0.0190462, 0.0348618, 0.0211036, 0.0352255, -0.0104237, 0.0083455
4: -0.0033445, -0.0013233, -0.0033721, -0.0011299, -0.0022146, 0.0020488
5: 0.0131005, 0.0151059, 0.0129861, 0.0149477, -0.0018473, 0.0021198
6: 0.0034079, 0.0051506, 0.0032308, 0.0051642, -0.0017563, 0.0019198
7: -0.0168130, -0.0127142, -0.0169073, -0.0132474, -0.0035656, 0.0041930
8: 0.0033905, 0.0066423, 0.0033157, 0.0062193, -0.0028288, 0.0033265
9: 0.0031264, 0.0096714, 0.0029157, 0.0089106, -0.0050433, 0.0055995

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040097, upper bound: 0.0040475
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.23 + 598.74 = 601.97 seconds
