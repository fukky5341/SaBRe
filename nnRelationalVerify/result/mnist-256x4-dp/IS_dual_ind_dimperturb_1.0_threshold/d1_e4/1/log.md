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
Threshold: 0.0302838


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0107326, 0.0107326)
1: (0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176)
2: (0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544)
3: (-0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004)
4: (-0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596)
5: (0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762)
6: (-0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913)
7: (-0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638)
8: (0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195)
9: (-0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.31 + 2.87 = 4.18 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0356280, upper bound: 0.0356280

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0337967, upper bound: 0.0342015
time: 2.07 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0343184, upper bound: 0.0343184
time: 2.13 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 4.33 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 4.33
Output dim: 8, lower bound: -0.0337967, upper bound: 0.0342015
IS_A2, status: Status.UNKNOWN, split count: 1, time: 4.33
Output dim: 8, lower bound: -0.0343184, upper bound: 0.0343184

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0001082, 0.0065945, -0.0013616, 0.0066181, -0.0063704, 0.0076178
1: 0.0011559, 0.0022750, 0.0009344, 0.0036890, -0.0025331, 0.0013406
2: 0.0107139, 0.0146090, 0.0098678, 0.0149406, -0.0042267, 0.0047412
3: -0.0035996, 0.0003584, -0.0036129, 0.0020388, -0.0056384, 0.0039714
4: -0.0042892, -0.0001402, -0.0078531, -0.0001248, -0.0041645, 0.0077129
5: 0.0042460, 0.0082959, 0.0037320, 0.0085449, -0.0042989, 0.0045640
6: -0.0054535, 0.0101254, -0.0055089, 0.0209459, -0.0263994, 0.0156343
7: -0.0164091, 0.0048705, -0.0169494, 0.0051979, -0.0216069, 0.0218199
8: 0.9767771, 0.9926447, 0.9684792, 0.9926965, -0.0159194, 0.0241656
9: -0.0092107, 0.0044784, -0.0105981, 0.0049334, -0.0141441, 0.0150765

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0318411, upper bound: 0.0311607
time: 6.93 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0330508, upper bound: 0.0334282
time: 2.21 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0000700, 0.0067379, -0.0018050, 0.0066146, -0.0063589, 0.0083103
1: 0.0012143, 0.0022957, 0.0010275, 0.0045713, -0.0033570, 0.0012682
2: 0.0106347, 0.0145214, 0.0092207, 0.0148012, -0.0041665, 0.0053007
3: -0.0036816, 0.0002926, -0.0036109, 0.0026529, -0.0063345, 0.0039035
4: -0.0042656, -0.0000514, -0.0095484, -0.0001268, -0.0041388, 0.0094969
5: 0.0041620, 0.0082302, 0.0033920, 0.0084402, -0.0042782, 0.0048383
6: -0.0057867, 0.0100368, -0.0055010, 0.0260570, -0.0318437, 0.0155377
7: -0.0162664, 0.0053243, -0.0167223, 0.0052996, -0.0215660, 0.0220466
8: 0.9771857, 0.9929643, 0.9657608, 0.9926889, -0.0155032, 0.0272036
9: -0.0095008, 0.0043583, -0.0114130, 0.0047421, -0.0142430, 0.0157712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0323418, upper bound: 0.0312518
time: 1.95 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0335439, upper bound: 0.0335439
time: 1.72 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.95 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.95
Output dim: 8, lower bound: -0.0318411, upper bound: 0.0311607
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.95
Output dim: 8, lower bound: -0.0330508, upper bound: 0.0334282
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.95
Output dim: 8, lower bound: -0.0323418, upper bound: 0.0312518
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.95
Output dim: 8, lower bound: -0.0335439, upper bound: 0.0335439

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0000239, 0.0065776, 0.0000459, 0.0065563, -0.0059885, 0.0060290
1: 0.0013258, 0.0022726, 0.0013289, 0.0022695, -0.0008652, 0.0008710
2: 0.0107233, 0.0143466, 0.0107350, 0.0143345, -0.0033333, 0.0033109
3: -0.0035900, 0.0001575, -0.0035778, 0.0001450, -0.0034474, 0.0034243
4: -0.0042075, -0.0001506, -0.0041939, -0.0001638, -0.0037070, 0.0037320
5: 0.0042559, 0.0080950, 0.0042684, 0.0080822, -0.0035318, 0.0035080
6: -0.0054142, 0.0098184, -0.0053647, 0.0097674, -0.0140130, 0.0139188
7: -0.0159286, 0.0048170, -0.0158590, 0.0047495, -0.0189562, 0.0190845
8: 0.9779934, 0.9926071, 0.9780424, 0.9925595, -0.0133532, 0.0134435
9: -0.0091765, 0.0040888, -0.0091333, 0.0040443, -0.0122032, 0.0121211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303511, upper bound: 0.0301393
time: 1.82 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309976, upper bound: 0.0302871
time: 1.96 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0000490, 0.0065879, -0.0001037, 0.0066017, -0.0062677, 0.0062982
1: 0.0012465, 0.0022741, 0.0011628, 0.0022761, -0.0010295, 0.0011113
2: 0.0107176, 0.0144732, 0.0107100, 0.0145986, -0.0038810, 0.0037632
3: -0.0035958, 0.0002563, -0.0036037, 0.0003507, -0.0039465, 0.0038600
4: -0.0042526, -0.0001443, -0.0042864, -0.0001357, -0.0041169, 0.0041422
5: 0.0042499, 0.0081940, 0.0042418, 0.0082882, -0.0040383, 0.0039522
6: -0.0054381, 0.0099879, -0.0054701, 0.0101149, -0.0155530, 0.0154580
7: -0.0161878, 0.0048495, -0.0163922, 0.0048931, -0.0210809, 0.0212417
8: 0.9774110, 0.9926299, 0.9768254, 0.9926607, -0.0152497, 0.0158046
9: -0.0091972, 0.0042921, -0.0092251, 0.0044642, -0.0136614, 0.0135172

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0315281, upper bound: 0.0323795
time: 1.90 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0322231, upper bound: 0.0326010
time: 1.78 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.0000622, 0.0067210, 0.0001062, 0.0065528, -0.0059978, 0.0063155
1: 0.0013313, 0.0022933, 0.0013376, 0.0022690, -0.0008665, 0.0009124
2: 0.0106440, 0.0143254, 0.0107370, 0.0143011, -0.0034917, 0.0033160
3: -0.0036720, 0.0001356, -0.0035757, 0.0001105, -0.0036113, 0.0034296
4: -0.0041837, -0.0000618, -0.0041565, -0.0001660, -0.0037127, 0.0039094
5: 0.0041719, 0.0080726, 0.0042705, 0.0080468, -0.0036996, 0.0035135
6: -0.0057475, 0.0097293, -0.0053564, 0.0096271, -0.0146790, 0.0139406
7: -0.0158072, 0.0052709, -0.0156680, 0.0047382, -0.0189859, 0.0199915
8: 0.9780790, 0.9929268, 0.9781770, 0.9925516, -0.0133740, 0.0140824
9: -0.0094667, 0.0040112, -0.0091261, 0.0039222, -0.0127831, 0.0121401

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0308337, upper bound: 0.0302462
time: 2.30 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0314845, upper bound: 0.0303557
time: 1.96 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0000129, 0.0067312, -0.0000486, 0.0065982, -0.0062553, 0.0064623
1: 0.0013018, 0.0022948, 0.0012472, 0.0022756, -0.0009738, 0.0010475
2: 0.0106384, 0.0143904, 0.0107119, 0.0144722, -0.0038338, 0.0036785
3: -0.0036777, 0.0001941, -0.0036017, 0.0002556, -0.0039333, 0.0037958
4: -0.0042303, -0.0000556, -0.0042523, -0.0001379, -0.0040924, 0.0041968
5: 0.0041660, 0.0081319, 0.0042438, 0.0081932, -0.0040273, 0.0038881
6: -0.0057711, 0.0099041, -0.0054621, 0.0099869, -0.0157579, 0.0153662
7: -0.0160529, 0.0053030, -0.0161861, 0.0048822, -0.0209351, 0.0214891
8: 0.9777973, 0.9929494, 0.9774157, 0.9926530, -0.0148557, 0.0155337
9: -0.0094872, 0.0041785, -0.0092181, 0.0042907, -0.0137779, 0.0133966

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0319725, upper bound: 0.0325017
time: 2.31 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0326952, upper bound: 0.0326953
time: 2.06 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 5.66 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.66
Output dim: 8, lower bound: -0.0303511, upper bound: 0.0301393
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.66
Output dim: 8, lower bound: -0.0309976, upper bound: 0.0302871
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.66
Output dim: 8, lower bound: -0.0315281, upper bound: 0.0323795
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.66
Output dim: 8, lower bound: -0.0322231, upper bound: 0.0326010
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.66
Output dim: 8, lower bound: -0.0308337, upper bound: 0.0302462
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.66
Output dim: 8, lower bound: -0.0314845, upper bound: 0.0303557
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.66
Output dim: 8, lower bound: -0.0319725, upper bound: 0.0325017
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.66
Output dim: 8, lower bound: -0.0326952, upper bound: 0.0326953

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0002836, 0.0065426, 0.0001557, 0.0065420, -0.0056999, 0.0058725
1: 0.0013633, 0.0022675, 0.0013448, 0.0022674, -0.0008235, 0.0008484
2: 0.0107426, 0.0142031, 0.0107430, 0.0142738, -0.0032467, 0.0031513
3: -0.0035699, 0.0000090, -0.0035696, 0.0000821, -0.0033579, 0.0032592
4: -0.0040467, -0.0001723, -0.0041259, -0.0001727, -0.0035283, 0.0036352
5: 0.0042764, 0.0079429, 0.0042768, 0.0080178, -0.0034401, 0.0033390
6: -0.0053329, 0.0092148, -0.0053313, 0.0095121, -0.0136493, 0.0132481
7: -0.0151065, 0.0047062, -0.0155113, 0.0047041, -0.0180427, 0.0185891
8: 0.9785725, 0.9925290, 0.9782873, 0.9925275, -0.0127097, 0.0130946
9: -0.0091056, 0.0035632, -0.0091043, 0.0038220, -0.0118864, 0.0115370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303511, upper bound: 0.0297311
time: 2.03 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303511, upper bound: 0.0301393
time: 1.93 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0002178, 0.0066600, 0.0001403, 0.0065435, -0.0057583, 0.0060589
1: 0.0013538, 0.0022845, 0.0013426, 0.0022676, -0.0008319, 0.0008753
2: 0.0106777, 0.0142394, 0.0107422, 0.0142823, -0.0033498, 0.0031836
3: -0.0036371, 0.0000466, -0.0035704, 0.0000910, -0.0034645, 0.0032927
4: -0.0040874, -0.0000996, -0.0041354, -0.0001718, -0.0035645, 0.0037505
5: 0.0042076, 0.0079814, 0.0042759, 0.0080269, -0.0035493, 0.0033732
6: -0.0056057, 0.0093677, -0.0053348, 0.0095479, -0.0140825, 0.0133840
7: -0.0153147, 0.0050777, -0.0155601, 0.0047088, -0.0182278, 0.0191791
8: 0.9784259, 0.9927907, 0.9782531, 0.9925309, -0.0128400, 0.0135102
9: -0.0093432, 0.0036963, -0.0091073, 0.0038532, -0.0122637, 0.0116554

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309976, upper bound: 0.0299010
time: 2.14 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309976, upper bound: 0.0302871
time: 1.94 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0002091, 0.0065533, 0.0000062, 0.0065877, -0.0058312, 0.0059859
1: 0.0013525, 0.0022691, 0.0013232, 0.0022740, -0.0008424, 0.0008648
2: 0.0107367, 0.0142442, 0.0107177, 0.0143564, -0.0033094, 0.0032239
3: -0.0035760, 0.0000516, -0.0035957, 0.0001676, -0.0034228, 0.0033343
4: -0.0040928, -0.0001657, -0.0042184, -0.0001444, -0.0036096, 0.0037054
5: 0.0042701, 0.0079865, 0.0042500, 0.0081054, -0.0035065, 0.0034159
6: -0.0053576, 0.0093879, -0.0054377, 0.0098595, -0.0139129, 0.0135533
7: -0.0153422, 0.0047399, -0.0159845, 0.0048489, -0.0184584, 0.0189481
8: 0.9784065, 0.9925528, 0.9779540, 0.9926296, -0.0130025, 0.0133475
9: -0.0091272, 0.0037139, -0.0091969, 0.0041246, -0.0121159, 0.0118028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0315281, upper bound: 0.0319951
time: 1.88 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0315281, upper bound: 0.0323795
time: 1.96 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0001429, 0.0066707, -0.0000089, 0.0065889, -0.0060555, 0.0061592
1: 0.0013429, 0.0022860, 0.0013080, 0.0022742, -0.0008512, 0.0009781
2: 0.0106718, 0.0142808, 0.0107170, 0.0143812, -0.0037094, 0.0032575
3: -0.0036432, 0.0000895, -0.0035964, 0.0001871, -0.0038303, 0.0033691
4: -0.0041338, -0.0000930, -0.0042278, -0.0001436, -0.0039902, 0.0038126
5: 0.0042013, 0.0080253, 0.0042493, 0.0081250, -0.0039236, 0.0034515
6: -0.0056306, 0.0095418, -0.0054405, 0.0098947, -0.0143156, 0.0149822
7: -0.0155518, 0.0051117, -0.0160379, 0.0048527, -0.0186507, 0.0211496
8: 0.9782588, 0.9928147, 0.9778404, 0.9926323, -0.0131380, 0.0149742
9: -0.0093649, 0.0038479, -0.0091993, 0.0041659, -0.0135308, 0.0119258

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0322231, upper bound: 0.0322208
time: 2.01 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0322231, upper bound: 0.0326010
time: 1.61 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0003245, 0.0066848, 0.0002168, 0.0065380, -0.0057121, 0.0061625
1: 0.0013692, 0.0022881, 0.0013536, 0.0022669, -0.0008252, 0.0008903
2: 0.0106640, 0.0141805, 0.0107452, 0.0142400, -0.0034071, 0.0031581
3: -0.0036513, -0.0000143, -0.0035673, 0.0000472, -0.0035238, 0.0032662
4: -0.0040214, -0.0000843, -0.0040881, -0.0001751, -0.0035359, 0.0038147
5: 0.0041931, 0.0079190, 0.0042791, 0.0079820, -0.0036100, 0.0033461
6: -0.0056634, 0.0091198, -0.0053221, 0.0093700, -0.0143234, 0.0132765
7: -0.0149771, 0.0051564, -0.0153179, 0.0046916, -0.0180814, 0.0195073
8: 0.9786637, 0.9928461, 0.9784237, 0.9925187, -0.0127369, 0.0137413
9: -0.0093935, 0.0034804, -0.0090963, 0.0036983, -0.0124735, 0.0115617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0306818, upper bound: 0.0300910
time: 7.13 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0306634, upper bound: 0.0300910
time: 1.87 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0002544, 0.0068173, 0.0001985, 0.0065400, -0.0057695, 0.0063527
1: 0.0013591, 0.0023072, 0.0013510, 0.0022671, -0.0008335, 0.0009178
2: 0.0105907, 0.0142192, 0.0107441, 0.0142501, -0.0035122, 0.0031898
3: -0.0037270, 0.0000257, -0.0035685, 0.0000577, -0.0036325, 0.0032991
4: -0.0040648, -0.0000022, -0.0040994, -0.0001739, -0.0035714, 0.0039324
5: 0.0041155, 0.0079600, 0.0042779, 0.0079928, -0.0037214, 0.0033798
6: -0.0059713, 0.0092827, -0.0053268, 0.0094127, -0.0147654, 0.0134100
7: -0.0151990, 0.0055757, -0.0153759, 0.0046980, -0.0182633, 0.0201092
8: 0.9785073, 0.9931415, 0.9783827, 0.9925233, -0.0128650, 0.0141653
9: -0.0096616, 0.0036223, -0.0091004, 0.0037354, -0.0128583, 0.0116780

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0313405, upper bound: 0.0301873
time: 1.96 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0313102, upper bound: 0.0301871
time: 1.78 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0002463, 0.0066953, 0.0000617, 0.0065839, -0.0058448, 0.0062586
1: 0.0013579, 0.0022896, 0.0013312, 0.0022735, -0.0008444, 0.0009042
2: 0.0106582, 0.0142237, 0.0107198, 0.0143258, -0.0034602, 0.0032315
3: -0.0036572, 0.0000303, -0.0035935, 0.0001359, -0.0035787, 0.0033421
4: -0.0040698, -0.0000778, -0.0041841, -0.0001468, -0.0036180, 0.0038742
5: 0.0041870, 0.0079647, 0.0042522, 0.0080729, -0.0036663, 0.0034239
6: -0.0056877, 0.0093014, -0.0054287, 0.0097306, -0.0145466, 0.0135850
7: -0.0152244, 0.0051894, -0.0158090, 0.0048367, -0.0185016, 0.0198113
8: 0.9784895, 0.9928694, 0.9780777, 0.9926209, -0.0130329, 0.0139555
9: -0.0094146, 0.0036386, -0.0091891, 0.0040123, -0.0126679, 0.0118304

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0318163, upper bound: 0.0323401
time: 2.26 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0317973, upper bound: 0.0323401
time: 2.16 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0001821, 0.0068280, 0.0000448, 0.0065855, -0.0059040, 0.0064495
1: 0.0013486, 0.0023087, 0.0013288, 0.0022737, -0.0008530, 0.0009318
2: 0.0105848, 0.0142592, 0.0107189, 0.0143351, -0.0035658, 0.0032642
3: -0.0037331, 0.0000671, -0.0035945, 0.0001456, -0.0036879, 0.0033760
4: -0.0041096, 0.0000044, -0.0041946, -0.0001457, -0.0036547, 0.0039924
5: 0.0041092, 0.0080024, 0.0042513, 0.0080828, -0.0037781, 0.0034586
6: -0.0059961, 0.0094508, -0.0054326, 0.0097700, -0.0149905, 0.0137226
7: -0.0154279, 0.0056095, -0.0158626, 0.0048420, -0.0186889, 0.0204157
8: 0.9783461, 0.9931653, 0.9780399, 0.9926247, -0.0131649, 0.0143813
9: -0.0096832, 0.0037687, -0.0091925, 0.0040466, -0.0130544, 0.0119502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0325566, upper bound: 0.0325398
time: 3.10 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0325398, upper bound: 0.0325398
time: 2.06 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 6.47 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.47
Output dim: 8, lower bound: -0.0303511, upper bound: 0.0297311
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.47
Output dim: 8, lower bound: -0.0303511, upper bound: 0.0301393
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.47
Output dim: 8, lower bound: -0.0309976, upper bound: 0.0299010
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.47
Output dim: 8, lower bound: -0.0309976, upper bound: 0.0302871
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.47
Output dim: 8, lower bound: -0.0315281, upper bound: 0.0319951
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.47
Output dim: 8, lower bound: -0.0315281, upper bound: 0.0323795
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.47
Output dim: 8, lower bound: -0.0322231, upper bound: 0.0322208
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.47
Output dim: 8, lower bound: -0.0322231, upper bound: 0.0326010
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.47
Output dim: 8, lower bound: -0.0306818, upper bound: 0.0300910
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.47
Output dim: 8, lower bound: -0.0306634, upper bound: 0.0300910
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.47
Output dim: 8, lower bound: -0.0313405, upper bound: 0.0301873
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.47
Output dim: 8, lower bound: -0.0313102, upper bound: 0.0301871
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.47
Output dim: 8, lower bound: -0.0318163, upper bound: 0.0323401
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.47
Output dim: 8, lower bound: -0.0317973, upper bound: 0.0323401
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.47
Output dim: 8, lower bound: -0.0325566, upper bound: 0.0325398
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.47
Output dim: 8, lower bound: -0.0325398, upper bound: 0.0325398

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0002836, 0.0065426, 0.0002981, 0.0065179, -0.0056759, 0.0057036
1: 0.0013633, 0.0022675, 0.0013654, 0.0022639, -0.0008200, 0.0008240
2: 0.0107426, 0.0142031, 0.0107563, 0.0141950, -0.0031534, 0.0031381
3: -0.0035699, 0.0000090, -0.0035558, 0.0000007, -0.0032614, 0.0032455
4: -0.0040467, -0.0001723, -0.0040377, -0.0001876, -0.0035135, 0.0035306
5: 0.0042764, 0.0079429, 0.0042909, 0.0079344, -0.0033412, 0.0033249
6: -0.0053329, 0.0092148, -0.0052754, 0.0091810, -0.0132567, 0.0131924
7: -0.0151065, 0.0047062, -0.0150605, 0.0046279, -0.0179669, 0.0180545
8: 0.9785725, 0.9925290, 0.9786050, 0.9924738, -0.0126563, 0.0127180
9: -0.0091056, 0.0035632, -0.0090555, 0.0035337, -0.0115446, 0.0114885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302010, upper bound: 0.0296003
time: 2.36 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302010, upper bound: 0.0295898
time: 1.90 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0002836, 0.0065426, 0.0003374, 0.0066535, -0.0059445, 0.0057822
1: 0.0013633, 0.0022675, 0.0013710, 0.0022835, -0.0008588, 0.0008354
2: 0.0107426, 0.0142031, 0.0106813, 0.0141733, -0.0031968, 0.0032866
3: -0.0035699, 0.0000090, -0.0036334, -0.0000217, -0.0033063, 0.0033991
4: -0.0040467, -0.0001723, -0.0040134, -0.0001036, -0.0036797, 0.0035793
5: 0.0042764, 0.0079429, 0.0042114, 0.0079114, -0.0033872, 0.0034823
6: -0.0053329, 0.0092148, -0.0055907, 0.0090898, -0.0134395, 0.0138167
7: -0.0151065, 0.0047062, -0.0149362, 0.0050573, -0.0188171, 0.0183034
8: 0.9785725, 0.9925290, 0.9786925, 0.9927763, -0.0132552, 0.0128933
9: -0.0091056, 0.0035632, -0.0093301, 0.0034543, -0.0117037, 0.0120322

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302010, upper bound: 0.0299976
time: 2.32 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302010, upper bound: 0.0299912
time: 1.97 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0002178, 0.0066600, 0.0002826, 0.0065191, -0.0057340, 0.0058913
1: 0.0013538, 0.0022845, 0.0013631, 0.0022641, -0.0008284, 0.0008511
2: 0.0106777, 0.0142394, 0.0107557, 0.0142036, -0.0032572, 0.0031702
3: -0.0036371, 0.0000466, -0.0035565, 0.0000096, -0.0033687, 0.0032788
4: -0.0040874, -0.0000996, -0.0040474, -0.0001869, -0.0035494, 0.0036468
5: 0.0042076, 0.0079814, 0.0042902, 0.0079435, -0.0034511, 0.0033590
6: -0.0056057, 0.0093677, -0.0052781, 0.0092172, -0.0136931, 0.0133274
7: -0.0153147, 0.0050777, -0.0151098, 0.0046316, -0.0181508, 0.0186488
8: 0.9784259, 0.9927907, 0.9785702, 0.9924765, -0.0127858, 0.0131366
9: -0.0093432, 0.0036963, -0.0090579, 0.0035653, -0.0119245, 0.0116061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0308382, upper bound: 0.0297624
time: 1.88 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0308382, upper bound: 0.0297468
time: 2.06 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0002178, 0.0066600, 0.0003191, 0.0066558, -0.0060037, 0.0059649
1: 0.0013538, 0.0022845, 0.0013684, 0.0022839, -0.0008674, 0.0008617
2: 0.0106777, 0.0142394, 0.0106801, 0.0141835, -0.0032978, 0.0033193
3: -0.0036371, 0.0000466, -0.0036346, -0.0000113, -0.0034108, 0.0034329
4: -0.0040874, -0.0000996, -0.0040248, -0.0001022, -0.0037164, 0.0036923
5: 0.0042076, 0.0079814, 0.0042101, 0.0079221, -0.0034942, 0.0035169
6: -0.0056057, 0.0093677, -0.0055958, 0.0091324, -0.0138640, 0.0139542
7: -0.0153147, 0.0050777, -0.0149943, 0.0050644, -0.0190043, 0.0188815
8: 0.9784259, 0.9927907, 0.9786516, 0.9927813, -0.0133871, 0.0133005
9: -0.0093432, 0.0036963, -0.0093346, 0.0034914, -0.0120734, 0.0121519

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0308382, upper bound: 0.0301247
time: 1.89 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0308382, upper bound: 0.0301118
time: 1.98 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0002091, 0.0065533, 0.0001495, 0.0065639, -0.0058078, 0.0058147
1: 0.0013525, 0.0022691, 0.0013439, 0.0022706, -0.0008391, 0.0008401
2: 0.0107367, 0.0142442, 0.0107309, 0.0142772, -0.0032148, 0.0032110
3: -0.0035760, 0.0000516, -0.0035821, 0.0000857, -0.0033249, 0.0033210
4: -0.0040928, -0.0001657, -0.0041297, -0.0001591, -0.0035951, 0.0035994
5: 0.0042701, 0.0079865, 0.0042639, 0.0080214, -0.0034063, 0.0034022
6: -0.0053576, 0.0093879, -0.0053822, 0.0095264, -0.0135150, 0.0134989
7: -0.0153422, 0.0047399, -0.0155309, 0.0047734, -0.0183844, 0.0184063
8: 0.9784065, 0.9925528, 0.9782736, 0.9925764, -0.0129504, 0.0129658
9: -0.0091272, 0.0037139, -0.0091486, 0.0038345, -0.0117695, 0.0117555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0313662, upper bound: 0.0318593
time: 1.93 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0313662, upper bound: 0.0318450
time: 2.33 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0002091, 0.0065533, 0.0001835, 0.0067065, -0.0060956, 0.0058970
1: 0.0013525, 0.0022691, 0.0013488, 0.0022912, -0.0008806, 0.0008519
2: 0.0107367, 0.0142442, 0.0106520, 0.0142584, -0.0032603, 0.0033701
3: -0.0035760, 0.0000516, -0.0036636, 0.0000663, -0.0033719, 0.0034855
4: -0.0040928, -0.0001657, -0.0041087, -0.0000709, -0.0037733, 0.0036503
5: 0.0042701, 0.0079865, 0.0041804, 0.0080016, -0.0034544, 0.0035708
6: -0.0053576, 0.0093879, -0.0057137, 0.0094476, -0.0137061, 0.0141679
7: -0.0153422, 0.0047399, -0.0154235, 0.0052249, -0.0192955, 0.0186666
8: 0.9784065, 0.9925528, 0.9783493, 0.9928944, -0.0135921, 0.0131491
9: -0.0091272, 0.0037139, -0.0094373, 0.0037659, -0.0119359, 0.0123381

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0313662, upper bound: 0.0322445
time: 1.62 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0313662, upper bound: 0.0322295
time: 1.96 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0001429, 0.0066707, 0.0001346, 0.0065648, -0.0058682, 0.0059944
1: 0.0013429, 0.0022860, 0.0013417, 0.0022707, -0.0008478, 0.0008660
2: 0.0106718, 0.0142808, 0.0107304, 0.0142854, -0.0033141, 0.0032444
3: -0.0036432, 0.0000895, -0.0035826, 0.0000942, -0.0034276, 0.0033555
4: -0.0041338, -0.0000930, -0.0041390, -0.0001586, -0.0036325, 0.0037106
5: 0.0042013, 0.0080253, 0.0042634, 0.0080302, -0.0035115, 0.0034376
6: -0.0056306, 0.0095418, -0.0053843, 0.0095611, -0.0139326, 0.0136393
7: -0.0155518, 0.0051117, -0.0155782, 0.0047762, -0.0185756, 0.0189750
8: 0.9782588, 0.9928147, 0.9782403, 0.9925783, -0.0130850, 0.0133664
9: -0.0093649, 0.0038479, -0.0091504, 0.0038648, -0.0121331, 0.0118777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0320672, upper bound: 0.0320824
time: 1.97 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0320672, upper bound: 0.0320636
time: 1.87 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0001429, 0.0066707, 0.0001691, 0.0067086, -0.0061573, 0.0060727
1: 0.0013429, 0.0022860, 0.0013467, 0.0022915, -0.0008895, 0.0008773
2: 0.0106718, 0.0142808, 0.0106509, 0.0142664, -0.0033574, 0.0034042
3: -0.0036432, 0.0000895, -0.0036648, 0.0000745, -0.0034724, 0.0035208
4: -0.0041338, -0.0000930, -0.0041176, -0.0000695, -0.0038114, 0.0037591
5: 0.0042013, 0.0080253, 0.0041792, 0.0080100, -0.0035574, 0.0036069
6: -0.0056306, 0.0095418, -0.0057186, 0.0094809, -0.0141147, 0.0143112
7: -0.0155518, 0.0051117, -0.0154689, 0.0052316, -0.0194905, 0.0192229
8: 0.9782588, 0.9928147, 0.9783173, 0.9928991, -0.0137296, 0.0135410
9: -0.0093649, 0.0038479, -0.0094416, 0.0037949, -0.0122917, 0.0124628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0320672, upper bound: 0.0324652
time: 1.98 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0320672, upper bound: 0.0324462
time: 1.85 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0003389, 0.0066278, 0.0002528, 0.0063881, -0.0055294, 0.0059787
1: 0.0013713, 0.0022798, 0.0013588, 0.0022452, -0.0007988, 0.0008637
2: 0.0106955, 0.0141725, 0.0108280, 0.0142201, -0.0033055, 0.0030571
3: -0.0036187, -0.0000226, -0.0034816, 0.0000266, -0.0034187, 0.0031618
4: -0.0040125, -0.0001196, -0.0040658, -0.0002679, -0.0034228, 0.0037009
5: 0.0042265, 0.0079105, 0.0043669, 0.0079609, -0.0035023, 0.0032391
6: -0.0055308, 0.0090862, -0.0049737, 0.0092863, -0.0138961, 0.0128520
7: -0.0149314, 0.0049758, -0.0152039, 0.0042171, -0.0175033, 0.0189253
8: 0.9786959, 0.9927189, 0.9785039, 0.9921844, -0.0123297, 0.0133314
9: -0.0092780, 0.0034512, -0.0087929, 0.0036254, -0.0121014, 0.0111921

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0286495, upper bound: 0.0285203
time: 1.70 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0288596, upper bound: 0.0285203
time: 1.69 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0003452, 0.0065926, 0.0001785, 0.0063687, -0.0055452, 0.0060839
1: 0.0013722, 0.0022747, 0.0013481, 0.0022424, -0.0008011, 0.0008790
2: 0.0107150, 0.0141690, 0.0108388, 0.0142612, -0.0033637, 0.0030658
3: -0.0035985, -0.0000262, -0.0034705, 0.0000691, -0.0034789, 0.0031708
4: -0.0040086, -0.0001414, -0.0041118, -0.0002799, -0.0034326, 0.0037661
5: 0.0042471, 0.0079069, 0.0043783, 0.0080045, -0.0035640, 0.0032484
6: -0.0054489, 0.0090718, -0.0049287, 0.0094592, -0.0141408, 0.0128886
7: -0.0149117, 0.0048643, -0.0154393, 0.0041557, -0.0175531, 0.0192585
8: 0.9787098, 0.9926404, 0.9783381, 0.9921412, -0.0123648, 0.0135661
9: -0.0092067, 0.0034386, -0.0087536, 0.0037759, -0.0123144, 0.0112239

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0286450, upper bound: 0.0285203
time: 1.94 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0288473, upper bound: 0.0285203
time: 1.56 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0002690, 0.0067611, 0.0002345, 0.0063902, -0.0055869, 0.0061632
1: 0.0013612, 0.0022991, 0.0013562, 0.0022455, -0.0008071, 0.0008904
2: 0.0106218, 0.0142112, 0.0108269, 0.0142302, -0.0034075, 0.0030888
3: -0.0036949, 0.0000174, -0.0034828, 0.0000371, -0.0035242, 0.0031946
4: -0.0040558, -0.0000371, -0.0040771, -0.0002666, -0.0034584, 0.0038151
5: 0.0041484, 0.0079515, 0.0043657, 0.0079717, -0.0036104, 0.0032728
6: -0.0058406, 0.0092488, -0.0049786, 0.0093289, -0.0143249, 0.0129854
7: -0.0151528, 0.0053977, -0.0152618, 0.0042237, -0.0176850, 0.0195093
8: 0.9785399, 0.9930161, 0.9784631, 0.9921891, -0.0124577, 0.0137428
9: -0.0095478, 0.0035928, -0.0087971, 0.0036625, -0.0124748, 0.0113083

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0293647, upper bound: 0.0286887
time: 2.16 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0298205, upper bound: 0.0286887
time: 2.03 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0002751, 0.0067257, 0.0001589, 0.0063708, -0.0056034, 0.0062754
1: 0.0013620, 0.0022940, 0.0013453, 0.0022427, -0.0008095, 0.0009066
2: 0.0106414, 0.0142078, 0.0108376, 0.0142720, -0.0034695, 0.0030980
3: -0.0036746, 0.0000139, -0.0034717, 0.0000803, -0.0035883, 0.0032041
4: -0.0040520, -0.0000590, -0.0041239, -0.0002787, -0.0034686, 0.0038846
5: 0.0041692, 0.0079479, 0.0043771, 0.0080160, -0.0036761, 0.0032825
6: -0.0057583, 0.0092345, -0.0049334, 0.0095047, -0.0145857, 0.0130238
7: -0.0151334, 0.0052856, -0.0155013, 0.0041622, -0.0177373, 0.0198644
8: 0.9785536, 0.9929371, 0.9782944, 0.9921458, -0.0124945, 0.0139929
9: -0.0094761, 0.0035803, -0.0087578, 0.0038156, -0.0127019, 0.0113417

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0293474, upper bound: 0.0286877
time: 10.05 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0297890, upper bound: 0.0286877
time: 1.88 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0002608, 0.0066382, 0.0001001, 0.0064391, -0.0056656, 0.0061581
1: 0.0013600, 0.0022813, 0.0013368, 0.0022526, -0.0008185, 0.0008897
2: 0.0106898, 0.0142157, 0.0107998, 0.0143045, -0.0034046, 0.0031324
3: -0.0036246, 0.0000221, -0.0035108, 0.0001139, -0.0035212, 0.0032397
4: -0.0040609, -0.0001131, -0.0041603, -0.0002363, -0.0035071, 0.0038119
5: 0.0042204, 0.0079563, 0.0043370, 0.0080504, -0.0036074, 0.0033189
6: -0.0055550, 0.0092679, -0.0050923, 0.0096413, -0.0143130, 0.0131685
7: -0.0151788, 0.0050086, -0.0156873, 0.0043786, -0.0179343, 0.0194931
8: 0.9785216, 0.9927421, 0.9781634, 0.9922982, -0.0126333, 0.0137314
9: -0.0092990, 0.0036094, -0.0088961, 0.0039345, -0.0124644, 0.0114677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0297793, upper bound: 0.0305537
time: 2.26 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0297649, upper bound: 0.0305225
time: 2.05 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0002670, 0.0066027, 0.0000299, 0.0064074, -0.0056808, 0.0061563
1: 0.0013609, 0.0022762, 0.0013266, 0.0022480, -0.0008207, 0.0008894
2: 0.0107094, 0.0142123, 0.0108174, 0.0143433, -0.0034037, 0.0031408
3: -0.0036043, 0.0000185, -0.0034926, 0.0001541, -0.0035202, 0.0032484
4: -0.0040570, -0.0001351, -0.0042037, -0.0002560, -0.0035165, 0.0038109
5: 0.0042412, 0.0079527, 0.0043556, 0.0080915, -0.0036064, 0.0033278
6: -0.0054726, 0.0092535, -0.0050185, 0.0098044, -0.0143090, 0.0132038
7: -0.0151592, 0.0048965, -0.0159095, 0.0042780, -0.0179825, 0.0194876
8: 0.9785355, 0.9926630, 0.9780069, 0.9922274, -0.0126672, 0.0137274
9: -0.0092273, 0.0035968, -0.0088318, 0.0040766, -0.0124609, 0.0114985

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0297855, upper bound: 0.0305536
time: 2.02 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0297556, upper bound: 0.0305225
time: 1.88 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0001965, 0.0067716, 0.0000832, 0.0064407, -0.0057247, 0.0063492
1: 0.0013507, 0.0023006, 0.0013343, 0.0022528, -0.0008271, 0.0009173
2: 0.0106160, 0.0142512, 0.0107989, 0.0143139, -0.0035103, 0.0031651
3: -0.0037009, 0.0000588, -0.0035117, 0.0001236, -0.0036305, 0.0032734
4: -0.0041006, -0.0000306, -0.0041708, -0.0002354, -0.0035437, 0.0039303
5: 0.0041423, 0.0079939, 0.0043361, 0.0080603, -0.0037193, 0.0033535
6: -0.0058651, 0.0094172, -0.0050961, 0.0096807, -0.0147573, 0.0133058
7: -0.0153822, 0.0054310, -0.0157409, 0.0043837, -0.0181214, 0.0200981
8: 0.9783783, 0.9930396, 0.9781256, 0.9923018, -0.0127651, 0.0141576
9: -0.0095691, 0.0037394, -0.0088994, 0.0039688, -0.0128513, 0.0115873

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0306210, upper bound: 0.0308453
time: 1.80 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0308561, upper bound: 0.0308392
time: 2.21 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0002026, 0.0067362, 0.0000121, 0.0064091, -0.0057410, 0.0063515
1: 0.0013516, 0.0022955, 0.0013240, 0.0022482, -0.0008294, 0.0009176
2: 0.0106356, 0.0142479, 0.0108164, 0.0143532, -0.0035116, 0.0031740
3: -0.0036806, 0.0000553, -0.0034936, 0.0001643, -0.0036318, 0.0032828
4: -0.0040969, -0.0000524, -0.0042148, -0.0002549, -0.0035538, 0.0039317
5: 0.0041630, 0.0079904, 0.0043546, 0.0081020, -0.0037207, 0.0033631
6: -0.0057828, 0.0094031, -0.0050226, 0.0098459, -0.0147626, 0.0133436
7: -0.0153629, 0.0053190, -0.0159659, 0.0042836, -0.0181729, 0.0201053
8: 0.9783919, 0.9929608, 0.9779670, 0.9922314, -0.0128014, 0.0141626
9: -0.0094975, 0.0037271, -0.0088354, 0.0041127, -0.0128559, 0.0116202

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0306182, upper bound: 0.0308446
time: 2.32 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0308388, upper bound: 0.0308388
time: 1.90 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.53 seconds
IS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.53
Output dim: 8, lower bound: -0.0302010, upper bound: 0.0296003
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.53
Output dim: 8, lower bound: -0.0302010, upper bound: 0.0295898
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.53
Output dim: 8, lower bound: -0.0302010, upper bound: 0.0299976
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 5.53
Output dim: 8, lower bound: -0.0302010, upper bound: 0.0299912
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.53
Output dim: 8, lower bound: -0.0308382, upper bound: 0.0297624
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.53
Output dim: 8, lower bound: -0.0308382, upper bound: 0.0297468
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.53
Output dim: 8, lower bound: -0.0308382, upper bound: 0.0301247
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.53
Output dim: 8, lower bound: -0.0308382, upper bound: 0.0301118
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.53
Output dim: 8, lower bound: -0.0313662, upper bound: 0.0318593
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.53
Output dim: 8, lower bound: -0.0313662, upper bound: 0.0318450
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.53
Output dim: 8, lower bound: -0.0313662, upper bound: 0.0322445
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.53
Output dim: 8, lower bound: -0.0313662, upper bound: 0.0322295
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.53
Output dim: 8, lower bound: -0.0320672, upper bound: 0.0320824
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.53
Output dim: 8, lower bound: -0.0320672, upper bound: 0.0320636
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.53
Output dim: 8, lower bound: -0.0320672, upper bound: 0.0324652
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.53
Output dim: 8, lower bound: -0.0320672, upper bound: 0.0324462
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.53
Output dim: 8, lower bound: -0.0286495, upper bound: 0.0285203
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.53
Output dim: 8, lower bound: -0.0288596, upper bound: 0.0285203
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.53
Output dim: 8, lower bound: -0.0286450, upper bound: 0.0285203
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 5.53
Output dim: 8, lower bound: -0.0288473, upper bound: 0.0285203
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.53
Output dim: 8, lower bound: -0.0293647, upper bound: 0.0286887
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.53
Output dim: 8, lower bound: -0.0298205, upper bound: 0.0286887
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.53
Output dim: 8, lower bound: -0.0293474, upper bound: 0.0286877
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 5.53
Output dim: 8, lower bound: -0.0297890, upper bound: 0.0286877
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.53
Output dim: 8, lower bound: -0.0297793, upper bound: 0.0305537
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.53
Output dim: 8, lower bound: -0.0297649, upper bound: 0.0305225
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.53
Output dim: 8, lower bound: -0.0297855, upper bound: 0.0305536
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.53
Output dim: 8, lower bound: -0.0297556, upper bound: 0.0305225
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.53
Output dim: 8, lower bound: -0.0306210, upper bound: 0.0308453
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.53
Output dim: 8, lower bound: -0.0308561, upper bound: 0.0308392
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.53
Output dim: 8, lower bound: -0.0306182, upper bound: 0.0308446
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.53
Output dim: 8, lower bound: -0.0308388, upper bound: 0.0308388

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0002565, 0.0065178, 0.0002972, 0.0064598, -0.0056302, 0.0055713
1: 0.0013594, 0.0022639, 0.0013652, 0.0022556, -0.0008134, 0.0008049
2: 0.0107564, 0.0142181, 0.0107884, 0.0141955, -0.0030802, 0.0031128
3: -0.0035557, 0.0000245, -0.0035226, 0.0000012, -0.0031857, 0.0032194
4: -0.0040635, -0.0001877, -0.0040383, -0.0002236, -0.0034852, 0.0034487
5: 0.0042909, 0.0079588, 0.0043249, 0.0079349, -0.0032636, 0.0032981
6: -0.0052751, 0.0092779, -0.0051404, 0.0091831, -0.0129492, 0.0130860
7: -0.0151924, 0.0046276, -0.0150633, 0.0044440, -0.0178221, 0.0176357
8: 0.9785120, 0.9924735, 0.9786029, 0.9923444, -0.0125542, 0.0124229
9: -0.0090553, 0.0036181, -0.0089380, 0.0035356, -0.0112767, 0.0113959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0289928, upper bound: 0.0279367
time: 2.13 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0289836, upper bound: 0.0279664
time: 2.04 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0001809, 0.0064882, 0.0003008, 0.0064319, -0.0057431, 0.0055859
1: 0.0013484, 0.0022597, 0.0013658, 0.0022515, -0.0008297, 0.0008070
2: 0.0107727, 0.0142598, 0.0108039, 0.0141936, -0.0030883, 0.0031752
3: -0.0035388, 0.0000677, -0.0035066, -0.0000008, -0.0031941, 0.0032839
4: -0.0041103, -0.0002060, -0.0040361, -0.0002408, -0.0035551, 0.0034578
5: 0.0043083, 0.0080031, 0.0043413, 0.0079329, -0.0032722, 0.0033643
6: -0.0052063, 0.0094535, -0.0050754, 0.0091749, -0.0129832, 0.0133485
7: -0.0154315, 0.0045338, -0.0150521, 0.0043556, -0.0181795, 0.0176820
8: 0.9783435, 0.9924075, 0.9786109, 0.9922820, -0.0128060, 0.0124556
9: -0.0089954, 0.0037710, -0.0088814, 0.0035284, -0.0113063, 0.0116244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0289928, upper bound: 0.0279429
time: 1.71 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0289836, upper bound: 0.0279540
time: 1.82 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0002565, 0.0065178, 0.0003338, 0.0065987, -0.0059014, 0.0057109
1: 0.0013594, 0.0022639, 0.0013705, 0.0022756, -0.0008526, 0.0008251
2: 0.0107564, 0.0142181, 0.0107116, 0.0141753, -0.0031574, 0.0032627
3: -0.0035557, 0.0000245, -0.0036020, -0.0000197, -0.0032656, 0.0033745
4: -0.0040635, -0.0001877, -0.0040157, -0.0001375, -0.0036530, 0.0035351
5: 0.0042909, 0.0079588, 0.0042435, 0.0079135, -0.0033454, 0.0034570
6: -0.0052751, 0.0092779, -0.0054633, 0.0090982, -0.0132737, 0.0137164
7: -0.0151924, 0.0046276, -0.0149476, 0.0048838, -0.0186806, 0.0180777
8: 0.9785120, 0.9924735, 0.9786845, 0.9926541, -0.0131590, 0.0127343
9: -0.0090553, 0.0036181, -0.0092192, 0.0034616, -0.0115594, 0.0119449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0289816, upper bound: 0.0281253
time: 2.00 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0289759, upper bound: 0.0284956
time: 1.90 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0001809, 0.0064882, 0.0003397, 0.0065641, -0.0060130, 0.0057226
1: 0.0013484, 0.0022597, 0.0013714, 0.0022706, -0.0008687, 0.0008267
2: 0.0107727, 0.0142598, 0.0107308, 0.0141721, -0.0031638, 0.0033244
3: -0.0035388, 0.0000677, -0.0035822, -0.0000230, -0.0032722, 0.0034383
4: -0.0041103, -0.0002060, -0.0040120, -0.0001590, -0.0037221, 0.0035424
5: 0.0043083, 0.0080031, 0.0042638, 0.0079101, -0.0033523, 0.0035224
6: -0.0052063, 0.0094535, -0.0053827, 0.0090845, -0.0133008, 0.0139759
7: -0.0154315, 0.0045338, -0.0149290, 0.0047740, -0.0190339, 0.0181145
8: 0.9783435, 0.9924075, 0.9786975, 0.9925768, -0.0134079, 0.0127602
9: -0.0089954, 0.0037710, -0.0091490, 0.0034497, -0.0115829, 0.0121708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0289816, upper bound: 0.0281298
time: 2.10 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0289759, upper bound: 0.0284863
time: 2.20 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0002473, 0.0064107, 0.0001641, 0.0065073, -0.0057056, 0.0055455
1: 0.0013580, 0.0022485, 0.0013460, 0.0022624, -0.0008243, 0.0008012
2: 0.0108155, 0.0142231, 0.0107621, 0.0142691, -0.0030660, 0.0031545
3: -0.0034945, 0.0000298, -0.0035498, 0.0000773, -0.0031710, 0.0032625
4: -0.0040692, -0.0002539, -0.0041207, -0.0001941, -0.0035319, 0.0034328
5: 0.0043536, 0.0079642, 0.0042971, 0.0080129, -0.0032486, 0.0033423
6: -0.0050263, 0.0092992, -0.0052508, 0.0094925, -0.0128893, 0.0132614
7: -0.0152215, 0.0042887, -0.0154847, 0.0045945, -0.0180608, 0.0175541
8: 0.9784915, 0.9922349, 0.9783061, 0.9924504, -0.0127224, 0.0123655
9: -0.0088387, 0.0036367, -0.0090342, 0.0038050, -0.0112246, 0.0115486

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0289239, upper bound: 0.0295102
time: 2.57 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0289037, upper bound: 0.0296312
time: 1.80 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0001769, 0.0063784, 0.0001700, 0.0064716, -0.0057308, 0.0055570
1: 0.0013479, 0.0022438, 0.0013469, 0.0022573, -0.0008279, 0.0008028
2: 0.0108334, 0.0142621, 0.0107819, 0.0142659, -0.0030723, 0.0031684
3: -0.0034760, 0.0000700, -0.0035293, 0.0000740, -0.0031775, 0.0032769
4: -0.0041128, -0.0002739, -0.0041171, -0.0002163, -0.0035475, 0.0034399
5: 0.0043726, 0.0080054, 0.0043180, 0.0080095, -0.0032553, 0.0033571
6: -0.0049512, 0.0094628, -0.0051677, 0.0094789, -0.0129160, 0.0133200
7: -0.0154442, 0.0041863, -0.0154662, 0.0044813, -0.0181406, 0.0175905
8: 0.9783346, 0.9921628, 0.9783192, 0.9923705, -0.0127786, 0.0123911
9: -0.0087732, 0.0037791, -0.0089618, 0.0037931, -0.0112478, 0.0115996

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0289239, upper bound: 0.0295184
time: 5.26 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0289042, upper bound: 0.0296289
time: 2.13 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0002473, 0.0064107, 0.0001978, 0.0066495, -0.0059964, 0.0056297
1: 0.0013580, 0.0022485, 0.0013509, 0.0022830, -0.0008663, 0.0008133
2: 0.0108155, 0.0142231, 0.0106835, 0.0142505, -0.0031125, 0.0033153
3: -0.0034945, 0.0000298, -0.0036310, 0.0000581, -0.0032191, 0.0034288
4: -0.0040692, -0.0002539, -0.0040998, -0.0001062, -0.0037119, 0.0034849
5: 0.0043536, 0.0079642, 0.0042138, 0.0079932, -0.0032979, 0.0035127
6: -0.0050263, 0.0092992, -0.0055812, 0.0094142, -0.0130851, 0.0139373
7: -0.0152215, 0.0042887, -0.0153781, 0.0050444, -0.0189814, 0.0178207
8: 0.9784915, 0.9922349, 0.9783812, 0.9927673, -0.0133709, 0.0125533
9: -0.0088387, 0.0036367, -0.0093219, 0.0037368, -0.0113951, 0.0121372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0289239, upper bound: 0.0298053
time: 2.01 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0289037, upper bound: 0.0301789
time: 1.95 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0001769, 0.0063784, 0.0002040, 0.0066143, -0.0060156, 0.0056410
1: 0.0013479, 0.0022438, 0.0013518, 0.0022779, -0.0008691, 0.0008150
2: 0.0108334, 0.0142621, 0.0107030, 0.0142471, -0.0031188, 0.0033259
3: -0.0034760, 0.0000700, -0.0036109, 0.0000545, -0.0032256, 0.0034398
4: -0.0041128, -0.0002739, -0.0040960, -0.0001279, -0.0037238, 0.0034919
5: 0.0043726, 0.0080054, 0.0042344, 0.0079895, -0.0033045, 0.0035239
6: -0.0049512, 0.0094628, -0.0054994, 0.0093999, -0.0131113, 0.0139819
7: -0.0154442, 0.0041863, -0.0153585, 0.0049330, -0.0190422, 0.0178564
8: 0.9783346, 0.9921628, 0.9783950, 0.9926888, -0.0134137, 0.0125784
9: -0.0087732, 0.0037791, -0.0092506, 0.0037243, -0.0114179, 0.0121761

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0289239, upper bound: 0.0298078
time: 1.89 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0289042, upper bound: 0.0301765
time: 1.81 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0001813, 0.0065280, 0.0001492, 0.0065082, -0.0057658, 0.0057284
1: 0.0013485, 0.0022654, 0.0013439, 0.0022625, -0.0008330, 0.0008276
2: 0.0107507, 0.0142596, 0.0107617, 0.0142774, -0.0031671, 0.0031878
3: -0.0035616, 0.0000675, -0.0035502, 0.0000859, -0.0032756, 0.0032970
4: -0.0041101, -0.0001814, -0.0041299, -0.0001936, -0.0035691, 0.0035460
5: 0.0042850, 0.0080029, 0.0042966, 0.0080217, -0.0033557, 0.0033776
6: -0.0052988, 0.0094527, -0.0052528, 0.0095273, -0.0133144, 0.0134014
7: -0.0154304, 0.0046598, -0.0155321, 0.0045971, -0.0182515, 0.0181331
8: 0.9783443, 0.9924964, 0.9782727, 0.9924521, -0.0128567, 0.0127733
9: -0.0090760, 0.0037703, -0.0090359, 0.0038353, -0.0115948, 0.0116705

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0300448, upper bound: 0.0300742
time: 1.86 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0300352, upper bound: 0.0300490
time: 1.90 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0001073, 0.0064988, 0.0001550, 0.0064726, -0.0058767, 0.0057389
1: 0.0013378, 0.0022612, 0.0013447, 0.0022574, -0.0008490, 0.0008291
2: 0.0107669, 0.0143005, 0.0107813, 0.0142742, -0.0031729, 0.0032491
3: -0.0035449, 0.0001098, -0.0035299, 0.0000826, -0.0032816, 0.0033604
4: -0.0041558, -0.0001994, -0.0041263, -0.0002156, -0.0036378, 0.0035525
5: 0.0043021, 0.0080462, 0.0043174, 0.0080183, -0.0033619, 0.0034426
6: -0.0052309, 0.0096246, -0.0051701, 0.0095138, -0.0133389, 0.0136592
7: -0.0156645, 0.0045673, -0.0155137, 0.0044845, -0.0186026, 0.0181664
8: 0.9781795, 0.9924312, 0.9782858, 0.9923729, -0.0131041, 0.0127968
9: -0.0090168, 0.0039200, -0.0089639, 0.0038235, -0.0116161, 0.0118950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0300448, upper bound: 0.0300750
time: 1.81 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0300352, upper bound: 0.0300352
time: 1.71 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0001813, 0.0065280, 0.0001834, 0.0066516, -0.0060579, 0.0058071
1: 0.0013485, 0.0022654, 0.0013488, 0.0022833, -0.0008752, 0.0008390
2: 0.0107507, 0.0142596, 0.0106824, 0.0142584, -0.0032106, 0.0033493
3: -0.0035616, 0.0000675, -0.0036322, 0.0000663, -0.0033206, 0.0034640
4: -0.0041101, -0.0001814, -0.0041087, -0.0001048, -0.0037500, 0.0035947
5: 0.0042850, 0.0080029, 0.0042126, 0.0080016, -0.0034018, 0.0035487
6: -0.0052988, 0.0094527, -0.0055861, 0.0094476, -0.0134974, 0.0140803
7: -0.0154304, 0.0046598, -0.0154236, 0.0050511, -0.0191762, 0.0183822
8: 0.9783443, 0.9924964, 0.9783492, 0.9927720, -0.0135081, 0.0129488
9: -0.0090760, 0.0037703, -0.0093262, 0.0037659, -0.0117541, 0.0122618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0300447, upper bound: 0.0303167
time: 1.68 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0300352, upper bound: 0.0306037
time: 2.19 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0001073, 0.0064988, 0.0001895, 0.0066165, -0.0061635, 0.0058175
1: 0.0013378, 0.0022612, 0.0013497, 0.0022782, -0.0008905, 0.0008405
2: 0.0107669, 0.0143005, 0.0107018, 0.0142551, -0.0032163, 0.0034077
3: -0.0035449, 0.0001098, -0.0036122, 0.0000628, -0.0033265, 0.0035244
4: -0.0041558, -0.0001994, -0.0041050, -0.0001266, -0.0038153, 0.0036011
5: 0.0043021, 0.0080462, 0.0042331, 0.0079980, -0.0034079, 0.0036106
6: -0.0052309, 0.0096246, -0.0055045, 0.0094335, -0.0135214, 0.0143258
7: -0.0156645, 0.0045673, -0.0154043, 0.0049399, -0.0195104, 0.0184149
8: 0.9781795, 0.9924312, 0.9783628, 0.9926937, -0.0137436, 0.0129719
9: -0.0090168, 0.0039200, -0.0092551, 0.0037536, -0.0117750, 0.0124755

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0300447, upper bound: 0.0303169
time: 1.74 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0300352, upper bound: 0.0305867
time: 1.65 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0003939, 0.0066238, 0.0001116, 0.0064380, -0.0055328, 0.0062405
1: 0.0013792, 0.0022793, 0.0013384, 0.0022524, -0.0007993, 0.0009016
2: 0.0106977, 0.0141421, 0.0108005, 0.0142982, -0.0034502, 0.0030589
3: -0.0036164, -0.0000541, -0.0035101, 0.0001074, -0.0035684, 0.0031637
4: -0.0039784, -0.0001220, -0.0041532, -0.0002371, -0.0034249, 0.0038630
5: 0.0042288, 0.0078783, 0.0043377, 0.0080437, -0.0036557, 0.0032411
6: -0.0055216, 0.0089585, -0.0050896, 0.0096147, -0.0145046, 0.0128597
7: -0.0147574, 0.0049632, -0.0156511, 0.0043749, -0.0175138, 0.0197540
8: 0.9788184, 0.9927101, 0.9781889, 0.9922956, -0.0123371, 0.0139152
9: -0.0092700, 0.0033399, -0.0088938, 0.0039114, -0.0126313, 0.0111988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294192, upper bound: 0.0296692
time: 1.68 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0294192, upper bound: 0.0303777
time: 1.80 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0004802, 0.0069775, 0.0002282, 0.0064250, -0.0055723, 0.0064337
1: 0.0013917, 0.0023303, 0.0013553, 0.0022505, -0.0008050, 0.0009295
2: 0.0105022, 0.0140944, 0.0108077, 0.0142337, -0.0035570, 0.0030808
3: -0.0038186, -0.0001034, -0.0035027, 0.0000407, -0.0036788, 0.0031863
4: -0.0039250, 0.0000969, -0.0040810, -0.0002451, -0.0034493, 0.0039826
5: 0.0040216, 0.0078277, 0.0043453, 0.0079754, -0.0037688, 0.0032642
6: -0.0063436, 0.0087579, -0.0050594, 0.0093436, -0.0149537, 0.0129515
7: -0.0144841, 0.0060828, -0.0152819, 0.0043338, -0.0176388, 0.0203656
8: 0.9790109, 0.9934986, 0.9784490, 0.9922667, -0.0124251, 0.0143460
9: -0.0099858, 0.0031652, -0.0088675, 0.0036753, -0.0130223, 0.0112787

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0295101, upper bound: 0.0296282
time: 2.62 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0295101, upper bound: 0.0303743
time: 1.98 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0004001, 0.0065883, 0.0000412, 0.0064062, -0.0055480, 0.0062103
1: 0.0013801, 0.0022741, 0.0013283, 0.0022478, -0.0008015, 0.0008972
2: 0.0107174, 0.0141387, 0.0108180, 0.0143371, -0.0034335, 0.0030673
3: -0.0035961, -0.0000576, -0.0034919, 0.0001476, -0.0035511, 0.0031724
4: -0.0039746, -0.0001440, -0.0041968, -0.0002567, -0.0034343, 0.0038443
5: 0.0042496, 0.0078747, 0.0043563, 0.0080849, -0.0036380, 0.0032500
6: -0.0054391, 0.0089440, -0.0050158, 0.0097782, -0.0144345, 0.0128950
7: -0.0147377, 0.0048508, -0.0158738, 0.0042744, -0.0175619, 0.0196585
8: 0.9788323, 0.9926309, 0.9780320, 0.9922248, -0.0123710, 0.0138479
9: -0.0091981, 0.0033273, -0.0088295, 0.0040538, -0.0125702, 0.0112296

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294310, upper bound: 0.0296692
time: 1.93 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0294310, upper bound: 0.0303774
time: 2.16 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0004861, 0.0069417, 0.0001589, 0.0063931, -0.0055849, 0.0064080
1: 0.0013925, 0.0023252, 0.0013453, 0.0022459, -0.0008069, 0.0009258
2: 0.0105220, 0.0140911, 0.0108253, 0.0142720, -0.0035428, 0.0030878
3: -0.0037982, -0.0001068, -0.0034844, 0.0000803, -0.0036641, 0.0031935
4: -0.0039214, 0.0000748, -0.0041239, -0.0002648, -0.0034572, 0.0039666
5: 0.0040426, 0.0078243, 0.0043640, 0.0080159, -0.0037538, 0.0032716
6: -0.0062605, 0.0087441, -0.0049853, 0.0095045, -0.0148939, 0.0129809
7: -0.0144654, 0.0059696, -0.0155011, 0.0042328, -0.0176788, 0.0202842
8: 0.9790241, 0.9934189, 0.9782946, 0.9921955, -0.0124533, 0.0142886
9: -0.0099134, 0.0031533, -0.0088029, 0.0038155, -0.0129703, 0.0113043

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0295108, upper bound: 0.0296289
time: 1.91 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0295108, upper bound: 0.0303740
time: 2.19 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0003319, 0.0067565, 0.0000947, 0.0064395, -0.0055843, 0.0063211
1: 0.0013703, 0.0022984, 0.0013360, 0.0022526, -0.0008068, 0.0009132
2: 0.0106244, 0.0141763, 0.0107996, 0.0143075, -0.0034948, 0.0030874
3: -0.0036922, -0.0000186, -0.0035110, 0.0001170, -0.0036145, 0.0031931
4: -0.0040168, -0.0000399, -0.0041636, -0.0002361, -0.0034568, 0.0039129
5: 0.0041511, 0.0079146, 0.0043368, 0.0080536, -0.0037029, 0.0032713
6: -0.0058299, 0.0091025, -0.0050933, 0.0096538, -0.0146921, 0.0129794
7: -0.0149535, 0.0053831, -0.0157044, 0.0043799, -0.0176768, 0.0200093
8: 0.9786803, 0.9930058, 0.9781514, 0.9922992, -0.0124519, 0.0140950
9: -0.0095384, 0.0034653, -0.0088970, 0.0039455, -0.0127945, 0.0113030

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303167, upper bound: 0.0300447
time: 1.88 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303167, upper bound: 0.0306986
time: 1.78 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0004041, 0.0071257, 0.0002096, 0.0064271, -0.0056384, 0.0066641
1: 0.0013807, 0.0023518, 0.0013526, 0.0022508, -0.0008146, 0.0009628
2: 0.0104202, 0.0141364, 0.0108065, 0.0142440, -0.0036844, 0.0031173
3: -0.0039034, -0.0000599, -0.0035039, 0.0000513, -0.0038106, 0.0032241
4: -0.0039721, 0.0001887, -0.0040925, -0.0002438, -0.0034902, 0.0041252
5: 0.0039348, 0.0078723, 0.0043441, 0.0079863, -0.0039038, 0.0033029
6: -0.0066882, 0.0089348, -0.0050644, 0.0093868, -0.0154892, 0.0131051
7: -0.0147251, 0.0065520, -0.0153408, 0.0043405, -0.0178481, 0.0210950
8: 0.9788412, 0.9938292, 0.9784074, 0.9922714, -0.0125726, 0.0148597
9: -0.0102859, 0.0033193, -0.0088718, 0.0037130, -0.0134887, 0.0114125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0306037, upper bound: 0.0300352
time: 1.58 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0306037, upper bound: 0.0306986
time: 1.82 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0003381, 0.0067211, 0.0000235, 0.0064080, -0.0056005, 0.0063243
1: 0.0013711, 0.0022933, 0.0013257, 0.0022481, -0.0008091, 0.0009137
2: 0.0106440, 0.0141729, 0.0108171, 0.0143469, -0.0034966, 0.0030964
3: -0.0036720, -0.0000221, -0.0034929, 0.0001577, -0.0036163, 0.0032024
4: -0.0040130, -0.0000618, -0.0042077, -0.0002556, -0.0034668, 0.0039149
5: 0.0041719, 0.0079110, 0.0043553, 0.0080953, -0.0037048, 0.0032808
6: -0.0057476, 0.0090882, -0.0050199, 0.0098194, -0.0146995, 0.0130171
7: -0.0149340, 0.0052710, -0.0159298, 0.0042799, -0.0177282, 0.0200194
8: 0.9786941, 0.9929269, 0.9779926, 0.9922288, -0.0124881, 0.0141021
9: -0.0094668, 0.0034529, -0.0088330, 0.0040896, -0.0128009, 0.0113359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303169, upper bound: 0.0300447
time: 2.02 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303169, upper bound: 0.0306985
time: 1.96 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0004100, 0.0070913, 0.0001384, 0.0063953, -0.0056509, 0.0066451
1: 0.0013815, 0.0023468, 0.0013423, 0.0022462, -0.0008164, 0.0009600
2: 0.0104393, 0.0141332, 0.0108241, 0.0142833, -0.0036739, 0.0031242
3: -0.0038837, -0.0000633, -0.0034857, 0.0000920, -0.0037997, 0.0032312
4: -0.0039685, 0.0001673, -0.0041366, -0.0002635, -0.0034980, 0.0041134
5: 0.0039550, 0.0078688, 0.0043627, 0.0080280, -0.0038927, 0.0033103
6: -0.0066080, 0.0089210, -0.0049905, 0.0095522, -0.0154450, 0.0131343
7: -0.0147063, 0.0064429, -0.0155660, 0.0042399, -0.0178878, 0.0210348
8: 0.9788544, 0.9937524, 0.9782488, 0.9922006, -0.0126005, 0.0148173
9: -0.0102161, 0.0033073, -0.0088074, 0.0038570, -0.0134502, 0.0114379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0305867, upper bound: 0.0300352
time: 1.90 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0305867, upper bound: 0.0306985
time: 1.80 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.99 seconds
IS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0289928, upper bound: 0.0279367
IS_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0289836, upper bound: 0.0279664
IS_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0289928, upper bound: 0.0279429
IS_A1_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0289836, upper bound: 0.0279540
IS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0289816, upper bound: 0.0281253
IS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0289759, upper bound: 0.0284956
IS_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0289816, upper bound: 0.0281298
IS_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0289759, upper bound: 0.0284863
IS_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0289239, upper bound: 0.0295102
IS_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0289037, upper bound: 0.0296312
IS_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0289239, upper bound: 0.0295184
IS_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0289042, upper bound: 0.0296289
IS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0289239, upper bound: 0.0298053
IS_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0289037, upper bound: 0.0301789
IS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0289239, upper bound: 0.0298078
IS_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0289042, upper bound: 0.0301765
IS_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0300448, upper bound: 0.0300742
IS_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0300352, upper bound: 0.0300490
IS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0300448, upper bound: 0.0300750
IS_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0300352, upper bound: 0.0300352
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0300447, upper bound: 0.0303167
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0300352, upper bound: 0.0306037
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0300447, upper bound: 0.0303169
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0300352, upper bound: 0.0305867
IS_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0294192, upper bound: 0.0296692
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0294192, upper bound: 0.0303777
IS_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0295101, upper bound: 0.0296282
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0295101, upper bound: 0.0303743
IS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0294310, upper bound: 0.0296692
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0294310, upper bound: 0.0303774
IS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0295108, upper bound: 0.0296289
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0295108, upper bound: 0.0303740
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0303167, upper bound: 0.0300447
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0303167, upper bound: 0.0306986
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0306037, upper bound: 0.0300352
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0306037, upper bound: 0.0306986
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0303169, upper bound: 0.0300447
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0303169, upper bound: 0.0306985
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0305867, upper bound: 0.0300352
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 8, lower bound: -0.0305867, upper bound: 0.0306985

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0001921, 0.0065269, 0.0003192, 0.0066374, -0.0060314, 0.0056608
1: 0.0013501, 0.0022653, 0.0013684, 0.0022812, -0.0008714, 0.0008178
2: 0.0107513, 0.0142536, 0.0106902, 0.0141834, -0.0031297, 0.0033346
3: -0.0035610, 0.0000613, -0.0036241, -0.0000113, -0.0032369, 0.0034488
4: -0.0041033, -0.0001820, -0.0040247, -0.0001136, -0.0037335, 0.0035041
5: 0.0042856, 0.0079965, 0.0042209, 0.0079221, -0.0033161, 0.0035332
6: -0.0052964, 0.0094274, -0.0055531, 0.0091321, -0.0131573, 0.0140187
7: -0.0153960, 0.0046565, -0.0149938, 0.0050061, -0.0190922, 0.0179191
8: 0.9783686, 0.9924940, 0.9786519, 0.9927403, -0.0134490, 0.0126226
9: -0.0090738, 0.0037483, -0.0092974, 0.0034911, -0.0114580, 0.0122081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0289209, upper bound: 0.0293007
time: 2.04 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0298821, upper bound: 0.0301432
time: 1.94 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0003192, 0.0065151, 0.0003970, 0.0069848, -0.0062042, 0.0056925
1: 0.0013684, 0.0022635, 0.0013797, 0.0023314, -0.0008963, 0.0008224
2: 0.0107578, 0.0141834, 0.0104981, 0.0141403, -0.0031473, 0.0034301
3: -0.0035542, -0.0000113, -0.0038228, -0.0000558, -0.0032550, 0.0035476
4: -0.0040247, -0.0001893, -0.0039765, 0.0001014, -0.0038405, 0.0035238
5: 0.0042925, 0.0079221, 0.0040174, 0.0078765, -0.0033347, 0.0036344
6: -0.0052690, 0.0091321, -0.0063607, 0.0089512, -0.0132310, 0.0144203
7: -0.0149938, 0.0046192, -0.0147474, 0.0061060, -0.0196392, 0.0180195
8: 0.9786519, 0.9924677, 0.9788254, 0.9935150, -0.0138342, 0.0126933
9: -0.0090500, 0.0034911, -0.0100007, 0.0033336, -0.0115222, 0.0125578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0289031, upper bound: 0.0296133
time: 1.71 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0298699, upper bound: 0.0304256
time: 1.82 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0001179, 0.0064978, 0.0003253, 0.0066021, -0.0061372, 0.0056711
1: 0.0013393, 0.0022610, 0.0013693, 0.0022761, -0.0008867, 0.0008193
2: 0.0107674, 0.0142947, 0.0107097, 0.0141800, -0.0031354, 0.0033931
3: -0.0035443, 0.0001038, -0.0036040, -0.0000148, -0.0032428, 0.0035093
4: -0.0041493, -0.0002001, -0.0040209, -0.0001355, -0.0037990, 0.0035105
5: 0.0043027, 0.0080400, 0.0042415, 0.0079185, -0.0033221, 0.0035952
6: -0.0052286, 0.0095999, -0.0054711, 0.0091178, -0.0131813, 0.0142646
7: -0.0156310, 0.0045642, -0.0149744, 0.0048945, -0.0194272, 0.0179518
8: 0.9782031, 0.9924290, 0.9786655, 0.9926617, -0.0136849, 0.0126456
9: -0.0090148, 0.0038985, -0.0092260, 0.0034787, -0.0114789, 0.0124223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0289210, upper bound: 0.0293000
time: 2.06 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0298821, upper bound: 0.0301420
time: 2.19 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0002486, 0.0064860, 0.0004030, 0.0069492, -0.0063052, 0.0056673
1: 0.0013582, 0.0022593, 0.0013805, 0.0023263, -0.0009109, 0.0008188
2: 0.0107739, 0.0142224, 0.0105178, 0.0141371, -0.0031333, 0.0034860
3: -0.0035376, 0.0000290, -0.0038024, -0.0000593, -0.0032406, 0.0036054
4: -0.0040684, -0.0002073, -0.0039728, 0.0000794, -0.0039030, 0.0035081
5: 0.0043096, 0.0079634, 0.0040382, 0.0078730, -0.0033199, 0.0036936
6: -0.0052013, 0.0092961, -0.0062778, 0.0089373, -0.0131723, 0.0146551
7: -0.0152172, 0.0045270, -0.0147286, 0.0059931, -0.0199590, 0.0179395
8: 0.9784945, 0.9924028, 0.9788387, 0.9934355, -0.0140595, 0.0126370
9: -0.0089910, 0.0036340, -0.0099285, 0.0033215, -0.0114710, 0.0127623

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0289031, upper bound: 0.0296123
time: 2.02 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0298699, upper bound: 0.0304096
time: 2.09 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0003939, 0.0066238, 0.0002327, 0.0065602, -0.0055191, 0.0058824
1: 0.0013792, 0.0022793, 0.0013559, 0.0022701, -0.0007974, 0.0008498
2: 0.0106977, 0.0141421, 0.0107329, 0.0142312, -0.0032522, 0.0030514
3: -0.0036164, -0.0000541, -0.0035800, 0.0000381, -0.0033636, 0.0031559
4: -0.0039784, -0.0001220, -0.0040783, -0.0001614, -0.0034164, 0.0036413
5: 0.0042288, 0.0078783, 0.0042661, 0.0079728, -0.0034459, 0.0032331
6: -0.0055216, 0.0089585, -0.0053737, 0.0093332, -0.0136722, 0.0128279
7: -0.0147574, 0.0049632, -0.0152678, 0.0047618, -0.0174705, 0.0186204
8: 0.9788184, 0.9927101, 0.9784589, 0.9925681, -0.0123066, 0.0131166
9: -0.0092700, 0.0033399, -0.0091411, 0.0036663, -0.0119064, 0.0111711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0284662, upper bound: 0.0293510
time: 1.98 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0292518, upper bound: 0.0302078
time: 1.94 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0004802, 0.0069775, 0.0003447, 0.0065492, -0.0055792, 0.0060091
1: 0.0013917, 0.0023303, 0.0013721, 0.0022685, -0.0008060, 0.0008681
2: 0.0105022, 0.0140944, 0.0107390, 0.0141693, -0.0033223, 0.0030846
3: -0.0038186, -0.0001034, -0.0035737, -0.0000259, -0.0034361, 0.0031902
4: -0.0039250, 0.0000969, -0.0040089, -0.0001682, -0.0034536, 0.0037197
5: 0.0040216, 0.0078277, 0.0042725, 0.0079071, -0.0035201, 0.0032683
6: -0.0063436, 0.0087579, -0.0053481, 0.0090727, -0.0139668, 0.0129676
7: -0.0144841, 0.0060828, -0.0149130, 0.0047270, -0.0176607, 0.0190216
8: 0.9790109, 0.9934986, 0.9787088, 0.9925436, -0.0124406, 0.0133992
9: -0.0099858, 0.0031652, -0.0091189, 0.0034394, -0.0121629, 0.0112927

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0285164, upper bound: 0.0293481
time: 1.88 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0293546, upper bound: 0.0302046
time: 1.90 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0004001, 0.0065883, 0.0001697, 0.0065285, -0.0056386, 0.0058632
1: 0.0013801, 0.0022741, 0.0013468, 0.0022655, -0.0008146, 0.0008471
2: 0.0107174, 0.0141387, 0.0107504, 0.0142660, -0.0032416, 0.0031174
3: -0.0035961, -0.0000576, -0.0035619, 0.0000742, -0.0033526, 0.0032242
4: -0.0039746, -0.0001440, -0.0041172, -0.0001810, -0.0034904, 0.0036294
5: 0.0042496, 0.0078747, 0.0042847, 0.0080096, -0.0034346, 0.0033031
6: -0.0054391, 0.0089440, -0.0053001, 0.0094796, -0.0136276, 0.0131055
7: -0.0147377, 0.0048508, -0.0154671, 0.0046615, -0.0178486, 0.0185596
8: 0.9788323, 0.9926309, 0.9783186, 0.9924974, -0.0125729, 0.0130738
9: -0.0091981, 0.0033273, -0.0090770, 0.0037937, -0.0118675, 0.0114129

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0284740, upper bound: 0.0293510
time: 2.22 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0292608, upper bound: 0.0302078
time: 1.94 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0004861, 0.0069417, 0.0002801, 0.0065164, -0.0055851, 0.0059952
1: 0.0013925, 0.0023252, 0.0013628, 0.0022637, -0.0008069, 0.0008661
2: 0.0105220, 0.0140911, 0.0107571, 0.0142050, -0.0033146, 0.0030878
3: -0.0037982, -0.0001068, -0.0035550, 0.0000110, -0.0034281, 0.0031936
4: -0.0039214, 0.0000748, -0.0040489, -0.0001885, -0.0034573, 0.0037111
5: 0.0040426, 0.0078243, 0.0042917, 0.0079450, -0.0035120, 0.0032717
6: -0.0062605, 0.0087441, -0.0052719, 0.0092230, -0.0139345, 0.0129813
7: -0.0144654, 0.0059696, -0.0151177, 0.0046232, -0.0176794, 0.0189776
8: 0.9790241, 0.9934189, 0.9785646, 0.9924706, -0.0124537, 0.0133682
9: -0.0099134, 0.0031533, -0.0090526, 0.0035703, -0.0121348, 0.0113047

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0285153, upper bound: 0.0293481
time: 2.20 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0293495, upper bound: 0.0302042
time: 1.89 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0003319, 0.0067565, 0.0001835, 0.0064216, -0.0055473, 0.0061544
1: 0.0013703, 0.0022984, 0.0013488, 0.0022500, -0.0008014, 0.0008891
2: 0.0106244, 0.0141763, 0.0108096, 0.0142584, -0.0034026, 0.0030670
3: -0.0036922, -0.0000186, -0.0035007, 0.0000663, -0.0035191, 0.0031720
4: -0.0040168, -0.0000399, -0.0041087, -0.0002472, -0.0034339, 0.0038097
5: 0.0041511, 0.0079146, 0.0043473, 0.0080015, -0.0036052, 0.0032496
6: -0.0058299, 0.0091025, -0.0050515, 0.0094475, -0.0143045, 0.0128935
7: -0.0149535, 0.0053831, -0.0154233, 0.0043229, -0.0175599, 0.0194814
8: 0.9786803, 0.9930058, 0.9783494, 0.9922590, -0.0123695, 0.0137231
9: -0.0095384, 0.0034653, -0.0088605, 0.0037658, -0.0124570, 0.0112283

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0292134, upper bound: 0.0290530
time: 1.63 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0301432, upper bound: 0.0298821
time: 1.93 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0003319, 0.0067565, 0.0002183, 0.0065622, -0.0055728, 0.0059903
1: 0.0013703, 0.0022984, 0.0013538, 0.0022704, -0.0008051, 0.0008654
2: 0.0106244, 0.0141763, 0.0107318, 0.0142392, -0.0033119, 0.0030811
3: -0.0036922, -0.0000186, -0.0035812, 0.0000464, -0.0034253, 0.0031866
4: -0.0040168, -0.0000399, -0.0040871, -0.0001601, -0.0034497, 0.0037081
5: 0.0041511, 0.0079146, 0.0042649, 0.0079812, -0.0035091, 0.0032645
6: -0.0058299, 0.0091025, -0.0053785, 0.0093666, -0.0139231, 0.0129528
7: -0.0149535, 0.0053831, -0.0153132, 0.0047683, -0.0176405, 0.0189621
8: 0.9786803, 0.9930058, 0.9784269, 0.9925728, -0.0124264, 0.0133573
9: -0.0095384, 0.0034653, -0.0091453, 0.0036953, -0.0121249, 0.0112798

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0292134, upper bound: 0.0296078
time: 1.80 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301432, upper bound: 0.0305322
time: 1.94 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0004041, 0.0071257, 0.0003115, 0.0064102, -0.0055940, 0.0064769
1: 0.0013807, 0.0023518, 0.0013673, 0.0022484, -0.0008082, 0.0009357
2: 0.0104202, 0.0141364, 0.0108158, 0.0141876, -0.0035809, 0.0030928
3: -0.0039034, -0.0000599, -0.0034942, -0.0000070, -0.0037035, 0.0031987
4: -0.0039721, 0.0001887, -0.0040294, -0.0002542, -0.0034628, 0.0040093
5: 0.0039348, 0.0078723, 0.0043539, 0.0079265, -0.0037941, 0.0032770
6: -0.0066882, 0.0089348, -0.0050252, 0.0091499, -0.0150540, 0.0130021
7: -0.0147251, 0.0065520, -0.0150181, 0.0042871, -0.0177077, 0.0205023
8: 0.9788412, 0.9938292, 0.9786348, 0.9922338, -0.0124737, 0.0144423
9: -0.0102859, 0.0033193, -0.0088377, 0.0035066, -0.0131097, 0.0113228

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294007, upper bound: 0.0290415
time: 1.92 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304256, upper bound: 0.0298699
time: 1.76 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0004041, 0.0071257, 0.0003295, 0.0065519, -0.0056457, 0.0062441
1: 0.0013807, 0.0023518, 0.0013699, 0.0022689, -0.0008156, 0.0009021
2: 0.0104202, 0.0141364, 0.0107375, 0.0141777, -0.0034522, 0.0031213
3: -0.0039034, -0.0000599, -0.0035752, -0.0000172, -0.0035705, 0.0032282
4: -0.0039721, 0.0001887, -0.0040183, -0.0001666, -0.0034948, 0.0038652
5: 0.0039348, 0.0078723, 0.0042710, 0.0079160, -0.0036578, 0.0033072
6: -0.0066882, 0.0089348, -0.0053544, 0.0091082, -0.0145131, 0.0131221
7: -0.0147251, 0.0065520, -0.0149613, 0.0047355, -0.0178712, 0.0197656
8: 0.9788412, 0.9938292, 0.9786748, 0.9925497, -0.0125888, 0.0139233
9: -0.0102859, 0.0033193, -0.0091243, 0.0034703, -0.0126387, 0.0114273

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294007, upper bound: 0.0296078
time: 2.29 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304256, upper bound: 0.0305322
time: 1.78 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0003381, 0.0067211, 0.0001107, 0.0063894, -0.0055628, 0.0061605
1: 0.0013711, 0.0022933, 0.0013383, 0.0022454, -0.0008037, 0.0008900
2: 0.0106440, 0.0141729, 0.0108274, 0.0142986, -0.0034060, 0.0030755
3: -0.0036720, -0.0000221, -0.0034823, 0.0001079, -0.0035226, 0.0031809
4: -0.0040130, -0.0000618, -0.0041537, -0.0002672, -0.0034435, 0.0038134
5: 0.0041719, 0.0079110, 0.0043662, 0.0080442, -0.0036088, 0.0032587
6: -0.0057476, 0.0090882, -0.0049766, 0.0096166, -0.0143187, 0.0129296
7: -0.0149340, 0.0052710, -0.0156537, 0.0042210, -0.0176089, 0.0195008
8: 0.9786941, 0.9929269, 0.9781870, 0.9921873, -0.0124041, 0.0137367
9: -0.0094668, 0.0034529, -0.0087954, 0.0039131, -0.0124693, 0.0112596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0292103, upper bound: 0.0290530
time: 1.79 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0301420, upper bound: 0.0298821
time: 1.66 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0003381, 0.0067211, 0.0001521, 0.0065306, -0.0056915, 0.0059913
1: 0.0013711, 0.0022933, 0.0013443, 0.0022658, -0.0008223, 0.0008656
2: 0.0106440, 0.0141729, 0.0107492, 0.0142758, -0.0033124, 0.0031467
3: -0.0036720, -0.0000221, -0.0035631, 0.0000842, -0.0034259, 0.0032544
4: -0.0040130, -0.0000618, -0.0041281, -0.0001797, -0.0035231, 0.0037087
5: 0.0041719, 0.0079110, 0.0042834, 0.0080199, -0.0035097, 0.0033340
6: -0.0057476, 0.0090882, -0.0053050, 0.0095204, -0.0139255, 0.0132285
7: -0.0149340, 0.0052710, -0.0155227, 0.0046683, -0.0180161, 0.0189653
8: 0.9786941, 0.9929269, 0.9782794, 0.9925023, -0.0126909, 0.0133596
9: -0.0094668, 0.0034529, -0.0090814, 0.0038293, -0.0121269, 0.0115200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0292103, upper bound: 0.0296078
time: 1.85 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0301420, upper bound: 0.0305322
time: 2.43 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0004100, 0.0070913, 0.0002425, 0.0063780, -0.0056056, 0.0064644
1: 0.0013815, 0.0023468, 0.0013573, 0.0022437, -0.0008099, 0.0009339
2: 0.0104393, 0.0141332, 0.0108336, 0.0142258, -0.0035740, 0.0030992
3: -0.0038837, -0.0000633, -0.0034758, 0.0000325, -0.0036964, 0.0032054
4: -0.0039685, 0.0001673, -0.0040721, -0.0002742, -0.0034700, 0.0040016
5: 0.0039550, 0.0078688, 0.0043728, 0.0079670, -0.0037868, 0.0032838
6: -0.0066080, 0.0089210, -0.0049502, 0.0093103, -0.0150250, 0.0130291
7: -0.0147063, 0.0064429, -0.0152365, 0.0041850, -0.0177445, 0.0204628
8: 0.9788544, 0.9937524, 0.9784810, 0.9921619, -0.0124996, 0.0144144
9: -0.0102161, 0.0033073, -0.0087724, 0.0036463, -0.0130845, 0.0113463

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0293892, upper bound: 0.0290415
time: 2.09 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304096, upper bound: 0.0298699
time: 1.74 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0004100, 0.0070913, 0.0002607, 0.0065191, -0.0056504, 0.0062277
1: 0.0013815, 0.0023468, 0.0013600, 0.0022641, -0.0008163, 0.0008997
2: 0.0104393, 0.0141332, 0.0107556, 0.0142157, -0.0034431, 0.0031240
3: -0.0038837, -0.0000633, -0.0035565, 0.0000221, -0.0035611, 0.0032310
4: -0.0039685, 0.0001673, -0.0040609, -0.0001869, -0.0034977, 0.0038550
5: 0.0039550, 0.0078688, 0.0042902, 0.0079563, -0.0036482, 0.0033100
6: -0.0066080, 0.0089210, -0.0052781, 0.0092681, -0.0144749, 0.0131331
7: -0.0147063, 0.0064429, -0.0151790, 0.0046316, -0.0178862, 0.0197135
8: 0.9788544, 0.9937524, 0.9785214, 0.9924765, -0.0125994, 0.0138866
9: -0.0102161, 0.0033073, -0.0090579, 0.0036095, -0.0126054, 0.0114369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0293892, upper bound: 0.0296078
time: 1.75 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304096, upper bound: 0.0305322
time: 1.80 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.87 seconds
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0289209, upper bound: 0.0293007
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0298821, upper bound: 0.0301432
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0289031, upper bound: 0.0296133
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0298699, upper bound: 0.0304256
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0289210, upper bound: 0.0293000
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0298821, upper bound: 0.0301420
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0289031, upper bound: 0.0296123
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0298699, upper bound: 0.0304096
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0284662, upper bound: 0.0293510
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0292518, upper bound: 0.0302078
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0285164, upper bound: 0.0293481
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0293546, upper bound: 0.0302046
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0284740, upper bound: 0.0293510
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0292608, upper bound: 0.0302078
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0285153, upper bound: 0.0293481
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0293495, upper bound: 0.0302042
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0292134, upper bound: 0.0290530
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0301432, upper bound: 0.0298821
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0292134, upper bound: 0.0296078
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0301432, upper bound: 0.0305322
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0294007, upper bound: 0.0290415
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0304256, upper bound: 0.0298699
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0294007, upper bound: 0.0296078
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0304256, upper bound: 0.0305322
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0292103, upper bound: 0.0290530
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0301420, upper bound: 0.0298821
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0292103, upper bound: 0.0296078
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0301420, upper bound: 0.0305322
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0293892, upper bound: 0.0290415
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0304096, upper bound: 0.0298699
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0293892, upper bound: 0.0296078
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0304096, upper bound: 0.0305322

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0003386, 0.0064130, 0.0004017, 0.0069605, -0.0060933, 0.0055754
1: 0.0013712, 0.0022488, 0.0013803, 0.0023279, -0.0008803, 0.0008055
2: 0.0108143, 0.0141727, 0.0105116, 0.0141378, -0.0030825, 0.0033688
3: -0.0034958, -0.0000224, -0.0038089, -0.0000585, -0.0031881, 0.0034842
4: -0.0040127, -0.0002526, -0.0039736, 0.0000864, -0.0037719, 0.0034513
5: 0.0043524, 0.0079107, 0.0040316, 0.0078737, -0.0032661, 0.0035695
6: -0.0050315, 0.0090871, -0.0063040, 0.0089403, -0.0129588, 0.0141625
7: -0.0149325, 0.0042958, -0.0147326, 0.0060288, -0.0192882, 0.0176487
8: 0.9786950, 0.9922398, 0.9788359, 0.9934608, -0.0135870, 0.0124321
9: -0.0088432, 0.0034519, -0.0099513, 0.0033241, -0.0112851, 0.0123334

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0265551, upper bound: 0.0275487
time: 1.69 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0264223, upper bound: 0.0267067
time: 1.65 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0002649, 0.0063846, 0.0004077, 0.0069250, -0.0061928, 0.0055538
1: 0.0013606, 0.0022447, 0.0013812, 0.0023228, -0.0008947, 0.0008024
2: 0.0108300, 0.0142134, 0.0105312, 0.0141345, -0.0030706, 0.0034239
3: -0.0034796, 0.0000197, -0.0037886, -0.0000619, -0.0031757, 0.0035411
4: -0.0040583, -0.0002701, -0.0039699, 0.0000644, -0.0038335, 0.0034379
5: 0.0043689, 0.0079539, 0.0040524, 0.0078702, -0.0032534, 0.0036277
6: -0.0049656, 0.0092583, -0.0062215, 0.0089264, -0.0129087, 0.0143938
7: -0.0151657, 0.0042060, -0.0147137, 0.0059165, -0.0196032, 0.0175805
8: 0.9785308, 0.9921767, 0.9788492, 0.9933816, -0.0138089, 0.0123841
9: -0.0087858, 0.0036010, -0.0098795, 0.0033120, -0.0112414, 0.0125348

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0263818, upper bound: 0.0274696
time: 1.65 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0262914, upper bound: 0.0266864
time: 1.68 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0003531, 0.0066488, 0.0002230, 0.0065389, -0.0055437, 0.0058687
1: 0.0013733, 0.0022829, 0.0013545, 0.0022670, -0.0008009, 0.0008479
2: 0.0106839, 0.0141647, 0.0107447, 0.0142366, -0.0032447, 0.0030650
3: -0.0036306, -0.0000307, -0.0035678, 0.0000437, -0.0033558, 0.0031699
4: -0.0040037, -0.0001066, -0.0040843, -0.0001746, -0.0034316, 0.0036328
5: 0.0042142, 0.0079022, 0.0042786, 0.0079784, -0.0034379, 0.0032475
6: -0.0055796, 0.0090533, -0.0053242, 0.0093557, -0.0136405, 0.0128850
7: -0.0148866, 0.0050422, -0.0152984, 0.0046944, -0.0175483, 0.0185772
8: 0.9787275, 0.9927657, 0.9784373, 0.9925207, -0.0123614, 0.0130861
9: -0.0093204, 0.0034225, -0.0090981, 0.0036859, -0.0118787, 0.0112209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304535, upper bound: 0.0304136
time: 1.86 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304535, upper bound: 0.0305322
time: 1.90 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0004254, 0.0070147, 0.0003162, 0.0063879, -0.0055628, 0.0063493
1: 0.0013838, 0.0023357, 0.0013680, 0.0022452, -0.0008037, 0.0009173
2: 0.0104816, 0.0141247, 0.0108281, 0.0141851, -0.0035103, 0.0030755
3: -0.0038399, -0.0000720, -0.0034815, -0.0000096, -0.0036306, 0.0031809
4: -0.0039590, 0.0001199, -0.0040266, -0.0002680, -0.0034435, 0.0039303
5: 0.0039999, 0.0078599, 0.0043670, 0.0079238, -0.0037194, 0.0032587
6: -0.0064301, 0.0088853, -0.0049733, 0.0091392, -0.0147575, 0.0129295
7: -0.0146577, 0.0062005, -0.0150035, 0.0042165, -0.0176089, 0.0200984
8: 0.9788886, 0.9935817, 0.9786451, 0.9921841, -0.0124041, 0.0141577
9: -0.0100611, 0.0032762, -0.0087925, 0.0034973, -0.0128515, 0.0112596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0268451, upper bound: 0.0271442
time: 1.45 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0267067, upper bound: 0.0264223
time: 1.69 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0004254, 0.0070147, 0.0003341, 0.0065285, -0.0056157, 0.0061213
1: 0.0013838, 0.0023357, 0.0013706, 0.0022655, -0.0008113, 0.0008843
2: 0.0104816, 0.0141247, 0.0107504, 0.0141751, -0.0033843, 0.0031048
3: -0.0038399, -0.0000720, -0.0035619, -0.0000199, -0.0035002, 0.0032111
4: -0.0039590, 0.0001199, -0.0040154, -0.0001810, -0.0034762, 0.0037892
5: 0.0039999, 0.0078599, 0.0042846, 0.0079133, -0.0035858, 0.0032896
6: -0.0064301, 0.0088853, -0.0053001, 0.0090974, -0.0142275, 0.0130523
7: -0.0146577, 0.0062005, -0.0149465, 0.0046616, -0.0177762, 0.0193766
8: 0.9788886, 0.9935817, 0.9786853, 0.9924976, -0.0125219, 0.0136493
9: -0.0100611, 0.0032762, -0.0090771, 0.0034609, -0.0123900, 0.0113666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0271131, upper bound: 0.0278584
time: 1.88 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0270176, upper bound: 0.0270643
time: 1.56 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0003593, 0.0066128, 0.0001558, 0.0065064, -0.0056612, 0.0058806
1: 0.0013742, 0.0022777, 0.0013448, 0.0022623, -0.0008179, 0.0008496
2: 0.0107038, 0.0141612, 0.0107626, 0.0142737, -0.0032512, 0.0031299
3: -0.0036101, -0.0000342, -0.0035492, 0.0000821, -0.0033626, 0.0032371
4: -0.0039999, -0.0001288, -0.0041259, -0.0001947, -0.0035043, 0.0036402
5: 0.0042353, 0.0078986, 0.0042976, 0.0080178, -0.0034449, 0.0033163
6: -0.0054960, 0.0090389, -0.0052487, 0.0095120, -0.0136682, 0.0131581
7: -0.0148670, 0.0049284, -0.0155112, 0.0045916, -0.0179202, 0.0186149
8: 0.9787413, 0.9926855, 0.9782875, 0.9924483, -0.0126233, 0.0131127
9: -0.0092477, 0.0034100, -0.0090324, 0.0038219, -0.0119029, 0.0114586

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304483, upper bound: 0.0304135
time: 1.64 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304483, upper bound: 0.0305322
time: 1.86 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0004314, 0.0069800, 0.0002460, 0.0063553, -0.0055719, 0.0063480
1: 0.0013846, 0.0023307, 0.0013578, 0.0022405, -0.0008050, 0.0009171
2: 0.0105008, 0.0141213, 0.0108462, 0.0142239, -0.0035096, 0.0030806
3: -0.0038200, -0.0000755, -0.0034628, 0.0000305, -0.0036298, 0.0031861
4: -0.0039552, 0.0000984, -0.0040700, -0.0002882, -0.0034491, 0.0039295
5: 0.0040202, 0.0078563, 0.0043861, 0.0079649, -0.0037186, 0.0032640
6: -0.0063494, 0.0088712, -0.0048975, 0.0093023, -0.0147545, 0.0129507
7: -0.0146385, 0.0060906, -0.0152256, 0.0041132, -0.0176377, 0.0200944
8: 0.9789022, 0.9935043, 0.9784886, 0.9921114, -0.0124244, 0.0141549
9: -0.0099909, 0.0032639, -0.0087265, 0.0036393, -0.0128489, 0.0112780

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0268376, upper bound: 0.0271432
time: 1.68 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0266864, upper bound: 0.0262914
time: 1.59 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0004314, 0.0069800, 0.0002643, 0.0064949, -0.0056207, 0.0061191
1: 0.0013846, 0.0023307, 0.0013605, 0.0022606, -0.0008120, 0.0008840
2: 0.0105008, 0.0141213, 0.0107690, 0.0142138, -0.0033831, 0.0031075
3: -0.0038200, -0.0000755, -0.0035426, 0.0000201, -0.0034990, 0.0032140
4: -0.0039552, 0.0000984, -0.0040587, -0.0002018, -0.0034793, 0.0037878
5: 0.0040202, 0.0078563, 0.0043044, 0.0079542, -0.0035846, 0.0032926
6: -0.0063494, 0.0088712, -0.0052219, 0.0092598, -0.0142225, 0.0130641
7: -0.0146385, 0.0060906, -0.0151677, 0.0045551, -0.0177922, 0.0193698
8: 0.9789022, 0.9935043, 0.9785295, 0.9924225, -0.0125332, 0.0136445
9: -0.0099909, 0.0032639, -0.0090090, 0.0036023, -0.0123855, 0.0113768

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0271058, upper bound: 0.0278583
time: 1.51 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0269978, upper bound: 0.0269537
time: 1.40 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 4.22 seconds
IS_A1_B2_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.22
Output dim: 8, lower bound: -0.0265551, upper bound: 0.0275487
IS_A1_B2_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 4.22
Output dim: 8, lower bound: -0.0264223, upper bound: 0.0267067
IS_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.22
Output dim: 8, lower bound: -0.0263818, upper bound: 0.0274696
IS_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 4.22
Output dim: 8, lower bound: -0.0262914, upper bound: 0.0266864
IS_A2_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 8, lower bound: -0.0304535, upper bound: 0.0304136
IS_A2_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 8, lower bound: -0.0304535, upper bound: 0.0305322
IS_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.22
Output dim: 8, lower bound: -0.0268451, upper bound: 0.0271442
IS_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 4.22
Output dim: 8, lower bound: -0.0267067, upper bound: 0.0264223
IS_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.22
Output dim: 8, lower bound: -0.0271131, upper bound: 0.0278584
IS_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 4.22
Output dim: 8, lower bound: -0.0270176, upper bound: 0.0270643
IS_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 8, lower bound: -0.0304483, upper bound: 0.0304135
IS_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 8, lower bound: -0.0304483, upper bound: 0.0305322
IS_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.22
Output dim: 8, lower bound: -0.0268376, upper bound: 0.0271432
IS_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 4.22
Output dim: 8, lower bound: -0.0266864, upper bound: 0.0262914
IS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.22
Output dim: 8, lower bound: -0.0271058, upper bound: 0.0278583
IS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 4.22
Output dim: 8, lower bound: -0.0269978, upper bound: 0.0269537

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0003531, 0.0066488, 0.0003465, 0.0065282, -0.0055319, 0.0057430
1: 0.0013733, 0.0022829, 0.0013724, 0.0022654, -0.0007992, 0.0008297
2: 0.0106839, 0.0141647, 0.0107506, 0.0141683, -0.0031752, 0.0030584
3: -0.0036306, -0.0000307, -0.0035617, -0.0000270, -0.0032839, 0.0031632
4: -0.0040037, -0.0001066, -0.0040078, -0.0001812, -0.0034243, 0.0035550
5: 0.0042142, 0.0079022, 0.0042849, 0.0079061, -0.0033642, 0.0032406
6: -0.0055796, 0.0090533, -0.0052993, 0.0090686, -0.0133483, 0.0128576
7: -0.0148866, 0.0050422, -0.0149073, 0.0046604, -0.0175109, 0.0181793
8: 0.9787275, 0.9927657, 0.9787128, 0.9924968, -0.0123350, 0.0128059
9: -0.0093204, 0.0034225, -0.0090763, 0.0034358, -0.0116243, 0.0111969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0279484, upper bound: 0.0272597
time: 1.92 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0272636, upper bound: 0.0271435
time: 1.65 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0003531, 0.0066488, 0.0004250, 0.0068748, -0.0058551, 0.0056922
1: 0.0013733, 0.0022829, 0.0013837, 0.0023155, -0.0008459, 0.0008224
2: 0.0106839, 0.0141647, 0.0105589, 0.0141249, -0.0031471, 0.0032371
3: -0.0036306, -0.0000307, -0.0037599, -0.0000718, -0.0032548, 0.0033480
4: -0.0040037, -0.0001066, -0.0039592, 0.0000334, -0.0036244, 0.0035236
5: 0.0042142, 0.0079022, 0.0040818, 0.0078601, -0.0033345, 0.0034299
6: -0.0055796, 0.0090533, -0.0061050, 0.0088862, -0.0132302, 0.0136088
7: -0.0148866, 0.0050422, -0.0146590, 0.0057578, -0.0185340, 0.0180184
8: 0.9787275, 0.9927657, 0.9788877, 0.9932699, -0.0130557, 0.0126925
9: -0.0093204, 0.0034225, -0.0097781, 0.0032770, -0.0115215, 0.0118511

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0279484, upper bound: 0.0272597
time: 1.70 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0272636, upper bound: 0.0271435
time: 1.61 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0003593, 0.0066128, 0.0002795, 0.0064952, -0.0055425, 0.0057576
1: 0.0013742, 0.0022777, 0.0013627, 0.0022607, -0.0008007, 0.0008318
2: 0.0107038, 0.0141612, 0.0107689, 0.0142053, -0.0031832, 0.0030643
3: -0.0036101, -0.0000342, -0.0035428, 0.0000113, -0.0032923, 0.0031693
4: -0.0039999, -0.0001288, -0.0040492, -0.0002017, -0.0034309, 0.0035641
5: 0.0042353, 0.0078986, 0.0043042, 0.0079453, -0.0033728, 0.0032468
6: -0.0054960, 0.0090389, -0.0052226, 0.0092243, -0.0133823, 0.0128824
7: -0.0148670, 0.0049284, -0.0151194, 0.0045559, -0.0175446, 0.0182255
8: 0.9787413, 0.9926855, 0.9785634, 0.9924232, -0.0123588, 0.0128384
9: -0.0092477, 0.0034100, -0.0090095, 0.0035714, -0.0116539, 0.0112185

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0278586, upper bound: 0.0271078
time: 2.13 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0272488, upper bound: 0.0270314
time: 1.88 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0003593, 0.0066128, 0.0003649, 0.0068411, -0.0058918, 0.0056788
1: 0.0013742, 0.0022777, 0.0013750, 0.0023106, -0.0008512, 0.0008204
2: 0.0107038, 0.0141612, 0.0105776, 0.0141581, -0.0031397, 0.0032574
3: -0.0036101, -0.0000342, -0.0037406, -0.0000375, -0.0032472, 0.0033690
4: -0.0039999, -0.0001288, -0.0039964, 0.0000125, -0.0036471, 0.0035153
5: 0.0042353, 0.0078986, 0.0041015, 0.0078953, -0.0033267, 0.0034514
6: -0.0054960, 0.0090389, -0.0060266, 0.0090258, -0.0131992, 0.0136942
7: -0.0148670, 0.0049284, -0.0148490, 0.0056510, -0.0186503, 0.0179761
8: 0.9787413, 0.9926855, 0.9787539, 0.9931945, -0.0131376, 0.0126628
9: -0.0092477, 0.0034100, -0.0097097, 0.0033985, -0.0114944, 0.0119255

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0278586, upper bound: 0.0271078
time: 1.65 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0272488, upper bound: 0.0270314
time: 1.53 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 4.49 seconds
IS_A2_B2_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.49
Output dim: 8, lower bound: -0.0279484, upper bound: 0.0272597
IS_A2_B2_A2_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 4.49
Output dim: 8, lower bound: -0.0272636, upper bound: 0.0271435
IS_A2_B2_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.49
Output dim: 8, lower bound: -0.0279484, upper bound: 0.0272597
IS_A2_B2_A2_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 4.49
Output dim: 8, lower bound: -0.0272636, upper bound: 0.0271435
IS_A2_B2_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.49
Output dim: 8, lower bound: -0.0278586, upper bound: 0.0271078
IS_A2_B2_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 4.49
Output dim: 8, lower bound: -0.0272488, upper bound: 0.0270314
IS_A2_B2_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.49
Output dim: 8, lower bound: -0.0278586, upper bound: 0.0271078
IS_A2_B2_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 4.49
Output dim: 8, lower bound: -0.0272488, upper bound: 0.0270314

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 4.18 + 427.44 = 431.62 seconds
