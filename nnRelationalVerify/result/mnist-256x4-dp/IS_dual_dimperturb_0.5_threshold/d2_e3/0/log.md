## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.04893569308


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0398926, 0.0250191, -0.0398926, 0.0250191, -0.0649118, 0.0649118)
1: (-0.0276871, 0.0250473, -0.0276871, 0.0250473, -0.0527345, 0.0527345)
2: (-0.0215551, 0.0600055, -0.0215551, 0.0600055, -0.0815607, 0.0815607)
3: (-0.0153743, 0.0370634, -0.0153743, 0.0370634, -0.0524377, 0.0524377)
4: (-0.0395439, 0.0351036, -0.0395439, 0.0351036, -0.0746475, 0.0746475)
5: (-0.0202170, 0.0789822, -0.0202170, 0.0789822, -0.0991993, 0.0991993)
6: (-0.0253401, 0.0288020, -0.0253401, 0.0288020, -0.0541420, 0.0541420)
7: (-0.0575527, 0.0263384, -0.0575527, 0.0263384, -0.0838911, 0.0838911)
8: (0.8489331, 1.0140631, 0.8489331, 1.0140631, -0.1651300, 0.1651300)
9: (-0.0242065, 0.1062990, -0.0242065, 0.1062990, -0.1305055, 0.1305055)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.35 + 2.64 = 3.99 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.1335548, upper bound: 0.1335548

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1305388, upper bound: 0.1286886
time: 1.30 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1286803, upper bound: 0.1286803
time: 1.28 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.72 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.72
Output dim: 8, lower bound: -0.1305388, upper bound: 0.1286886
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.72
Output dim: 8, lower bound: -0.1286803, upper bound: 0.1286803

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0344434, 0.0118043, -0.0390800, 0.0229062, -0.0573496, 0.0508843
1: -0.0217098, 0.0159400, -0.0267407, 0.0236876, -0.0453974, 0.0426806
2: -0.0128664, 0.0496545, -0.0202558, 0.0584141, -0.0712806, 0.0699104
3: -0.0129082, 0.0268208, -0.0148744, 0.0355375, -0.0484456, 0.0416953
4: -0.0319459, 0.0270046, -0.0384097, 0.0338749, -0.0658208, 0.0654143
5: -0.0161501, 0.0662604, -0.0195061, 0.0769624, -0.0931125, 0.0857666
6: -0.0157710, 0.0251687, -0.0238347, 0.0282625, -0.0440335, 0.0490034
7: -0.0454779, 0.0195187, -0.0556933, 0.0253190, -0.0707969, 0.0752120
8: 0.8709604, 1.0136979, 0.8522841, 1.0140086, -0.1430483, 0.1614137
9: -0.0162956, 0.0897103, -0.0229747, 0.1038222, -0.1201178, 0.1126851

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=22, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1279912, upper bound: 0.1263679
time: 1.46 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1279500, upper bound: 0.1259962
time: 1.46 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0301989, 0.0084538, -0.0357671, 0.0142497, -0.0444486, 0.0442209
1: -0.0168175, 0.0167089, -0.0228487, 0.0180581, -0.0348756, 0.0395576
2: -0.0092240, 0.0460262, -0.0150517, 0.0519274, -0.0611513, 0.0610779
3: -0.0135696, 0.0237774, -0.0132787, 0.0293365, -0.0429061, 0.0370562
4: -0.0314688, 0.0227356, -0.0337483, 0.0288653, -0.0603341, 0.0564840
5: -0.0204378, 0.0583873, -0.0170836, 0.0692588, -0.0896967, 0.0754709
6: -0.0114492, 0.0527692, -0.0176532, 0.0263086, -0.0377578, 0.0704224
7: -0.0396300, 0.0123938, -0.0479732, 0.0210727, -0.0607026, 0.0603669
8: 0.8597847, 1.0152612, 0.8651496, 1.0138948, -0.1541101, 0.1501116
9: -0.0214782, 0.0777000, -0.0182789, 0.0937439, -0.1152221, 0.0959789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=22, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1263592, upper bound: 0.1261673
time: 1.38 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1259916, upper bound: 0.1259916
time: 1.40 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.15 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 4.15
Output dim: 8, lower bound: -0.1279912, upper bound: 0.1263679
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 4.15
Output dim: 8, lower bound: -0.1279500, upper bound: 0.1259962
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.15
Output dim: 8, lower bound: -0.1263592, upper bound: 0.1261673
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.15
Output dim: 8, lower bound: -0.1259916, upper bound: 0.1259916

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -0.0283627, 0.0052712, -0.0390800, 0.0229062, -0.0512690, 0.0443512
1: -0.0171391, 0.0112086, -0.0267407, 0.0236876, -0.0408266, 0.0379493
2: -0.0027906, 0.0410275, -0.0202558, 0.0584141, -0.0612048, 0.0612833
3: -0.0115621, 0.0174708, -0.0148744, 0.0355375, -0.0470995, 0.0323452
4: -0.0241775, 0.0212617, -0.0384097, 0.0338749, -0.0580523, 0.0596714
5: -0.0129021, 0.0448202, -0.0195061, 0.0769624, -0.0898645, 0.0643263
6: -0.0108887, 0.0153921, -0.0238347, 0.0282625, -0.0391512, 0.0392268
7: -0.0401489, 0.0140886, -0.0556933, 0.0253190, -0.0654679, 0.0697819
8: 0.9087639, 1.0130953, 0.8522841, 1.0140086, -0.1052447, 0.1608111
9: -0.0054042, 0.0711022, -0.0229747, 0.1038222, -0.1092265, 0.0940769

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=22, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1279912, upper bound: 0.1263679
time: 1.44 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1279912, upper bound: 0.1263679
time: 1.32 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -0.0256752, 0.0039049, -0.0356230, 0.0145238, -0.0401989, 0.0395279
1: -0.0149258, 0.0097488, -0.0231666, 0.0184710, -0.0333968, 0.0329153
2: 0.0008055, 0.0376480, -0.0139352, 0.0515456, -0.0507401, 0.0515833
3: -0.0104017, 0.0143854, -0.0128768, 0.0286713, -0.0390730, 0.0272622
4: -0.0207179, 0.0183405, -0.0333104, 0.0292143, -0.0499321, 0.0516509
5: -0.0102629, 0.0380660, -0.0157794, 0.0641333, -0.0743962, 0.0538454
6: -0.0087661, 0.0135641, -0.0183340, 0.0201839, -0.0289501, 0.0318981
7: -0.0379869, 0.0108813, -0.0494273, 0.0220844, -0.0600713, 0.0603085
8: 0.9198378, 1.0135031, 0.8765277, 1.0134892, -0.0936515, 0.1369754
9: -0.0053238, 0.0628072, -0.0149502, 0.0930040, -0.0983278, 0.0777574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=22, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1279500, upper bound: 0.1259962
time: 1.24 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1279500, upper bound: 0.1259962
time: 1.31 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0301989, 0.0084538, -0.0295691, 0.0062243, -0.0364231, 0.0380229
1: -0.0168175, 0.0167089, -0.0181476, 0.0118456, -0.0286631, 0.0348565
2: -0.0092240, 0.0460262, -0.0043370, 0.0424792, -0.0517032, 0.0503632
3: -0.0135696, 0.0237774, -0.0118206, 0.0187949, -0.0323645, 0.0355981
4: -0.0314688, 0.0227356, -0.0257971, 0.0223213, -0.0537901, 0.0485328
5: -0.0204378, 0.0583873, -0.0133590, 0.0475217, -0.0679595, 0.0717463
6: -0.0114492, 0.0527692, -0.0117602, 0.0161701, -0.0276193, 0.0645294
7: -0.0396300, 0.0123938, -0.0411506, 0.0155392, -0.0551692, 0.0535443
8: 0.8597847, 1.0152612, 0.9041864, 1.0132862, -0.1535015, 0.1110748
9: -0.0214782, 0.0777000, -0.0061321, 0.0747284, -0.0962066, 0.0838322

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1160391, upper bound: 0.1226163
time: 1.30 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1238108, upper bound: 0.1235380
time: 1.33 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0263035, 0.0053601, -0.0270953, 0.0042992, -0.0306026, 0.0324555
1: -0.0145479, 0.0127356, -0.0161284, 0.0105017, -0.0250496, 0.0288639
2: -0.0030448, 0.0405761, -0.0009516, 0.0393342, -0.0423789, 0.0415277
3: -0.0126330, 0.0185921, -0.0106524, 0.0158850, -0.0285180, 0.0292445
4: -0.0239832, 0.0197722, -0.0226791, 0.0195784, -0.0435616, 0.0424513
5: -0.0172459, 0.0474765, -0.0107386, 0.0408899, -0.0581358, 0.0582151
6: -0.0093038, 0.0349055, -0.0097941, 0.0144533, -0.0237571, 0.0446995
7: -0.0373645, 0.0091802, -0.0391881, 0.0126564, -0.0500209, 0.0483683
8: 0.8906958, 1.0147502, 0.9148760, 1.0136831, -0.1229873, 0.0998742
9: -0.0125087, 0.0662156, -0.0054505, 0.0670621, -0.0795708, 0.0716660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1259916, upper bound: 0.1259916
time: 1.36 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1259916, upper bound: 0.1259916
time: 1.09 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.99 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.99
Output dim: 8, lower bound: -0.1279912, upper bound: 0.1263679
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.99
Output dim: 8, lower bound: -0.1279912, upper bound: 0.1263679
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.99
Output dim: 8, lower bound: -0.1279500, upper bound: 0.1259962
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.99
Output dim: 8, lower bound: -0.1279500, upper bound: 0.1259962
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.99
Output dim: 8, lower bound: -0.1160391, upper bound: 0.1226163
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.99
Output dim: 8, lower bound: -0.1238108, upper bound: 0.1235380
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.99
Output dim: 8, lower bound: -0.1259916, upper bound: 0.1259916
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.99
Output dim: 8, lower bound: -0.1259916, upper bound: 0.1259916

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0283627, 0.0052712, -0.0344434, 0.0118043, -0.0401670, 0.0397147
1: -0.0171391, 0.0112086, -0.0217098, 0.0159400, -0.0330790, 0.0329184
2: -0.0027906, 0.0410275, -0.0128664, 0.0496545, -0.0524452, 0.0538939
3: -0.0115621, 0.0174708, -0.0129082, 0.0268208, -0.0383829, 0.0303790
4: -0.0241775, 0.0212617, -0.0319459, 0.0270046, -0.0511821, 0.0532076
5: -0.0129021, 0.0448202, -0.0161501, 0.0662604, -0.0791626, 0.0609703
6: -0.0108887, 0.0153921, -0.0157710, 0.0251687, -0.0360574, 0.0311632
7: -0.0401489, 0.0140886, -0.0454779, 0.0195187, -0.0596676, 0.0595665
8: 0.9087639, 1.0130953, 0.8709604, 1.0136979, -0.1049339, 0.1421349
9: -0.0054042, 0.0711022, -0.0162956, 0.0897103, -0.0951146, 0.0873977

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1242831, upper bound: 0.1160525
time: 1.33 seconds

## Relational analysis of IS_A1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1255804, upper bound: 0.1238290
time: 1.28 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0283627, 0.0052712, -0.0301989, 0.0084538, -0.0368165, 0.0354701
1: -0.0171391, 0.0112086, -0.0168175, 0.0167089, -0.0338480, 0.0280261
2: -0.0027906, 0.0410275, -0.0092240, 0.0460262, -0.0488168, 0.0502515
3: -0.0115621, 0.0174708, -0.0135696, 0.0237774, -0.0353395, 0.0310404
4: -0.0241775, 0.0212617, -0.0314688, 0.0227356, -0.0469131, 0.0527305
5: -0.0129021, 0.0448202, -0.0204378, 0.0583873, -0.0712894, 0.0652580
6: -0.0108887, 0.0153921, -0.0114492, 0.0527692, -0.0636579, 0.0268413
7: -0.0401489, 0.0140886, -0.0396300, 0.0123938, -0.0525426, 0.0537185
8: 0.9087639, 1.0130953, 0.8597847, 1.0152612, -0.1064972, 0.1533105
9: -0.0054042, 0.0711022, -0.0214782, 0.0777000, -0.0831043, 0.0925803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1242831, upper bound: 0.1160525
time: 1.21 seconds

## Relational analysis of IS_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1255804, upper bound: 0.1238290
time: 1.13 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0256752, 0.0039049, -0.0309964, 0.0076272, -0.0333024, 0.0349014
1: -0.0149258, 0.0097488, -0.0191940, 0.0124573, -0.0273831, 0.0289428
2: 0.0008055, 0.0376480, -0.0071482, 0.0446808, -0.0438753, 0.0447963
3: -0.0104017, 0.0143854, -0.0118823, 0.0212389, -0.0316407, 0.0262677
4: -0.0207179, 0.0183405, -0.0268881, 0.0234310, -0.0441489, 0.0452285
5: -0.0102629, 0.0380660, -0.0134199, 0.0555945, -0.0658573, 0.0514859
6: -0.0087661, 0.0135641, -0.0127294, 0.0175040, -0.0262702, 0.0262935
7: -0.0379869, 0.0108813, -0.0420187, 0.0162618, -0.0542487, 0.0529000
8: 0.9198378, 1.0135031, 0.8929895, 1.0131890, -0.0933512, 0.1205136
9: -0.0053238, 0.0628072, -0.0094077, 0.0790628, -0.0843865, 0.0722150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1242423, upper bound: 0.1159503
time: 1.19 seconds

## Relational analysis of IS_A1_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1255306, upper bound: 0.1234351
time: 1.16 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0256752, 0.0039049, -0.0249345, 0.0044222, -0.0300974, 0.0288394
1: -0.0149258, 0.0097488, -0.0143362, 0.0093659, -0.0242917, 0.0240849
2: 0.0008055, 0.0376480, -0.0005631, 0.0380501, -0.0372447, 0.0382112
3: -0.0104017, 0.0143854, -0.0125424, 0.0161634, -0.0265651, 0.0269278
4: -0.0207179, 0.0183405, -0.0183352, 0.0195955, -0.0403134, 0.0366757
5: -0.0102629, 0.0380660, -0.0154458, 0.0462012, -0.0564641, 0.0535118
6: -0.0087661, 0.0135641, -0.0090709, 0.0141083, -0.0228744, 0.0226351
7: -0.0379869, 0.0108813, -0.0369499, 0.0085476, -0.0465345, 0.0478312
8: 0.9198378, 1.0135031, 0.9107170, 1.0147042, -0.0948665, 0.1027861
9: -0.0053238, 0.0628072, -0.0059440, 0.0618375, -0.0671613, 0.0687513

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1182898, upper bound: 0.1200531
time: 1.18 seconds

## Relational analysis of IS_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1172856, upper bound: 0.1154684
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0033211, 0.0038597, -0.0231660, 0.0037976, -0.0071187, 0.0270257
1: -0.0039174, 0.0065713, -0.0131559, 0.0084956, -0.0124130, 0.0197272
2: 0.0079075, 0.0218855, 0.0035078, 0.0347463, -0.0268388, 0.0183776
3: -0.0109692, 0.0054274, -0.0108671, 0.0126413, -0.0236105, 0.0162945
4: -0.0086862, 0.0090956, -0.0175362, 0.0169365, -0.0256227, 0.0266317
5: -0.0127789, 0.0150990, -0.0117263, 0.0336006, -0.0463795, 0.0268252
6: -0.0003169, 0.0116918, -0.0073739, 0.0121109, -0.0124279, 0.0190657
7: -0.0270863, -0.0029260, -0.0361180, 0.0077763, -0.0348625, 0.0331920
8: 0.9485660, 1.0140324, 0.9278288, 1.0130222, -0.0644562, 0.0862036
9: -0.0046747, 0.0203464, -0.0048891, 0.0557170, -0.0603917, 0.0252355

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1160391, upper bound: 0.1169204
time: 1.37 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1160391, upper bound: 0.1226163
time: 1.49 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0179518, 0.0041029, -0.0257301, 0.0038384, -0.0217902, 0.0298329
1: -0.0103998, 0.0072695, -0.0150061, 0.0097987, -0.0201986, 0.0222755
2: 0.0065886, 0.0294104, 0.0009566, 0.0375582, -0.0309695, 0.0284538
3: -0.0104946, 0.0106155, -0.0103903, 0.0142399, -0.0247345, 0.0210058
4: -0.0112466, 0.0143567, -0.0209843, 0.0184019, -0.0296485, 0.0353410
5: -0.0102437, 0.0284399, -0.0103252, 0.0370808, -0.0473245, 0.0387651
6: -0.0043548, 0.0120345, -0.0087910, 0.0134994, -0.0178543, 0.0208255
7: -0.0329334, 0.0004320, -0.0381048, 0.0111608, -0.0440941, 0.0385367
8: 0.9383868, 1.0181904, 0.9209781, 1.0124488, -0.0740620, 0.0972123
9: -0.0065009, 0.0423704, -0.0049322, 0.0628989, -0.0693998, 0.0473026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1234147, upper bound: 0.1169204
time: 1.26 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1234147, upper bound: 0.1235380
time: 1.40 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0263035, 0.0053601, -0.0256752, 0.0039049, -0.0302084, 0.0310353
1: -0.0145479, 0.0127356, -0.0149258, 0.0097488, -0.0242966, 0.0276614
2: -0.0030448, 0.0405761, 0.0008055, 0.0376480, -0.0406928, 0.0397707
3: -0.0126330, 0.0185921, -0.0104017, 0.0143854, -0.0270184, 0.0289939
4: -0.0239832, 0.0197722, -0.0207179, 0.0183405, -0.0423236, 0.0404901
5: -0.0172459, 0.0474765, -0.0102629, 0.0380660, -0.0553119, 0.0577393
6: -0.0093038, 0.0349055, -0.0087661, 0.0135641, -0.0228679, 0.0436716
7: -0.0373645, 0.0091802, -0.0379869, 0.0108813, -0.0482457, 0.0471672
8: 0.8906958, 1.0147502, 0.9198378, 1.0135031, -0.1228073, 0.0949125
9: -0.0125087, 0.0662156, -0.0053238, 0.0628072, -0.0753159, 0.0715393

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1159460, upper bound: 0.1225889
time: 5.47 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1234250, upper bound: 0.1234250
time: 1.20 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0263035, 0.0053601, -0.0200224, 0.0039166, -0.0302200, 0.0253825
1: -0.0145479, 0.0127356, -0.0113016, 0.0072315, -0.0217793, 0.0240372
2: -0.0030448, 0.0405761, 0.0057362, 0.0313499, -0.0343947, 0.0348399
3: -0.0126330, 0.0185921, -0.0112181, 0.0114130, -0.0240460, 0.0298103
4: -0.0239832, 0.0197722, -0.0131724, 0.0160236, -0.0400067, 0.0329446
5: -0.0172459, 0.0474765, -0.0126786, 0.0304634, -0.0477093, 0.0601550
6: -0.0093038, 0.0349055, -0.0057513, 0.0117873, -0.0210911, 0.0406568
7: -0.0373645, 0.0091802, -0.0338370, 0.0033192, -0.0406837, 0.0430173
8: 0.8906958, 1.0147502, 0.9348752, 1.0149735, -0.1242778, 0.0798750
9: -0.0125087, 0.0662156, -0.0053377, 0.0467713, -0.0592801, 0.0715533

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1159460, upper bound: 0.1225889
time: 1.25 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1234250, upper bound: 0.1234250
time: 1.32 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.99 seconds
IS_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 8, lower bound: -0.1242831, upper bound: 0.1160525
IS_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 8, lower bound: -0.1255804, upper bound: 0.1238290
IS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 8, lower bound: -0.1242831, upper bound: 0.1160525
IS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 8, lower bound: -0.1255804, upper bound: 0.1238290
IS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 8, lower bound: -0.1242423, upper bound: 0.1159503
IS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 8, lower bound: -0.1255306, upper bound: 0.1234351
IS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 8, lower bound: -0.1182898, upper bound: 0.1200531
IS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 8, lower bound: -0.1172856, upper bound: 0.1154684
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 8, lower bound: -0.1160391, upper bound: 0.1169204
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 8, lower bound: -0.1160391, upper bound: 0.1226163
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 8, lower bound: -0.1234147, upper bound: 0.1169204
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 8, lower bound: -0.1234147, upper bound: 0.1235380
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 8, lower bound: -0.1159460, upper bound: 0.1225889
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 8, lower bound: -0.1234250, upper bound: 0.1234250
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 8, lower bound: -0.1159460, upper bound: 0.1225889
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 8, lower bound: -0.1234250, upper bound: 0.1234250

## BFS IS instance: IS_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0222442, 0.0037409, -0.0072251, 0.0036251, -0.0258320, 0.0109660
1: -0.0126182, 0.0079720, -0.0059727, 0.0063467, -0.0189649, 0.0139447
2: 0.0041589, 0.0337156, 0.0079692, 0.0223770, -0.0182181, 0.0257464
3: -0.0106639, 0.0122839, -0.0097229, 0.0070658, -0.0177297, 0.0220068
4: -0.0162607, 0.0164429, -0.0086490, 0.0094077, -0.0256684, 0.0250919
5: -0.0114064, 0.0326907, -0.0099315, 0.0188024, -0.0302087, 0.0426222
6: -0.0067988, 0.0117115, 0.0000636, 0.0116621, -0.0184608, 0.0116479
7: -0.0354565, 0.0063591, -0.0289394, -0.0035058, -0.0319507, 0.0352985
8: 0.9298722, 1.0128350, 0.9459007, 1.0124959, -0.0826237, 0.0669343
9: -0.0048406, 0.0529821, -0.0045467, 0.0266323, -0.0314729, 0.0575288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_A1_B1_B1_A1

### Relational analysis result of IS_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1200982, upper bound: 0.1190862
time: 1.11 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2

### Relational analysis result of IS_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1200982, upper bound: 0.1190862
time: 1.51 seconds

## BFS IS instance: IS_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0244792, 0.0036958, -0.0231080, 0.0039024, -0.0283817, 0.0268038
1: -0.0140101, 0.0091251, -0.0131813, 0.0082304, -0.0222405, 0.0223064
2: 0.0024070, 0.0361006, 0.0030613, 0.0348265, -0.0324196, 0.0330393
3: -0.0101658, 0.0131392, -0.0093704, 0.0131248, -0.0232906, 0.0225095
4: -0.0192980, 0.0172916, -0.0171948, 0.0156432, -0.0349412, 0.0344863
5: -0.0099007, 0.0348757, -0.0072567, 0.0354626, -0.0453633, 0.0421324
6: -0.0079236, 0.0127376, -0.0069304, 0.0122355, -0.0201591, 0.0196680
7: -0.0370921, 0.0095397, -0.0360463, 0.0069290, -0.0440212, 0.0455860
8: 0.9249642, 1.0122606, 0.9255506, 1.0166478, -0.0916836, 0.0867100
9: -0.0048385, 0.0592071, -0.0064350, 0.0551394, -0.0599779, 0.0656421

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_A1_B1_B2_A1

### Relational analysis result of IS_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1200982, upper bound: 0.1272000
time: 1.27 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2

### Relational analysis result of IS_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1200982, upper bound: 0.1277441
time: 1.38 seconds

## BFS IS instance: IS_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0222442, 0.0037409, -0.0033211, 0.0038597, -0.0261039, 0.0070619
1: -0.0126182, 0.0079720, -0.0039174, 0.0065713, -0.0191895, 0.0118894
2: 0.0041589, 0.0337156, 0.0079075, 0.0218855, -0.0177266, 0.0258081
3: -0.0106639, 0.0122839, -0.0109692, 0.0054274, -0.0160913, 0.0232531
4: -0.0162607, 0.0164429, -0.0086862, 0.0090956, -0.0253563, 0.0251291
5: -0.0114064, 0.0326907, -0.0127789, 0.0150990, -0.0265053, 0.0454697
6: -0.0067988, 0.0117115, -0.0003169, 0.0116918, -0.0184905, 0.0120284
7: -0.0354565, 0.0063591, -0.0270863, -0.0029260, -0.0325305, 0.0334453
8: 0.9298722, 1.0128350, 0.9485660, 1.0140324, -0.0841602, 0.0642690
9: -0.0048406, 0.0529821, -0.0046747, 0.0203464, -0.0251870, 0.0576568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_A1_B2_B1_A1

### Relational analysis result of IS_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1186079, upper bound: 0.1160525
time: 1.47 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2

### Relational analysis result of IS_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1186079, upper bound: 0.1160525
time: 1.29 seconds

## BFS IS instance: IS_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0244792, 0.0036958, -0.0179518, 0.0041029, -0.0285821, 0.0216476
1: -0.0140101, 0.0091251, -0.0103998, 0.0072695, -0.0212796, 0.0195250
2: 0.0024070, 0.0361006, 0.0065886, 0.0294104, -0.0270034, 0.0295120
3: -0.0101658, 0.0131392, -0.0104946, 0.0106155, -0.0207814, 0.0236338
4: -0.0192980, 0.0172916, -0.0112466, 0.0143567, -0.0336548, 0.0285381
5: -0.0099007, 0.0348757, -0.0102437, 0.0284399, -0.0383406, 0.0451194
6: -0.0079236, 0.0127376, -0.0043548, 0.0120345, -0.0199581, 0.0170924
7: -0.0370921, 0.0095397, -0.0329334, 0.0004320, -0.0375241, 0.0424731
8: 0.9249642, 1.0122606, 0.9383868, 1.0181904, -0.0932262, 0.0738738
9: -0.0048385, 0.0592071, -0.0065009, 0.0423704, -0.0472089, 0.0657080

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_A1_B2_B2_A1

### Relational analysis result of IS_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1186079, upper bound: 0.1234454
time: 1.12 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2

### Relational analysis result of IS_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1186079, upper bound: 0.1238290
time: 1.23 seconds

## BFS IS instance: IS_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0203205, 0.0036477, -0.0055109, 0.0035379, -0.0238251, 0.0091585
1: -0.0115080, 0.0069606, -0.0051597, 0.0062618, -0.0177698, 0.0121203
2: 0.0055126, 0.0315145, 0.0080066, 0.0216483, -0.0161356, 0.0235079
3: -0.0096371, 0.0115435, -0.0091529, 0.0064133, -0.0160504, 0.0206964
4: -0.0136078, 0.0147863, -0.0086134, 0.0079961, -0.0216039, 0.0233997
5: -0.0090001, 0.0308124, -0.0084758, 0.0172388, -0.0262388, 0.0392882
6: -0.0053675, 0.0117563, 0.0006027, 0.0116580, -0.0170256, 0.0111536
7: -0.0340909, 0.0032326, -0.0282064, -0.0037373, -0.0303536, 0.0314390
8: 0.9340908, 1.0132447, 0.9470538, 1.0119916, -0.0779008, 0.0661909
9: -0.0051416, 0.0470409, -0.0045112, 0.0234765, -0.0286181, 0.0515522

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_A2_B1_B1_A1

### Relational analysis result of IS_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1189821, upper bound: 0.1189821
time: 1.54 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2

### Relational analysis result of IS_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1189821, upper bound: 0.1189821
time: 1.18 seconds

## BFS IS instance: IS_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0225371, 0.0035790, -0.0209120, 0.0038041, -0.0263412, 0.0244909
1: -0.0128153, 0.0079432, -0.0118779, 0.0071466, -0.0199619, 0.0198211
2: 0.0039414, 0.0339391, 0.0050848, 0.0320811, -0.0281397, 0.0288542
3: -0.0092183, 0.0124095, -0.0087201, 0.0117842, -0.0210024, 0.0211296
4: -0.0166845, 0.0153754, -0.0144434, 0.0140419, -0.0307264, 0.0298188
5: -0.0076848, 0.0330242, -0.0058374, 0.0314382, -0.0391230, 0.0388616
6: -0.0065313, 0.0117877, -0.0053698, 0.0120103, -0.0185416, 0.0171575
7: -0.0356990, 0.0064457, -0.0345459, 0.0038111, -0.0395100, 0.0409916
8: 0.9291230, 1.0127466, 0.9326854, 1.0161886, -0.0870656, 0.0800612
9: -0.0051389, 0.0533699, -0.0063148, 0.0483558, -0.0534947, 0.0596847

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_A2_B1_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1189821, upper bound: 0.1263716
time: 1.25 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1189821, upper bound: 0.1263716
time: 1.18 seconds

## BFS IS instance: IS_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0256752, 0.0039049, -0.0244263, 0.0040301, -0.0297053, 0.0283312
1: -0.0149258, 0.0097488, -0.0140498, 0.0089372, -0.0238630, 0.0237985
2: 0.0008055, 0.0376480, -0.0000440, 0.0373316, -0.0365262, 0.0376920
3: -0.0104017, 0.0143854, -0.0109903, 0.0158123, -0.0262140, 0.0253757
4: -0.0207179, 0.0183405, -0.0177394, 0.0183563, -0.0390742, 0.0360799
5: -0.0102629, 0.0380660, -0.0121268, 0.0450991, -0.0553620, 0.0501928
6: -0.0087661, 0.0135641, -0.0083641, 0.0137848, -0.0225509, 0.0219282
7: -0.0379869, 0.0108813, -0.0366306, 0.0075932, -0.0455801, 0.0475119
8: 0.9198378, 1.0135031, 0.9125261, 1.0126464, -0.0928087, 0.1009769
9: -0.0053238, 0.0628072, -0.0055209, 0.0599475, -0.0652713, 0.0683281

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_A2_B2_B1_B1

### Relational analysis result of IS_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1126806, upper bound: 0.1115017
time: 1.14 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2

### Relational analysis result of IS_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1126806, upper bound: 0.1152694
time: 1.11 seconds

## BFS IS instance: IS_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0253566, 0.0036561, -0.0522889, 0.0157735, -0.0411301, 0.0559450
1: -0.0146736, 0.0094929, -0.0307357, 0.0246762, -0.0393498, 0.0402286
2: 0.0011767, 0.0372220, -0.0248688, 0.0700028, -0.0688261, 0.0620908
3: -0.0094576, 0.0140755, -0.0173666, 0.0319212, -0.0413788, 0.0314421
4: -0.0202817, 0.0175179, -0.0536265, 0.0316563, -0.0519380, 0.0711444
5: -0.0082536, 0.0375097, -0.0125833, 0.0941620, -0.1024156, 0.0500930
6: -0.0083365, 0.0133469, -0.0252834, 0.0276105, -0.0359471, 0.0386304
7: -0.0377336, 0.0103098, -0.0560055, 0.0463036, -0.0840372, 0.0663153
8: 0.9208320, 1.0120943, 0.8250294, 1.0106617, -0.0898297, 0.1870648
9: -0.0050473, 0.0616337, -0.0270296, 0.1414998, -0.1465471, 0.0886633

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1116318, upper bound: 0.1083416
time: 1.69 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1116316, upper bound: 0.1090537
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0033211, 0.0038597, -0.0053073, 0.0035574, -0.0068299, 0.0091669
1: -0.0039174, 0.0065713, -0.0050478, 0.0062834, -0.0102008, 0.0116191
2: 0.0079075, 0.0218855, 0.0079911, 0.0216298, -0.0137223, 0.0138944
3: -0.0109692, 0.0054274, -0.0092329, 0.0063237, -0.0172929, 0.0146603
4: -0.0086862, 0.0090956, -0.0086246, 0.0079589, -0.0166451, 0.0177202
5: -0.0127789, 0.0150990, -0.0086537, 0.0170461, -0.0298250, 0.0237526
6: -0.0003169, 0.0116918, 0.0005652, 0.0116629, -0.0119799, 0.0111265
7: -0.0270863, -0.0029260, -0.0281055, -0.0036815, -0.0234047, 0.0251795
8: 0.9485660, 1.0140324, 0.9471899, 1.0121381, -0.0635721, 0.0668424
9: -0.0046747, 0.0203464, -0.0045350, 0.0230722, -0.0272009, 0.0248814

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1117144, upper bound: 0.1105269
time: 1.13 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1084311, upper bound: 0.1100302
time: 1.15 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0033211, 0.0038597, -0.0205847, 0.0038100, -0.0071310, 0.0244444
1: -0.0039174, 0.0065713, -0.0116877, 0.0070842, -0.0110016, 0.0182590
2: 0.0079075, 0.0218855, 0.0053157, 0.0317080, -0.0238005, 0.0165698
3: -0.0109692, 0.0054274, -0.0086948, 0.0116576, -0.0226268, 0.0141222
4: -0.0086862, 0.0090956, -0.0139912, 0.0138274, -0.0225136, 0.0230867
5: -0.0127789, 0.0150990, -0.0057759, 0.0311165, -0.0438954, 0.0208749
6: -0.0003169, 0.0116918, -0.0051490, 0.0120133, -0.0123303, 0.0168407
7: -0.0270863, -0.0029260, -0.0343120, 0.0032988, -0.0303851, 0.0313860
8: 0.9485660, 1.0140324, 0.9334079, 1.0162596, -0.0676935, 0.0806245
9: -0.0046747, 0.0203464, -0.0063339, 0.0473633, -0.0520380, 0.0266802

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1117144, upper bound: 0.1114326
time: 1.13 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1084311, upper bound: 0.1107404
time: 1.06 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0179518, 0.0041029, -0.0053073, 0.0035574, -0.0214731, 0.0094101
1: -0.0103998, 0.0072695, -0.0050478, 0.0062834, -0.0166832, 0.0123173
2: 0.0065886, 0.0294104, 0.0079911, 0.0216298, -0.0150412, 0.0214193
3: -0.0104946, 0.0106155, -0.0092329, 0.0063237, -0.0168184, 0.0198485
4: -0.0112466, 0.0143567, -0.0086246, 0.0079589, -0.0192055, 0.0229813
5: -0.0102437, 0.0284399, -0.0086537, 0.0170461, -0.0272897, 0.0370936
6: -0.0043548, 0.0120345, 0.0005652, 0.0116629, -0.0160178, 0.0114693
7: -0.0329334, 0.0004320, -0.0281055, -0.0036815, -0.0292518, 0.0285375
8: 0.9383868, 1.0181904, 0.9471899, 1.0121381, -0.0737513, 0.0710005
9: -0.0065009, 0.0423704, -0.0045350, 0.0230722, -0.0295731, 0.0469054

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1160503, upper bound: 0.1104035
time: 1.36 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1091634, upper bound: 0.1096392
time: 1.93 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0179518, 0.0041029, -0.0205847, 0.0038100, -0.0217618, 0.0246876
1: -0.0103998, 0.0072695, -0.0116877, 0.0070842, -0.0174841, 0.0189572
2: 0.0065886, 0.0294104, 0.0053157, 0.0317080, -0.0251194, 0.0240947
3: -0.0104946, 0.0106155, -0.0086948, 0.0116576, -0.0221523, 0.0193104
4: -0.0112466, 0.0143567, -0.0139912, 0.0138274, -0.0250740, 0.0283479
5: -0.0102437, 0.0284399, -0.0057759, 0.0311165, -0.0413602, 0.0342158
6: -0.0043548, 0.0120345, -0.0051490, 0.0120133, -0.0163682, 0.0171835
7: -0.0329334, 0.0004320, -0.0343120, 0.0032988, -0.0362322, 0.0347439
8: 0.9383868, 1.0181904, 0.9334079, 1.0162596, -0.0778728, 0.0847825
9: -0.0065009, 0.0423704, -0.0063339, 0.0473633, -0.0538642, 0.0487043

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1160503, upper bound: 0.1110161
time: 1.21 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1091634, upper bound: 0.1099232
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0018165, 0.0037732, -0.0203205, 0.0036477, -0.0054642, 0.0240938
1: -0.0030981, 0.0064890, -0.0115080, 0.0069606, -0.0100587, 0.0179970
2: 0.0079450, 0.0216626, 0.0055126, 0.0315145, -0.0235695, 0.0161500
3: -0.0104089, 0.0047804, -0.0096371, 0.0115435, -0.0219524, 0.0144175
4: -0.0086661, 0.0077916, -0.0136078, 0.0147863, -0.0234524, 0.0213993
5: -0.0113327, 0.0136738, -0.0090001, 0.0308124, -0.0421450, 0.0226739
6: 0.0001149, 0.0116877, -0.0053675, 0.0117563, -0.0116414, 0.0170553
7: -0.0263402, -0.0031444, -0.0340909, 0.0032326, -0.0295728, 0.0309465
8: 0.9495575, 1.0135293, 0.9340908, 1.0132447, -0.0636872, 0.0794385
9: -0.0046552, 0.0172282, -0.0051416, 0.0470409, -0.0516961, 0.0223698

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B2_B1_A1_B1

### Relational analysis result of IS_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1159503, upper bound: 0.1178807
time: 1.21 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2

### Relational analysis result of IS_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1159503, upper bound: 0.1242423
time: 1.23 seconds

## BFS IS instance: IS_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0160074, 0.0040264, -0.0225371, 0.0035790, -0.0195864, 0.0265313
1: -0.0096048, 0.0071122, -0.0128153, 0.0079432, -0.0175480, 0.0199275
2: 0.0066269, 0.0280350, 0.0039414, 0.0339391, -0.0273121, 0.0240937
3: -0.0099210, 0.0099695, -0.0092183, 0.0124095, -0.0223305, 0.0191878
4: -0.0097058, 0.0129456, -0.0166845, 0.0153754, -0.0250812, 0.0296301
5: -0.0088824, 0.0266991, -0.0076848, 0.0330242, -0.0419066, 0.0343839
6: -0.0030332, 0.0120301, -0.0065313, 0.0117877, -0.0148209, 0.0185614
7: -0.0322144, -0.0010697, -0.0356990, 0.0064457, -0.0386602, 0.0346292
8: 0.9398386, 1.0177325, 0.9291230, 1.0127466, -0.0729080, 0.0886095
9: -0.0064433, 0.0392239, -0.0051389, 0.0533699, -0.0598131, 0.0443628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B2_B1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1225976, upper bound: 0.1178807
time: 1.41 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1225976, upper bound: 0.1178807
time: 1.52 seconds

## BFS IS instance: IS_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0018165, 0.0037732, -0.0150375, 0.0038627, -0.0056741, 0.0188108
1: -0.0030981, 0.0064890, -0.0091887, 0.0067401, -0.0098382, 0.0156777
2: 0.0079450, 0.0216626, 0.0075000, 0.0275250, -0.0195139, 0.0141626
3: -0.0104089, 0.0047804, -0.0107253, 0.0096413, -0.0200502, 0.0155057
4: -0.0086661, 0.0077916, -0.0090938, 0.0134740, -0.0221401, 0.0168854
5: -0.0113327, 0.0136738, -0.0117568, 0.0257944, -0.0371270, 0.0254306
6: 0.0001149, 0.0116877, -0.0028629, 0.0117740, -0.0116591, 0.0145506
7: -0.0263402, -0.0031444, -0.0318392, -0.0018012, -0.0245390, 0.0286948
8: 0.9495575, 1.0135293, 0.9405329, 1.0147133, -0.0651557, 0.0729964
9: -0.0046552, 0.0172282, -0.0052297, 0.0382676, -0.0424607, 0.0224579

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B2_B2_A1_B1

### Relational analysis result of IS_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1159460, upper bound: 0.1159460
time: 1.32 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2

### Relational analysis result of IS_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1159460, upper bound: 0.1225889
time: 1.41 seconds

## BFS IS instance: IS_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0160074, 0.0040264, -0.0173903, 0.0037712, -0.0197229, 0.0213841
1: -0.0096048, 0.0071122, -0.0101688, 0.0067457, -0.0163505, 0.0172810
2: 0.0066269, 0.0280350, 0.0075003, 0.0289781, -0.0223512, 0.0205348
3: -0.0099210, 0.0099695, -0.0102180, 0.0104227, -0.0203436, 0.0201875
4: -0.0097058, 0.0129456, -0.0107971, 0.0140174, -0.0237232, 0.0237427
5: -0.0088824, 0.0266991, -0.0103613, 0.0279253, -0.0368077, 0.0370604
6: -0.0030332, 0.0120301, -0.0039019, 0.0117719, -0.0148051, 0.0159321
7: -0.0322144, -0.0010697, -0.0327230, -0.0005835, -0.0316309, 0.0316533
8: 0.9398386, 1.0177325, 0.9388971, 1.0142077, -0.0743691, 0.0788354
9: -0.0064433, 0.0392239, -0.0052442, 0.0414314, -0.0478746, 0.0444681

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B2_B2_A2_B1

### Relational analysis result of IS_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1225889, upper bound: 0.1159460
time: 1.10 seconds

## Relational analysis of IS_A2_B2_B2_A2_B2

### Relational analysis result of IS_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1225889, upper bound: 0.1159460
time: 1.34 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.84 seconds
IS_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -0.1200982, upper bound: 0.1190862
IS_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -0.1200982, upper bound: 0.1190862
IS_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -0.1200982, upper bound: 0.1272000
IS_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -0.1200982, upper bound: 0.1277441
IS_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -0.1186079, upper bound: 0.1160525
IS_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -0.1186079, upper bound: 0.1160525
IS_A1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -0.1186079, upper bound: 0.1234454
IS_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -0.1186079, upper bound: 0.1238290
IS_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -0.1189821, upper bound: 0.1189821
IS_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -0.1189821, upper bound: 0.1189821
IS_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -0.1189821, upper bound: 0.1263716
IS_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -0.1189821, upper bound: 0.1263716
IS_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -0.1126806, upper bound: 0.1115017
IS_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -0.1126806, upper bound: 0.1152694
IS_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -0.1116318, upper bound: 0.1083416
IS_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -0.1116316, upper bound: 0.1090537
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -0.1117144, upper bound: 0.1105269
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -0.1084311, upper bound: 0.1100302
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -0.1117144, upper bound: 0.1114326
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -0.1084311, upper bound: 0.1107404
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -0.1160503, upper bound: 0.1104035
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -0.1091634, upper bound: 0.1096392
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -0.1160503, upper bound: 0.1110161
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -0.1091634, upper bound: 0.1099232
IS_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -0.1159503, upper bound: 0.1178807
IS_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -0.1159503, upper bound: 0.1242423
IS_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -0.1225976, upper bound: 0.1178807
IS_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -0.1225976, upper bound: 0.1178807
IS_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -0.1159460, upper bound: 0.1159460
IS_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -0.1159460, upper bound: 0.1225889
IS_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -0.1225889, upper bound: 0.1159460
IS_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 8, lower bound: -0.1225889, upper bound: 0.1159460

## BFS IS instance: IS_A1_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0045608, 0.0035324, -0.0072251, 0.0036251, -0.0081498, 0.0106902
1: -0.0046369, 0.0062533, -0.0059727, 0.0063467, -0.0109836, 0.0122260
2: 0.0080222, 0.0215743, 0.0079692, 0.0223770, -0.0143548, 0.0136051
3: -0.0091214, 0.0059949, -0.0097229, 0.0070658, -0.0161872, 0.0157178
4: -0.0086054, 0.0074994, -0.0086490, 0.0094077, -0.0180130, 0.0161484
5: -0.0083864, 0.0163383, -0.0099315, 0.0188024, -0.0271888, 0.0262698
6: 0.0007054, 0.0116568, 0.0000636, 0.0116621, -0.0109567, 0.0115932
7: -0.0277350, -0.0037607, -0.0289394, -0.0035058, -0.0242292, 0.0251787
8: 0.9476901, 1.0119501, 0.9459007, 1.0124959, -0.0648057, 0.0660495
9: -0.0045054, 0.0216250, -0.0045467, 0.0266323, -0.0303338, 0.0259217

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_A1_B1_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1131345, upper bound: 0.1142569
time: 1.23 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_B2

### Relational analysis result of IS_A1_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1127955, upper bound: 0.1113065
time: 1.20 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0195158, 0.0037846, -0.0072251, 0.0036251, -0.0231175, 0.0110097
1: -0.0110764, 0.0069401, -0.0059727, 0.0063467, -0.0174231, 0.0129128
2: 0.0060706, 0.0305190, 0.0079692, 0.0223770, -0.0163063, 0.0225499
3: -0.0085345, 0.0112431, -0.0097229, 0.0070658, -0.0156003, 0.0209660
4: -0.0125515, 0.0132460, -0.0086490, 0.0094077, -0.0219591, 0.0218950
5: -0.0054809, 0.0300612, -0.0099315, 0.0188024, -0.0242833, 0.0399927
6: -0.0044684, 0.0120024, 0.0000636, 0.0116621, -0.0161304, 0.0119389
7: -0.0335687, 0.0016651, -0.0289394, -0.0035058, -0.0300629, 0.0306046
8: 0.9357346, 1.0160751, 0.9459007, 1.0124959, -0.0767613, 0.0701745
9: -0.0062916, 0.0442557, -0.0045467, 0.0266323, -0.0329238, 0.0488024

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_A1_B1_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1131345, upper bound: 0.1142569
time: 1.32 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2_B2

### Relational analysis result of IS_A1_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1127955, upper bound: 0.1113065
time: 1.14 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0045608, 0.0035324, -0.0231080, 0.0039024, -0.0084632, 0.0266063
1: -0.0046369, 0.0062533, -0.0131813, 0.0082304, -0.0128673, 0.0194346
2: 0.0080222, 0.0215743, 0.0030613, 0.0348265, -0.0268043, 0.0185129
3: -0.0091214, 0.0059949, -0.0093704, 0.0131248, -0.0222461, 0.0153653
4: -0.0086054, 0.0074994, -0.0171948, 0.0156432, -0.0242486, 0.0246942
5: -0.0083864, 0.0163383, -0.0072567, 0.0354626, -0.0438491, 0.0235950
6: 0.0007054, 0.0116568, -0.0069304, 0.0122355, -0.0115301, 0.0185872
7: -0.0277350, -0.0037607, -0.0360463, 0.0069290, -0.0346641, 0.0322855
8: 0.9476901, 1.0119501, 0.9255506, 1.0166478, -0.0689577, 0.0863996
9: -0.0045054, 0.0216250, -0.0064350, 0.0551394, -0.0596448, 0.0280600

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1130535, upper bound: 0.1202042
time: 1.58 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B2

### Relational analysis result of IS_A1_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1126266, upper bound: 0.1139029
time: 1.12 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0195158, 0.0037846, -0.0231080, 0.0039024, -0.0234183, 0.0268926
1: -0.0110764, 0.0069401, -0.0131813, 0.0082304, -0.0193068, 0.0201214
2: 0.0060706, 0.0305190, 0.0030613, 0.0348265, -0.0287559, 0.0274577
3: -0.0085345, 0.0112431, -0.0093704, 0.0131248, -0.0216593, 0.0206135
4: -0.0125515, 0.0132460, -0.0171948, 0.0156432, -0.0281947, 0.0304408
5: -0.0054809, 0.0300612, -0.0072567, 0.0354626, -0.0409435, 0.0373179
6: -0.0044684, 0.0120024, -0.0069304, 0.0122355, -0.0167038, 0.0189328
7: -0.0335687, 0.0016651, -0.0360463, 0.0069290, -0.0404977, 0.0377114
8: 0.9357346, 1.0160751, 0.9255506, 1.0166478, -0.0809132, 0.0905246
9: -0.0062916, 0.0442557, -0.0064350, 0.0551394, -0.0614310, 0.0506907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1130535, upper bound: 0.1160501
time: 1.48 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2_B2

### Relational analysis result of IS_A1_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1126266, upper bound: 0.1130722
time: 1.13 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0045608, 0.0035324, -0.0033211, 0.0038597, -0.0084204, 0.0067950
1: -0.0046369, 0.0062533, -0.0039174, 0.0065713, -0.0112082, 0.0101707
2: 0.0080222, 0.0215743, 0.0079075, 0.0218855, -0.0138633, 0.0136668
3: -0.0091214, 0.0059949, -0.0109692, 0.0054274, -0.0145487, 0.0169641
4: -0.0086054, 0.0074994, -0.0086862, 0.0090956, -0.0177009, 0.0161856
5: -0.0083864, 0.0163383, -0.0127789, 0.0150990, -0.0234854, 0.0291172
6: 0.0007054, 0.0116568, -0.0003169, 0.0116918, -0.0109864, 0.0119737
7: -0.0277350, -0.0037607, -0.0270863, -0.0029260, -0.0248090, 0.0233255
8: 0.9476901, 1.0119501, 0.9485660, 1.0140324, -0.0663422, 0.0633841
9: -0.0045054, 0.0216250, -0.0046747, 0.0203464, -0.0247055, 0.0258517

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1120127, upper bound: 0.1118984
time: 1.09 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1112390, upper bound: 0.1084720
time: 1.04 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0195158, 0.0037846, -0.0033211, 0.0038597, -0.0233755, 0.0071057
1: -0.0110764, 0.0069401, -0.0039174, 0.0065713, -0.0176477, 0.0108575
2: 0.0060706, 0.0305190, 0.0079075, 0.0218855, -0.0158148, 0.0226115
3: -0.0085345, 0.0112431, -0.0109692, 0.0054274, -0.0139619, 0.0222123
4: -0.0125515, 0.0132460, -0.0086862, 0.0090956, -0.0216470, 0.0219322
5: -0.0054809, 0.0300612, -0.0127789, 0.0150990, -0.0205799, 0.0428401
6: -0.0044684, 0.0120024, -0.0003169, 0.0116918, -0.0161601, 0.0123194
7: -0.0335687, 0.0016651, -0.0270863, -0.0029260, -0.0306427, 0.0287514
8: 0.9357346, 1.0160751, 0.9485660, 1.0140324, -0.0782978, 0.0675091
9: -0.0062916, 0.0442557, -0.0046747, 0.0203464, -0.0266379, 0.0489304

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1120127, upper bound: 0.1118984
time: 1.03 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1112390, upper bound: 0.1084720
time: 1.06 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0045608, 0.0035324, -0.0179518, 0.0041029, -0.0086636, 0.0214386
1: -0.0046369, 0.0062533, -0.0103998, 0.0072695, -0.0119064, 0.0166531
2: 0.0080222, 0.0215743, 0.0065886, 0.0294104, -0.0213882, 0.0149856
3: -0.0091214, 0.0059949, -0.0104946, 0.0106155, -0.0197369, 0.0164895
4: -0.0086054, 0.0074994, -0.0112466, 0.0143567, -0.0229621, 0.0187460
5: -0.0083864, 0.0163383, -0.0102437, 0.0284399, -0.0368263, 0.0265820
6: 0.0007054, 0.0116568, -0.0043548, 0.0120345, -0.0113291, 0.0160116
7: -0.0277350, -0.0037607, -0.0329334, 0.0004320, -0.0281670, 0.0291726
8: 0.9476901, 1.0119501, 0.9383868, 1.0181904, -0.0705003, 0.0735633
9: -0.0045054, 0.0216250, -0.0065009, 0.0423704, -0.0468758, 0.0281259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1117636, upper bound: 0.1162920
time: 1.12 seconds

## Relational analysis of IS_A1_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1106817, upper bound: 0.1091937
time: 1.07 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0195158, 0.0037846, -0.0179518, 0.0041029, -0.0236187, 0.0217364
1: -0.0110764, 0.0069401, -0.0103998, 0.0072695, -0.0183458, 0.0173399
2: 0.0060706, 0.0305190, 0.0065886, 0.0294104, -0.0233397, 0.0239304
3: -0.0085345, 0.0112431, -0.0104946, 0.0106155, -0.0191500, 0.0217377
4: -0.0125515, 0.0132460, -0.0112466, 0.0143567, -0.0269082, 0.0244926
5: -0.0054809, 0.0300612, -0.0102437, 0.0284399, -0.0339208, 0.0403049
6: -0.0044684, 0.0120024, -0.0043548, 0.0120345, -0.0165028, 0.0163573
7: -0.0335687, 0.0016651, -0.0329334, 0.0004320, -0.0340006, 0.0345985
8: 0.9357346, 1.0160751, 0.9383868, 1.0181904, -0.0824558, 0.0776883
9: -0.0062916, 0.0442557, -0.0065009, 0.0423704, -0.0486619, 0.0507566

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1117636, upper bound: 0.1126700
time: 1.08 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1106817, upper bound: 0.1088615
time: 1.05 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0030529, 0.0034917, -0.0055109, 0.0035379, -0.0065644, 0.0088953
1: -0.0038226, 0.0062906, -0.0051597, 0.0062618, -0.0100844, 0.0114503
2: 0.0077696, 0.0213024, 0.0080066, 0.0216483, -0.0138787, 0.0132958
3: -0.0082696, 0.0053400, -0.0091529, 0.0064133, -0.0146828, 0.0144929
4: -0.0087865, 0.0056963, -0.0086134, 0.0079961, -0.0167826, 0.0143097
5: -0.0060420, 0.0149357, -0.0084758, 0.0172388, -0.0232807, 0.0234116
6: 0.0011369, 0.0117346, 0.0006027, 0.0116580, -0.0105212, 0.0111320
7: -0.0270008, -0.0037161, -0.0282064, -0.0037373, -0.0232635, 0.0242480
8: 0.9486815, 1.0123346, 0.9470538, 1.0119916, -0.0633101, 0.0652807
9: -0.0048824, 0.0182671, -0.0045112, 0.0234765, -0.0274788, 0.0227759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_A2_B1_B1_A1_B1

### Relational analysis result of IS_A1_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1115381, upper bound: 0.1139435
time: 1.23 seconds

## Relational analysis of IS_A1_A2_B1_B1_A1_B2

### Relational analysis result of IS_A1_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1113761, upper bound: 0.1111350
time: 1.05 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0177458, 0.0037962, -0.0055109, 0.0035379, -0.0212612, 0.0093071
1: -0.0103552, 0.0069181, -0.0051597, 0.0062618, -0.0166170, 0.0120777
2: 0.0065006, 0.0288837, 0.0080066, 0.0216483, -0.0151477, 0.0208771
3: -0.0079280, 0.0105618, -0.0091529, 0.0064133, -0.0143413, 0.0197147
4: -0.0110604, 0.0117997, -0.0086134, 0.0079961, -0.0190565, 0.0204131
5: -0.0037567, 0.0283331, -0.0084758, 0.0172388, -0.0209954, 0.0368089
6: -0.0031581, 0.0120779, 0.0006027, 0.0116580, -0.0148161, 0.0114753
7: -0.0328915, -0.0010486, -0.0282064, -0.0037373, -0.0291542, 0.0271578
8: 0.9385613, 1.0167232, 0.9470538, 1.0119916, -0.0734303, 0.0696693
9: -0.0066337, 0.0407473, -0.0045112, 0.0234765, -0.0301101, 0.0452586

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_A2_B1_B1_A2_B1

### Relational analysis result of IS_A1_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1115381, upper bound: 0.1139435
time: 1.13 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2_B2

### Relational analysis result of IS_A1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1113761, upper bound: 0.1111350
time: 1.04 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0030529, 0.0034917, -0.0209120, 0.0038041, -0.0068569, 0.0243086
1: -0.0038226, 0.0062906, -0.0118779, 0.0071466, -0.0109692, 0.0180356
2: 0.0077696, 0.0213024, 0.0050848, 0.0320811, -0.0243115, 0.0162175
3: -0.0082696, 0.0053400, -0.0087201, 0.0117842, -0.0200538, 0.0140601
4: -0.0087865, 0.0056963, -0.0144434, 0.0140419, -0.0228183, 0.0201397
5: -0.0060420, 0.0149357, -0.0058374, 0.0314382, -0.0374802, 0.0207731
6: 0.0011369, 0.0117346, -0.0053698, 0.0120103, -0.0108734, 0.0171045
7: -0.0270008, -0.0037161, -0.0345459, 0.0038111, -0.0308119, 0.0308298
8: 0.9486815, 1.0123346, 0.9326854, 1.0161886, -0.0675071, 0.0796492
9: -0.0048824, 0.0182671, -0.0063148, 0.0483558, -0.0532382, 0.0245819

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_A2_B1_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1114129, upper bound: 0.1189928
time: 1.32 seconds

## Relational analysis of IS_A1_A2_B1_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1111350, upper bound: 0.1136995
time: 0.95 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0177458, 0.0037962, -0.0209120, 0.0038041, -0.0215499, 0.0247082
1: -0.0103552, 0.0069181, -0.0118779, 0.0071466, -0.0175018, 0.0187959
2: 0.0065006, 0.0288837, 0.0050848, 0.0320811, -0.0255805, 0.0237989
3: -0.0079280, 0.0105618, -0.0087201, 0.0117842, -0.0197122, 0.0192819
4: -0.0110604, 0.0117997, -0.0144434, 0.0140419, -0.0251022, 0.0262431
5: -0.0037567, 0.0283331, -0.0058374, 0.0314382, -0.0351949, 0.0341705
6: -0.0031581, 0.0120779, -0.0053698, 0.0120103, -0.0151684, 0.0174478
7: -0.0328915, -0.0010486, -0.0345459, 0.0038111, -0.0367026, 0.0334973
8: 0.9385613, 1.0167232, 0.9326854, 1.0161886, -0.0776273, 0.0840378
9: -0.0066337, 0.0407473, -0.0063148, 0.0483558, -0.0549894, 0.0470622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_A2_B1_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1114129, upper bound: 0.1156671
time: 1.44 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1111350, upper bound: 0.1128825
time: 1.08 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0203205, 0.0036477, -0.0014593, 0.0034727, -0.0237686, 0.0051070
1: -0.0115080, 0.0069606, -0.0029142, 0.0061746, -0.0176826, 0.0098748
2: 0.0055126, 0.0315145, 0.0081321, 0.0213962, -0.0158836, 0.0233824
3: -0.0096371, 0.0115435, -0.0089263, 0.0047468, -0.0143839, 0.0204698
4: -0.0136078, 0.0147863, -0.0085392, 0.0060088, -0.0196166, 0.0233255
5: -0.0090001, 0.0308124, -0.0080598, 0.0132005, -0.0222006, 0.0388721
6: -0.0053675, 0.0117563, 0.0010578, 0.0116359, -0.0170034, 0.0106984
7: -0.0340909, 0.0032326, -0.0260925, -0.0039642, -0.0301267, 0.0293250
8: 0.9340908, 1.0132447, 0.9497153, 1.0114659, -0.0773751, 0.0635294
9: -0.0051416, 0.0470409, -0.0044040, 0.0160343, -0.0211760, 0.0514449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_A2_B2_B1_B1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1107421, upper bound: 0.1115017
time: 1.03 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1107421, upper bound: 0.1115017
time: 1.10 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0225371, 0.0035790, -0.0154271, 0.0037739, -0.0262373, 0.0190061
1: -0.0128153, 0.0079432, -0.0093805, 0.0068034, -0.0196187, 0.0173237
2: 0.0039414, 0.0339391, 0.0068454, 0.0275025, -0.0235611, 0.0270937
3: -0.0092183, 0.0124095, -0.0086044, 0.0097856, -0.0190039, 0.0210139
4: -0.0166845, 0.0153754, -0.0095041, 0.0116912, -0.0283757, 0.0248796
5: -0.0076848, 0.0330242, -0.0059525, 0.0262114, -0.0338962, 0.0389767
6: -0.0065313, 0.0117877, -0.0022087, 0.0119764, -0.0185076, 0.0139964
7: -0.0356990, 0.0064457, -0.0320122, -0.0021025, -0.0335965, 0.0384580
8: 0.9291230, 1.0127466, 0.9402128, 1.0158402, -0.0867171, 0.0725337
9: -0.0051389, 0.0533699, -0.0061489, 0.0378785, -0.0430174, 0.0595188

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_A2_B2_B1_B2_A1

### Relational analysis result of IS_A1_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1107421, upper bound: 0.1152656
time: 1.04 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_A2

### Relational analysis result of IS_A1_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1107421, upper bound: 0.1152694
time: 1.13 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0200577, 0.0034541, -0.0244569, 0.0033006, -0.0233584, 0.0279110
1: -0.0113671, 0.0066800, -0.0139614, 0.0089104, -0.0202776, 0.0206415
2: 0.0056932, 0.0311550, 0.0025750, 0.0359852, -0.0302919, 0.0285800
3: -0.0087354, 0.0114473, -0.0086218, 0.0131660, -0.0219014, 0.0200691
4: -0.0132528, 0.0140859, -0.0193591, 0.0155492, -0.0288020, 0.0334449
5: -0.0070487, 0.0305741, -0.0066712, 0.0349633, -0.0420120, 0.0372453
6: -0.0049717, 0.0117068, -0.0074161, 0.0125686, -0.0175403, 0.0191229
7: -0.0339176, 0.0026152, -0.0371088, 0.0091385, -0.0430561, 0.0397240
8: 0.9346261, 1.0118430, 0.9247676, 1.0093809, -0.0747548, 0.0870754
9: -0.0048868, 0.0460119, -0.0041879, 0.0586625, -0.0635493, 0.0501998

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_A2_B2_B2_B1_A1

### Relational analysis result of IS_A1_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1097157, upper bound: 0.1083282
time: 1.12 seconds

## Relational analysis of IS_A1_A2_B2_B2_B1_A2

### Relational analysis result of IS_A1_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1097157, upper bound: 0.1083282
time: 1.38 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0222789, 0.0033760, -0.0430708, 0.0103677, -0.0326466, 0.0464468
1: -0.0126784, 0.0076855, -0.0250748, 0.0196402, -0.0323186, 0.0327603
2: 0.0041182, 0.0335861, -0.0137587, 0.0582798, -0.0541615, 0.0473448
3: -0.0082624, 0.0123157, -0.0120960, 0.0236724, -0.0319348, 0.0244117
4: -0.0163369, 0.0146507, -0.0433694, 0.0244695, -0.0408064, 0.0580201
5: -0.0056357, 0.0327927, -0.0048613, 0.0661123, -0.0717480, 0.0376540
6: -0.0061503, 0.0117157, -0.0190986, 0.0218156, -0.0279659, 0.0308143
7: -0.0355306, 0.0058480, -0.0500955, 0.0353679, -0.0708985, 0.0559435
8: 0.9296432, 1.0113101, 0.8678553, 1.0134218, -0.0837786, 0.1434548
9: -0.0048781, 0.0523549, -0.0143167, 0.1139988, -0.1188770, 0.0666716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=17, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_A2_B2_B2_B2_A1

### Relational analysis result of IS_A1_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1097157, upper bound: 0.1090537
time: 1.00 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_A2

### Relational analysis result of IS_A1_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1097157, upper bound: 0.1090537
time: 1.13 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0029930, 0.0035612, -0.0053073, 0.0035574, -0.0065030, 0.0088409
1: -0.0037575, 0.0062576, -0.0050478, 0.0062834, -0.0100409, 0.0113054
2: 0.0080996, 0.0216237, 0.0079911, 0.0216298, -0.0135302, 0.0136326
3: -0.0094765, 0.0052950, -0.0092329, 0.0063237, -0.0158002, 0.0145279
4: -0.0085558, 0.0074387, -0.0086246, 0.0079589, -0.0165147, 0.0160633
5: -0.0094841, 0.0148236, -0.0086537, 0.0170461, -0.0265302, 0.0234773
6: 0.0006428, 0.0116399, 0.0005652, 0.0116629, -0.0110201, 0.0110747
7: -0.0269421, -0.0037441, -0.0281055, -0.0036815, -0.0232606, 0.0243615
8: 0.9487606, 1.0119759, 0.9471899, 1.0121381, -0.0633776, 0.0647860
9: -0.0044235, 0.0190531, -0.0045350, 0.0230722, -0.0268907, 0.0235335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1088480, upper bound: 0.1100937
time: 1.18 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1088480, upper bound: 0.1100937
time: 1.21 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0218978, 0.0033159, -0.0050586, 0.0033645, -0.0251904, 0.0083180
1: -0.0142142, 0.0059936, -0.0049229, 0.0060656, -0.0201795, 0.0109165
2: 0.0006986, 0.0224870, 0.0081674, 0.0214790, -0.0207804, 0.0143195
3: -0.0084944, 0.0136500, -0.0083465, 0.0062211, -0.0147156, 0.0219965
4: -0.0084113, 0.0172161, -0.0085048, 0.0070307, -0.0154420, 0.0257209
5: -0.0079856, 0.0328348, -0.0067117, 0.0168309, -0.0248165, 0.0395465
6: -0.0047100, 0.0115437, 0.0011739, 0.0116155, -0.0163255, 0.0103698
7: -0.0363706, -0.0013027, -0.0279929, -0.0042466, -0.0321241, 0.0266902
8: 0.9360312, 1.0099288, 0.9473420, 1.0106995, -0.0746683, 0.0625868
9: -0.0039574, 0.0548396, -0.0043051, 0.0222527, -0.0262101, 0.0584351

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_A1_B1_A2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0866325, upper bound: 0.0941533
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1078232, upper bound: 0.1090695
time: 1.09 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0029930, 0.0035612, -0.0205847, 0.0038100, -0.0068030, 0.0241322
1: -0.0037575, 0.0062576, -0.0116877, 0.0070842, -0.0108418, 0.0179453
2: 0.0080996, 0.0216237, 0.0053157, 0.0317080, -0.0236084, 0.0163080
3: -0.0094765, 0.0052950, -0.0086948, 0.0116576, -0.0211341, 0.0139898
4: -0.0085558, 0.0074387, -0.0139912, 0.0138274, -0.0223832, 0.0214299
5: -0.0094841, 0.0148236, -0.0057759, 0.0311165, -0.0406006, 0.0205995
6: 0.0006428, 0.0116399, -0.0051490, 0.0120133, -0.0113705, 0.0167889
7: -0.0269421, -0.0037441, -0.0343120, 0.0032988, -0.0302410, 0.0305679
8: 0.9487606, 1.0119759, 0.9334079, 1.0162596, -0.0674990, 0.0785680
9: -0.0044235, 0.0190531, -0.0063339, 0.0473633, -0.0517868, 0.0253870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1084311, upper bound: 0.1107404
time: 1.55 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1084311, upper bound: 0.1107404
time: 2.44 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0218978, 0.0033159, -0.0203010, 0.0036155, -0.0255133, 0.0235737
1: -0.0142142, 0.0059936, -0.0115383, 0.0067917, -0.0210059, 0.0175319
2: 0.0006986, 0.0224870, 0.0055096, 0.0313073, -0.0306087, 0.0169774
3: -0.0084944, 0.0136500, -0.0077251, 0.0115550, -0.0200495, 0.0213751
4: -0.0084113, 0.0172161, -0.0136100, 0.0130151, -0.0214264, 0.0308261
5: -0.0079856, 0.0328348, -0.0036181, 0.0308637, -0.0388493, 0.0364530
6: -0.0047100, 0.0115437, -0.0046953, 0.0119665, -0.0166765, 0.0162390
7: -0.0363706, -0.0013027, -0.0341282, 0.0026052, -0.0389758, 0.0328255
8: 0.9360312, 1.0099288, 0.9339756, 1.0147887, -0.0787575, 0.0759532
9: -0.0039574, 0.0548396, -0.0060875, 0.0462095, -0.0501668, 0.0609271

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A1_B2_A2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0986220, upper bound: 0.1042314
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1072259, upper bound: 0.1096160
time: 1.57 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0175789, 0.0038475, -0.0053073, 0.0035574, -0.0211009, 0.0090680
1: -0.0102611, 0.0069237, -0.0050478, 0.0062834, -0.0165445, 0.0119715
2: 0.0068085, 0.0289904, 0.0079911, 0.0216298, -0.0148213, 0.0209993
3: -0.0091627, 0.0104923, -0.0092329, 0.0063237, -0.0154864, 0.0197253
4: -0.0109390, 0.0132039, -0.0086246, 0.0079589, -0.0188979, 0.0218285
5: -0.0073045, 0.0281260, -0.0086537, 0.0170461, -0.0243506, 0.0367797
6: -0.0036795, 0.0119803, 0.0005652, 0.0116629, -0.0153425, 0.0114151
7: -0.0328062, -0.0006157, -0.0281055, -0.0036815, -0.0291247, 0.0274898
8: 0.9387430, 1.0163075, 0.9471899, 1.0121381, -0.0733951, 0.0691175
9: -0.0062034, 0.0412842, -0.0045350, 0.0230722, -0.0292756, 0.0452421

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1092523, upper bound: 0.1096392
time: 1.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1092523, upper bound: 0.1096392
time: 1.11 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0437107, 0.0037373, -0.0050586, 0.0033645, -0.0470197, 0.0087959
1: -0.0210832, 0.0204734, -0.0049229, 0.0060656, -0.0270851, 0.0253963
2: -0.0110408, 0.0517711, 0.0081674, 0.0214790, -0.0325199, 0.0436037
3: -0.0088168, 0.0206522, -0.0083465, 0.0062211, -0.0150380, 0.0289987
4: -0.0328849, 0.0244999, -0.0085048, 0.0070307, -0.0399156, 0.0330046
5: -0.0051254, 0.0540413, -0.0067117, 0.0168309, -0.0219562, 0.0607530
6: -0.0194701, 0.0151854, 0.0011739, 0.0116155, -0.0310856, 0.0140115
7: -0.0429649, 0.0392273, -0.0279929, -0.0042466, -0.0387183, 0.0672202
8: 0.8965118, 1.0138938, 0.9473420, 1.0106995, -0.1141877, 0.0665519
9: -0.0095789, 0.0877255, -0.0043051, 0.0222527, -0.0318317, 0.0920307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0982920, upper bound: 0.1028860
time: 1.14 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1080304, upper bound: 0.1085189
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0175789, 0.0038475, -0.0205847, 0.0038100, -0.0213889, 0.0242109
1: -0.0102611, 0.0069237, -0.0116877, 0.0070842, -0.0173453, 0.0186114
2: 0.0068085, 0.0289904, 0.0053157, 0.0317080, -0.0248995, 0.0236747
3: -0.0091627, 0.0104923, -0.0086948, 0.0116576, -0.0208204, 0.0191872
4: -0.0109390, 0.0132039, -0.0139912, 0.0138274, -0.0247664, 0.0271951
5: -0.0073045, 0.0281260, -0.0057759, 0.0311165, -0.0384210, 0.0339019
6: -0.0036795, 0.0119803, -0.0051490, 0.0120133, -0.0156929, 0.0171293
7: -0.0328062, -0.0006157, -0.0343120, 0.0032988, -0.0361051, 0.0336963
8: 0.9387430, 1.0163075, 0.9334079, 1.0162596, -0.0764870, 0.0828996
9: -0.0062034, 0.0412842, -0.0063339, 0.0473633, -0.0535667, 0.0476180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0986249, upper bound: 0.0908082
time: 2.39 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1151183, upper bound: 0.1099181
time: 1.05 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0437107, 0.0037373, -0.0203010, 0.0036155, -0.0473262, 0.0240383
1: -0.0210832, 0.0204734, -0.0115383, 0.0067917, -0.0278749, 0.0320117
2: -0.0110408, 0.0517711, 0.0055096, 0.0313073, -0.0423481, 0.0462615
3: -0.0088168, 0.0206522, -0.0077251, 0.0115550, -0.0203718, 0.0283773
4: -0.0328849, 0.0244999, -0.0136100, 0.0130151, -0.0459000, 0.0381098
5: -0.0051254, 0.0540413, -0.0036181, 0.0308637, -0.0359891, 0.0576595
6: -0.0194701, 0.0151854, -0.0046953, 0.0119665, -0.0314366, 0.0198807
7: -0.0429649, 0.0392273, -0.0341282, 0.0026052, -0.0455701, 0.0733555
8: 0.8965118, 1.0138938, 0.9339756, 1.0147887, -0.1182770, 0.0799183
9: -0.0095789, 0.0877255, -0.0060875, 0.0462095, -0.0557884, 0.0938131

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A2_B2_A2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0980260, upper bound: 0.1028860
time: 1.12 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1079434, upper bound: 0.1088047
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0018165, 0.0037732, -0.0030529, 0.0034917, -0.0052130, 0.0068261
1: -0.0030981, 0.0064890, -0.0038226, 0.0062906, -0.0093888, 0.0103117
2: 0.0079450, 0.0216626, 0.0077696, 0.0213024, -0.0133573, 0.0138930
3: -0.0104089, 0.0047804, -0.0082696, 0.0053400, -0.0157489, 0.0130500
4: -0.0086661, 0.0077916, -0.0087865, 0.0056963, -0.0143624, 0.0165781
5: -0.0113327, 0.0136738, -0.0060420, 0.0149357, -0.0262684, 0.0197158
6: 0.0001149, 0.0116877, 0.0011369, 0.0117346, -0.0116197, 0.0105509
7: -0.0263402, -0.0031444, -0.0270008, -0.0037161, -0.0226241, 0.0238564
8: 0.9495575, 1.0135293, 0.9486815, 1.0123346, -0.0627770, 0.0648478
9: -0.0046552, 0.0172282, -0.0048824, 0.0182671, -0.0228399, 0.0221106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1115017, upper bound: 0.1109709
time: 1.13 seconds

## Relational analysis of IS_A2_B2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1083416, upper bound: 0.1102918
time: 1.13 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0018165, 0.0037732, -0.0177458, 0.0037962, -0.0056127, 0.0215191
1: -0.0030981, 0.0064890, -0.0103552, 0.0069181, -0.0100162, 0.0168443
2: 0.0079450, 0.0216626, 0.0065006, 0.0288837, -0.0209387, 0.0151621
3: -0.0104089, 0.0047804, -0.0079280, 0.0105618, -0.0209707, 0.0127084
4: -0.0086661, 0.0077916, -0.0110604, 0.0117997, -0.0204658, 0.0188520
5: -0.0113327, 0.0136738, -0.0037567, 0.0283331, -0.0396658, 0.0174305
6: 0.0001149, 0.0116877, -0.0031581, 0.0120779, -0.0119630, 0.0148458
7: -0.0263402, -0.0031444, -0.0328915, -0.0010486, -0.0252916, 0.0297471
8: 0.9495575, 1.0135293, 0.9385613, 1.0167232, -0.0671656, 0.0749680
9: -0.0046552, 0.0172282, -0.0066337, 0.0407473, -0.0454025, 0.0238619

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=6, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1115017, upper bound: 0.1126806
time: 1.18 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1083416, upper bound: 0.1116318
time: 1.38 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0160074, 0.0040264, -0.0030529, 0.0034917, -0.0194118, 0.0070480
1: -0.0096048, 0.0071122, -0.0038226, 0.0062906, -0.0158682, 0.0109348
2: 0.0066269, 0.0280350, 0.0077696, 0.0213024, -0.0146754, 0.0202655
3: -0.0099210, 0.0099695, -0.0082696, 0.0053400, -0.0152610, 0.0182391
4: -0.0097058, 0.0129456, -0.0087865, 0.0056963, -0.0154021, 0.0217321
5: -0.0088824, 0.0266991, -0.0060420, 0.0149357, -0.0238181, 0.0327410
6: -0.0030332, 0.0120301, 0.0011369, 0.0117346, -0.0147678, 0.0108933
7: -0.0322144, -0.0010697, -0.0270008, -0.0037161, -0.0284423, 0.0259310
8: 0.9398386, 1.0177325, 0.9486815, 1.0123346, -0.0724960, 0.0690510
9: -0.0064433, 0.0392239, -0.0048824, 0.0182671, -0.0247104, 0.0433357

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1152656, upper bound: 0.1107421
time: 1.17 seconds

## Relational analysis of IS_A2_B2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1090537, upper bound: 0.1097157
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0160074, 0.0040264, -0.0177458, 0.0037962, -0.0198036, 0.0215952
1: -0.0096048, 0.0071122, -0.0103552, 0.0069181, -0.0165228, 0.0174675
2: 0.0066269, 0.0280350, 0.0065006, 0.0288837, -0.0222568, 0.0215345
3: -0.0099210, 0.0099695, -0.0079280, 0.0105618, -0.0204827, 0.0178975
4: -0.0097058, 0.0129456, -0.0110604, 0.0117997, -0.0215054, 0.0240060
5: -0.0088824, 0.0266991, -0.0037567, 0.0283331, -0.0372155, 0.0304557
6: -0.0030332, 0.0120301, -0.0031581, 0.0120779, -0.0151111, 0.0151882
7: -0.0322144, -0.0010697, -0.0328915, -0.0010486, -0.0311658, 0.0318218
8: 0.9398386, 1.0177325, 0.9385613, 1.0167232, -0.0755153, 0.0791712
9: -0.0064433, 0.0392239, -0.0066337, 0.0407473, -0.0471906, 0.0458576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1152656, upper bound: 0.1119953
time: 1.25 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1090537, upper bound: 0.1102579
time: 1.09 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0018165, 0.0037732, -0.0009213, 0.0037073, -0.0054782, 0.0046945
1: -0.0030981, 0.0064890, -0.0026224, 0.0065050, -0.0096032, 0.0091114
2: 0.0079450, 0.0216626, 0.0076852, 0.0212936, -0.0133486, 0.0139775
3: -0.0104089, 0.0047804, -0.0094824, 0.0046227, -0.0150316, 0.0142628
4: -0.0086661, 0.0077916, -0.0088514, 0.0060914, -0.0147575, 0.0166430
5: -0.0113327, 0.0136738, -0.0088992, 0.0125430, -0.0238756, 0.0225731
6: 0.0001149, 0.0116877, 0.0005402, 0.0117634, -0.0116485, 0.0111475
7: -0.0263402, -0.0031444, -0.0256288, -0.0031393, -0.0232009, 0.0224844
8: 0.9495575, 1.0135293, 0.9503613, 1.0137819, -0.0642244, 0.0631680
9: -0.0046552, 0.0172282, -0.0050216, 0.0154503, -0.0201054, 0.0222028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1113061, upper bound: 0.1086417
time: 1.12 seconds

## Relational analysis of IS_A2_B2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1082809, upper bound: 0.1084804
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0018165, 0.0037732, -0.0127855, 0.0040169, -0.0057375, 0.0165587
1: -0.0030981, 0.0064890, -0.0082867, 0.0071503, -0.0102485, 0.0147758
2: 0.0079450, 0.0216626, 0.0063757, 0.0258024, -0.0178574, 0.0152869
3: -0.0104089, 0.0047804, -0.0090949, 0.0089116, -0.0193205, 0.0138753
4: -0.0086661, 0.0077916, -0.0098079, 0.0106183, -0.0192844, 0.0175995
5: -0.0113327, 0.0136738, -0.0066803, 0.0238335, -0.0351661, 0.0203541
6: 0.0001149, 0.0116877, -0.0011428, 0.0121129, -0.0119980, 0.0128306
7: -0.0263402, -0.0031444, -0.0310260, -0.0015891, -0.0247511, 0.0278816
8: 0.9495575, 1.0135293, 0.9420384, 1.0182815, -0.0687239, 0.0714909
9: -0.0046552, 0.0172282, -0.0067846, 0.0341062, -0.0383031, 0.0240128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1113061, upper bound: 0.1092253
time: 2.64 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1082809, upper bound: 0.1090539
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0160074, 0.0040264, -0.0009213, 0.0037073, -0.0196761, 0.0049374
1: -0.0096048, 0.0071122, -0.0026224, 0.0065050, -0.0161098, 0.0097346
2: 0.0066269, 0.0280350, 0.0076852, 0.0212936, -0.0146667, 0.0199199
3: -0.0099210, 0.0099695, -0.0094824, 0.0046227, -0.0145437, 0.0194519
4: -0.0097058, 0.0129456, -0.0088514, 0.0060914, -0.0157972, 0.0217970
5: -0.0088824, 0.0266991, -0.0088992, 0.0125430, -0.0214254, 0.0355983
6: -0.0030332, 0.0120301, 0.0005402, 0.0117634, -0.0147966, 0.0114899
7: -0.0322144, -0.0010697, -0.0256288, -0.0031393, -0.0290024, 0.0245591
8: 0.9398386, 1.0177325, 0.9503613, 1.0137819, -0.0739433, 0.0673712
9: -0.0064433, 0.0392239, -0.0050216, 0.0154503, -0.0218935, 0.0433888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1150911, upper bound: 0.1085321
time: 1.09 seconds

## Relational analysis of IS_A2_B2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1090242, upper bound: 0.1082809
time: 1.21 seconds

## BFS IS instance: IS_A2_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0160074, 0.0040264, -0.0127855, 0.0040169, -0.0197794, 0.0166364
1: -0.0096048, 0.0071122, -0.0082867, 0.0071503, -0.0167551, 0.0153990
2: 0.0066269, 0.0280350, 0.0063757, 0.0258024, -0.0191755, 0.0216593
3: -0.0099210, 0.0099695, -0.0090949, 0.0089116, -0.0188325, 0.0190644
4: -0.0097058, 0.0129456, -0.0098079, 0.0106183, -0.0203240, 0.0227535
5: -0.0088824, 0.0266991, -0.0066803, 0.0238335, -0.0327159, 0.0333793
6: -0.0030332, 0.0120301, -0.0011428, 0.0121129, -0.0151461, 0.0131730
7: -0.0322144, -0.0010697, -0.0310260, -0.0015891, -0.0306253, 0.0299562
8: 0.9398386, 1.0177325, 0.9420384, 1.0182815, -0.0777196, 0.0756941
9: -0.0064433, 0.0392239, -0.0067846, 0.0341062, -0.0405494, 0.0452027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1150911, upper bound: 0.1090197
time: 1.26 seconds

## Relational analysis of IS_A2_B2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1090243, upper bound: 0.1085856
time: 1.11 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.83 seconds
IS_A1_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1131345, upper bound: 0.1142569
IS_A1_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1127955, upper bound: 0.1113065
IS_A1_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1131345, upper bound: 0.1142569
IS_A1_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1127955, upper bound: 0.1113065
IS_A1_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1130535, upper bound: 0.1202042
IS_A1_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1126266, upper bound: 0.1139029
IS_A1_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1130535, upper bound: 0.1160501
IS_A1_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1126266, upper bound: 0.1130722
IS_A1_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1120127, upper bound: 0.1118984
IS_A1_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1112390, upper bound: 0.1084720
IS_A1_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1120127, upper bound: 0.1118984
IS_A1_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1112390, upper bound: 0.1084720
IS_A1_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1117636, upper bound: 0.1162920
IS_A1_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1106817, upper bound: 0.1091937
IS_A1_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1117636, upper bound: 0.1126700
IS_A1_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1106817, upper bound: 0.1088615
IS_A1_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1115381, upper bound: 0.1139435
IS_A1_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1113761, upper bound: 0.1111350
IS_A1_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1115381, upper bound: 0.1139435
IS_A1_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1113761, upper bound: 0.1111350
IS_A1_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1114129, upper bound: 0.1189928
IS_A1_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1111350, upper bound: 0.1136995
IS_A1_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1114129, upper bound: 0.1156671
IS_A1_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1111350, upper bound: 0.1128825
IS_A1_A2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1107421, upper bound: 0.1115017
IS_A1_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1107421, upper bound: 0.1115017
IS_A1_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1107421, upper bound: 0.1152656
IS_A1_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1107421, upper bound: 0.1152694
IS_A1_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1097157, upper bound: 0.1083282
IS_A1_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1097157, upper bound: 0.1083282
IS_A1_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1097157, upper bound: 0.1090537
IS_A1_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1097157, upper bound: 0.1090537
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1088480, upper bound: 0.1100937
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1088480, upper bound: 0.1100937
IS_A2_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.0866325, upper bound: 0.0941533
IS_A2_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1078232, upper bound: 0.1090695
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1084311, upper bound: 0.1107404
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1084311, upper bound: 0.1107404
IS_A2_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.0986220, upper bound: 0.1042314
IS_A2_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1072259, upper bound: 0.1096160
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1092523, upper bound: 0.1096392
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1092523, upper bound: 0.1096392
IS_A2_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.0982920, upper bound: 0.1028860
IS_A2_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1080304, upper bound: 0.1085189
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.0986249, upper bound: 0.0908082
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1151183, upper bound: 0.1099181
IS_A2_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.0980260, upper bound: 0.1028860
IS_A2_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1079434, upper bound: 0.1088047
IS_A2_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1115017, upper bound: 0.1109709
IS_A2_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1083416, upper bound: 0.1102918
IS_A2_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1115017, upper bound: 0.1126806
IS_A2_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1083416, upper bound: 0.1116318
IS_A2_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1152656, upper bound: 0.1107421
IS_A2_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1090537, upper bound: 0.1097157
IS_A2_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1152656, upper bound: 0.1119953
IS_A2_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1090537, upper bound: 0.1102579
IS_A2_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1113061, upper bound: 0.1086417
IS_A2_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1082809, upper bound: 0.1084804
IS_A2_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1113061, upper bound: 0.1092253
IS_A2_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1082809, upper bound: 0.1090539
IS_A2_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1150911, upper bound: 0.1085321
IS_A2_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1090242, upper bound: 0.1082809
IS_A2_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1150911, upper bound: 0.1090197
IS_A2_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 8, lower bound: -0.1090243, upper bound: 0.1085856

## BFS IS instance: IS_A1_A1_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0045608, 0.0035324, -0.0068219, 0.0033322, -0.0078117, 0.0102879
1: -0.0046369, 0.0062533, -0.0058224, 0.0060354, -0.0106723, 0.0120757
2: 0.0080222, 0.0215743, 0.0081652, 0.0219081, -0.0138860, 0.0134090
3: -0.0091214, 0.0059949, -0.0082344, 0.0069409, -0.0160623, 0.0142293
4: -0.0086054, 0.0074994, -0.0085099, 0.0079931, -0.0165984, 0.0160093
5: -0.0083864, 0.0163383, -0.0066023, 0.0184757, -0.0268621, 0.0229407
6: 0.0007054, 0.0116568, 0.0010500, 0.0116118, -0.0109065, 0.0106068
7: -0.0277350, -0.0037607, -0.0288039, -0.0043283, -0.0234067, 0.0250432
8: 0.9476901, 1.0119501, 0.9461516, 1.0104891, -0.0627990, 0.0657985
9: -0.0045054, 0.0216250, -0.0042948, 0.0254496, -0.0290952, 0.0255615

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_A1_B1_B1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1127955, upper bound: 0.1116181
time: 1.14 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1127955, upper bound: 0.1116181
time: 1.12 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0043204, 0.0033403, -0.0318988, 0.0030645, -0.0072806, 0.0351507
1: -0.0045169, 0.0060363, -0.0162033, 0.0126463, -0.0171632, 0.0220276
2: 0.0081973, 0.0214228, -0.0026901, 0.0381863, -0.0299890, 0.0241129
3: -0.0082354, 0.0058962, -0.0078452, 0.0152369, -0.0234723, 0.0137415
4: -0.0084861, 0.0065518, -0.0221942, 0.0187027, -0.0270150, 0.0287460
5: -0.0064485, 0.0161316, -0.0049443, 0.0410450, -0.0474936, 0.0210759
6: 0.0013103, 0.0116094, -0.0119444, 0.0115108, -0.0102005, 0.0235171
7: -0.0276268, -0.0043234, -0.0381642, 0.0206654, -0.0482922, 0.0338363
8: 0.9478363, 1.0105119, 0.9288251, 1.0082068, -0.0603706, 0.0816868
9: -0.0042755, 0.0208152, -0.0040864, 0.0614225, -0.0648121, 0.0249016

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_A1_B1_B1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1036792, upper bound: 0.0972489
time: 1.49 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1117177, upper bound: 0.1105601
time: 1.11 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0195158, 0.0037846, -0.0068219, 0.0033322, -0.0227794, 0.0106065
1: -0.0110764, 0.0069401, -0.0058224, 0.0060354, -0.0171118, 0.0127625
2: 0.0060706, 0.0305190, 0.0081652, 0.0219081, -0.0158375, 0.0223538
3: -0.0085345, 0.0112431, -0.0082344, 0.0069409, -0.0154754, 0.0194775
4: -0.0125515, 0.0132460, -0.0085099, 0.0079931, -0.0205445, 0.0217559
5: -0.0054809, 0.0300612, -0.0066023, 0.0184757, -0.0239566, 0.0366636
6: -0.0044684, 0.0120024, 0.0010500, 0.0116118, -0.0160802, 0.0109525
7: -0.0335687, 0.0016651, -0.0288039, -0.0043283, -0.0292403, 0.0304691
8: 0.9357346, 1.0160751, 0.9461516, 1.0104891, -0.0747545, 0.0699235
9: -0.0062916, 0.0442557, -0.0042948, 0.0254496, -0.0317412, 0.0485505

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_A1_B1_B1_A2_B1_A1

### Relational analysis result of IS_A1_A1_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1148819, upper bound: 0.1113065
time: 1.39 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2_B1_A2

### Relational analysis result of IS_A1_A1_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1148819, upper bound: 0.1113065
time: 1.17 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0192396, 0.0035914, -0.0318988, 0.0030645, -0.0222117, 0.0354902
1: -0.0109740, 0.0066711, -0.0162033, 0.0126463, -0.0236204, 0.0228745
2: 0.0062592, 0.0301830, -0.0026901, 0.0381863, -0.0319270, 0.0328731
3: -0.0075755, 0.0111434, -0.0078452, 0.0152369, -0.0228124, 0.0189886
4: -0.0123154, 0.0124311, -0.0221942, 0.0187027, -0.0310182, 0.0346253
5: -0.0033397, 0.0298159, -0.0049443, 0.0410450, -0.0443847, 0.0347602
6: -0.0040105, 0.0119572, -0.0119444, 0.0115108, -0.0155213, 0.0239016
7: -0.0334725, 0.0009605, -0.0381642, 0.0206654, -0.0541379, 0.0391247
8: 0.9361358, 1.0146081, 0.9288251, 1.0082068, -0.0720711, 0.0857831
9: -0.0060476, 0.0434009, -0.0040864, 0.0614225, -0.0674701, 0.0474873

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_A1_B1_B1_A2_B2_B1

### Relational analysis result of IS_A1_A1_B1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1096676, upper bound: 0.1035643
time: 1.31 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2_B2_B2

### Relational analysis result of IS_A1_A1_B1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1137056, upper bound: 0.1100887
time: 1.04 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0045608, 0.0035324, -0.0226736, 0.0036342, -0.0081949, 0.0261665
1: -0.0046369, 0.0062533, -0.0129405, 0.0078287, -0.0124656, 0.0191807
2: 0.0080222, 0.0215743, 0.0035001, 0.0341709, -0.0261487, 0.0180742
3: -0.0091214, 0.0059949, -0.0080314, 0.0128207, -0.0219420, 0.0140263
4: -0.0086054, 0.0074994, -0.0166920, 0.0144934, -0.0230987, 0.0241914
5: -0.0083864, 0.0163383, -0.0043004, 0.0345180, -0.0429044, 0.0206387
6: 0.0007054, 0.0116568, -0.0062984, 0.0119785, -0.0112732, 0.0179552
7: -0.0277350, -0.0037607, -0.0357802, 0.0060686, -0.0338037, 0.0320195
8: 0.9476901, 1.0119501, 0.9271098, 1.0147977, -0.0671076, 0.0848404
9: -0.0045054, 0.0216250, -0.0061112, 0.0534249, -0.0579303, 0.0277361

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1051321, upper bound: 0.1041240
time: 1.37 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1119707, upper bound: 0.1191869
time: 1.32 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0043204, 0.0033403, -0.0506294, 0.0142860, -0.0186064, 0.0539697
1: -0.0045169, 0.0060363, -0.0296670, 0.0238411, -0.0283580, 0.0357033
2: 0.0081973, 0.0214228, -0.0215281, 0.0676704, -0.0594732, 0.0429508
3: -0.0082354, 0.0058962, -0.0146881, 0.0290987, -0.0373341, 0.0205844
4: -0.0084861, 0.0065518, -0.0525106, 0.0282725, -0.0367586, 0.0590624
5: -0.0064485, 0.0161316, -0.0040090, 0.0835307, -0.0899792, 0.0201406
6: 0.0013103, 0.0116094, -0.0237658, 0.0259806, -0.0246703, 0.0353752
7: -0.0276268, -0.0043234, -0.0552041, 0.0451173, -0.0727441, 0.0508806
8: 0.9478363, 1.0105119, 0.8390644, 1.0122657, -0.0644294, 0.1714475
9: -0.0042755, 0.0208152, -0.0224045, 0.1363686, -0.1406441, 0.0432197

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_B1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1074380, upper bound: 0.1053478
time: 1.18 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_B2

### Relational analysis result of IS_A1_A1_B1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1114354, upper bound: 0.1127014
time: 1.27 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0195158, 0.0037846, -0.0226736, 0.0036342, -0.0231500, 0.0264582
1: -0.0110764, 0.0069401, -0.0129405, 0.0078287, -0.0189051, 0.0198806
2: 0.0060706, 0.0305190, 0.0035001, 0.0341709, -0.0281003, 0.0270189
3: -0.0085345, 0.0112431, -0.0080314, 0.0128207, -0.0213552, 0.0192744
4: -0.0125515, 0.0132460, -0.0166920, 0.0144934, -0.0270448, 0.0299380
5: -0.0054809, 0.0300612, -0.0043004, 0.0345180, -0.0399989, 0.0343616
6: -0.0044684, 0.0120024, -0.0062984, 0.0119785, -0.0164469, 0.0183008
7: -0.0335687, 0.0016651, -0.0357802, 0.0060686, -0.0396373, 0.0374454
8: 0.9357346, 1.0160751, 0.9271098, 1.0147977, -0.0790631, 0.0889654
9: -0.0062916, 0.0442557, -0.0061112, 0.0534249, -0.0597165, 0.0503669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_A1_B1_B2_A2_B1_B1

### Relational analysis result of IS_A1_A1_B1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1110905, upper bound: 0.1101886
time: 0.93 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1140679, upper bound: 0.1148367
time: 1.19 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0192396, 0.0035914, -0.0506294, 0.0142860, -0.0335256, 0.0542207
1: -0.0109740, 0.0066711, -0.0296670, 0.0238411, -0.0348151, 0.0363381
2: 0.0062592, 0.0301830, -0.0215281, 0.0676704, -0.0614112, 0.0517110
3: -0.0075755, 0.0111434, -0.0146881, 0.0290987, -0.0366742, 0.0258315
4: -0.0123154, 0.0124311, -0.0525106, 0.0282725, -0.0405879, 0.0649417
5: -0.0033397, 0.0298159, -0.0040090, 0.0835307, -0.0868703, 0.0338249
6: -0.0040105, 0.0119572, -0.0237658, 0.0259806, -0.0299911, 0.0357230
7: -0.0334725, 0.0009605, -0.0552041, 0.0451173, -0.0785899, 0.0561645
8: 0.9361358, 1.0146081, 0.8390644, 1.0122657, -0.0761299, 0.1755437
9: -0.0060476, 0.0434009, -0.0224045, 0.1363686, -0.1424162, 0.0658055

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_A1_B1_B2_A2_B2_B1

### Relational analysis result of IS_A1_A1_B1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1096973, upper bound: 0.1049064
time: 0.93 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2_B2_B2

### Relational analysis result of IS_A1_A1_B1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1137160, upper bound: 0.1118636
time: 1.14 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0045608, 0.0035324, -0.0029930, 0.0035612, -0.0080946, 0.0064682
1: -0.0046369, 0.0062533, -0.0037575, 0.0062576, -0.0108945, 0.0100109
2: 0.0080222, 0.0215743, 0.0080996, 0.0216237, -0.0136015, 0.0134747
3: -0.0091214, 0.0059949, -0.0094765, 0.0052950, -0.0144163, 0.0154714
4: -0.0086054, 0.0074994, -0.0085558, 0.0074387, -0.0160441, 0.0160552
5: -0.0083864, 0.0163383, -0.0094841, 0.0148236, -0.0232101, 0.0258225
6: 0.0007054, 0.0116568, 0.0006428, 0.0116399, -0.0109345, 0.0110140
7: -0.0277350, -0.0037607, -0.0269421, -0.0037441, -0.0239910, 0.0231814
8: 0.9476901, 1.0119501, 0.9487606, 1.0119759, -0.0642858, 0.0631896
9: -0.0045054, 0.0216250, -0.0044235, 0.0190531, -0.0233568, 0.0255415

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_A1_B2_B1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1112487, upper bound: 0.1088727
time: 1.22 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1112487, upper bound: 0.1088727
time: 1.18 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0043204, 0.0033403, -0.0218978, 0.0033159, -0.0075800, 0.0251571
1: -0.0045169, 0.0060363, -0.0142142, 0.0059936, -0.0105105, 0.0200964
2: 0.0081973, 0.0214228, 0.0006986, 0.0224870, -0.0142897, 0.0207241
3: -0.0082354, 0.0058962, -0.0084944, 0.0136500, -0.0218854, 0.0143907
4: -0.0084861, 0.0065518, -0.0084113, 0.0172161, -0.0256510, 0.0149631
5: -0.0064485, 0.0161316, -0.0079856, 0.0328348, -0.0392834, 0.0241172
6: 0.0013103, 0.0116094, -0.0047100, 0.0115437, -0.0102334, 0.0163194
7: -0.0276268, -0.0043234, -0.0363706, -0.0013027, -0.0263241, 0.0320472
8: 0.9478363, 1.0105119, 0.9360312, 1.0099288, -0.0620925, 0.0744807
9: -0.0042755, 0.0208152, -0.0039574, 0.0548396, -0.0582842, 0.0247726

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_A1_B2_B1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0950305, upper bound: 0.0866565
time: 1.06 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1101912, upper bound: 0.1078447
time: 1.17 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0195158, 0.0037846, -0.0029930, 0.0035612, -0.0230622, 0.0067776
1: -0.0110764, 0.0069401, -0.0037575, 0.0062576, -0.0173340, 0.0106977
2: 0.0060706, 0.0305190, 0.0080996, 0.0216237, -0.0155530, 0.0224195
3: -0.0085345, 0.0112431, -0.0094765, 0.0052950, -0.0138295, 0.0207195
4: -0.0125515, 0.0132460, -0.0085558, 0.0074387, -0.0199902, 0.0218018
5: -0.0054809, 0.0300612, -0.0094841, 0.0148236, -0.0203046, 0.0395454
6: -0.0044684, 0.0120024, 0.0006428, 0.0116399, -0.0161082, 0.0113596
7: -0.0335687, 0.0016651, -0.0269421, -0.0037441, -0.0298246, 0.0286073
8: 0.9357346, 1.0160751, 0.9487606, 1.0119759, -0.0762413, 0.0673146
9: -0.0062916, 0.0442557, -0.0044235, 0.0190531, -0.0253446, 0.0486792

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_A1_B2_B1_A2_B1_A1

### Relational analysis result of IS_A1_A1_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1124427, upper bound: 0.1084720
time: 0.92 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_B1_A2

### Relational analysis result of IS_A1_A1_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1124427, upper bound: 0.1084720
time: 0.97 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0192396, 0.0035914, -0.0218978, 0.0033159, -0.0225112, 0.0254891
1: -0.0109740, 0.0066711, -0.0142142, 0.0059936, -0.0169677, 0.0208853
2: 0.0062592, 0.0301830, 0.0006986, 0.0224870, -0.0162278, 0.0294843
3: -0.0075755, 0.0111434, -0.0084944, 0.0136500, -0.0212255, 0.0196378
4: -0.0123154, 0.0124311, -0.0084113, 0.0172161, -0.0295315, 0.0208424
5: -0.0033397, 0.0298159, -0.0079856, 0.0328348, -0.0361745, 0.0378015
6: -0.0040105, 0.0119572, -0.0047100, 0.0115437, -0.0155542, 0.0166672
7: -0.0334725, 0.0009605, -0.0363706, -0.0013027, -0.0321698, 0.0373311
8: 0.9361358, 1.0146081, 0.9360312, 1.0099288, -0.0737931, 0.0785769
9: -0.0060476, 0.0434009, -0.0039574, 0.0548396, -0.0608872, 0.0473583

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_A1_B2_B1_A2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1052059, upper bound: 0.0986596
time: 1.06 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1112619, upper bound: 0.1072640
time: 0.97 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0045608, 0.0035324, -0.0175789, 0.0038475, -0.0083217, 0.0210665
1: -0.0046369, 0.0062533, -0.0102611, 0.0069237, -0.0115606, 0.0165144
2: 0.0080222, 0.0215743, 0.0068085, 0.0289904, -0.0209682, 0.0147657
3: -0.0091214, 0.0059949, -0.0091627, 0.0104923, -0.0196137, 0.0151576
4: -0.0086054, 0.0074994, -0.0109390, 0.0132039, -0.0218093, 0.0184384
5: -0.0083864, 0.0163383, -0.0073045, 0.0281260, -0.0365124, 0.0236428
6: 0.0007054, 0.0116568, -0.0036795, 0.0119803, -0.0112749, 0.0153363
7: -0.0277350, -0.0037607, -0.0328062, -0.0006157, -0.0271193, 0.0290455
8: 0.9476901, 1.0119501, 0.9387430, 1.0163075, -0.0686173, 0.0732071
9: -0.0045054, 0.0216250, -0.0062034, 0.0412842, -0.0450955, 0.0278284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A1

### Relational analysis result of IS_A1_A1_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1106817, upper bound: 0.1092836
time: 2.18 seconds

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A2

### Relational analysis result of IS_A1_A1_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1106817, upper bound: 0.1092836
time: 1.33 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0043204, 0.0033403, -0.0437107, 0.0037373, -0.0080577, 0.0469865
1: -0.0045169, 0.0060363, -0.0210832, 0.0204734, -0.0249903, 0.0270038
2: 0.0081973, 0.0214228, -0.0110408, 0.0517711, -0.0435738, 0.0324636
3: -0.0082354, 0.0058962, -0.0088168, 0.0206522, -0.0288876, 0.0147131
4: -0.0084861, 0.0065518, -0.0328849, 0.0244999, -0.0329051, 0.0394366
5: -0.0064485, 0.0161316, -0.0051254, 0.0540413, -0.0604899, 0.0212569
6: 0.0013103, 0.0116094, -0.0194701, 0.0151854, -0.0138751, 0.0310795
7: -0.0276268, -0.0043234, -0.0429649, 0.0392273, -0.0668541, 0.0386415
8: 0.9478363, 1.0105119, 0.8965118, 1.0138938, -0.0660576, 0.1140001
9: -0.0042755, 0.0208152, -0.0095789, 0.0877255, -0.0920011, 0.0303942

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_A1_B2_B2_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1035555, upper bound: 0.0983161
time: 0.95 seconds

## Relational analysis of IS_A1_A1_B2_B2_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1094888, upper bound: 0.1080613
time: 1.00 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0195158, 0.0037846, -0.0175789, 0.0038475, -0.0231397, 0.0213635
1: -0.0110764, 0.0069401, -0.0102611, 0.0069237, -0.0180001, 0.0172012
2: 0.0060706, 0.0305190, 0.0068085, 0.0289904, -0.0229197, 0.0237105
3: -0.0085345, 0.0112431, -0.0091627, 0.0104923, -0.0190268, 0.0204058
4: -0.0125515, 0.0132460, -0.0109390, 0.0132039, -0.0257554, 0.0241851
5: -0.0054809, 0.0300612, -0.0073045, 0.0281260, -0.0336069, 0.0373657
6: -0.0044684, 0.0120024, -0.0036795, 0.0119803, -0.0164487, 0.0156819
7: -0.0335687, 0.0016651, -0.0328062, -0.0006157, -0.0329529, 0.0344714
8: 0.9357346, 1.0160751, 0.9387430, 1.0163075, -0.0805729, 0.0760672
9: -0.0062916, 0.0442557, -0.0062034, 0.0412842, -0.0475757, 0.0504591

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A1

### Relational analysis result of IS_A1_A1_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0968259, upper bound: 0.0999164
time: 1.04 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A2

### Relational analysis result of IS_A1_A1_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1123864, upper bound: 0.1117345
time: 1.12 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0192396, 0.0035914, -0.0437107, 0.0037373, -0.0229768, 0.0473021
1: -0.0109740, 0.0066711, -0.0210832, 0.0204734, -0.0314474, 0.0277543
2: 0.0062592, 0.0301830, -0.0110408, 0.0517711, -0.0455119, 0.0412238
3: -0.0075755, 0.0111434, -0.0088168, 0.0206522, -0.0282277, 0.0199602
4: -0.0123154, 0.0124311, -0.0328849, 0.0244999, -0.0368153, 0.0453160
5: -0.0033397, 0.0298159, -0.0051254, 0.0540413, -0.0573810, 0.0349413
6: -0.0040105, 0.0119572, -0.0194701, 0.0151854, -0.0191959, 0.0314273
7: -0.0334725, 0.0009605, -0.0429649, 0.0392273, -0.0726998, 0.0439253
8: 0.9361358, 1.0146081, 0.8965118, 1.0138938, -0.0777581, 0.1180964
9: -0.0060476, 0.0434009, -0.0095789, 0.0877255, -0.0937732, 0.0529799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_A1_B2_B2_A2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1049022, upper bound: 0.0980559
time: 1.01 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1112619, upper bound: 0.1076649
time: 0.93 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0030529, 0.0034917, -0.0051999, 0.0032512, -0.0062327, 0.0085852
1: -0.0038226, 0.0062906, -0.0050100, 0.0059551, -0.0097777, 0.0113006
2: 0.0077696, 0.0213024, 0.0081984, 0.0213926, -0.0136230, 0.0131040
3: -0.0082696, 0.0053400, -0.0076872, 0.0062888, -0.0145583, 0.0130272
4: -0.0087865, 0.0056963, -0.0084846, 0.0065460, -0.0153325, 0.0141809
5: -0.0060420, 0.0149357, -0.0051782, 0.0169810, -0.0230229, 0.0201139
6: 0.0011369, 0.0117346, 0.0015606, 0.0116078, -0.0104709, 0.0101740
7: -0.0270008, -0.0037161, -0.0280714, -0.0045467, -0.0224541, 0.0241086
8: 0.9486815, 1.0123346, 0.9472360, 1.0099926, -0.0613111, 0.0650986
9: -0.0048824, 0.0182671, -0.0042680, 0.0222831, -0.0262358, 0.0224761

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_A2_B1_B1_A1_B1_A1

### Relational analysis result of IS_A1_A2_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1113761, upper bound: 0.1113761
time: 1.12 seconds

## Relational analysis of IS_A1_A2_B1_B1_A1_B1_A2

### Relational analysis result of IS_A1_A2_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1113761, upper bound: 0.1113761
time: 1.04 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0028440, 0.0033110, -0.0237278, 0.0029796, -0.0057434, 0.0269083
1: -0.0037197, 0.0060833, -0.0152494, 0.0056336, -0.0093533, 0.0210170
2: 0.0079383, 0.0211577, -0.0010504, 0.0223047, -0.0143664, 0.0222080
3: -0.0074448, 0.0052550, -0.0066361, 0.0144722, -0.0219170, 0.0118910
4: -0.0086708, 0.0047785, -0.0089014, 0.0172070, -0.0254803, 0.0136798
5: -0.0041927, 0.0147585, -0.0034648, 0.0346180, -0.0388107, 0.0182233
6: 0.0017096, 0.0116880, -0.0051666, 0.0115069, -0.0097972, 0.0167635
7: -0.0269080, -0.0042504, -0.0373041, -0.0001012, -0.0268068, 0.0325076
8: 0.9488067, 1.0109490, 0.9347709, 1.0076787, -0.0588720, 0.0761781
9: -0.0046564, 0.0175214, -0.0037791, 0.0578290, -0.0614333, 0.0213005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_A2_B1_B1_A1_B2_B1

### Relational analysis result of IS_A1_A2_B1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1029643, upper bound: 0.0970492
time: 1.45 seconds

## Relational analysis of IS_A1_A2_B1_B1_A1_B2_B2

### Relational analysis result of IS_A1_A2_B1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1103274, upper bound: 0.1103274
time: 1.24 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0177458, 0.0037962, -0.0051999, 0.0032512, -0.0209295, 0.0089961
1: -0.0103552, 0.0069181, -0.0050100, 0.0059551, -0.0163103, 0.0119281
2: 0.0065006, 0.0288837, 0.0081984, 0.0213926, -0.0148920, 0.0206854
3: -0.0079280, 0.0105618, -0.0076872, 0.0062888, -0.0142168, 0.0182490
4: -0.0110604, 0.0117997, -0.0084846, 0.0065460, -0.0176064, 0.0202843
5: -0.0037567, 0.0283331, -0.0051782, 0.0169810, -0.0207377, 0.0335113
6: -0.0031581, 0.0120779, 0.0015606, 0.0116078, -0.0147659, 0.0105173
7: -0.0328915, -0.0010486, -0.0280714, -0.0045467, -0.0283448, 0.0270228
8: 0.9385613, 1.0167232, 0.9472360, 1.0099926, -0.0714313, 0.0694872
9: -0.0066337, 0.0407473, -0.0042680, 0.0222831, -0.0289168, 0.0450153

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_A2_B1_B1_A2_B1_A1

### Relational analysis result of IS_A1_A2_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1136995, upper bound: 0.1111350
time: 1.21 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2_B1_A2

### Relational analysis result of IS_A1_A2_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1136995, upper bound: 0.1111350
time: 1.11 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0175105, 0.0036136, -0.0237278, 0.0029796, -0.0204133, 0.0271927
1: -0.0102697, 0.0066998, -0.0152494, 0.0056336, -0.0159033, 0.0219492
2: 0.0066770, 0.0286204, -0.0010504, 0.0223047, -0.0156277, 0.0296708
3: -0.0070754, 0.0104885, -0.0066361, 0.0144722, -0.0215475, 0.0171245
4: -0.0108686, 0.0110369, -0.0089014, 0.0172070, -0.0280756, 0.0199383
5: -0.0017614, 0.0281447, -0.0034648, 0.0346180, -0.0363795, 0.0316095
6: -0.0027206, 0.0120337, -0.0051666, 0.0115069, -0.0142275, 0.0172003
7: -0.0328140, -0.0017763, -0.0373041, -0.0001012, -0.0327127, 0.0355278
8: 0.9387287, 1.0153177, 0.9347709, 1.0076787, -0.0689500, 0.0805468
9: -0.0063952, 0.0400428, -0.0037791, 0.0578290, -0.0642242, 0.0438219

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_A2_B1_B1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1037274, upper bound: 0.0966930
time: 0.97 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1125675, upper bound: 0.1100775
time: 1.00 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0030529, 0.0034917, -0.0205653, 0.0035603, -0.0066132, 0.0239622
1: -0.0038226, 0.0062906, -0.0116970, 0.0067612, -0.0105838, 0.0178407
2: 0.0077696, 0.0213024, 0.0053211, 0.0315848, -0.0238152, 0.0159813
3: -0.0082696, 0.0053400, -0.0074531, 0.0116596, -0.0199292, 0.0127931
4: -0.0087865, 0.0056963, -0.0139789, 0.0129780, -0.0216398, 0.0196752
5: -0.0060420, 0.0149357, -0.0029828, 0.0311322, -0.0371741, 0.0179185
6: 0.0011369, 0.0117346, -0.0047944, 0.0119537, -0.0108168, 0.0165290
7: -0.0270008, -0.0037161, -0.0343234, 0.0029490, -0.0299498, 0.0306073
8: 0.9486815, 1.0123346, 0.9333725, 1.0143588, -0.0656773, 0.0789621
9: -0.0048824, 0.0182671, -0.0060158, 0.0469196, -0.0518019, 0.0242829

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_A2_B1_B2_A1_B1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1079591, upper bound: 0.1122083
time: 1.15 seconds

## Relational analysis of IS_A1_A2_B1_B2_A1_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1101933, upper bound: 0.1178666
time: 1.46 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0028440, 0.0033110, -0.0457211, 0.0098522, -0.0126963, 0.0489158
1: -0.0037197, 0.0060833, -0.0264469, 0.0215808, -0.0253005, 0.0322669
2: 0.0079383, 0.0211577, -0.0124754, 0.0599188, -0.0519805, 0.0336330
3: -0.0074448, 0.0052550, -0.0132466, 0.0214486, -0.0288934, 0.0185016
4: -0.0086708, 0.0047785, -0.0488367, 0.0243930, -0.0326525, 0.0536151
5: -0.0041927, 0.0147585, -0.0023952, 0.0560871, -0.0602798, 0.0171537
6: 0.0017096, 0.0116880, -0.0203899, 0.0216249, -0.0199153, 0.0320036
7: -0.0269080, -0.0042504, -0.0524669, 0.0421625, -0.0690705, 0.0482165
8: 0.9488067, 1.0109490, 0.8773209, 1.0117702, -0.0629635, 0.1336281
9: -0.0046564, 0.0175214, -0.0106404, 0.1211803, -0.1258367, 0.0281619

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_A2_B1_B2_A1_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1066593, upper bound: 0.1051630
time: 1.14 seconds

## Relational analysis of IS_A1_A2_B1_B2_A1_B2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1099237, upper bound: 0.1124840
time: 1.12 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0177458, 0.0037962, -0.0205653, 0.0035603, -0.0213062, 0.0243615
1: -0.0103552, 0.0069181, -0.0116970, 0.0067612, -0.0171164, 0.0186151
2: 0.0065006, 0.0288837, 0.0053211, 0.0315848, -0.0250842, 0.0235626
3: -0.0079280, 0.0105618, -0.0074531, 0.0116596, -0.0195876, 0.0180149
4: -0.0110604, 0.0117997, -0.0139789, 0.0129780, -0.0240384, 0.0257785
5: -0.0037567, 0.0283331, -0.0029828, 0.0311322, -0.0348889, 0.0313158
6: -0.0031581, 0.0120779, -0.0047944, 0.0119537, -0.0151117, 0.0168723
7: -0.0328915, -0.0010486, -0.0343234, 0.0029490, -0.0358405, 0.0332748
8: 0.9385613, 1.0167232, 0.9333725, 1.0143588, -0.0757974, 0.0833507
9: -0.0066337, 0.0407473, -0.0060158, 0.0469196, -0.0535532, 0.0467631

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_A2_B1_B2_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1102346, upper bound: 0.1097881
time: 0.93 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1126152, upper bound: 0.1144794
time: 1.35 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0175105, 0.0036136, -0.0457211, 0.0098522, -0.0273628, 0.0490437
1: -0.0102697, 0.0066998, -0.0264469, 0.0215808, -0.0318505, 0.0331467
2: 0.0066770, 0.0286204, -0.0124754, 0.0599188, -0.0532418, 0.0410958
3: -0.0070754, 0.0104885, -0.0132466, 0.0214486, -0.0285240, 0.0237350
4: -0.0108686, 0.0110369, -0.0488367, 0.0243930, -0.0352616, 0.0598736
5: -0.0017614, 0.0281447, -0.0023952, 0.0560871, -0.0578485, 0.0305399
6: -0.0027206, 0.0120337, -0.0203899, 0.0216249, -0.0243455, 0.0324236
7: -0.0328140, -0.0017763, -0.0524669, 0.0421625, -0.0749765, 0.0506906
8: 0.9387287, 1.0153177, 0.8773209, 1.0117702, -0.0723936, 0.1379968
9: -0.0063952, 0.0400428, -0.0106404, 0.1211803, -0.1275754, 0.0506833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1090485, upper bound: 0.1048051
time: 1.18 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1124840, upper bound: 0.1116854
time: 1.18 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0030529, 0.0034917, -0.0014593, 0.0034727, -0.0065079, 0.0048609
1: -0.0038226, 0.0062906, -0.0029142, 0.0061746, -0.0099972, 0.0092048
2: 0.0077696, 0.0213024, 0.0081321, 0.0213962, -0.0136266, 0.0131703
3: -0.0082696, 0.0053400, -0.0089263, 0.0047468, -0.0130164, 0.0142663
4: -0.0087865, 0.0056963, -0.0085392, 0.0060088, -0.0147953, 0.0142355
5: -0.0060420, 0.0149357, -0.0080598, 0.0132005, -0.0192425, 0.0229955
6: 0.0011369, 0.0117346, 0.0010578, 0.0116359, -0.0104990, 0.0106768
7: -0.0270008, -0.0037161, -0.0260925, -0.0039642, -0.0230366, 0.0223764
8: 0.9486815, 1.0123346, 0.9497153, 1.0114659, -0.0627844, 0.0626193
9: -0.0048824, 0.0182671, -0.0044040, 0.0160343, -0.0208164, 0.0225213

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1109709, upper bound: 0.1115017
time: 1.17 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1109709, upper bound: 0.1115017
time: 1.19 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0177458, 0.0037962, -0.0014593, 0.0034727, -0.0212047, 0.0052555
1: -0.0103552, 0.0069181, -0.0029142, 0.0061746, -0.0165298, 0.0098323
2: 0.0065006, 0.0288837, 0.0081321, 0.0213962, -0.0148956, 0.0207516
3: -0.0079280, 0.0105618, -0.0089263, 0.0047468, -0.0126748, 0.0194881
4: -0.0110604, 0.0117997, -0.0085392, 0.0060088, -0.0170692, 0.0203388
5: -0.0037567, 0.0283331, -0.0080598, 0.0132005, -0.0169572, 0.0363929
6: -0.0031581, 0.0120779, 0.0010578, 0.0116359, -0.0147939, 0.0110201
7: -0.0328915, -0.0010486, -0.0260925, -0.0039642, -0.0289274, 0.0250438
8: 0.9385613, 1.0167232, 0.9497153, 1.0114659, -0.0729046, 0.0670078
9: -0.0066337, 0.0407473, -0.0044040, 0.0160343, -0.0226680, 0.0451513

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1109709, upper bound: 0.1115017
time: 1.30 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1109709, upper bound: 0.1115017
time: 1.24 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0030529, 0.0034917, -0.0154271, 0.0037739, -0.0067540, 0.0188322
1: -0.0038226, 0.0062906, -0.0093805, 0.0068034, -0.0106260, 0.0156271
2: 0.0077696, 0.0213024, 0.0068454, 0.0275025, -0.0197329, 0.0144570
3: -0.0082696, 0.0053400, -0.0086044, 0.0097856, -0.0180552, 0.0139444
4: -0.0087865, 0.0056963, -0.0095041, 0.0116912, -0.0204777, 0.0152004
5: -0.0060420, 0.0149357, -0.0059525, 0.0262114, -0.0322534, 0.0208882
6: 0.0011369, 0.0117346, -0.0022087, 0.0119764, -0.0108395, 0.0139434
7: -0.0270008, -0.0037161, -0.0320122, -0.0021025, -0.0248983, 0.0282389
8: 0.9486815, 1.0123346, 0.9402128, 1.0158402, -0.0671587, 0.0721217
9: -0.0048824, 0.0182671, -0.0061489, 0.0378785, -0.0419311, 0.0244161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_A2_B2_B1_B2_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1107421, upper bound: 0.1152656
time: 0.98 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1107421, upper bound: 0.1152656
time: 1.11 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0177458, 0.0037962, -0.0154271, 0.0037739, -0.0213002, 0.0192233
1: -0.0103552, 0.0069181, -0.0093805, 0.0068034, -0.0171587, 0.0162986
2: 0.0065006, 0.0288837, 0.0068454, 0.0275025, -0.0210020, 0.0220383
3: -0.0079280, 0.0105618, -0.0086044, 0.0097856, -0.0177136, 0.0191662
4: -0.0110604, 0.0117997, -0.0095041, 0.0116912, -0.0227516, 0.0213038
5: -0.0037567, 0.0283331, -0.0059525, 0.0262114, -0.0299681, 0.0342856
6: -0.0031581, 0.0120779, -0.0022087, 0.0119764, -0.0151344, 0.0142866
7: -0.0328915, -0.0010486, -0.0320122, -0.0021025, -0.0307891, 0.0309636
8: 0.9385613, 1.0167232, 0.9402128, 1.0158402, -0.0772789, 0.0748101
9: -0.0066337, 0.0407473, -0.0061489, 0.0378785, -0.0444592, 0.0468963

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_A2_B2_B1_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1002778, upper bound: 0.0951601
time: 1.16 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1096904, upper bound: 0.1112686
time: 1.18 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028440, 0.0033110, -0.0244569, 0.0033006, -0.0061447, 0.0276448
1: -0.0037197, 0.0060833, -0.0139614, 0.0089104, -0.0126301, 0.0197751
2: 0.0079383, 0.0211577, 0.0025750, 0.0359852, -0.0280469, 0.0185827
3: -0.0074448, 0.0052550, -0.0086218, 0.0131660, -0.0206108, 0.0138768
4: -0.0086708, 0.0047785, -0.0193591, 0.0155492, -0.0239339, 0.0241375
5: -0.0041927, 0.0147585, -0.0066712, 0.0349633, -0.0391560, 0.0214297
6: 0.0017096, 0.0116880, -0.0074161, 0.0125686, -0.0108590, 0.0190143
7: -0.0269080, -0.0042504, -0.0371088, 0.0091385, -0.0360465, 0.0328584
8: 0.9488067, 1.0109490, 0.9247676, 1.0093809, -0.0605742, 0.0861814
9: -0.0046564, 0.0175214, -0.0041879, 0.0586625, -0.0633189, 0.0217093

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=8, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_A2_B2_B2_B1_A1_B1

### Relational analysis result of IS_A1_A2_B2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0923579, upper bound: 0.0854193
time: 0.96 seconds

## Relational analysis of IS_A1_A2_B2_B2_B1_A1_B2

### Relational analysis result of IS_A1_A2_B2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1092201, upper bound: 0.1073236
time: 1.37 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0175105, 0.0036136, -0.0244569, 0.0033006, -0.0208112, 0.0279293
1: -0.0102697, 0.0066998, -0.0139614, 0.0089104, -0.0191801, 0.0206612
2: 0.0066770, 0.0286204, 0.0025750, 0.0359852, -0.0293082, 0.0260454
3: -0.0070754, 0.0104885, -0.0086218, 0.0131660, -0.0202414, 0.0191103
4: -0.0108686, 0.0110369, -0.0193591, 0.0155492, -0.0264178, 0.0303960
5: -0.0017614, 0.0281447, -0.0066712, 0.0349633, -0.0367248, 0.0348158
6: -0.0027206, 0.0120337, -0.0074161, 0.0125686, -0.0152893, 0.0194498
7: -0.0328140, -0.0017763, -0.0371088, 0.0091385, -0.0419525, 0.0353325
8: 0.9387287, 1.0153177, 0.9247676, 1.0093809, -0.0706522, 0.0905501
9: -0.0063952, 0.0400428, -0.0041879, 0.0586625, -0.0650576, 0.0442308

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=8, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_A2_B2_B2_B1_A2_B1

### Relational analysis result of IS_A1_A2_B2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0923579, upper bound: 0.0854193
time: 1.06 seconds

## Relational analysis of IS_A1_A2_B2_B2_B1_A2_B2

### Relational analysis result of IS_A1_A2_B2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1092201, upper bound: 0.1073237
time: 1.07 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0028440, 0.0033110, -0.0430708, 0.0103677, -0.0132117, 0.0463818
1: -0.0037197, 0.0060833, -0.0250748, 0.0196402, -0.0233599, 0.0311581
2: 0.0079383, 0.0211577, -0.0137587, 0.0582798, -0.0503415, 0.0349164
3: -0.0074448, 0.0052550, -0.0120960, 0.0236724, -0.0311172, 0.0173510
4: -0.0086708, 0.0047785, -0.0433694, 0.0244695, -0.0331403, 0.0481478
5: -0.0041927, 0.0147585, -0.0048613, 0.0661123, -0.0703050, 0.0196198
6: 0.0017096, 0.0116880, -0.0190986, 0.0218156, -0.0201059, 0.0307388
7: -0.0269080, -0.0042504, -0.0500955, 0.0353679, -0.0622759, 0.0458450
8: 0.9488067, 1.0109490, 0.8678553, 1.0134218, -0.0646151, 0.1430937
9: -0.0046564, 0.0175214, -0.0143167, 0.1139988, -0.1186553, 0.0318381

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=17, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_A2_B2_B2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028084, upper bound: 0.0979650
time: 1.03 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1085064, upper bound: 0.1078270
time: 1.02 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0175105, 0.0036136, -0.0430708, 0.0103677, -0.0278782, 0.0465975
1: -0.0102697, 0.0066998, -0.0250748, 0.0196402, -0.0299099, 0.0317746
2: 0.0066770, 0.0286204, -0.0137587, 0.0582798, -0.0516028, 0.0423791
3: -0.0070754, 0.0104885, -0.0120960, 0.0236724, -0.0307478, 0.0225845
4: -0.0108686, 0.0110369, -0.0433694, 0.0244695, -0.0353381, 0.0544063
5: -0.0017614, 0.0281447, -0.0048613, 0.0661123, -0.0678737, 0.0330059
6: -0.0027206, 0.0120337, -0.0190986, 0.0218156, -0.0245362, 0.0311323
7: -0.0328140, -0.0017763, -0.0500955, 0.0353679, -0.0681819, 0.0483192
8: 0.9387287, 1.0153177, 0.8678553, 1.0134218, -0.0746323, 0.1474624
9: -0.0063952, 0.0400428, -0.0143167, 0.1139988, -0.1203940, 0.0543595

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=17, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_A2_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_A2_B2_B2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028084, upper bound: 0.0979650
time: 1.15 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1085064, upper bound: 0.1074971
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0029930, 0.0035612, -0.0049917, 0.0032690, -0.0061692, 0.0085262
1: -0.0037575, 0.0062576, -0.0048952, 0.0059755, -0.0097330, 0.0111528
2: 0.0080996, 0.0216237, 0.0081825, 0.0213760, -0.0132764, 0.0134412
3: -0.0094765, 0.0052950, -0.0077580, 0.0061970, -0.0156734, 0.0130530
4: -0.0085558, 0.0074387, -0.0084959, 0.0064813, -0.0150371, 0.0159346
5: -0.0094841, 0.0148236, -0.0053339, 0.0167832, -0.0262674, 0.0201575
6: 0.0006428, 0.0116399, 0.0015273, 0.0116127, -0.0109699, 0.0101126
7: -0.0269421, -0.0037441, -0.0279679, -0.0044940, -0.0224481, 0.0242239
8: 0.9487606, 1.0119759, 0.9473757, 1.0101304, -0.0613698, 0.0646002
9: -0.0044235, 0.0190531, -0.0042916, 0.0218724, -0.0256135, 0.0232328

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1007451, upper bound: 0.0937313
time: 1.10 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1120889, upper bound: 0.1095692
time: 1.44 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0029930, 0.0035612, -0.0236363, 0.0030022, -0.0058697, 0.0271752
1: -0.0037575, 0.0062576, -0.0151987, 0.0056594, -0.0094169, 0.0214170
2: 0.0080996, 0.0216237, -0.0009637, 0.0223034, -0.0142039, 0.0225874
3: -0.0094765, 0.0052950, -0.0067404, 0.0144317, -0.0239081, 0.0120354
4: -0.0085558, 0.0074387, -0.0088657, 0.0171705, -0.0254712, 0.0163044
5: -0.0094841, 0.0148236, -0.0037143, 0.0345305, -0.0440147, 0.0185379
6: 0.0006428, 0.0116399, -0.0051277, 0.0115113, -0.0108685, 0.0167676
7: -0.0269421, -0.0037441, -0.0372583, -0.0001868, -0.0267553, 0.0335142
8: 0.9487606, 1.0119759, 0.9348328, 1.0078447, -0.0590841, 0.0771431
9: -0.0044235, 0.0190531, -0.0038006, 0.0576610, -0.0612634, 0.0228537

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1007451, upper bound: 0.0937313
time: 1.35 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1120889, upper bound: 0.1095692
time: 1.59 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0088907, 0.0033843, -0.0015033, 0.0033396, -0.0121543, 0.0048508
1: -0.0070307, 0.0060372, -0.0029554, 0.0060295, -0.0130450, 0.0089926
2: 0.0083363, 0.0218088, 0.0082252, 0.0212463, -0.0129100, 0.0135836
3: -0.0088360, 0.0079079, -0.0082084, 0.0047500, -0.0135860, 0.0161163
4: -0.0083745, 0.0097643, -0.0084745, 0.0051873, -0.0135618, 0.0182387
5: -0.0082346, 0.0204616, -0.0063080, 0.0133065, -0.0215411, 0.0267697
6: 0.0004920, 0.0115527, 0.0014992, 0.0116094, -0.0111174, 0.0100535
7: -0.0298935, -0.0042756, -0.0261480, -0.0043403, -0.0252883, 0.0218724
8: 0.9447759, 1.0103499, 0.9496799, 1.0105027, -0.0657268, 0.0606700
9: -0.0040010, 0.0298266, -0.0042760, 0.0156359, -0.0196368, 0.0332862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A1_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0866325, upper bound: 0.0941533
time: 1.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0866325, upper bound: 0.0941533
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0199803, 0.0032972, -0.0050586, 0.0033645, -0.0232373, 0.0082975
1: -0.0131573, 0.0059682, -0.0049229, 0.0060656, -0.0189747, 0.0108911
2: 0.0025074, 0.0223593, 0.0081674, 0.0214790, -0.0189716, 0.0141918
3: -0.0083981, 0.0128047, -0.0083465, 0.0062211, -0.0146193, 0.0211512
4: -0.0083925, 0.0159596, -0.0085048, 0.0070307, -0.0154232, 0.0244272
5: -0.0076804, 0.0310143, -0.0067117, 0.0168309, -0.0245113, 0.0377260
6: -0.0036992, 0.0115401, 0.0011739, 0.0116155, -0.0153147, 0.0103662
7: -0.0354176, -0.0029254, -0.0279929, -0.0042466, -0.0311711, 0.0250675
8: 0.9373178, 1.0097879, 0.9473420, 1.0106995, -0.0733817, 0.0624459
9: -0.0039400, 0.0510822, -0.0043051, 0.0222527, -0.0261928, 0.0543287

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A1_B1_A2_A2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0978632, upper bound: 0.1029768
time: 1.11 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_A2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1066044, upper bound: 0.1079780
time: 1.22 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0029930, 0.0035612, -0.0202348, 0.0035620, -0.0065550, 0.0237824
1: -0.0037575, 0.0062576, -0.0115055, 0.0067060, -0.0104636, 0.0177631
2: 0.0080996, 0.0216237, 0.0055540, 0.0312044, -0.0231048, 0.0160697
3: -0.0094765, 0.0052950, -0.0073972, 0.0115320, -0.0210084, 0.0126921
4: -0.0085558, 0.0074387, -0.0135224, 0.0127540, -0.0212947, 0.0209612
5: -0.0094841, 0.0148236, -0.0028507, 0.0308082, -0.0402923, 0.0176744
6: 0.0006428, 0.0116399, -0.0045645, 0.0119572, -0.0113143, 0.0162044
7: -0.0269421, -0.0037441, -0.0340878, 0.0024168, -0.0293590, 0.0303437
8: 0.9487606, 1.0119759, 0.9341005, 1.0144036, -0.0656430, 0.0778754
9: -0.0044235, 0.0190531, -0.0060347, 0.0459065, -0.0503300, 0.0250878

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1048011, upper bound: 0.1006397
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105424, upper bound: 0.1103586
time: 1.16 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0029930, 0.0035612, -0.0457147, 0.0098504, -0.0128434, 0.0492684
1: -0.0037575, 0.0062576, -0.0264429, 0.0215778, -0.0253353, 0.0327005
2: 0.0080996, 0.0216237, -0.0124708, 0.0599116, -0.0518120, 0.0340944
3: -0.0094765, 0.0052950, -0.0132507, 0.0214460, -0.0309224, 0.0185457
4: -0.0085558, 0.0074387, -0.0488276, 0.0243961, -0.0326698, 0.0562663
5: -0.0094841, 0.0148236, -0.0024567, 0.0560805, -0.0655646, 0.0172803
6: 0.0006428, 0.0116399, -0.0203882, 0.0216226, -0.0209797, 0.0320281
7: -0.0269421, -0.0037441, -0.0524620, 0.0421541, -0.0690963, 0.0487180
8: 0.9487606, 1.0119759, 0.8773362, 1.0118749, -0.0631143, 0.1346397
9: -0.0044235, 0.0190531, -0.0106374, 0.1211638, -0.1255873, 0.0296905

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1048011, upper bound: 0.1006397
time: 1.07 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105424, upper bound: 0.1103586
time: 1.16 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0139207, 0.0031581, -0.0163289, 0.0035394, -0.0173212, 0.0193636
1: -0.0098403, 0.0058469, -0.0097790, 0.0065593, -0.0163996, 0.0156259
2: 0.0082037, 0.0217275, 0.0069461, 0.0278607, -0.0196570, 0.0147814
3: -0.0072010, 0.0101468, -0.0071426, 0.0100968, -0.0172978, 0.0172895
4: -0.0084174, 0.0110171, -0.0099400, 0.0106195, -0.0190369, 0.0209571
5: -0.0042204, 0.0253010, -0.0022685, 0.0270778, -0.0312982, 0.0275696
6: -0.0001442, 0.0115676, -0.0020572, 0.0119558, -0.0121000, 0.0136248
7: -0.0324269, -0.0047908, -0.0323715, -0.0027630, -0.0296638, 0.0275807
8: 0.9413556, 1.0091732, 0.9395477, 1.0143075, -0.0729519, 0.0696255
9: -0.0040732, 0.0387371, -0.0060216, 0.0383951, -0.0423191, 0.0447587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A1_B2_A2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0986220, upper bound: 0.1042314
time: 1.55 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0986220, upper bound: 0.1042314
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0195223, 0.0032216, -0.0201942, 0.0036125, -0.0231348, 0.0233488
1: -0.0129145, 0.0058925, -0.0114763, 0.0067742, -0.0196887, 0.0173687
2: 0.0029312, 0.0222193, 0.0055849, 0.0311851, -0.0282539, 0.0166344
3: -0.0078746, 0.0126084, -0.0076998, 0.0115137, -0.0193883, 0.0203083
4: -0.0083811, 0.0152305, -0.0134625, 0.0129457, -0.0213268, 0.0286929
5: -0.0062891, 0.0305962, -0.0035634, 0.0307587, -0.0370478, 0.0341596
6: -0.0032664, 0.0115396, -0.0046221, 0.0119660, -0.0152324, 0.0161617
7: -0.0351987, -0.0035101, -0.0340519, 0.0024342, -0.0376329, 0.0305418
8: 0.9376134, 1.0093395, 0.9342114, 1.0147709, -0.0771575, 0.0751280
9: -0.0039377, 0.0499790, -0.0060854, 0.0458851, -0.0498229, 0.0560645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_A1_B2_A2_A2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0826738, upper bound: 0.0889897
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_A2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1062215, upper bound: 0.1085899
time: 1.52 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0175789, 0.0038475, -0.0049917, 0.0032690, -0.0207671, 0.0087533
1: -0.0102611, 0.0069237, -0.0048952, 0.0059755, -0.0162366, 0.0118189
2: 0.0068085, 0.0289904, 0.0081825, 0.0213760, -0.0145675, 0.0208079
3: -0.0091627, 0.0104923, -0.0077580, 0.0061970, -0.0153597, 0.0182503
4: -0.0109390, 0.0132039, -0.0084959, 0.0064813, -0.0174203, 0.0216998
5: -0.0073045, 0.0281260, -0.0053339, 0.0167832, -0.0240877, 0.0334599
6: -0.0036795, 0.0119803, 0.0015273, 0.0116127, -0.0152922, 0.0104530
7: -0.0328062, -0.0006157, -0.0279679, -0.0044940, -0.0283122, 0.0273522
8: 0.9387430, 1.0163075, 0.9473757, 1.0101304, -0.0713874, 0.0689318
9: -0.0062034, 0.0412842, -0.0042916, 0.0218724, -0.0280758, 0.0449414

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1014867, upper bound: 0.0931794
time: 1.18 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1151183, upper bound: 0.1093742
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0175789, 0.0038475, -0.0236363, 0.0030022, -0.0204677, 0.0274028
1: -0.0102611, 0.0069237, -0.0151987, 0.0056594, -0.0159205, 0.0221224
2: 0.0068085, 0.0289904, -0.0009637, 0.0223034, -0.0154949, 0.0299541
3: -0.0091627, 0.0104923, -0.0067404, 0.0144317, -0.0235944, 0.0172327
4: -0.0109390, 0.0132039, -0.0088657, 0.0171705, -0.0281095, 0.0220696
5: -0.0073045, 0.0281260, -0.0037143, 0.0345305, -0.0418350, 0.0318403
6: -0.0036795, 0.0119803, -0.0051277, 0.0115113, -0.0151908, 0.0171080
7: -0.0328062, -0.0006157, -0.0372583, -0.0001868, -0.0326194, 0.0366426
8: 0.9387430, 1.0163075, 0.9348328, 1.0078447, -0.0691017, 0.0814747
9: -0.0062034, 0.0412842, -0.0038006, 0.0576610, -0.0638644, 0.0450848

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1014867, upper bound: 0.0931794
time: 1.05 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1151183, upper bound: 0.1093742
time: 1.50 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0326985, 0.0034253, -0.0022423, 0.0032857, -0.0359006, 0.0056675
1: -0.0165527, 0.0136993, -0.0033820, 0.0059834, -0.0223714, 0.0170814
2: -0.0032663, 0.0419784, 0.0082151, 0.0211845, -0.0244508, 0.0337633
3: -0.0068817, 0.0163853, -0.0078232, 0.0049863, -0.0118680, 0.0242085
4: -0.0236224, 0.0181222, -0.0084793, 0.0050050, -0.0286274, 0.0262325
5: -0.0013356, 0.0431845, -0.0053672, 0.0141768, -0.0155125, 0.0485517
6: -0.0121617, 0.0129614, 0.0016652, 0.0116106, -0.0236783, 0.0112962
7: -0.0387108, 0.0217049, -0.0266036, -0.0044687, -0.0342421, 0.0483085
8: 0.9142694, 1.0133413, 0.9492178, 1.0102041, -0.0959346, 0.0641235
9: -0.0059197, 0.0672948, -0.0042813, 0.0165580, -0.0224777, 0.0715762

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A2_B1_A2_A1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0982920, upper bound: 0.1028860
time: 1.19 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_A1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0982920, upper bound: 0.1028860
time: 1.18 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0405089, 0.0035339, -0.0049757, 0.0033614, -0.0438023, 0.0085096
1: -0.0197691, 0.0184920, -0.0048776, 0.0060625, -0.0256978, 0.0233696
2: -0.0087822, 0.0489266, 0.0081689, 0.0214697, -0.0302519, 0.0407577
3: -0.0078733, 0.0194137, -0.0083257, 0.0061849, -0.0140582, 0.0277394
4: -0.0301908, 0.0224915, -0.0085040, 0.0069633, -0.0371541, 0.0309472
5: -0.0033317, 0.0508923, -0.0066579, 0.0167529, -0.0200846, 0.0575502
6: -0.0172969, 0.0145283, 0.0011935, 0.0116153, -0.0289123, 0.0133347
7: -0.0417310, 0.0341085, -0.0279521, -0.0042551, -0.0374759, 0.0620606
8: 0.9016622, 1.0133090, 0.9473971, 1.0106807, -0.1090184, 0.0659119
9: -0.0077360, 0.0817166, -0.0043045, 0.0220834, -0.0298194, 0.0860211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A2_B1_A2_A2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1080304, upper bound: 0.1085189
time: 1.10 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_A2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1080304, upper bound: 0.1085189
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0123690, 0.0038217, -0.0050457, 0.0039091, -0.0160300, 0.0086404
1: -0.0081084, 0.0068406, -0.0049089, 0.0069594, -0.0144088, 0.0117496
2: 0.0068625, 0.0255872, 0.0067222, 0.0215449, -0.0146824, 0.0187254
3: -0.0088975, 0.0087708, -0.0088197, 0.0062114, -0.0151090, 0.0175252
4: -0.0094693, 0.0107432, -0.0095098, 0.0070665, -0.0165358, 0.0201787
5: -0.0068370, 0.0234459, -0.0063240, 0.0168069, -0.0236439, 0.0297699
6: -0.0009000, 0.0119743, 0.0001781, 0.0120272, -0.0129272, 0.0117962
7: -0.0308652, -0.0023334, -0.0279803, -0.0020926, -0.0274846, 0.0256470
8: 0.9423360, 1.0161040, 0.9473590, 1.0170119, -0.0739722, 0.0687450
9: -0.0061131, 0.0336965, -0.0062996, 0.0223642, -0.0279465, 0.0379059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0986249, upper bound: 0.0908082
time: 1.23 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0986249, upper bound: 0.0908082
time: 1.71 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0175789, 0.0038475, -0.0181752, 0.0037917, -0.0213706, 0.0217695
1: -0.0102611, 0.0069237, -0.0105220, 0.0068779, -0.0171390, 0.0174457
2: 0.0068085, 0.0289904, 0.0067474, 0.0293448, -0.0225363, 0.0222430
3: -0.0091627, 0.0104923, -0.0085291, 0.0107218, -0.0198845, 0.0190214
4: -0.0109390, 0.0132039, -0.0114248, 0.0126263, -0.0235653, 0.0246287
5: -0.0073045, 0.0281260, -0.0054895, 0.0287326, -0.0360371, 0.0336155
6: -0.0036795, 0.0119803, -0.0036684, 0.0120021, -0.0156817, 0.0156487
7: -0.0328062, -0.0006157, -0.0330481, -0.0002512, -0.0325550, 0.0324323
8: 0.9387430, 1.0163075, 0.9379075, 1.0161368, -0.0763080, 0.0784000
9: -0.0062034, 0.0412842, -0.0062922, 0.0418447, -0.0480481, 0.0475764

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1151183, upper bound: 0.1099181
time: 1.28 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1151183, upper bound: 0.1099181
time: 1.20 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0326985, 0.0034253, -0.0163289, 0.0035394, -0.0359581, 0.0197542
1: -0.0165527, 0.0136993, -0.0097790, 0.0065593, -0.0231120, 0.0234783
2: -0.0032663, 0.0419784, 0.0069461, 0.0278607, -0.0311271, 0.0350323
3: -0.0068817, 0.0163853, -0.0071426, 0.0100968, -0.0169785, 0.0235279
4: -0.0236224, 0.0181222, -0.0099400, 0.0106195, -0.0342419, 0.0280622
5: -0.0013356, 0.0431845, -0.0022685, 0.0270778, -0.0284134, 0.0454530
6: -0.0121617, 0.0129614, -0.0020572, 0.0119558, -0.0241175, 0.0150186
7: -0.0387108, 0.0217049, -0.0323715, -0.0027630, -0.0359478, 0.0540764
8: 0.9142694, 1.0133413, 0.9395477, 1.0143075, -0.1000381, 0.0716024
9: -0.0059197, 0.0672948, -0.0060216, 0.0383951, -0.0443148, 0.0733164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A2_B2_A2_A1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0968112, upper bound: 0.1001533
time: 1.03 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A2_B2_A2_A1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0980260, upper bound: 0.1028860
time: 1.16 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_A1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0980260, upper bound: 0.1028860
time: 1.17 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0405089, 0.0035339, -0.0201942, 0.0036125, -0.0441214, 0.0237281
1: -0.0197691, 0.0184920, -0.0114763, 0.0067742, -0.0265433, 0.0299682
2: -0.0087822, 0.0489266, 0.0055849, 0.0311851, -0.0399673, 0.0433418
3: -0.0078733, 0.0194137, -0.0076998, 0.0115137, -0.0193870, 0.0271136
4: -0.0301908, 0.0224915, -0.0134625, 0.0129457, -0.0431365, 0.0359540
5: -0.0033317, 0.0508923, -0.0035634, 0.0307587, -0.0340904, 0.0544556
6: -0.0172969, 0.0145283, -0.0046221, 0.0119660, -0.0292629, 0.0191504
7: -0.0417310, 0.0341085, -0.0340519, 0.0024342, -0.0441652, 0.0681604
8: 0.9016622, 1.0133090, 0.9342114, 1.0147709, -0.1131086, 0.0790976
9: -0.0077360, 0.0817166, -0.0060854, 0.0458851, -0.0536212, 0.0878020

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_A2_B2_A2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0918585, upper bound: 0.0873914
time: 1.04 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1069070, upper bound: 0.1078033
time: 1.17 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0015934, 0.0034747, -0.0030529, 0.0034917, -0.0049931, 0.0065091
1: -0.0029917, 0.0061774, -0.0038226, 0.0062906, -0.0092823, 0.0100000
2: 0.0081310, 0.0214071, 0.0077696, 0.0213024, -0.0131714, 0.0136375
3: -0.0089322, 0.0047590, -0.0082696, 0.0053400, -0.0142722, 0.0130286
4: -0.0085392, 0.0060931, -0.0087865, 0.0056963, -0.0142354, 0.0148796
5: -0.0080637, 0.0134000, -0.0060420, 0.0149357, -0.0229994, 0.0194419
6: 0.0010452, 0.0116359, 0.0011369, 0.0117346, -0.0106894, 0.0104990
7: -0.0261968, -0.0039572, -0.0270008, -0.0037161, -0.0224808, 0.0230436
8: 0.9496489, 1.0114819, 0.9486815, 1.0123346, -0.0626857, 0.0628004
9: -0.0044040, 0.0162063, -0.0048824, 0.0182671, -0.0225332, 0.0209846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1086644, upper bound: 0.1102918
time: 1.03 seconds

## Relational analysis of IS_A2_B2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1086644, upper bound: 0.1102918
time: 1.32 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0137078, 0.0032326, -0.0028440, 0.0033110, -0.0168965, 0.0060340
1: -0.0100065, 0.0059051, -0.0037197, 0.0060833, -0.0158934, 0.0096248
2: 0.0083883, 0.0222912, 0.0079383, 0.0211577, -0.0127694, 0.0143529
3: -0.0079565, 0.0058575, -0.0074448, 0.0052550, -0.0132115, 0.0133023
4: -0.0083037, 0.0158676, -0.0086708, 0.0047785, -0.0130821, 0.0242699
5: -0.0055736, 0.0314514, -0.0041927, 0.0147585, -0.0203321, 0.0356440
6: -0.0008486, 0.0115397, 0.0017096, 0.0116880, -0.0125366, 0.0098301
7: -0.0356464, -0.0046918, -0.0269080, -0.0042504, -0.0309650, 0.0222162
8: 0.9436315, 1.0094063, 0.9488067, 1.0109490, -0.0673175, 0.0605996
9: -0.0039382, 0.0319856, -0.0046564, 0.0175214, -0.0214596, 0.0356581

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B2_B1_A1_B1_A2_A1

### Relational analysis result of IS_A2_B2_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0865077, upper bound: 0.0943210
time: 0.90 seconds

## Relational analysis of IS_A2_B2_B1_A1_B1_A2_A2

### Relational analysis result of IS_A2_B2_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1076397, upper bound: 0.1092201
time: 1.08 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0015934, 0.0034747, -0.0177458, 0.0037962, -0.0053896, 0.0212059
1: -0.0029917, 0.0061774, -0.0103552, 0.0069181, -0.0099098, 0.0165326
2: 0.0081310, 0.0214071, 0.0065006, 0.0288837, -0.0207527, 0.0149065
3: -0.0089322, 0.0047590, -0.0079280, 0.0105618, -0.0194940, 0.0126870
4: -0.0085392, 0.0060931, -0.0110604, 0.0117997, -0.0203388, 0.0171535
5: -0.0080637, 0.0134000, -0.0037567, 0.0283331, -0.0363968, 0.0171567
6: 0.0010452, 0.0116359, -0.0031581, 0.0120779, -0.0110327, 0.0147939
7: -0.0261968, -0.0039572, -0.0328915, -0.0010486, -0.0251482, 0.0289343
8: 0.9496489, 1.0114819, 0.9385613, 1.0167232, -0.0670743, 0.0729206
9: -0.0044040, 0.0162063, -0.0066337, 0.0407473, -0.0451513, 0.0228400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=6, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1083416, upper bound: 0.1116318
time: 1.15 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1083416, upper bound: 0.1116318
time: 1.14 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0137078, 0.0032326, -0.0175105, 0.0036136, -0.0171810, 0.0207040
1: -0.0100065, 0.0059051, -0.0102697, 0.0066998, -0.0167063, 0.0161748
2: 0.0083883, 0.0222912, 0.0066770, 0.0286204, -0.0202321, 0.0156142
3: -0.0079565, 0.0058575, -0.0070754, 0.0104885, -0.0184450, 0.0129329
4: -0.0083037, 0.0158676, -0.0108686, 0.0110369, -0.0193406, 0.0267362
5: -0.0055736, 0.0314514, -0.0017614, 0.0281447, -0.0337183, 0.0332128
6: -0.0008486, 0.0115397, -0.0027206, 0.0120337, -0.0128823, 0.0142603
7: -0.0356464, -0.0046918, -0.0328140, -0.0017763, -0.0338701, 0.0281221
8: 0.9436315, 1.0094063, 0.9387287, 1.0153177, -0.0716861, 0.0706776
9: -0.0039382, 0.0319856, -0.0063952, 0.0400428, -0.0439810, 0.0383807

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B2_B1_A1_B2_A2_A1

### Relational analysis result of IS_A2_B2_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0854193, upper bound: 0.0923579
time: 1.36 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2_A2_A2

### Relational analysis result of IS_A2_B2_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1073237, upper bound: 0.1105236
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0156411, 0.0037755, -0.0030529, 0.0034917, -0.0190463, 0.0067550
1: -0.0094690, 0.0068055, -0.0038226, 0.0062906, -0.0157130, 0.0106281
2: 0.0068440, 0.0276422, 0.0077696, 0.0213024, -0.0144584, 0.0198726
3: -0.0086130, 0.0098564, -0.0082696, 0.0053400, -0.0139530, 0.0181260
4: -0.0095072, 0.0117891, -0.0087865, 0.0056963, -0.0152035, 0.0205756
5: -0.0059633, 0.0264039, -0.0060420, 0.0149357, -0.0208990, 0.0324459
6: -0.0023315, 0.0119764, 0.0011369, 0.0117346, -0.0140661, 0.0108395
7: -0.0320920, -0.0020450, -0.0270008, -0.0037161, -0.0283168, 0.0249558
8: 0.9400651, 1.0158503, 0.9486815, 1.0123346, -0.0722694, 0.0671688
9: -0.0061514, 0.0381885, -0.0048824, 0.0182671, -0.0244185, 0.0422387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1090831, upper bound: 0.1097157
time: 1.15 seconds

## Relational analysis of IS_A2_B2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1090831, upper bound: 0.1097157
time: 1.41 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0418727, 0.0034804, -0.0028440, 0.0033110, -0.0450769, 0.0062517
1: -0.0203305, 0.0181668, -0.0037197, 0.0060833, -0.0261892, 0.0218865
2: -0.0097452, 0.0447819, 0.0079383, 0.0211577, -0.0309029, 0.0367031
3: -0.0081115, 0.0185357, -0.0074448, 0.0052550, -0.0133665, 0.0259806
4: -0.0300388, 0.0232558, -0.0086708, 0.0047785, -0.0348172, 0.0316183
5: -0.0037476, 0.0500180, -0.0041927, 0.0147585, -0.0185061, 0.0542107
6: -0.0181970, 0.0122213, 0.0017096, 0.0116880, -0.0298269, 0.0105116
7: -0.0418856, 0.0362727, -0.0269080, -0.0042504, -0.0372820, 0.0631807
8: 0.9219365, 1.0134249, 0.9488067, 1.0109490, -0.0890125, 0.0646182
9: -0.0085155, 0.0758684, -0.0046564, 0.0175214, -0.0260369, 0.0797404

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_B1_A2_B1_A2_A1

### Relational analysis result of IS_A2_B2_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0981316, upper bound: 0.1028084
time: 0.99 seconds

## Relational analysis of IS_A2_B2_B1_A2_B1_A2_A2

### Relational analysis result of IS_A2_B2_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1078603, upper bound: 0.1085064
time: 1.29 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0156411, 0.0037755, -0.0177458, 0.0037962, -0.0194373, 0.0213013
1: -0.0094690, 0.0068055, -0.0103552, 0.0069181, -0.0163871, 0.0171607
2: 0.0068440, 0.0276422, 0.0065006, 0.0288837, -0.0220397, 0.0211417
3: -0.0086130, 0.0098564, -0.0079280, 0.0105618, -0.0191748, 0.0177844
4: -0.0095072, 0.0117891, -0.0110604, 0.0117997, -0.0213069, 0.0228494
5: -0.0059633, 0.0264039, -0.0037567, 0.0283331, -0.0342964, 0.0301606
6: -0.0023315, 0.0119764, -0.0031581, 0.0120779, -0.0144094, 0.0151344
7: -0.0320920, -0.0020450, -0.0328915, -0.0010486, -0.0310434, 0.0308465
8: 0.9400651, 1.0158503, 0.9385613, 1.0167232, -0.0749502, 0.0772890
9: -0.0061514, 0.0381885, -0.0066337, 0.0407473, -0.0468988, 0.0447676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B2_B1_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0959335, upper bound: 0.1002991
time: 0.96 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1143392, upper bound: 0.1108955
time: 1.21 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0418727, 0.0034804, -0.0175105, 0.0036136, -0.0452026, 0.0207685
1: -0.0203305, 0.0181668, -0.0102697, 0.0066998, -0.0270303, 0.0284365
2: -0.0097452, 0.0447819, 0.0066770, 0.0286204, -0.0383657, 0.0381050
3: -0.0081115, 0.0185357, -0.0070754, 0.0104885, -0.0186000, 0.0255800
4: -0.0300388, 0.0232558, -0.0108686, 0.0110369, -0.0410756, 0.0341244
5: -0.0037476, 0.0500180, -0.0017614, 0.0281447, -0.0318922, 0.0517794
6: -0.0181970, 0.0122213, -0.0027206, 0.0120337, -0.0302307, 0.0149419
7: -0.0418856, 0.0362727, -0.0328140, -0.0017763, -0.0401093, 0.0690867
8: 0.9219365, 1.0134249, 0.9387287, 1.0153177, -0.0884096, 0.0746353
9: -0.0085155, 0.0758684, -0.0063952, 0.0400428, -0.0485583, 0.0821963

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_B1_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0979650, upper bound: 0.1028098
time: 1.03 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1078270, upper bound: 0.1090359
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0015934, 0.0034747, -0.0009213, 0.0037073, -0.0052584, 0.0043960
1: -0.0029917, 0.0061774, -0.0026224, 0.0065050, -0.0094968, 0.0087997
2: 0.0081310, 0.0214071, 0.0076852, 0.0212936, -0.0131626, 0.0137219
3: -0.0089322, 0.0047590, -0.0094824, 0.0046227, -0.0135549, 0.0142414
4: -0.0085392, 0.0060931, -0.0088514, 0.0060914, -0.0146306, 0.0149445
5: -0.0080637, 0.0134000, -0.0088992, 0.0125430, -0.0206067, 0.0222992
6: 0.0010452, 0.0116359, 0.0005402, 0.0117634, -0.0107181, 0.0110956
7: -0.0261968, -0.0039572, -0.0256288, -0.0031393, -0.0230576, 0.0216716
8: 0.9496489, 1.0114819, 0.9503613, 1.0137819, -0.0641330, 0.0611206
9: -0.0044040, 0.0162063, -0.0050216, 0.0154503, -0.0198542, 0.0210511

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1086398, upper bound: 0.1086398
time: 1.06 seconds

## Relational analysis of IS_A2_B2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1086398, upper bound: 0.1086398
time: 1.40 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0137078, 0.0032326, -0.0008801, 0.0035198, -0.0171560, 0.0040895
1: -0.0100065, 0.0059051, -0.0026143, 0.0062913, -0.0161387, 0.0085194
2: 0.0083883, 0.0222912, 0.0078463, 0.0211436, -0.0127553, 0.0144449
3: -0.0079565, 0.0058575, -0.0086251, 0.0046089, -0.0125654, 0.0144826
4: -0.0083037, 0.0158676, -0.0087397, 0.0051603, -0.0134640, 0.0242113
5: -0.0055736, 0.0314514, -0.0070077, 0.0125339, -0.0181075, 0.0384591
6: -0.0008486, 0.0115397, 0.0011012, 0.0117177, -0.0125664, 0.0104385
7: -0.0356464, -0.0046918, -0.0256091, -0.0036923, -0.0315442, 0.0209173
8: 0.9436315, 1.0094063, 0.9504176, 1.0123553, -0.0687238, 0.0589887
9: -0.0039382, 0.0319856, -0.0048006, 0.0149929, -0.0189311, 0.0357331

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B2_B2_A1_B1_A2_A1

### Relational analysis result of IS_A2_B2_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0864871, upper bound: 0.0934949
time: 0.91 seconds

## Relational analysis of IS_A2_B2_B2_A1_B1_A2_A2

### Relational analysis result of IS_A2_B2_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1076189, upper bound: 0.1076189
time: 1.11 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0015934, 0.0034747, -0.0127855, 0.0040169, -0.0055177, 0.0162459
1: -0.0029917, 0.0061774, -0.0082867, 0.0071503, -0.0101420, 0.0144641
2: 0.0081310, 0.0214071, 0.0063757, 0.0258024, -0.0176714, 0.0150314
3: -0.0089322, 0.0047590, -0.0090949, 0.0089116, -0.0178438, 0.0138539
4: -0.0085392, 0.0060931, -0.0098079, 0.0106183, -0.0191574, 0.0159010
5: -0.0080637, 0.0134000, -0.0066803, 0.0238335, -0.0318972, 0.0200802
6: 0.0010452, 0.0116359, -0.0011428, 0.0121129, -0.0110676, 0.0127787
7: -0.0261968, -0.0039572, -0.0310260, -0.0015891, -0.0246077, 0.0270688
8: 0.9496489, 1.0114819, 0.9420384, 1.0182815, -0.0686326, 0.0694435
9: -0.0044040, 0.0162063, -0.0067846, 0.0341062, -0.0379988, 0.0229909

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1082809, upper bound: 0.1090539
time: 0.97 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1082809, upper bound: 0.1090539
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0137078, 0.0032326, -0.0125483, 0.0038325, -0.0174231, 0.0157425
1: -0.0100065, 0.0059051, -0.0082004, 0.0069340, -0.0169405, 0.0141055
2: 0.0083883, 0.0222912, 0.0065438, 0.0255262, -0.0171379, 0.0157474
3: -0.0079565, 0.0058575, -0.0081904, 0.0088391, -0.0167956, 0.0140479
4: -0.0083037, 0.0158676, -0.0096802, 0.0097755, -0.0180792, 0.0255478
5: -0.0055736, 0.0314514, -0.0046118, 0.0236459, -0.0292195, 0.0360632
6: -0.0008486, 0.0115397, -0.0005186, 0.0120698, -0.0129185, 0.0120583
7: -0.0356464, -0.0046918, -0.0309482, -0.0021779, -0.0334686, 0.0262563
8: 0.9436315, 1.0094063, 0.9421825, 1.0168518, -0.0732203, 0.0672238
9: -0.0039382, 0.0319856, -0.0065604, 0.0333830, -0.0373211, 0.0385459

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_B2_A1_B2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0985033, upper bound: 0.1034645
time: 1.05 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1070807, upper bound: 0.1078301
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0156411, 0.0037755, -0.0009213, 0.0037073, -0.0193104, 0.0046446
1: -0.0094690, 0.0068055, -0.0026224, 0.0065050, -0.0159638, 0.0094278
2: 0.0068440, 0.0276422, 0.0076852, 0.0212936, -0.0144496, 0.0194883
3: -0.0086130, 0.0098564, -0.0094824, 0.0046227, -0.0132357, 0.0193388
4: -0.0095072, 0.0117891, -0.0088514, 0.0060914, -0.0155986, 0.0206405
5: -0.0059633, 0.0264039, -0.0088992, 0.0125430, -0.0185062, 0.0353031
6: -0.0023315, 0.0119764, 0.0005402, 0.0117634, -0.0140948, 0.0114361
7: -0.0320920, -0.0020450, -0.0256288, -0.0031393, -0.0288767, 0.0235838
8: 0.9400651, 1.0158503, 0.9503613, 1.0137819, -0.0737168, 0.0654890
9: -0.0061514, 0.0381885, -0.0050216, 0.0154503, -0.0216017, 0.0422845

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1090539, upper bound: 0.1082809
time: 1.31 seconds

## Relational analysis of IS_A2_B2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1090539, upper bound: 0.1082809
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0418727, 0.0034804, -0.0008801, 0.0035198, -0.0453363, 0.0043086
1: -0.0203305, 0.0181668, -0.0026143, 0.0062913, -0.0264350, 0.0207811
2: -0.0097452, 0.0447819, 0.0078463, 0.0211436, -0.0308888, 0.0363351
3: -0.0081115, 0.0185357, -0.0086251, 0.0046089, -0.0127204, 0.0271608
4: -0.0300388, 0.0232558, -0.0087397, 0.0051603, -0.0351991, 0.0315468
5: -0.0037476, 0.0500180, -0.0070077, 0.0125339, -0.0162815, 0.0570257
6: -0.0181970, 0.0122213, 0.0011012, 0.0117177, -0.0297336, 0.0111200
7: -0.0418856, 0.0362727, -0.0256091, -0.0036923, -0.0378610, 0.0618818
8: 0.9219365, 1.0134249, 0.9504176, 1.0123553, -0.0904188, 0.0630072
9: -0.0085155, 0.0758684, -0.0048006, 0.0149929, -0.0235084, 0.0797986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_B2_A2_B1_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0981085, upper bound: 0.1022134
time: 1.33 seconds

## Relational analysis of IS_A2_B2_B2_A2_B1_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1078301, upper bound: 0.1070807
time: 1.28 seconds

## BFS IS instance: IS_A2_B2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0156411, 0.0037755, -0.0127855, 0.0040169, -0.0194136, 0.0163429
1: -0.0094690, 0.0068055, -0.0082867, 0.0071503, -0.0166193, 0.0150922
2: 0.0068440, 0.0276422, 0.0063757, 0.0258024, -0.0189584, 0.0212665
3: -0.0086130, 0.0098564, -0.0090949, 0.0089116, -0.0175246, 0.0189513
4: -0.0095072, 0.0117891, -0.0098079, 0.0106183, -0.0201255, 0.0215969
5: -0.0059633, 0.0264039, -0.0066803, 0.0238335, -0.0297968, 0.0330841
6: -0.0023315, 0.0119764, -0.0011428, 0.0121129, -0.0144444, 0.0131192
7: -0.0320920, -0.0020450, -0.0310260, -0.0015891, -0.0305029, 0.0289810
8: 0.9400651, 1.0158503, 0.9420384, 1.0182815, -0.0771437, 0.0738119
9: -0.0061514, 0.0381885, -0.0067846, 0.0341062, -0.0402576, 0.0441028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1090242, upper bound: 0.1085856
time: 1.15 seconds

## Relational analysis of IS_A2_B2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1090242, upper bound: 0.1085856
time: 1.11 seconds

## BFS IS instance: IS_A2_B2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0418727, 0.0034804, -0.0125483, 0.0038325, -0.0454429, 0.0158092
1: -0.0203305, 0.0181668, -0.0082004, 0.0069340, -0.0272645, 0.0263672
2: -0.0097452, 0.0447819, 0.0065438, 0.0255262, -0.0352715, 0.0382381
3: -0.0081115, 0.0185357, -0.0081904, 0.0088391, -0.0169506, 0.0266947
4: -0.0300388, 0.0232558, -0.0096802, 0.0097755, -0.0398143, 0.0329360
5: -0.0037476, 0.0500180, -0.0046118, 0.0236459, -0.0273934, 0.0546298
6: -0.0181970, 0.0122213, -0.0005186, 0.0120698, -0.0302625, 0.0127398
7: -0.0418856, 0.0362727, -0.0309482, -0.0021779, -0.0397078, 0.0672209
8: 0.9219365, 1.0134249, 0.9421825, 1.0168518, -0.0905843, 0.0712423
9: -0.0085155, 0.0758684, -0.0065604, 0.0333830, -0.0418985, 0.0815380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_B2_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0979190, upper bound: 0.1022134
time: 1.05 seconds

## Relational analysis of IS_A2_B2_B2_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1077996, upper bound: 0.1073809
time: 1.70 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 6.27 seconds
IS_A1_A1_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1127955, upper bound: 0.1116181
IS_A1_A1_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1127955, upper bound: 0.1116181
IS_A1_A1_B1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1036792, upper bound: 0.0972489
IS_A1_A1_B1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1117177, upper bound: 0.1105601
IS_A1_A1_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1148819, upper bound: 0.1113065
IS_A1_A1_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1148819, upper bound: 0.1113065
IS_A1_A1_B1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1096676, upper bound: 0.1035643
IS_A1_A1_B1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1137056, upper bound: 0.1100887
IS_A1_A1_B1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1051321, upper bound: 0.1041240
IS_A1_A1_B1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1119707, upper bound: 0.1191869
IS_A1_A1_B1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1074380, upper bound: 0.1053478
IS_A1_A1_B1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1114354, upper bound: 0.1127014
IS_A1_A1_B1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1110905, upper bound: 0.1101886
IS_A1_A1_B1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1140679, upper bound: 0.1148367
IS_A1_A1_B1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1096973, upper bound: 0.1049064
IS_A1_A1_B1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1137160, upper bound: 0.1118636
IS_A1_A1_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1112487, upper bound: 0.1088727
IS_A1_A1_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1112487, upper bound: 0.1088727
IS_A1_A1_B2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.0950305, upper bound: 0.0866565
IS_A1_A1_B2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1101912, upper bound: 0.1078447
IS_A1_A1_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1124427, upper bound: 0.1084720
IS_A1_A1_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1124427, upper bound: 0.1084720
IS_A1_A1_B2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1052059, upper bound: 0.0986596
IS_A1_A1_B2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1112619, upper bound: 0.1072640
IS_A1_A1_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1106817, upper bound: 0.1092836
IS_A1_A1_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1106817, upper bound: 0.1092836
IS_A1_A1_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1035555, upper bound: 0.0983161
IS_A1_A1_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1094888, upper bound: 0.1080613
IS_A1_A1_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.0968259, upper bound: 0.0999164
IS_A1_A1_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1123864, upper bound: 0.1117345
IS_A1_A1_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1049022, upper bound: 0.0980559
IS_A1_A1_B2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1112619, upper bound: 0.1076649
IS_A1_A2_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1113761, upper bound: 0.1113761
IS_A1_A2_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1113761, upper bound: 0.1113761
IS_A1_A2_B1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1029643, upper bound: 0.0970492
IS_A1_A2_B1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1103274, upper bound: 0.1103274
IS_A1_A2_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1136995, upper bound: 0.1111350
IS_A1_A2_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1136995, upper bound: 0.1111350
IS_A1_A2_B1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1037274, upper bound: 0.0966930
IS_A1_A2_B1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1125675, upper bound: 0.1100775
IS_A1_A2_B1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1079591, upper bound: 0.1122083
IS_A1_A2_B1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1101933, upper bound: 0.1178666
IS_A1_A2_B1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1066593, upper bound: 0.1051630
IS_A1_A2_B1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1099237, upper bound: 0.1124840
IS_A1_A2_B1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1102346, upper bound: 0.1097881
IS_A1_A2_B1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1126152, upper bound: 0.1144794
IS_A1_A2_B1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1090485, upper bound: 0.1048051
IS_A1_A2_B1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1124840, upper bound: 0.1116854
IS_A1_A2_B2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1109709, upper bound: 0.1115017
IS_A1_A2_B2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1109709, upper bound: 0.1115017
IS_A1_A2_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1109709, upper bound: 0.1115017
IS_A1_A2_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1109709, upper bound: 0.1115017
IS_A1_A2_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1107421, upper bound: 0.1152656
IS_A1_A2_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1107421, upper bound: 0.1152656
IS_A1_A2_B2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1002778, upper bound: 0.0951601
IS_A1_A2_B2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1096904, upper bound: 0.1112686
IS_A1_A2_B2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.0923579, upper bound: 0.0854193
IS_A1_A2_B2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1092201, upper bound: 0.1073236
IS_A1_A2_B2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.0923579, upper bound: 0.0854193
IS_A1_A2_B2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1092201, upper bound: 0.1073237
IS_A1_A2_B2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1028084, upper bound: 0.0979650
IS_A1_A2_B2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1085064, upper bound: 0.1078270
IS_A1_A2_B2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1028084, upper bound: 0.0979650
IS_A1_A2_B2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1085064, upper bound: 0.1074971
IS_A2_B1_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1007451, upper bound: 0.0937313
IS_A2_B1_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1120889, upper bound: 0.1095692
IS_A2_B1_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1007451, upper bound: 0.0937313
IS_A2_B1_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1120889, upper bound: 0.1095692
IS_A2_B1_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.0866325, upper bound: 0.0941533
IS_A2_B1_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.0866325, upper bound: 0.0941533
IS_A2_B1_A1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.0978632, upper bound: 0.1029768
IS_A2_B1_A1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1066044, upper bound: 0.1079780
IS_A2_B1_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1048011, upper bound: 0.1006397
IS_A2_B1_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1105424, upper bound: 0.1103586
IS_A2_B1_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1048011, upper bound: 0.1006397
IS_A2_B1_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1105424, upper bound: 0.1103586
IS_A2_B1_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.0986220, upper bound: 0.1042314
IS_A2_B1_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.0986220, upper bound: 0.1042314
IS_A2_B1_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.0826738, upper bound: 0.0889897
IS_A2_B1_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1062215, upper bound: 0.1085899
IS_A2_B1_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1014867, upper bound: 0.0931794
IS_A2_B1_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1151183, upper bound: 0.1093742
IS_A2_B1_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1014867, upper bound: 0.0931794
IS_A2_B1_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1151183, upper bound: 0.1093742
IS_A2_B1_A2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.0982920, upper bound: 0.1028860
IS_A2_B1_A2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.0982920, upper bound: 0.1028860
IS_A2_B1_A2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1080304, upper bound: 0.1085189
IS_A2_B1_A2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1080304, upper bound: 0.1085189
IS_A2_B1_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.0986249, upper bound: 0.0908082
IS_A2_B1_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.0986249, upper bound: 0.0908082
IS_A2_B1_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1151183, upper bound: 0.1099181
IS_A2_B1_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1151183, upper bound: 0.1099181
IS_A2_B1_A2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.0980260, upper bound: 0.1028860
IS_A2_B1_A2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.0980260, upper bound: 0.1028860
IS_A2_B1_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.0918585, upper bound: 0.0873914
IS_A2_B1_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1069070, upper bound: 0.1078033
IS_A2_B2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1086644, upper bound: 0.1102918
IS_A2_B2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1086644, upper bound: 0.1102918
IS_A2_B2_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.0865077, upper bound: 0.0943210
IS_A2_B2_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1076397, upper bound: 0.1092201
IS_A2_B2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1083416, upper bound: 0.1116318
IS_A2_B2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1083416, upper bound: 0.1116318
IS_A2_B2_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.0854193, upper bound: 0.0923579
IS_A2_B2_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1073237, upper bound: 0.1105236
IS_A2_B2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1090831, upper bound: 0.1097157
IS_A2_B2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1090831, upper bound: 0.1097157
IS_A2_B2_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.0981316, upper bound: 0.1028084
IS_A2_B2_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1078603, upper bound: 0.1085064
IS_A2_B2_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.0959335, upper bound: 0.1002991
IS_A2_B2_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1143392, upper bound: 0.1108955
IS_A2_B2_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.0979650, upper bound: 0.1028098
IS_A2_B2_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1078270, upper bound: 0.1090359
IS_A2_B2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1086398, upper bound: 0.1086398
IS_A2_B2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1086398, upper bound: 0.1086398
IS_A2_B2_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.0864871, upper bound: 0.0934949
IS_A2_B2_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1076189, upper bound: 0.1076189
IS_A2_B2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1082809, upper bound: 0.1090539
IS_A2_B2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1082809, upper bound: 0.1090539
IS_A2_B2_B2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.0985033, upper bound: 0.1034645
IS_A2_B2_B2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1070807, upper bound: 0.1078301
IS_A2_B2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1090539, upper bound: 0.1082809
IS_A2_B2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1090539, upper bound: 0.1082809
IS_A2_B2_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.0981085, upper bound: 0.1022134
IS_A2_B2_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1078301, upper bound: 0.1070807
IS_A2_B2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1090242, upper bound: 0.1085856
IS_A2_B2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1090242, upper bound: 0.1085856
IS_A2_B2_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.0979190, upper bound: 0.1022134
IS_A2_B2_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.27
Output dim: 8, lower bound: -0.1077996, upper bound: 0.1073809

## BFS IS instance: IS_A1_A1_B1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0042551, 0.0032452, -0.0068219, 0.0033322, -0.0075069, 0.0099549
1: -0.0044903, 0.0059471, -0.0058224, 0.0060354, -0.0105257, 0.0117694
2: 0.0082113, 0.0213195, 0.0081652, 0.0219081, -0.0136968, 0.0131543
3: -0.0076520, 0.0058729, -0.0082344, 0.0069409, -0.0145929, 0.0141073
4: -0.0084777, 0.0059815, -0.0085099, 0.0079931, -0.0164708, 0.0144914
5: -0.0050728, 0.0160858, -0.0066023, 0.0184757, -0.0235485, 0.0226881
6: 0.0016502, 0.0116066, 0.0010500, 0.0116118, -0.0099616, 0.0105566
7: -0.0276028, -0.0045676, -0.0288039, -0.0043283, -0.0232745, 0.0242363
8: 0.9478686, 1.0099478, 0.9461516, 1.0104891, -0.0626205, 0.0637962
9: -0.0042621, 0.0204311, -0.0042948, 0.0254496, -0.0287950, 0.0242747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_A1_B1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1054592, upper bound: 0.1024675
time: 1.20 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_A1_B1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1120627, upper bound: 0.1142406
time: 1.31 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0228356, 0.0029786, -0.0068219, 0.0033322, -0.0260885, 0.0096571
1: -0.0147584, 0.0056306, -0.0058224, 0.0060354, -0.0207529, 0.0114529
2: -0.0002094, 0.0222400, 0.0081652, 0.0219081, -0.0221175, 0.0140748
3: -0.0066357, 0.0140793, -0.0082344, 0.0069409, -0.0135766, 0.0223137
4: -0.0085486, 0.0166003, -0.0085099, 0.0079931, -0.0165416, 0.0251102
5: -0.0034339, 0.0337723, -0.0066023, 0.0184757, -0.0219096, 0.0403746
6: -0.0046760, 0.0115056, 0.0010500, 0.0116118, -0.0162878, 0.0104556
7: -0.0368614, -0.0010724, -0.0288039, -0.0043283, -0.0325330, 0.0277315
8: 0.9353687, 1.0076625, 0.9461516, 1.0104891, -0.0751204, 0.0615109
9: -0.0037730, 0.0560703, -0.0042948, 0.0254496, -0.0292226, 0.0596573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_A1_B1_B1_A1_B1_A2_A1

### Relational analysis result of IS_A1_A1_B1_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0998381, upper bound: 0.1069852
time: 1.23 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_B1_A2_A2

### Relational analysis result of IS_A1_A1_B1_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1120627, upper bound: 0.1142406
time: 1.22 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0010487, 0.0033154, -0.0152090, 0.0031650, -0.0041335, 0.0184241
1: -0.0026946, 0.0060005, -0.0092984, 0.0058329, -0.0085274, 0.0150766
2: 0.0082501, 0.0211926, 0.0084166, 0.0272579, -0.0182803, 0.0127760
3: -0.0080984, 0.0047087, -0.0078232, 0.0097176, -0.0178160, 0.0125319
4: -0.0084595, 0.0047932, -0.0090669, 0.0111720, -0.0193939, 0.0138601
5: -0.0060823, 0.0126353, -0.0055655, 0.0260329, -0.0321152, 0.0182008
6: 0.0016002, 0.0116033, -0.0017782, 0.0115243, -0.0099241, 0.0133772
7: -0.0257966, -0.0044134, -0.0319382, -0.0040312, -0.0217654, 0.0269813
8: 0.9499036, 1.0103157, 0.9403499, 1.0088862, -0.0589826, 0.0699658
9: -0.0042464, 0.0149937, -0.0039843, 0.0372755, -0.0403899, 0.0189780

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_A1_B1_B1_A1_B2_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0987458, upper bound: 0.0911766
time: 1.50 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_B2_B1_B2

### Relational analysis result of IS_A1_A1_B1_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1022640, upper bound: 0.0956535
time: 1.13 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0043204, 0.0033403, -0.0293650, 0.0030477, -0.0072623, 0.0325818
1: -0.0045169, 0.0060363, -0.0151573, 0.0112346, -0.0157515, 0.0208303
2: 0.0081973, 0.0214228, -0.0008992, 0.0365087, -0.0282279, 0.0223220
3: -0.0082354, 0.0058962, -0.0076563, 0.0144001, -0.0226355, 0.0135525
4: -0.0084861, 0.0065518, -0.0202002, 0.0174273, -0.0255988, 0.0267520
5: -0.0064485, 0.0161316, -0.0046603, 0.0387708, -0.0452193, 0.0207919
6: 0.0013103, 0.0116094, -0.0103274, 0.0115072, -0.0101969, 0.0218268
7: -0.0276268, -0.0043234, -0.0372210, 0.0166859, -0.0443127, 0.0325217
8: 0.9478363, 1.0105119, 0.9305710, 1.0080781, -0.0602418, 0.0796312
9: -0.0042755, 0.0208152, -0.0040378, 0.0576913, -0.0607508, 0.0248530

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_A1_B1_B1_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B1_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1067486, upper bound: 0.1025937
time: 1.09 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_B2_B2_B2

### Relational analysis result of IS_A1_A1_B1_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1105773, upper bound: 0.1093361
time: 1.18 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0191781, 0.0035389, -0.0068219, 0.0033322, -0.0224417, 0.0103608
1: -0.0109528, 0.0065933, -0.0058224, 0.0060354, -0.0169882, 0.0124157
2: 0.0063003, 0.0300966, 0.0081652, 0.0219081, -0.0156078, 0.0219314
3: -0.0072627, 0.0111222, -0.0082344, 0.0069409, -0.0142036, 0.0193566
4: -0.0122624, 0.0121697, -0.0085099, 0.0079931, -0.0202554, 0.0206796
5: -0.0025787, 0.0297650, -0.0066023, 0.0184757, -0.0210544, 0.0363674
6: -0.0038765, 0.0119483, 0.0010500, 0.0116118, -0.0154884, 0.0108983
7: -0.0334526, 0.0007704, -0.0288039, -0.0043283, -0.0291243, 0.0295743
8: 0.9362192, 1.0142251, 0.9461516, 1.0104891, -0.0742699, 0.0680735
9: -0.0059955, 0.0431701, -0.0042948, 0.0254496, -0.0314452, 0.0474649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_A1_B1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1060354, upper bound: 0.1016304
time: 0.95 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_A1_B1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1141403, upper bound: 0.1133128
time: 1.42 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0441656, 0.0033292, -0.0068219, 0.0033322, -0.0474285, 0.0101510
1: -0.0212953, 0.0206310, -0.0058224, 0.0060354, -0.0273307, 0.0264534
2: -0.0113763, 0.0521468, 0.0081652, 0.0219081, -0.0332844, 0.0439815
3: -0.0066498, 0.0208448, -0.0082344, 0.0069409, -0.0135907, 0.0290792
4: -0.0332595, 0.0235818, -0.0085099, 0.0079931, -0.0412526, 0.0320917
5: -0.0002750, 0.0545497, -0.0066023, 0.0184757, -0.0187506, 0.0611520
6: -0.0193851, 0.0151946, 0.0010500, 0.0116118, -0.0309969, 0.0141447
7: -0.0431641, 0.0397022, -0.0288039, -0.0043283, -0.0388358, 0.0685061
8: 0.8956803, 1.0117061, 0.9461516, 1.0104891, -0.1148088, 0.0655545
9: -0.0097559, 0.0879855, -0.0042948, 0.0254496, -0.0352055, 0.0922803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_A1_B1_B1_A2_B1_A2_A1

### Relational analysis result of IS_A1_A1_B1_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1071904, upper bound: 0.1090870
time: 1.17 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2_B1_A2_A2

### Relational analysis result of IS_A1_A1_B1_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1140628, upper bound: 0.1130515
time: 1.16 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0152878, 0.0035173, -0.0211824, 0.0029213, -0.0180571, 0.0245469
1: -0.0093501, 0.0065308, -0.0117939, 0.0065764, -0.0159265, 0.0183247
2: 0.0069722, 0.0271591, 0.0048759, 0.0309826, -0.0240104, 0.0222832
3: -0.0070329, 0.0097534, -0.0061509, 0.0117053, -0.0187382, 0.0159043
4: -0.0093891, 0.0100495, -0.0137543, 0.0124880, -0.0218771, 0.0238038
5: -0.0020274, 0.0261453, -0.0012205, 0.0314585, -0.0334859, 0.0273657
6: -0.0013963, 0.0119500, -0.0048468, 0.0115458, -0.0129421, 0.0167968
7: -0.0319848, -0.0030682, -0.0341883, 0.0035964, -0.0355812, 0.0311201
8: 0.9402636, 1.0141330, 0.9361848, 1.0076226, -0.0673590, 0.0779482
9: -0.0059844, 0.0368281, -0.0040812, 0.0452063, -0.0511907, 0.0409092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_A1_B1_B1_A2_B2_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0986531, upper bound: 0.0907252
time: 1.43 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2_B2_B1_B2

### Relational analysis result of IS_A1_A1_B1_B1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1084846, upper bound: 0.1023449
time: 1.02 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0191321, 0.0035885, -0.0286614, 0.0029639, -0.0219890, 0.0322499
1: -0.0109300, 0.0066608, -0.0148740, 0.0107901, -0.0217200, 0.0215348
2: 0.0063350, 0.0300859, -0.0004061, 0.0359997, -0.0296647, 0.0304920
3: -0.0075533, 0.0111018, -0.0070089, 0.0141715, -0.0217248, 0.0181107
4: -0.0122250, 0.0123613, -0.0196433, 0.0166764, -0.0289014, 0.0320046
5: -0.0032882, 0.0297104, -0.0031456, 0.0381549, -0.0414431, 0.0328560
6: -0.0039371, 0.0119569, -0.0097549, 0.0115067, -0.0154438, 0.0217117
7: -0.0334312, 0.0007911, -0.0369656, 0.0154800, -0.0489113, 0.0377567
8: 0.9363085, 1.0145905, 0.9310437, 1.0075866, -0.0712781, 0.0834471
9: -0.0060456, 0.0431972, -0.0039916, 0.0564459, -0.0624915, 0.0471888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_A1_B1_B1_A2_B2_B2_B1

### Relational analysis result of IS_A1_A1_B1_B1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1029837, upper bound: 0.0952053
time: 1.04 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2_B2_B2_B2

### Relational analysis result of IS_A1_A1_B1_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1126567, upper bound: 0.1090357
time: 0.92 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0012123, 0.0035067, -0.0067004, 0.0037255, -0.0048211, 0.0101529
1: -0.0027756, 0.0062155, -0.0057758, 0.0067405, -0.0095161, 0.0119913
2: 0.0080812, 0.0213464, 0.0069060, 0.0217756, -0.0136944, 0.0144404
3: -0.0089845, 0.0047242, -0.0080846, 0.0069027, -0.0158872, 0.0128088
4: -0.0085758, 0.0058159, -0.0093850, 0.0074604, -0.0160362, 0.0152010
5: -0.0080004, 0.0128439, -0.0047906, 0.0183745, -0.0263748, 0.0176344
6: 0.0010091, 0.0116508, 0.0006110, 0.0119734, -0.0109644, 0.0110398
7: -0.0259058, -0.0038584, -0.0287620, -0.0026541, -0.0232517, 0.0247559
8: 0.9498342, 1.0117511, 0.9462292, 1.0155271, -0.0656930, 0.0655218
9: -0.0044764, 0.0156366, -0.0060432, 0.0251009, -0.0287493, 0.0216798

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_B1_A1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1051321, upper bound: 0.1041240
time: 1.44 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_B1_A2

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1051321, upper bound: 0.1041240
time: 1.34 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0045608, 0.0035324, -0.0199069, 0.0036126, -0.0081734, 0.0233564
1: -0.0046369, 0.0062533, -0.0113015, 0.0067543, -0.0113912, 0.0173893
2: 0.0080222, 0.0215743, 0.0057907, 0.0308919, -0.0228697, 0.0157836
3: -0.0091214, 0.0059949, -0.0078071, 0.0113990, -0.0205204, 0.0138020
4: -0.0086054, 0.0074994, -0.0130599, 0.0130520, -0.0216574, 0.0205594
5: -0.0083864, 0.0163383, -0.0039545, 0.0304631, -0.0388495, 0.0202928
6: 0.0007054, 0.0116568, -0.0045409, 0.0119525, -0.0112472, 0.0161977
7: -0.0277350, -0.0037607, -0.0338370, 0.0020865, -0.0298216, 0.0300762
8: 0.9476901, 1.0119501, 0.9348754, 1.0146763, -0.0669862, 0.0770747
9: -0.0045054, 0.0216250, -0.0060275, 0.0451523, -0.0496577, 0.0276525

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_B2_A1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1119707, upper bound: 0.1191869
time: 1.90 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_B2_A2

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1119707, upper bound: 0.1191869
time: 4.25 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0015887, 0.0032620, -0.0363140, 0.0060595, -0.0076482, 0.0394741
1: -0.0030185, 0.0059548, -0.0209589, 0.0157984, -0.0188168, 0.0266330
2: 0.0082425, 0.0211273, -0.0058315, 0.0492503, -0.0410078, 0.0269588
3: -0.0077166, 0.0047571, -0.0079207, 0.0178007, -0.0255173, 0.0126777
4: -0.0084622, 0.0045736, -0.0358209, 0.0191940, -0.0270910, 0.0403945
5: -0.0051208, 0.0134689, 0.0013628, 0.0468022, -0.0519230, 0.0121061
6: 0.0017767, 0.0116045, -0.0142097, 0.0175497, -0.0157730, 0.0256561
7: -0.0262329, -0.0045414, -0.0457162, 0.0272032, -0.0534361, 0.0411748
8: 0.9496257, 1.0100192, 0.8981763, 1.0116464, -0.0620207, 0.1118429
9: -0.0042518, 0.0153739, -0.0056634, 0.0929911, -0.0972429, 0.0210373

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_B1_B1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0957677, upper bound: 0.0880335
time: 1.06 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1063347, upper bound: 0.1040880
time: 1.06 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0042367, 0.0033373, -0.0464252, 0.0118589, -0.0160956, 0.0497625
1: -0.0044710, 0.0060332, -0.0271004, 0.0215093, -0.0259803, 0.0331336
2: 0.0081987, 0.0214133, -0.0167219, 0.0622965, -0.0540978, 0.0381352
3: -0.0082146, 0.0058595, -0.0125935, 0.0255870, -0.0338016, 0.0184530
4: -0.0084854, 0.0064854, -0.0477164, 0.0254186, -0.0339040, 0.0542018
5: -0.0063949, 0.0160526, -0.0017185, 0.0718681, -0.0782630, 0.0177711
6: 0.0013280, 0.0116092, -0.0209495, 0.0234342, -0.0221063, 0.0325324
7: -0.0275855, -0.0043319, -0.0524577, 0.0399717, -0.0675572, 0.0481258
8: 0.9478921, 1.0104932, 0.8573842, 1.0116768, -0.0637847, 0.1531090
9: -0.0042749, 0.0206435, -0.0170553, 0.1237351, -0.1280100, 0.0376988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1013909, upper bound: 0.0948112
time: 1.14 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_B2_B2

### Relational analysis result of IS_A1_A1_B1_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1103922, upper bound: 0.1116374
time: 1.28 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0155588, 0.0037061, -0.0110375, 0.0034813, -0.0187453, 0.0144583
1: -0.0094504, 0.0067523, -0.0076027, 0.0065157, -0.0159661, 0.0143551
2: 0.0067980, 0.0274457, 0.0069034, 0.0242620, -0.0174640, 0.0205423
3: -0.0079187, 0.0098370, -0.0066315, 0.0083534, -0.0162722, 0.0161463
4: -0.0095264, 0.0108443, -0.0094090, 0.0075099, -0.0170363, 0.0202532
5: -0.0040780, 0.0263634, -0.0007001, 0.0223464, -0.0264244, 0.0270635
6: -0.0018815, 0.0119938, 0.0011213, 0.0119762, -0.0138577, 0.0108725
7: -0.0320753, -0.0023776, -0.0304092, -0.0032671, -0.0288081, 0.0280316
8: 0.9400961, 1.0155754, 0.9431801, 1.0140958, -0.0711651, 0.0709157
9: -0.0062189, 0.0375890, -0.0060766, 0.0303372, -0.0359608, 0.0423521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_A1_B1_B2_A2_B1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1015928, upper bound: 0.0963134
time: 1.25 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2_B1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1099532, upper bound: 0.1090240
time: 1.08 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0194083, 0.0037817, -0.0193597, 0.0035508, -0.0229591, 0.0231414
1: -0.0110323, 0.0069295, -0.0110260, 0.0066207, -0.0176530, 0.0179556
2: 0.0061465, 0.0304222, 0.0061730, 0.0302699, -0.0241235, 0.0242492
3: -0.0085125, 0.0112015, -0.0073342, 0.0111916, -0.0197041, 0.0185357
4: -0.0124610, 0.0131764, -0.0124157, 0.0123359, -0.0247969, 0.0255921
5: -0.0054288, 0.0299555, -0.0027718, 0.0299405, -0.0353693, 0.0327273
6: -0.0043954, 0.0120021, -0.0040209, 0.0119502, -0.0163456, 0.0160229
7: -0.0335273, 0.0014978, -0.0335214, 0.0010777, -0.0346050, 0.0350191
8: 0.9359074, 1.0160574, 0.9359319, 1.0143085, -0.0784010, 0.0801255
9: -0.0062893, 0.0440525, -0.0060070, 0.0435464, -0.0498357, 0.0500595

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_A1_B1_B2_A2_B1_B2_A1

### Relational analysis result of IS_A1_A1_B1_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1140679, upper bound: 0.1148367
time: 1.09 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2_B1_B2_A2

### Relational analysis result of IS_A1_A1_B1_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1140679, upper bound: 0.1148367
time: 1.29 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0152878, 0.0035173, -0.0363140, 0.0060595, -0.0213473, 0.0395367
1: -0.0093501, 0.0065308, -0.0209589, 0.0157984, -0.0251485, 0.0274898
2: 0.0069722, 0.0271591, -0.0058315, 0.0492503, -0.0422781, 0.0329906
3: -0.0070329, 0.0097534, -0.0079207, 0.0178007, -0.0248336, 0.0176741
4: -0.0093891, 0.0100495, -0.0358209, 0.0191940, -0.0285832, 0.0458704
5: -0.0020274, 0.0261453, 0.0013628, 0.0468022, -0.0488296, 0.0247825
6: -0.0013963, 0.0119500, -0.0142097, 0.0175497, -0.0189460, 0.0261597
7: -0.0319848, -0.0030682, -0.0457162, 0.0272032, -0.0591880, 0.0426480
8: 0.9402636, 1.0141330, 0.8981763, 1.0116464, -0.0690732, 0.1159567
9: -0.0059844, 0.0368281, -0.0056634, 0.0929911, -0.0989755, 0.0424914

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_A1_B1_B2_A2_B2_B1_B1

### Relational analysis result of IS_A1_A1_B1_B2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0965480, upper bound: 0.0878224
time: 1.43 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2_B2_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1085079, upper bound: 0.1036468
time: 1.25 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 4.26 seconds
IS_A1_A1_B1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.26
Output dim: 8, lower bound: -0.1054592, upper bound: 0.1024675
IS_A1_A1_B1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.26
Output dim: 8, lower bound: -0.1120627, upper bound: 0.1142406
IS_A1_A1_B1_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 4.26
Output dim: 8, lower bound: -0.0998381, upper bound: 0.1069852
IS_A1_A1_B1_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 4.26
Output dim: 8, lower bound: -0.1120627, upper bound: 0.1142406
IS_A1_A1_B1_B1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 4.26
Output dim: 8, lower bound: -0.0987458, upper bound: 0.0911766
IS_A1_A1_B1_B1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 4.26
Output dim: 8, lower bound: -0.1022640, upper bound: 0.0956535
IS_A1_A1_B1_B1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 4.26
Output dim: 8, lower bound: -0.1067486, upper bound: 0.1025937
IS_A1_A1_B1_B1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 4.26
Output dim: 8, lower bound: -0.1105773, upper bound: 0.1093361
IS_A1_A1_B1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.26
Output dim: 8, lower bound: -0.1060354, upper bound: 0.1016304
IS_A1_A1_B1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.26
Output dim: 8, lower bound: -0.1141403, upper bound: 0.1133128
IS_A1_A1_B1_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 4.26
Output dim: 8, lower bound: -0.1071904, upper bound: 0.1090870
IS_A1_A1_B1_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 4.26
Output dim: 8, lower bound: -0.1140628, upper bound: 0.1130515
IS_A1_A1_B1_B1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 4.26
Output dim: 8, lower bound: -0.0986531, upper bound: 0.0907252
IS_A1_A1_B1_B1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 4.26
Output dim: 8, lower bound: -0.1084846, upper bound: 0.1023449
IS_A1_A1_B1_B1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 4.26
Output dim: 8, lower bound: -0.1029837, upper bound: 0.0952053
IS_A1_A1_B1_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 4.26
Output dim: 8, lower bound: -0.1126567, upper bound: 0.1090357
IS_A1_A1_B1_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.26
Output dim: 8, lower bound: -0.1051321, upper bound: 0.1041240
IS_A1_A1_B1_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.26
Output dim: 8, lower bound: -0.1051321, upper bound: 0.1041240
IS_A1_A1_B1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.26
Output dim: 8, lower bound: -0.1119707, upper bound: 0.1191869
IS_A1_A1_B1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.26
Output dim: 8, lower bound: -0.1119707, upper bound: 0.1191869
IS_A1_A1_B1_B2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 4.26
Output dim: 8, lower bound: -0.0957677, upper bound: 0.0880335
IS_A1_A1_B1_B2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 4.26
Output dim: 8, lower bound: -0.1063347, upper bound: 0.1040880
IS_A1_A1_B1_B2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 4.26
Output dim: 8, lower bound: -0.1013909, upper bound: 0.0948112
IS_A1_A1_B1_B2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 4.26
Output dim: 8, lower bound: -0.1103922, upper bound: 0.1116374
IS_A1_A1_B1_B2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 4.26
Output dim: 8, lower bound: -0.1015928, upper bound: 0.0963134
IS_A1_A1_B1_B2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 4.26
Output dim: 8, lower bound: -0.1099532, upper bound: 0.1090240
IS_A1_A1_B1_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.26
Output dim: 8, lower bound: -0.1140679, upper bound: 0.1148367
IS_A1_A1_B1_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.26
Output dim: 8, lower bound: -0.1140679, upper bound: 0.1148367
IS_A1_A1_B1_B2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 4.26
Output dim: 8, lower bound: -0.0965480, upper bound: 0.0878224
IS_A1_A1_B1_B2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 4.26
Output dim: 8, lower bound: -0.1085079, upper bound: 0.1036468
IS_A1_A1_B1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1137160, upper bound: 0.1118636
IS_A1_A1_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1112487, upper bound: 0.1088727
IS_A1_A1_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1112487, upper bound: 0.1088727
IS_A1_A1_B2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.0950305, upper bound: 0.0866565
IS_A1_A1_B2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1101912, upper bound: 0.1078447
IS_A1_A1_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1124427, upper bound: 0.1084720
IS_A1_A1_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1124427, upper bound: 0.1084720
IS_A1_A1_B2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1052059, upper bound: 0.0986596
IS_A1_A1_B2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1112619, upper bound: 0.1072640
IS_A1_A1_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1106817, upper bound: 0.1092836
IS_A1_A1_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1106817, upper bound: 0.1092836
IS_A1_A1_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1035555, upper bound: 0.0983161
IS_A1_A1_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1094888, upper bound: 0.1080613
IS_A1_A1_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.0968259, upper bound: 0.0999164
IS_A1_A1_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1123864, upper bound: 0.1117345
IS_A1_A1_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1049022, upper bound: 0.0980559
IS_A1_A1_B2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1112619, upper bound: 0.1076649
IS_A1_A2_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1113761, upper bound: 0.1113761
IS_A1_A2_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1113761, upper bound: 0.1113761
IS_A1_A2_B1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1029643, upper bound: 0.0970492
IS_A1_A2_B1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1103274, upper bound: 0.1103274
IS_A1_A2_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1136995, upper bound: 0.1111350
IS_A1_A2_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1136995, upper bound: 0.1111350
IS_A1_A2_B1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1037274, upper bound: 0.0966930
IS_A1_A2_B1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1125675, upper bound: 0.1100775
IS_A1_A2_B1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1079591, upper bound: 0.1122083
IS_A1_A2_B1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1101933, upper bound: 0.1178666
IS_A1_A2_B1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1066593, upper bound: 0.1051630
IS_A1_A2_B1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1099237, upper bound: 0.1124840
IS_A1_A2_B1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1102346, upper bound: 0.1097881
IS_A1_A2_B1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1126152, upper bound: 0.1144794
IS_A1_A2_B1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1090485, upper bound: 0.1048051
IS_A1_A2_B1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1124840, upper bound: 0.1116854
IS_A1_A2_B2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1109709, upper bound: 0.1115017
IS_A1_A2_B2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1109709, upper bound: 0.1115017
IS_A1_A2_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1109709, upper bound: 0.1115017
IS_A1_A2_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1109709, upper bound: 0.1115017
IS_A1_A2_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1107421, upper bound: 0.1152656
IS_A1_A2_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1107421, upper bound: 0.1152656
IS_A1_A2_B2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1002778, upper bound: 0.0951601
IS_A1_A2_B2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1096904, upper bound: 0.1112686
IS_A1_A2_B2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.0923579, upper bound: 0.0854193
IS_A1_A2_B2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1092201, upper bound: 0.1073236
IS_A1_A2_B2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.0923579, upper bound: 0.0854193
IS_A1_A2_B2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1092201, upper bound: 0.1073237
IS_A1_A2_B2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1028084, upper bound: 0.0979650
IS_A1_A2_B2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1085064, upper bound: 0.1078270
IS_A1_A2_B2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1028084, upper bound: 0.0979650
IS_A1_A2_B2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1085064, upper bound: 0.1074971
IS_A2_B1_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1007451, upper bound: 0.0937313
IS_A2_B1_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1120889, upper bound: 0.1095692
IS_A2_B1_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1007451, upper bound: 0.0937313
IS_A2_B1_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1120889, upper bound: 0.1095692
IS_A2_B1_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.0866325, upper bound: 0.0941533
IS_A2_B1_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.0866325, upper bound: 0.0941533
IS_A2_B1_A1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.0978632, upper bound: 0.1029768
IS_A2_B1_A1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1066044, upper bound: 0.1079780
IS_A2_B1_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1048011, upper bound: 0.1006397
IS_A2_B1_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1105424, upper bound: 0.1103586
IS_A2_B1_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1048011, upper bound: 0.1006397
IS_A2_B1_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1105424, upper bound: 0.1103586
IS_A2_B1_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.0986220, upper bound: 0.1042314
IS_A2_B1_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.0986220, upper bound: 0.1042314
IS_A2_B1_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.0826738, upper bound: 0.0889897
IS_A2_B1_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1062215, upper bound: 0.1085899
IS_A2_B1_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1014867, upper bound: 0.0931794
IS_A2_B1_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1151183, upper bound: 0.1093742
IS_A2_B1_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1014867, upper bound: 0.0931794
IS_A2_B1_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1151183, upper bound: 0.1093742
IS_A2_B1_A2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.0982920, upper bound: 0.1028860
IS_A2_B1_A2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.0982920, upper bound: 0.1028860
IS_A2_B1_A2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1080304, upper bound: 0.1085189
IS_A2_B1_A2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1080304, upper bound: 0.1085189
IS_A2_B1_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.0986249, upper bound: 0.0908082
IS_A2_B1_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.0986249, upper bound: 0.0908082
IS_A2_B1_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1151183, upper bound: 0.1099181
IS_A2_B1_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1151183, upper bound: 0.1099181
IS_A2_B1_A2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.0980260, upper bound: 0.1028860
IS_A2_B1_A2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.0980260, upper bound: 0.1028860
IS_A2_B1_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.0918585, upper bound: 0.0873914
IS_A2_B1_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1069070, upper bound: 0.1078033
IS_A2_B2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1086644, upper bound: 0.1102918
IS_A2_B2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1086644, upper bound: 0.1102918
IS_A2_B2_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.0865077, upper bound: 0.0943210
IS_A2_B2_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1076397, upper bound: 0.1092201
IS_A2_B2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1083416, upper bound: 0.1116318
IS_A2_B2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1083416, upper bound: 0.1116318
IS_A2_B2_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.0854193, upper bound: 0.0923579
IS_A2_B2_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1073237, upper bound: 0.1105236
IS_A2_B2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1090831, upper bound: 0.1097157
IS_A2_B2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1090831, upper bound: 0.1097157
IS_A2_B2_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.0981316, upper bound: 0.1028084
IS_A2_B2_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1078603, upper bound: 0.1085064
IS_A2_B2_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.0959335, upper bound: 0.1002991
IS_A2_B2_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1143392, upper bound: 0.1108955
IS_A2_B2_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.0979650, upper bound: 0.1028098
IS_A2_B2_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1078270, upper bound: 0.1090359
IS_A2_B2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1086398, upper bound: 0.1086398
IS_A2_B2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1086398, upper bound: 0.1086398
IS_A2_B2_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.0864871, upper bound: 0.0934949
IS_A2_B2_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1076189, upper bound: 0.1076189
IS_A2_B2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1082809, upper bound: 0.1090539
IS_A2_B2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1082809, upper bound: 0.1090539
IS_A2_B2_B2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.0985033, upper bound: 0.1034645
IS_A2_B2_B2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1070807, upper bound: 0.1078301
IS_A2_B2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1090539, upper bound: 0.1082809
IS_A2_B2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1090539, upper bound: 0.1082809
IS_A2_B2_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.0981085, upper bound: 0.1022134
IS_A2_B2_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1078301, upper bound: 0.1070807
IS_A2_B2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1090242, upper bound: 0.1085856
IS_A2_B2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1090242, upper bound: 0.1085856
IS_A2_B2_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.0979190, upper bound: 0.1022134
IS_A2_B2_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 8, lower bound: -0.1077996, upper bound: 0.1073809

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.99 + 597.45 = 601.43 seconds
