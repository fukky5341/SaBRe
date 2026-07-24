## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.01412451


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0095516, 0.0030898, -0.0095516, 0.0030898, -0.0126414, 0.0126414)
1: (-0.0068929, 0.0003836, -0.0068929, 0.0003836, -0.0072765, 0.0072765)
2: (0.0272588, 0.0502782, 0.0272588, 0.0502782, -0.0230194, 0.0230194)
3: (-0.0038932, 0.0076452, -0.0038932, 0.0076452, -0.0115384, 0.0115384)
4: (-0.0109744, 0.0068214, -0.0109744, 0.0068214, -0.0177958, 0.0177958)
5: (0.0044616, 0.0200061, 0.0044616, 0.0200061, -0.0155445, 0.0155445)
6: (-0.0252129, 0.0082960, -0.0252129, 0.0082960, -0.0335089, 0.0335089)
7: (0.9576753, 0.9795982, 0.9576753, 0.9795982, -0.0219229, 0.0219229)
8: (-0.0270380, 0.0117635, -0.0270380, 0.0117635, -0.0388014, 0.0388014)
9: (-0.0108122, 0.0133251, -0.0108122, 0.0133251, -0.0241373, 0.0241373)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.76 + 1.53 = 3.29 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0160471, upper bound: 0.0160471

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0160471, upper bound: 0.0160471
time: 0.75 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0160471, upper bound: 0.0160471
time: 0.83 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.75 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.75
Output dim: 7, lower bound: -0.0160471, upper bound: 0.0160471
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.75
Output dim: 7, lower bound: -0.0160471, upper bound: 0.0160471

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0093494, 0.0030164, -0.0095064, 0.0030734, -0.0124228, 0.0125227
1: -0.0068128, 0.0002957, -0.0068750, 0.0003639, -0.0071767, 0.0071707
2: 0.0273761, 0.0498052, 0.0272851, 0.0501723, -0.0227962, 0.0225201
3: -0.0038663, 0.0074259, -0.0038872, 0.0075962, -0.0114625, 0.0113131
4: -0.0107470, 0.0066112, -0.0109235, 0.0067744, -0.0175214, 0.0175347
5: 0.0046297, 0.0198007, 0.0044992, 0.0199601, -0.0153304, 0.0153014
6: -0.0245788, 0.0080191, -0.0250709, 0.0082340, -0.0328128, 0.0330900
7: 0.9581653, 0.9795296, 0.9577850, 0.9795828, -0.0214175, 0.0217447
8: -0.0266620, 0.0112945, -0.0269538, 0.0116585, -0.0383205, 0.0382483
9: -0.0105408, 0.0130499, -0.0107515, 0.0132635, -0.0238042, 0.0238014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0159225
time: 0.76 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157206
time: 0.74 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0094985, 0.0030705, -0.0095372, 0.0030846, -0.0125831, 0.0126078
1: -0.0068719, 0.0003606, -0.0068872, 0.0003774, -0.0072493, 0.0072478
2: 0.0272896, 0.0501540, 0.0272672, 0.0502445, -0.0229549, 0.0228869
3: -0.0038861, 0.0075877, -0.0038913, 0.0076296, -0.0115158, 0.0114789
4: -0.0109147, 0.0067663, -0.0109582, 0.0068065, -0.0177212, 0.0177244
5: 0.0045057, 0.0199522, 0.0044736, 0.0199915, -0.0154858, 0.0154786
6: -0.0250464, 0.0082233, -0.0251678, 0.0082763, -0.0333227, 0.0333911
7: 0.9578038, 0.9795802, 0.9577101, 0.9795933, -0.0217895, 0.0218701
8: -0.0269393, 0.0116404, -0.0270112, 0.0117301, -0.0386694, 0.0386516
9: -0.0107410, 0.0132528, -0.0107929, 0.0133055, -0.0240464, 0.0240457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0159552
time: 0.83 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157206
time: 0.67 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.15 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0159225
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157206
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0159552
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157206

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0093373, 0.0030120, -0.0094566, 0.0030553, -0.0123926, 0.0124686
1: -0.0068080, 0.0002905, -0.0068553, 0.0003423, -0.0071503, 0.0071457
2: 0.0273831, 0.0497769, 0.0273139, 0.0500560, -0.0226729, 0.0224630
3: -0.0038647, 0.0074128, -0.0038806, 0.0075422, -0.0114069, 0.0112933
4: -0.0107334, 0.0065986, -0.0108676, 0.0067227, -0.0174561, 0.0174662
5: 0.0046398, 0.0197884, 0.0045406, 0.0199096, -0.0152698, 0.0152478
6: -0.0245408, 0.0080025, -0.0249150, 0.0081659, -0.0327067, 0.0329175
7: 0.9581947, 0.9795254, 0.9579054, 0.9795659, -0.0213712, 0.0216200
8: -0.0266395, 0.0112664, -0.0268614, 0.0115431, -0.0381826, 0.0381277
9: -0.0105245, 0.0130335, -0.0106847, 0.0131958, -0.0237203, 0.0237182

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157104, upper bound: 0.0158891
time: 0.74 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0158891
time: 0.69 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0093164, 0.0030044, -0.0105805, 0.0034632, -0.0127795, 0.0135849
1: -0.0067997, 0.0002814, -0.0073005, 0.0008308, -0.0076306, 0.0075818
2: 0.0273952, 0.0497280, 0.0266624, 0.0526845, -0.0252893, 0.0230656
3: -0.0038619, 0.0073901, -0.0040299, 0.0087610, -0.0126230, 0.0114200
4: -0.0107099, 0.0065769, -0.0121308, 0.0078910, -0.0186009, 0.0187077
5: 0.0046572, 0.0197671, 0.0036062, 0.0210515, -0.0163943, 0.0161609
6: -0.0244752, 0.0079739, -0.0284391, 0.0097048, -0.0341801, 0.0364130
7: 0.9582453, 0.9795183, 0.9551816, 0.9799474, -0.0217021, 0.0243367
8: -0.0266007, 0.0112179, -0.0289507, 0.0141496, -0.0407503, 0.0401685
9: -0.0104964, 0.0130050, -0.0121935, 0.0147248, -0.0252212, 0.0251985

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0156835
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157206
time: 0.62 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0094863, 0.0030661, -0.0094870, 0.0030663, -0.0125527, 0.0125531
1: -0.0068670, 0.0003552, -0.0068673, 0.0003555, -0.0072226, 0.0072225
2: 0.0272967, 0.0501255, 0.0272963, 0.0501271, -0.0228304, 0.0228292
3: -0.0038845, 0.0075744, -0.0038846, 0.0075752, -0.0114597, 0.0114590
4: -0.0109010, 0.0067536, -0.0109017, 0.0067543, -0.0176552, 0.0176553
5: 0.0045159, 0.0199398, 0.0045153, 0.0199405, -0.0154246, 0.0154245
6: -0.0250081, 0.0082066, -0.0250103, 0.0082075, -0.0332156, 0.0332169
7: 0.9578335, 0.9795760, 0.9578318, 0.9795762, -0.0217427, 0.0217442
8: -0.0269166, 0.0116120, -0.0269179, 0.0116136, -0.0385302, 0.0385299
9: -0.0107246, 0.0132362, -0.0107255, 0.0132372, -0.0239617, 0.0239617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157104, upper bound: 0.0159360
time: 0.81 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0159360
time: 0.69 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0094644, 0.0030582, -0.0106089, 0.0034735, -0.0129379, 0.0136671
1: -0.0068584, 0.0003457, -0.0073117, 0.0008432, -0.0077016, 0.0076575
2: 0.0273094, 0.0500743, 0.0266459, 0.0527510, -0.0254416, 0.0234285
3: -0.0038816, 0.0075507, -0.0040337, 0.0087919, -0.0126735, 0.0115844
4: -0.0108764, 0.0067308, -0.0121628, 0.0079206, -0.0187969, 0.0188936
5: 0.0045341, 0.0199176, 0.0035826, 0.0210804, -0.0165463, 0.0163350
6: -0.0249396, 0.0081766, -0.0285282, 0.0097438, -0.0346834, 0.0367049
7: 0.9578865, 0.9795686, 0.9551128, 0.9799570, -0.0220706, 0.0244558
8: -0.0268760, 0.0115613, -0.0290035, 0.0142156, -0.0410915, 0.0405648
9: -0.0106952, 0.0132065, -0.0122316, 0.0147634, -0.0254587, 0.0254381

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157116
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157206
time: 0.61 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.84 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.84
Output dim: 7, lower bound: -0.0157104, upper bound: 0.0158891
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.84
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0158891
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.84
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0156835
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.84
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157206
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.84
Output dim: 7, lower bound: -0.0157104, upper bound: 0.0159360
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.84
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0159360
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.84
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157116
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.84
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157206

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0093009, 0.0029988, -0.0092786, 0.0029907, -0.0122916, 0.0122774
1: -0.0067936, 0.0002746, -0.0067848, 0.0002650, -0.0070585, 0.0070594
2: 0.0274042, 0.0496918, 0.0274171, 0.0496397, -0.0222355, 0.0222747
3: -0.0038599, 0.0073733, -0.0038569, 0.0073492, -0.0112091, 0.0112302
4: -0.0106925, 0.0065608, -0.0106675, 0.0065377, -0.0172302, 0.0172283
5: 0.0046701, 0.0197514, 0.0046886, 0.0197288, -0.0150587, 0.0150628
6: -0.0244267, 0.0079526, -0.0243569, 0.0079222, -0.0323489, 0.0323095
7: 0.9582829, 0.9795132, 0.9583369, 0.9795055, -0.0212226, 0.0211763
8: -0.0265718, 0.0111819, -0.0265305, 0.0111304, -0.0377022, 0.0377124
9: -0.0104756, 0.0129839, -0.0104458, 0.0129537, -0.0234293, 0.0234297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156848, upper bound: 0.0157243
time: 0.79 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156848, upper bound: 0.0158523
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0093313, 0.0030098, -0.0094263, 0.0030443, -0.0123756, 0.0124361
1: -0.0068056, 0.0002879, -0.0068432, 0.0003291, -0.0071348, 0.0071311
2: 0.0273866, 0.0497628, 0.0273315, 0.0499850, -0.0225984, 0.0224313
3: -0.0038639, 0.0074063, -0.0038765, 0.0075093, -0.0113732, 0.0112828
4: -0.0107267, 0.0065924, -0.0108335, 0.0066911, -0.0174178, 0.0174259
5: 0.0046448, 0.0197823, 0.0045658, 0.0198788, -0.0152340, 0.0152165
6: -0.0245220, 0.0079943, -0.0248198, 0.0081244, -0.0326463, 0.0328141
7: 0.9582093, 0.9795234, 0.9579790, 0.9795557, -0.0213463, 0.0215444
8: -0.0266284, 0.0112524, -0.0268050, 0.0114728, -0.0381011, 0.0380574
9: -0.0105164, 0.0130253, -0.0106440, 0.0131545, -0.0236709, 0.0236693

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157243
time: 0.75 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0158523
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0091108, 0.0029298, -0.0105408, 0.0034487, -0.0125596, 0.0134706
1: -0.0067183, 0.0001920, -0.0072847, 0.0008136, -0.0075319, 0.0074768
2: 0.0275144, 0.0492472, 0.0266854, 0.0525916, -0.0250773, 0.0225619
3: -0.0038346, 0.0071672, -0.0040246, 0.0087180, -0.0125526, 0.0111918
4: -0.0104789, 0.0063632, -0.0120862, 0.0078497, -0.0183286, 0.0184494
5: 0.0048281, 0.0195583, 0.0036392, 0.0210111, -0.0161831, 0.0159191
6: -0.0238307, 0.0076924, -0.0283146, 0.0096505, -0.0334812, 0.0360070
7: 0.9587435, 0.9794486, 0.9552779, 0.9799339, -0.0211904, 0.0241707
8: -0.0262185, 0.0107412, -0.0288769, 0.0140575, -0.0402761, 0.0396180
9: -0.0102205, 0.0127254, -0.0121402, 0.0146708, -0.0248912, 0.0248656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0155394
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0156508
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0092875, 0.0029939, -0.0105741, 0.0034608, -0.0127483, 0.0135680
1: -0.0067883, 0.0002688, -0.0072979, 0.0008281, -0.0076163, 0.0075668
2: 0.0274120, 0.0496604, 0.0266661, 0.0526695, -0.0252575, 0.0229944
3: -0.0038581, 0.0073588, -0.0040290, 0.0087541, -0.0126122, 0.0113878
4: -0.0106775, 0.0065468, -0.0121237, 0.0078843, -0.0185618, 0.0186705
5: 0.0046812, 0.0197378, 0.0036115, 0.0210450, -0.0163638, 0.0161263
6: -0.0243846, 0.0079343, -0.0284190, 0.0096961, -0.0340807, 0.0363533
7: 0.9583154, 0.9795086, 0.9551972, 0.9799452, -0.0216298, 0.0243114
8: -0.0265470, 0.0111509, -0.0289388, 0.0141348, -0.0406817, 0.0400897
9: -0.0104576, 0.0129657, -0.0121849, 0.0147161, -0.0251737, 0.0251506

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0156745
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157206
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0094470, 0.0030518, -0.0093080, 0.0030014, -0.0124484, 0.0123599
1: -0.0068515, 0.0003382, -0.0067964, 0.0002778, -0.0071292, 0.0071346
2: 0.0273195, 0.0500335, 0.0274001, 0.0497085, -0.0223890, 0.0226335
3: -0.0038793, 0.0075318, -0.0038608, 0.0073811, -0.0112604, 0.0113926
4: -0.0108568, 0.0067127, -0.0107006, 0.0065682, -0.0174250, 0.0174133
5: 0.0045486, 0.0198998, 0.0046641, 0.0197587, -0.0152101, 0.0152357
6: -0.0248849, 0.0081528, -0.0244491, 0.0079625, -0.0328473, 0.0326019
7: 0.9579288, 0.9795626, 0.9582655, 0.9795155, -0.0215867, 0.0212971
8: -0.0268435, 0.0115209, -0.0265852, 0.0111985, -0.0380421, 0.0381061
9: -0.0106718, 0.0131827, -0.0104852, 0.0129937, -0.0236655, 0.0236680

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156848, upper bound: 0.0157939
time: 0.83 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156848, upper bound: 0.0159104
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0094801, 0.0030638, -0.0094567, 0.0030553, -0.0125354, 0.0125205
1: -0.0068646, 0.0003525, -0.0068553, 0.0003424, -0.0072069, 0.0072078
2: 0.0273003, 0.0501109, 0.0273139, 0.0500561, -0.0227558, 0.0227970
3: -0.0038837, 0.0075677, -0.0038806, 0.0075423, -0.0114259, 0.0114482
4: -0.0108939, 0.0067470, -0.0108676, 0.0067227, -0.0176167, 0.0176147
5: 0.0045211, 0.0199334, 0.0045405, 0.0199097, -0.0153886, 0.0153929
6: -0.0249886, 0.0081980, -0.0249152, 0.0081660, -0.0331545, 0.0331132
7: 0.9578485, 0.9795738, 0.9579054, 0.9795659, -0.0217174, 0.0216684
8: -0.0269050, 0.0115975, -0.0268615, 0.0115433, -0.0384482, 0.0384590
9: -0.0107162, 0.0132277, -0.0106848, 0.0131959, -0.0239121, 0.0239125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157939
time: 0.82 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0159104
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0092869, 0.0029937, -0.0105688, 0.0034589, -0.0127458, 0.0135625
1: -0.0067880, 0.0002686, -0.0072958, 0.0008257, -0.0076138, 0.0075644
2: 0.0274123, 0.0496590, 0.0266692, 0.0526570, -0.0252447, 0.0229899
3: -0.0038580, 0.0073582, -0.0040283, 0.0087483, -0.0126063, 0.0113865
4: -0.0106768, 0.0065462, -0.0121177, 0.0078788, -0.0185556, 0.0186639
5: 0.0046817, 0.0197372, 0.0036159, 0.0210395, -0.0163579, 0.0161212
6: -0.0243828, 0.0079335, -0.0284023, 0.0096888, -0.0340716, 0.0363358
7: 0.9583169, 0.9795083, 0.9552101, 0.9799433, -0.0216265, 0.0242983
8: -0.0265459, 0.0111495, -0.0289288, 0.0141224, -0.0406682, 0.0400784
9: -0.0104569, 0.0129649, -0.0121777, 0.0147088, -0.0251657, 0.0251426

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0156117
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0156914
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0094348, 0.0030474, -0.0106025, 0.0034712, -0.0129059, 0.0136499
1: -0.0068466, 0.0003328, -0.0073092, 0.0008404, -0.0076870, 0.0076421
2: 0.0273266, 0.0500049, 0.0266496, 0.0527360, -0.0254095, 0.0233554
3: -0.0038777, 0.0075185, -0.0040328, 0.0087850, -0.0126626, 0.0115513
4: -0.0108430, 0.0067000, -0.0121556, 0.0079139, -0.0187569, 0.0188556
5: 0.0045587, 0.0198874, 0.0035879, 0.0210739, -0.0165151, 0.0162996
6: -0.0248465, 0.0081360, -0.0285082, 0.0097350, -0.0345815, 0.0366442
7: 0.9579583, 0.9795585, 0.9551283, 0.9799548, -0.0219964, 0.0244303
8: -0.0268208, 0.0114925, -0.0289916, 0.0142007, -0.0410215, 0.0404841
9: -0.0106554, 0.0131661, -0.0122231, 0.0147548, -0.0254102, 0.0253892

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157093
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157206
time: 0.66 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.96 seconds
IS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 7, lower bound: -0.0156848, upper bound: 0.0157243
IS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 7, lower bound: -0.0156848, upper bound: 0.0158523
IS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157243
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0158523
IS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0155394
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0156508
IS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0156745
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157206
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 7, lower bound: -0.0156848, upper bound: 0.0157939
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 7, lower bound: -0.0156848, upper bound: 0.0159104
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157939
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0159104
IS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0156117
IS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0156914
IS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157093
IS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157206

## BFS IS instance: IS_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0091325, 0.0029377, -0.0092178, 0.0029686, -0.0121011, 0.0121555
1: -0.0067269, 0.0002015, -0.0067607, 0.0002385, -0.0069654, 0.0069621
2: 0.0275018, 0.0492979, 0.0274524, 0.0494974, -0.0219956, 0.0218456
3: -0.0038375, 0.0071907, -0.0038488, 0.0072832, -0.0111207, 0.0110395
4: -0.0105032, 0.0063857, -0.0105991, 0.0064744, -0.0169776, 0.0169849
5: 0.0048101, 0.0195803, 0.0047391, 0.0196670, -0.0148569, 0.0148412
6: -0.0238986, 0.0077221, -0.0241661, 0.0078389, -0.0317375, 0.0318882
7: 0.9586910, 0.9794560, 0.9584843, 0.9794849, -0.0207939, 0.0209717
8: -0.0262588, 0.0107914, -0.0264174, 0.0109892, -0.0372481, 0.0372088
9: -0.0102496, 0.0127549, -0.0103641, 0.0128709, -0.0231205, 0.0231189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156848, upper bound: 0.0157243
time: 0.86 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156848, upper bound: 0.0157243
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0092456, 0.0029787, -0.0092648, 0.0029857, -0.0122313, 0.0122435
1: -0.0067717, 0.0002506, -0.0067793, 0.0002589, -0.0070306, 0.0070299
2: 0.0274362, 0.0495624, 0.0274251, 0.0496072, -0.0221710, 0.0221373
3: -0.0038525, 0.0073134, -0.0038551, 0.0073342, -0.0111867, 0.0111685
4: -0.0106304, 0.0065033, -0.0106519, 0.0065232, -0.0171536, 0.0171552
5: 0.0047160, 0.0196952, 0.0047001, 0.0197147, -0.0149987, 0.0149951
6: -0.0242533, 0.0078770, -0.0243134, 0.0079032, -0.0321565, 0.0321903
7: 0.9584169, 0.9794943, 0.9583704, 0.9795009, -0.0210840, 0.0211239
8: -0.0264691, 0.0110538, -0.0265047, 0.0110982, -0.0375673, 0.0375585
9: -0.0104014, 0.0129087, -0.0104271, 0.0129348, -0.0233362, 0.0233359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156848, upper bound: 0.0158523
time: 0.79 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156848, upper bound: 0.0158523
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0091632, 0.0029488, -0.0093626, 0.0030212, -0.0121844, 0.0123114
1: -0.0067390, 0.0002148, -0.0068180, 0.0003015, -0.0070405, 0.0070328
2: 0.0274840, 0.0493697, 0.0273684, 0.0498361, -0.0223521, 0.0220013
3: -0.0038416, 0.0072240, -0.0038681, 0.0074403, -0.0112818, 0.0110921
4: -0.0105378, 0.0064177, -0.0107619, 0.0066250, -0.0171627, 0.0171795
5: 0.0047845, 0.0196115, 0.0046187, 0.0198141, -0.0150296, 0.0149927
6: -0.0239949, 0.0077641, -0.0246202, 0.0080371, -0.0320321, 0.0323843
7: 0.9586167, 0.9794664, 0.9581333, 0.9795341, -0.0209174, 0.0213330
8: -0.0263159, 0.0108626, -0.0266866, 0.0113251, -0.0376410, 0.0375492
9: -0.0102908, 0.0127966, -0.0105585, 0.0130679, -0.0233587, 0.0233551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157243
time: 0.80 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157243
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0092755, 0.0029896, -0.0094121, 0.0030392, -0.0123147, 0.0124017
1: -0.0067835, 0.0002636, -0.0068376, 0.0003230, -0.0071065, 0.0071013
2: 0.0274189, 0.0496325, 0.0273397, 0.0499520, -0.0225331, 0.0222928
3: -0.0038565, 0.0073458, -0.0038746, 0.0074940, -0.0113505, 0.0112205
4: -0.0106640, 0.0065344, -0.0108176, 0.0066764, -0.0173405, 0.0173520
5: 0.0046911, 0.0197256, 0.0045776, 0.0198644, -0.0151733, 0.0151481
6: -0.0243472, 0.0079179, -0.0247755, 0.0081050, -0.0324522, 0.0326934
7: 0.9583443, 0.9795045, 0.9580132, 0.9795508, -0.0212064, 0.0214912
8: -0.0265247, 0.0111232, -0.0267787, 0.0114400, -0.0379647, 0.0379018
9: -0.0104416, 0.0129494, -0.0106250, 0.0131353, -0.0235769, 0.0235744

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_B2_A2_A1

### Relational analysis result of IS_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0158523
time: 0.83 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2

### Relational analysis result of IS_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0158523
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0088690, 0.0028421, -0.0104756, 0.0034251, -0.0122941, 0.0133177
1: -0.0066225, 0.0000869, -0.0072589, 0.0007852, -0.0074077, 0.0073459
2: 0.0276545, 0.0486817, 0.0267232, 0.0524392, -0.0247846, 0.0219586
3: -0.0038025, 0.0069050, -0.0040159, 0.0086473, -0.0124498, 0.0109209
4: -0.0102071, 0.0061118, -0.0120129, 0.0077819, -0.0179891, 0.0181248
5: 0.0050291, 0.0193126, 0.0036934, 0.0209449, -0.0159158, 0.0156192
6: -0.0230726, 0.0073613, -0.0281101, 0.0095612, -0.0326338, 0.0354714
7: 0.9593295, 0.9793665, 0.9554359, 0.9799118, -0.0205823, 0.0239307
8: -0.0257690, 0.0101804, -0.0287557, 0.0139063, -0.0396754, 0.0389360
9: -0.0098959, 0.0123964, -0.0120526, 0.0145821, -0.0244779, 0.0244491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0155394
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0155394
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0090581, 0.0029107, -0.0105262, 0.0034434, -0.0125015, 0.0134369
1: -0.0066974, 0.0001691, -0.0072790, 0.0008072, -0.0075046, 0.0074481
2: 0.0275449, 0.0491240, 0.0266938, 0.0525575, -0.0250126, 0.0224301
3: -0.0038276, 0.0071101, -0.0040227, 0.0087022, -0.0125298, 0.0111327
4: -0.0104196, 0.0063084, -0.0120698, 0.0078345, -0.0182542, 0.0183782
5: 0.0048719, 0.0195047, 0.0036513, 0.0209963, -0.0161244, 0.0158534
6: -0.0236655, 0.0076202, -0.0282688, 0.0096305, -0.0332960, 0.0358890
7: 0.9588713, 0.9794308, 0.9553132, 0.9799289, -0.0210575, 0.0241176
8: -0.0261206, 0.0106189, -0.0288497, 0.0140237, -0.0401443, 0.0394686
9: -0.0101497, 0.0126537, -0.0121206, 0.0146509, -0.0248007, 0.0247742

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0156508
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0156508
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0091186, 0.0029327, -0.0105092, 0.0034373, -0.0125559, 0.0134419
1: -0.0067214, 0.0001954, -0.0072723, 0.0007999, -0.0075212, 0.0074677
2: 0.0275099, 0.0492655, 0.0267037, 0.0525179, -0.0250080, 0.0225619
3: -0.0038356, 0.0071757, -0.0040204, 0.0086838, -0.0125194, 0.0111961
4: -0.0104877, 0.0063713, -0.0120508, 0.0078169, -0.0183046, 0.0184221
5: 0.0048216, 0.0195662, 0.0036654, 0.0209791, -0.0161575, 0.0159008
6: -0.0238552, 0.0077031, -0.0282156, 0.0096073, -0.0334625, 0.0359187
7: 0.9587246, 0.9794513, 0.9553543, 0.9799232, -0.0211986, 0.0240970
8: -0.0262331, 0.0107593, -0.0288182, 0.0139844, -0.0402174, 0.0395775
9: -0.0102310, 0.0127360, -0.0120978, 0.0146278, -0.0248588, 0.0248338

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156914, upper bound: 0.0156745
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156914, upper bound: 0.0156745
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0092317, 0.0029737, -0.0105594, 0.0034555, -0.0126872, 0.0135331
1: -0.0067662, 0.0002446, -0.0072921, 0.0008217, -0.0075879, 0.0075367
2: 0.0274443, 0.0495301, 0.0266746, 0.0526351, -0.0251909, 0.0228555
3: -0.0038507, 0.0072984, -0.0040271, 0.0087382, -0.0125889, 0.0113254
4: -0.0106148, 0.0064889, -0.0121071, 0.0078691, -0.0184839, 0.0185960
5: 0.0047275, 0.0196812, 0.0036237, 0.0210300, -0.0163025, 0.0160574
6: -0.0242099, 0.0078580, -0.0283729, 0.0096759, -0.0338859, 0.0362309
7: 0.9584504, 0.9794897, 0.9552328, 0.9799401, -0.0214897, 0.0242569
8: -0.0264433, 0.0110216, -0.0289114, 0.0141007, -0.0405440, 0.0399331
9: -0.0103828, 0.0128899, -0.0121651, 0.0146961, -0.0250789, 0.0250551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156914, upper bound: 0.0157206
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156914, upper bound: 0.0157206
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0092909, 0.0029952, -0.0092469, 0.0029792, -0.0122701, 0.0122420
1: -0.0067896, 0.0002703, -0.0067722, 0.0002512, -0.0070408, 0.0070425
2: 0.0274100, 0.0496683, 0.0274355, 0.0495655, -0.0221555, 0.0222329
3: -0.0038585, 0.0073625, -0.0038527, 0.0073148, -0.0111733, 0.0112152
4: -0.0106813, 0.0065504, -0.0106318, 0.0065047, -0.0171859, 0.0171822
5: 0.0046784, 0.0197412, 0.0047149, 0.0196965, -0.0150182, 0.0150263
6: -0.0243953, 0.0079390, -0.0242574, 0.0078787, -0.0322740, 0.0321964
7: 0.9583072, 0.9795097, 0.9584138, 0.9794948, -0.0211876, 0.0210959
8: -0.0265533, 0.0111587, -0.0264715, 0.0110567, -0.0376100, 0.0376302
9: -0.0104622, 0.0129703, -0.0104031, 0.0129105, -0.0233727, 0.0233735

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156848, upper bound: 0.0157939
time: 0.84 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156848, upper bound: 0.0157939
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0093910, 0.0030315, -0.0092941, 0.0029963, -0.0123873, 0.0123256
1: -0.0068293, 0.0003138, -0.0067909, 0.0002717, -0.0071010, 0.0071047
2: 0.0273519, 0.0499025, 0.0274081, 0.0496760, -0.0223240, 0.0224944
3: -0.0038718, 0.0074711, -0.0038590, 0.0073660, -0.0112379, 0.0113300
4: -0.0107938, 0.0066545, -0.0106849, 0.0065538, -0.0173476, 0.0173394
5: 0.0045951, 0.0198429, 0.0046757, 0.0197445, -0.0151494, 0.0151673
6: -0.0247092, 0.0080761, -0.0244055, 0.0079434, -0.0326526, 0.0324815
7: 0.9580645, 0.9795437, 0.9582993, 0.9795108, -0.0214463, 0.0212444
8: -0.0267394, 0.0113909, -0.0265593, 0.0111663, -0.0379057, 0.0379502
9: -0.0105966, 0.0131066, -0.0104666, 0.0129747, -0.0235713, 0.0235731

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0159104
time: 0.79 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0159104
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0093233, 0.0030069, -0.0093924, 0.0030320, -0.0123553, 0.0123993
1: -0.0068025, 0.0002844, -0.0068298, 0.0003144, -0.0071169, 0.0071142
2: 0.0273912, 0.0497442, 0.0273512, 0.0499057, -0.0225145, 0.0223930
3: -0.0038628, 0.0073976, -0.0038720, 0.0074725, -0.0113354, 0.0112696
4: -0.0107177, 0.0065841, -0.0107953, 0.0066559, -0.0173736, 0.0173794
5: 0.0046514, 0.0197742, 0.0045940, 0.0198443, -0.0151929, 0.0151801
6: -0.0244969, 0.0079833, -0.0247135, 0.0080779, -0.0325748, 0.0326969
7: 0.9582286, 0.9795207, 0.9580612, 0.9795441, -0.0213155, 0.0214595
8: -0.0266135, 0.0112339, -0.0267419, 0.0113941, -0.0380076, 0.0379758
9: -0.0105057, 0.0130144, -0.0105984, 0.0131084, -0.0236141, 0.0236129

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157939
time: 0.85 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157939
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0094235, 0.0030433, -0.0094424, 0.0030502, -0.0124736, 0.0124857
1: -0.0068421, 0.0003279, -0.0068496, 0.0003362, -0.0071783, 0.0071776
2: 0.0273331, 0.0499785, 0.0273221, 0.0500228, -0.0226897, 0.0226564
3: -0.0038762, 0.0075063, -0.0038787, 0.0075269, -0.0114030, 0.0113850
4: -0.0108303, 0.0066882, -0.0108516, 0.0067079, -0.0175383, 0.0175399
5: 0.0045681, 0.0198759, 0.0045524, 0.0198952, -0.0153271, 0.0153236
6: -0.0248111, 0.0081205, -0.0248705, 0.0081465, -0.0329576, 0.0329910
7: 0.9579858, 0.9795547, 0.9579399, 0.9795612, -0.0215754, 0.0216148
8: -0.0267998, 0.0114663, -0.0268350, 0.0115102, -0.0383100, 0.0383013
9: -0.0106402, 0.0131507, -0.0106657, 0.0131765, -0.0238167, 0.0238164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0159104
time: 0.74 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0159104
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0090496, 0.0029076, -0.0105028, 0.0034350, -0.0124845, 0.0134104
1: -0.0066940, 0.0001654, -0.0072697, 0.0007971, -0.0074911, 0.0074351
2: 0.0275499, 0.0491039, 0.0267074, 0.0525028, -0.0249529, 0.0223965
3: -0.0038265, 0.0071008, -0.0040196, 0.0086768, -0.0125033, 0.0111203
4: -0.0104100, 0.0062995, -0.0120435, 0.0078102, -0.0182202, 0.0183430
5: 0.0048790, 0.0194960, 0.0036708, 0.0209725, -0.0160935, 0.0158252
6: -0.0236386, 0.0076085, -0.0281955, 0.0095985, -0.0332371, 0.0358040
7: 0.9588920, 0.9794279, 0.9553700, 0.9799209, -0.0210289, 0.0240579
8: -0.0261046, 0.0105991, -0.0288062, 0.0139694, -0.0400741, 0.0394053
9: -0.0101382, 0.0126420, -0.0120892, 0.0146191, -0.0247573, 0.0247312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A1_A1_B1

### Relational analysis result of IS_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0156117
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2

### Relational analysis result of IS_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0156117
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0092323, 0.0029739, -0.0105541, 0.0034536, -0.0126859, 0.0135280
1: -0.0067664, 0.0002448, -0.0072900, 0.0008194, -0.0075858, 0.0075348
2: 0.0274439, 0.0495314, 0.0266777, 0.0526228, -0.0251788, 0.0228537
3: -0.0038508, 0.0072990, -0.0040264, 0.0087324, -0.0125832, 0.0113253
4: -0.0106154, 0.0064895, -0.0121012, 0.0078635, -0.0184790, 0.0185907
5: 0.0047271, 0.0196817, 0.0036282, 0.0210246, -0.0162976, 0.0160536
6: -0.0242117, 0.0078588, -0.0283562, 0.0096687, -0.0338804, 0.0362150
7: 0.9584491, 0.9794899, 0.9552456, 0.9799383, -0.0214893, 0.0242442
8: -0.0264444, 0.0110229, -0.0289016, 0.0140884, -0.0405327, 0.0399245
9: -0.0103836, 0.0128907, -0.0121580, 0.0146888, -0.0250724, 0.0250487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A1_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0156914
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0156914
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0092773, 0.0029902, -0.0105369, 0.0034473, -0.0127247, 0.0135271
1: -0.0067843, 0.0002644, -0.0072832, 0.0008119, -0.0075961, 0.0075476
2: 0.0274178, 0.0496367, 0.0266876, 0.0525825, -0.0251647, 0.0229491
3: -0.0038567, 0.0073478, -0.0040241, 0.0087138, -0.0125705, 0.0113719
4: -0.0106661, 0.0065363, -0.0120818, 0.0078457, -0.0185117, 0.0186182
5: 0.0046896, 0.0197275, 0.0036424, 0.0210072, -0.0163176, 0.0160850
6: -0.0243529, 0.0079204, -0.0283024, 0.0096451, -0.0339980, 0.0362228
7: 0.9583400, 0.9795051, 0.9552874, 0.9799325, -0.0215925, 0.0242177
8: -0.0265281, 0.0111273, -0.0288696, 0.0140485, -0.0405766, 0.0399970
9: -0.0104440, 0.0129519, -0.0121350, 0.0146655, -0.0251095, 0.0250869

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157093
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157093
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0093782, 0.0030268, -0.0105877, 0.0034658, -0.0128439, 0.0136146
1: -0.0068242, 0.0003082, -0.0073034, 0.0008340, -0.0076582, 0.0076116
2: 0.0273594, 0.0498725, 0.0266581, 0.0527015, -0.0253421, 0.0232143
3: -0.0038701, 0.0074571, -0.0040308, 0.0087689, -0.0126390, 0.0114880
4: -0.0107794, 0.0066411, -0.0121390, 0.0078985, -0.0186779, 0.0187801
5: 0.0046058, 0.0198299, 0.0036002, 0.0210588, -0.0164530, 0.0162297
6: -0.0246690, 0.0080585, -0.0284618, 0.0097148, -0.0343838, 0.0365203
7: 0.9580956, 0.9795393, 0.9551641, 0.9799498, -0.0218542, 0.0243753
8: -0.0267155, 0.0113612, -0.0289641, 0.0141664, -0.0408819, 0.0403253
9: -0.0105794, 0.0130891, -0.0122032, 0.0147347, -0.0253141, 0.0252923

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157206
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2

### Relational analysis result of IS_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157206
time: 0.84 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.24 seconds
IS_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 7, lower bound: -0.0156848, upper bound: 0.0157243
IS_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 7, lower bound: -0.0156848, upper bound: 0.0157243
IS_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 7, lower bound: -0.0156848, upper bound: 0.0158523
IS_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 7, lower bound: -0.0156848, upper bound: 0.0158523
IS_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157243
IS_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157243
IS_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0158523
IS_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0158523
IS_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0155394
IS_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0155394
IS_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0156508
IS_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0156508
IS_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 7, lower bound: -0.0156914, upper bound: 0.0156745
IS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 7, lower bound: -0.0156914, upper bound: 0.0156745
IS_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 7, lower bound: -0.0156914, upper bound: 0.0157206
IS_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 7, lower bound: -0.0156914, upper bound: 0.0157206
IS_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 7, lower bound: -0.0156848, upper bound: 0.0157939
IS_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 7, lower bound: -0.0156848, upper bound: 0.0157939
IS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0159104
IS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0159104
IS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157939
IS_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157939
IS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0159104
IS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0159104
IS_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0156117
IS_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0156117
IS_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0156914
IS_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0156914
IS_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157093
IS_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157093
IS_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157206
IS_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157206

## BFS IS instance: IS_A1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0090919, 0.0029230, -0.0092178, 0.0029686, -0.0120605, 0.0121407
1: -0.0067108, 0.0001838, -0.0067607, 0.0002385, -0.0069493, 0.0069445
2: 0.0275253, 0.0492030, 0.0274524, 0.0494974, -0.0219721, 0.0217506
3: -0.0038321, 0.0071467, -0.0038488, 0.0072832, -0.0111153, 0.0109955
4: -0.0104576, 0.0063436, -0.0105991, 0.0064744, -0.0169320, 0.0169427
5: 0.0048438, 0.0195391, 0.0047391, 0.0196670, -0.0148232, 0.0147999
6: -0.0237714, 0.0076665, -0.0241661, 0.0078389, -0.0316103, 0.0318326
7: 0.9587895, 0.9794422, 0.9584843, 0.9794849, -0.0206954, 0.0209579
8: -0.0261834, 0.0106973, -0.0264174, 0.0109892, -0.0371726, 0.0371147
9: -0.0101951, 0.0126997, -0.0103641, 0.0128709, -0.0230660, 0.0230637

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157243
time: 0.89 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157243
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0103882, 0.0033934, -0.0092178, 0.0029686, -0.0133568, 0.0126112
1: -0.0072243, 0.0007472, -0.0067607, 0.0002385, -0.0074628, 0.0075079
2: 0.0267738, 0.0522347, 0.0274524, 0.0494974, -0.0227236, 0.0247824
3: -0.0040043, 0.0085525, -0.0038488, 0.0072832, -0.0112875, 0.0124013
4: -0.0119147, 0.0076911, -0.0105991, 0.0064744, -0.0183891, 0.0182902
5: 0.0037661, 0.0208561, 0.0047391, 0.0196670, -0.0159009, 0.0161170
6: -0.0278360, 0.0094415, -0.0241661, 0.0078389, -0.0356749, 0.0336076
7: 0.9556477, 0.9798821, 0.9584843, 0.9794849, -0.0238371, 0.0213978
8: -0.0285932, 0.0137036, -0.0264174, 0.0109892, -0.0395824, 0.0401210
9: -0.0119353, 0.0144632, -0.0103641, 0.0128709, -0.0248062, 0.0248272

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157243
time: 1.00 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157243
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0092068, 0.0029647, -0.0092648, 0.0029857, -0.0121925, 0.0122294
1: -0.0067563, 0.0002338, -0.0067793, 0.0002589, -0.0070153, 0.0070130
2: 0.0274587, 0.0494718, 0.0274251, 0.0496072, -0.0221485, 0.0220466
3: -0.0038474, 0.0072713, -0.0038551, 0.0073342, -0.0111815, 0.0111264
4: -0.0105868, 0.0064630, -0.0106519, 0.0065232, -0.0171100, 0.0171149
5: 0.0047482, 0.0196558, 0.0047001, 0.0197147, -0.0149664, 0.0149557
6: -0.0241318, 0.0078239, -0.0243134, 0.0079032, -0.0320349, 0.0321372
7: 0.9585109, 0.9794813, 0.9583704, 0.9795009, -0.0209900, 0.0211108
8: -0.0263970, 0.0109638, -0.0265047, 0.0110982, -0.0374952, 0.0374685
9: -0.0103494, 0.0128560, -0.0104271, 0.0129348, -0.0232842, 0.0232831

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155956, upper bound: 0.0158523
time: 0.90 seconds

## Relational analysis of IS_A1_B1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155956, upper bound: 0.0158523
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0103679, 0.0033860, -0.0092648, 0.0029857, -0.0133535, 0.0126507
1: -0.0072163, 0.0007384, -0.0067793, 0.0002589, -0.0074752, 0.0075177
2: 0.0267856, 0.0521872, 0.0274251, 0.0496072, -0.0228216, 0.0247620
3: -0.0040016, 0.0085304, -0.0038551, 0.0073342, -0.0113358, 0.0123855
4: -0.0118918, 0.0076700, -0.0106519, 0.0065232, -0.0184151, 0.0183219
5: 0.0037830, 0.0208354, 0.0047001, 0.0197147, -0.0159317, 0.0161353
6: -0.0277723, 0.0094137, -0.0243134, 0.0079032, -0.0356754, 0.0337271
7: 0.9556970, 0.9798752, 0.9583704, 0.9795009, -0.0238039, 0.0215048
8: -0.0285554, 0.0136565, -0.0265047, 0.0110982, -0.0396535, 0.0401612
9: -0.0119080, 0.0144355, -0.0104271, 0.0129348, -0.0248428, 0.0248627

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0158523
time: 0.80 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0158523
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0091227, 0.0029341, -0.0093626, 0.0030212, -0.0121439, 0.0122968
1: -0.0067230, 0.0001972, -0.0068180, 0.0003015, -0.0070244, 0.0070152
2: 0.0275075, 0.0492750, 0.0273684, 0.0498361, -0.0223286, 0.0219066
3: -0.0038362, 0.0071801, -0.0038681, 0.0074403, -0.0112764, 0.0110481
4: -0.0104922, 0.0063755, -0.0107619, 0.0066250, -0.0171172, 0.0171374
5: 0.0048182, 0.0195704, 0.0046187, 0.0198141, -0.0149959, 0.0149516
6: -0.0238679, 0.0077087, -0.0246202, 0.0080371, -0.0319051, 0.0323288
7: 0.9587148, 0.9794527, 0.9581333, 0.9795341, -0.0208192, 0.0213194
8: -0.0262406, 0.0107687, -0.0266866, 0.0113251, -0.0375657, 0.0374553
9: -0.0102364, 0.0127415, -0.0105585, 0.0130679, -0.0233043, 0.0233000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_B2_A1_A1_A1

### Relational analysis result of IS_A1_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156848, upper bound: 0.0155431
time: 0.91 seconds

## Relational analysis of IS_A1_B1_B2_A1_A1_A2

### Relational analysis result of IS_A1_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156848, upper bound: 0.0157243
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0104200, 0.0034049, -0.0093626, 0.0030212, -0.0134412, 0.0127675
1: -0.0072369, 0.0007611, -0.0068180, 0.0003015, -0.0075384, 0.0075791
2: 0.0267554, 0.0523092, 0.0273684, 0.0498361, -0.0230807, 0.0249407
3: -0.0040086, 0.0085870, -0.0038681, 0.0074403, -0.0114488, 0.0124551
4: -0.0119505, 0.0077241, -0.0107619, 0.0066250, -0.0185754, 0.0184860
5: 0.0037396, 0.0208884, 0.0046187, 0.0198141, -0.0160745, 0.0162697
6: -0.0279358, 0.0094851, -0.0246202, 0.0080371, -0.0359730, 0.0341053
7: 0.9555706, 0.9798929, 0.9581333, 0.9795341, -0.0239635, 0.0217596
8: -0.0286523, 0.0137774, -0.0266866, 0.0113251, -0.0399774, 0.0404640
9: -0.0119780, 0.0145064, -0.0105585, 0.0130679, -0.0250459, 0.0250650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157243
time: 0.71 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157243
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0092370, 0.0029756, -0.0094121, 0.0030392, -0.0122761, 0.0123877
1: -0.0067683, 0.0002469, -0.0068376, 0.0003230, -0.0070912, 0.0070845
2: 0.0274412, 0.0495423, 0.0273397, 0.0499520, -0.0225107, 0.0222026
3: -0.0038514, 0.0073040, -0.0038746, 0.0074940, -0.0113453, 0.0111787
4: -0.0106207, 0.0064943, -0.0108176, 0.0066764, -0.0172971, 0.0173119
5: 0.0047232, 0.0196864, 0.0045776, 0.0198644, -0.0151412, 0.0151089
6: -0.0242262, 0.0078651, -0.0247755, 0.0081050, -0.0323313, 0.0326406
7: 0.9584379, 0.9794914, 0.9580132, 0.9795508, -0.0211129, 0.0214782
8: -0.0264530, 0.0110337, -0.0267787, 0.0114400, -0.0378930, 0.0378124
9: -0.0103898, 0.0128970, -0.0106250, 0.0131353, -0.0235251, 0.0235220

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_B2_A2_A1_A1

### Relational analysis result of IS_A1_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156848, upper bound: 0.0157137
time: 0.87 seconds

## Relational analysis of IS_A1_B1_B2_A2_A1_A2

### Relational analysis result of IS_A1_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156848, upper bound: 0.0158523
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0103997, 0.0033975, -0.0094121, 0.0030392, -0.0134388, 0.0128097
1: -0.0072288, 0.0007522, -0.0068376, 0.0003230, -0.0075518, 0.0075899
2: 0.0267672, 0.0522616, 0.0273397, 0.0499520, -0.0231848, 0.0249219
3: -0.0040059, 0.0085649, -0.0038746, 0.0074940, -0.0114998, 0.0124396
4: -0.0119276, 0.0077030, -0.0108176, 0.0066764, -0.0186040, 0.0185206
5: 0.0037565, 0.0208677, 0.0045776, 0.0198644, -0.0161079, 0.0162902
6: -0.0278721, 0.0094572, -0.0247755, 0.0081050, -0.0359771, 0.0342327
7: 0.9556199, 0.9798860, 0.9580132, 0.9795508, -0.0239308, 0.0218728
8: -0.0286145, 0.0137302, -0.0267787, 0.0114400, -0.0400545, 0.0405089
9: -0.0119507, 0.0144788, -0.0106250, 0.0131353, -0.0250860, 0.0251038

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0158523
time: 0.70 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0158523
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0088690, 0.0028421, -0.0103616, 0.0033837, -0.0122527, 0.0132037
1: -0.0066225, 0.0000869, -0.0072138, 0.0007357, -0.0073582, 0.0073007
2: 0.0276545, 0.0486817, 0.0267893, 0.0521725, -0.0245180, 0.0218925
3: -0.0038025, 0.0069050, -0.0040008, 0.0085237, -0.0123261, 0.0109058
4: -0.0102071, 0.0061118, -0.0118848, 0.0076634, -0.0178705, 0.0179967
5: 0.0050291, 0.0193126, 0.0037882, 0.0208291, -0.0158000, 0.0155244
6: -0.0230726, 0.0073613, -0.0277527, 0.0094051, -0.0324777, 0.0351140
7: 0.9593295, 0.9793665, 0.9557121, 0.9798731, -0.0205435, 0.0236545
8: -0.0257690, 0.0101804, -0.0285437, 0.0136420, -0.0394110, 0.0387241
9: -0.0098959, 0.0123964, -0.0118996, 0.0144270, -0.0243229, 0.0242961

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0155196
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0155394
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0088690, 0.0028421, -0.0104669, 0.0034219, -0.0122910, 0.0133090
1: -0.0066225, 0.0000869, -0.0072555, 0.0007815, -0.0074040, 0.0073424
2: 0.0276545, 0.0486817, 0.0267282, 0.0524189, -0.0247643, 0.0219535
3: -0.0038025, 0.0069050, -0.0040148, 0.0086379, -0.0124404, 0.0109198
4: -0.0102071, 0.0061118, -0.0120032, 0.0077729, -0.0179800, 0.0181150
5: 0.0050291, 0.0193126, 0.0037006, 0.0209361, -0.0159070, 0.0156120
6: -0.0230726, 0.0073613, -0.0280829, 0.0095493, -0.0326219, 0.0354442
7: 0.9593295, 0.9793665, 0.9554570, 0.9799088, -0.0205792, 0.0239096
8: -0.0257690, 0.0101804, -0.0287395, 0.0138862, -0.0396552, 0.0389199
9: -0.0098959, 0.0123964, -0.0120410, 0.0145703, -0.0244662, 0.0244374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0155196
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0155394
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0090581, 0.0029107, -0.0104099, 0.0034013, -0.0124594, 0.0133206
1: -0.0066974, 0.0001691, -0.0072329, 0.0007567, -0.0074541, 0.0074020
2: 0.0275449, 0.0491240, 0.0267612, 0.0522856, -0.0247407, 0.0223627
3: -0.0038276, 0.0071101, -0.0040072, 0.0085761, -0.0124037, 0.0111173
4: -0.0104196, 0.0063084, -0.0119391, 0.0077137, -0.0181333, 0.0182476
5: 0.0048719, 0.0195047, 0.0037480, 0.0208782, -0.0160063, 0.0157567
6: -0.0236655, 0.0076202, -0.0279043, 0.0094713, -0.0331368, 0.0355245
7: 0.9588713, 0.9794308, 0.9555951, 0.9798895, -0.0210182, 0.0238357
8: -0.0261206, 0.0106189, -0.0286336, 0.0137541, -0.0398746, 0.0392525
9: -0.0101497, 0.0126537, -0.0119645, 0.0144927, -0.0246425, 0.0246182

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A1_A2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156749, upper bound: 0.0156508
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156749, upper bound: 0.0156508
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0090581, 0.0029107, -0.0105182, 0.0034405, -0.0124986, 0.0134289
1: -0.0066974, 0.0001691, -0.0072758, 0.0008038, -0.0075012, 0.0074449
2: 0.0275449, 0.0491240, 0.0266985, 0.0525387, -0.0249938, 0.0224255
3: -0.0038276, 0.0071101, -0.0040216, 0.0086935, -0.0125211, 0.0111317
4: -0.0104196, 0.0063084, -0.0120608, 0.0078262, -0.0182459, 0.0183692
5: 0.0048719, 0.0195047, 0.0036580, 0.0209882, -0.0161163, 0.0158467
6: -0.0236655, 0.0076202, -0.0282437, 0.0096195, -0.0332850, 0.0358639
7: 0.9588713, 0.9794308, 0.9553328, 0.9799262, -0.0210549, 0.0240980
8: -0.0261206, 0.0106189, -0.0288348, 0.0140051, -0.0401257, 0.0394538
9: -0.0101497, 0.0126537, -0.0121098, 0.0146400, -0.0247897, 0.0247635

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A1_A2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156749, upper bound: 0.0156508
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156749, upper bound: 0.0156508
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0091186, 0.0029327, -0.0104752, 0.0034250, -0.0125436, 0.0134079
1: -0.0067214, 0.0001954, -0.0072588, 0.0007851, -0.0075065, 0.0074542
2: 0.0275099, 0.0492655, 0.0267234, 0.0524383, -0.0249284, 0.0225421
3: -0.0038356, 0.0071757, -0.0040159, 0.0086469, -0.0124825, 0.0111916
4: -0.0104877, 0.0063713, -0.0120125, 0.0077816, -0.0182692, 0.0183838
5: 0.0048216, 0.0195662, 0.0036937, 0.0209445, -0.0161229, 0.0158725
6: -0.0238552, 0.0077031, -0.0281090, 0.0095607, -0.0334159, 0.0358121
7: 0.9587246, 0.9794513, 0.9554368, 0.9799116, -0.0211869, 0.0240145
8: -0.0262331, 0.0107593, -0.0287550, 0.0139055, -0.0401385, 0.0395142
9: -0.0102310, 0.0127360, -0.0120521, 0.0145816, -0.0248126, 0.0247881

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156846, upper bound: 0.0156701
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156846, upper bound: 0.0156745
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0091186, 0.0029327, -0.0104851, 0.0034285, -0.0125472, 0.0134178
1: -0.0067214, 0.0001954, -0.0072627, 0.0007894, -0.0075108, 0.0074581
2: 0.0275099, 0.0492655, 0.0267176, 0.0524614, -0.0249516, 0.0225479
3: -0.0038356, 0.0071757, -0.0040172, 0.0086576, -0.0124933, 0.0111929
4: -0.0104877, 0.0063713, -0.0120237, 0.0077919, -0.0182795, 0.0183950
5: 0.0048216, 0.0195662, 0.0036855, 0.0209546, -0.0161330, 0.0158807
6: -0.0238552, 0.0077031, -0.0281401, 0.0095742, -0.0334295, 0.0358432
7: 0.9587246, 0.9794513, 0.9554127, 0.9799150, -0.0211903, 0.0240386
8: -0.0262331, 0.0107593, -0.0287734, 0.0139284, -0.0401615, 0.0395326
9: -0.0102310, 0.0127360, -0.0120654, 0.0145950, -0.0248260, 0.0248014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156846, upper bound: 0.0156701
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156846, upper bound: 0.0156745
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0092317, 0.0029737, -0.0105230, 0.0034423, -0.0126740, 0.0134967
1: -0.0067662, 0.0002446, -0.0072777, 0.0008059, -0.0075720, 0.0075223
2: 0.0274443, 0.0495301, 0.0266957, 0.0525501, -0.0251058, 0.0228344
3: -0.0038507, 0.0072984, -0.0040222, 0.0086987, -0.0125494, 0.0113206
4: -0.0106148, 0.0064889, -0.0120662, 0.0078313, -0.0184461, 0.0185552
5: 0.0047275, 0.0196812, 0.0036540, 0.0209931, -0.0162656, 0.0160272
6: -0.0242099, 0.0078580, -0.0282588, 0.0096261, -0.0338361, 0.0361168
7: 0.9584504, 0.9794897, 0.9553210, 0.9799278, -0.0214774, 0.0241687
8: -0.0264433, 0.0110216, -0.0288438, 0.0140163, -0.0404597, 0.0398655
9: -0.0103828, 0.0128899, -0.0121163, 0.0146466, -0.0250294, 0.0250062

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156846, upper bound: 0.0157206
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156846, upper bound: 0.0157206
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0092317, 0.0029737, -0.0105355, 0.0034468, -0.0126785, 0.0135092
1: -0.0067662, 0.0002446, -0.0072827, 0.0008113, -0.0075775, 0.0075273
2: 0.0274443, 0.0495301, 0.0266884, 0.0525792, -0.0251349, 0.0228416
3: -0.0038507, 0.0072984, -0.0040239, 0.0087122, -0.0125629, 0.0113222
4: -0.0106148, 0.0064889, -0.0120802, 0.0078442, -0.0184590, 0.0185692
5: 0.0047275, 0.0196812, 0.0036436, 0.0210057, -0.0162782, 0.0160375
6: -0.0242099, 0.0078580, -0.0282979, 0.0096432, -0.0338531, 0.0361559
7: 0.9584504, 0.9794897, 0.9552907, 0.9799320, -0.0214815, 0.0241989
8: -0.0264433, 0.0110216, -0.0288670, 0.0140452, -0.0404886, 0.0398886
9: -0.0103828, 0.0128899, -0.0121330, 0.0146635, -0.0250464, 0.0250229

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156117, upper bound: 0.0157206
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156117, upper bound: 0.0157206
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0092502, 0.0029804, -0.0092469, 0.0029792, -0.0122294, 0.0122273
1: -0.0067735, 0.0002526, -0.0067722, 0.0002512, -0.0070247, 0.0070248
2: 0.0274336, 0.0495732, 0.0274355, 0.0495655, -0.0221319, 0.0221377
3: -0.0038531, 0.0073184, -0.0038527, 0.0073148, -0.0111679, 0.0111711
4: -0.0106356, 0.0065081, -0.0106318, 0.0065047, -0.0171402, 0.0171399
5: 0.0047122, 0.0196999, 0.0047149, 0.0196965, -0.0149844, 0.0149850
6: -0.0242678, 0.0078833, -0.0242574, 0.0078787, -0.0321465, 0.0321407
7: 0.9584057, 0.9794959, 0.9584138, 0.9794948, -0.0210891, 0.0210821
8: -0.0264777, 0.0110644, -0.0264715, 0.0110567, -0.0375344, 0.0375359
9: -0.0104076, 0.0129150, -0.0104031, 0.0129105, -0.0233181, 0.0233181

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157939
time: 0.81 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157939
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0104876, 0.0034295, -0.0092469, 0.0029792, -0.0134668, 0.0126763
1: -0.0072637, 0.0007905, -0.0067722, 0.0002512, -0.0075149, 0.0075626
2: 0.0267162, 0.0524673, 0.0274355, 0.0495655, -0.0228493, 0.0250318
3: -0.0040175, 0.0086603, -0.0038527, 0.0073148, -0.0113323, 0.0125130
4: -0.0120265, 0.0077945, -0.0106318, 0.0065047, -0.0185312, 0.0184263
5: 0.0036834, 0.0209571, 0.0047149, 0.0196965, -0.0160132, 0.0162422
6: -0.0281479, 0.0095777, -0.0242574, 0.0078787, -0.0360266, 0.0338351
7: 0.9554067, 0.9799159, 0.9584138, 0.9794948, -0.0240881, 0.0215021
8: -0.0287781, 0.0139343, -0.0264715, 0.0110567, -0.0398348, 0.0404058
9: -0.0120688, 0.0145985, -0.0104031, 0.0129105, -0.0249793, 0.0250016

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157939
time: 0.81 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157939
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0093910, 0.0030315, -0.0090804, 0.0029188, -0.0123098, 0.0121119
1: -0.0068293, 0.0003138, -0.0067062, 0.0001788, -0.0070081, 0.0070200
2: 0.0273519, 0.0499025, 0.0275320, 0.0491761, -0.0218242, 0.0223705
3: -0.0038718, 0.0074711, -0.0038306, 0.0071342, -0.0110061, 0.0113016
4: -0.0107938, 0.0066545, -0.0104447, 0.0063316, -0.0171254, 0.0170992
5: 0.0045951, 0.0198429, 0.0048534, 0.0195274, -0.0149322, 0.0149896
6: -0.0247092, 0.0080761, -0.0237353, 0.0076507, -0.0323600, 0.0318114
7: 0.9580645, 0.9795437, 0.9588172, 0.9794384, -0.0213739, 0.0207265
8: -0.0267394, 0.0113909, -0.0261620, 0.0106706, -0.0374100, 0.0375529
9: -0.0105966, 0.0131066, -0.0101796, 0.0126840, -0.0232806, 0.0232862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0159104
time: 0.79 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0159104
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0093910, 0.0030315, -0.0092559, 0.0029825, -0.0123735, 0.0122874
1: -0.0068293, 0.0003138, -0.0067757, 0.0002551, -0.0070844, 0.0070896
2: 0.0273519, 0.0499025, 0.0274303, 0.0495865, -0.0222346, 0.0224723
3: -0.0038718, 0.0074711, -0.0038539, 0.0073245, -0.0111964, 0.0113249
4: -0.0107938, 0.0066545, -0.0106420, 0.0065140, -0.0173079, 0.0172964
5: 0.0045951, 0.0198429, 0.0047075, 0.0197057, -0.0151105, 0.0151355
6: -0.0247092, 0.0080761, -0.0242856, 0.0078910, -0.0326003, 0.0323617
7: 0.9580645, 0.9795437, 0.9583920, 0.9794978, -0.0214334, 0.0211517
8: -0.0267394, 0.0113909, -0.0264882, 0.0110776, -0.0378170, 0.0378791
9: -0.0105966, 0.0131066, -0.0104152, 0.0129227, -0.0235193, 0.0235218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0159104
time: 0.89 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0159104
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0093233, 0.0030069, -0.0092073, 0.0029648, -0.0122881, 0.0122142
1: -0.0068025, 0.0002844, -0.0067565, 0.0002340, -0.0070364, 0.0070409
2: 0.0273912, 0.0497442, 0.0274585, 0.0494729, -0.0220817, 0.0222857
3: -0.0038628, 0.0073976, -0.0038474, 0.0072718, -0.0111347, 0.0112450
4: -0.0107177, 0.0065841, -0.0105873, 0.0064635, -0.0171812, 0.0171714
5: 0.0046514, 0.0197742, 0.0047479, 0.0196563, -0.0150049, 0.0150263
6: -0.0244969, 0.0079833, -0.0241333, 0.0078245, -0.0323214, 0.0321166
7: 0.9582286, 0.9795207, 0.9585097, 0.9794814, -0.0212528, 0.0210111
8: -0.0266135, 0.0112339, -0.0263979, 0.0109649, -0.0375784, 0.0376318
9: -0.0105057, 0.0130144, -0.0103500, 0.0128566, -0.0233623, 0.0233644

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157939
time: 0.75 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157939
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0093233, 0.0030069, -0.0093546, 0.0030183, -0.0123416, 0.0123615
1: -0.0068025, 0.0002844, -0.0068149, 0.0002980, -0.0071004, 0.0070992
2: 0.0273912, 0.0497442, 0.0273731, 0.0498174, -0.0224262, 0.0223711
3: -0.0038628, 0.0073976, -0.0038670, 0.0074316, -0.0112944, 0.0112646
4: -0.0107177, 0.0065841, -0.0107529, 0.0066166, -0.0173343, 0.0173370
5: 0.0046514, 0.0197742, 0.0046254, 0.0198060, -0.0151545, 0.0151487
6: -0.0244969, 0.0079833, -0.0245951, 0.0080262, -0.0325231, 0.0325784
7: 0.9582286, 0.9795207, 0.9581528, 0.9795313, -0.0213027, 0.0213680
8: -0.0266135, 0.0112339, -0.0266717, 0.0113065, -0.0379200, 0.0379056
9: -0.0105057, 0.0130144, -0.0105477, 0.0130570, -0.0235627, 0.0235622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157939
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157939
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0094235, 0.0030433, -0.0092566, 0.0029827, -0.0124062, 0.0122999
1: -0.0068421, 0.0003279, -0.0067760, 0.0002554, -0.0070975, 0.0071040
2: 0.0273331, 0.0499785, 0.0274299, 0.0495882, -0.0222551, 0.0225486
3: -0.0038762, 0.0075063, -0.0038540, 0.0073253, -0.0112015, 0.0113603
4: -0.0108303, 0.0066882, -0.0106428, 0.0065148, -0.0173451, 0.0173310
5: 0.0045681, 0.0198759, 0.0047069, 0.0197064, -0.0151383, 0.0151691
6: -0.0248111, 0.0081205, -0.0242879, 0.0078920, -0.0327031, 0.0324084
7: 0.9579858, 0.9795547, 0.9583902, 0.9794980, -0.0215122, 0.0211645
8: -0.0267998, 0.0114663, -0.0264896, 0.0110793, -0.0378790, 0.0379559
9: -0.0106402, 0.0131507, -0.0104162, 0.0129237, -0.0235640, 0.0235669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0159104
time: 0.76 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0159104
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0094235, 0.0030433, -0.0094046, 0.0030364, -0.0124599, 0.0124479
1: -0.0068421, 0.0003279, -0.0068347, 0.0003197, -0.0071619, 0.0071626
2: 0.0273331, 0.0499785, 0.0273441, 0.0499343, -0.0226012, 0.0226344
3: -0.0038762, 0.0075063, -0.0038736, 0.0074858, -0.0113620, 0.0113799
4: -0.0108303, 0.0066882, -0.0108091, 0.0066686, -0.0174989, 0.0174973
5: 0.0045681, 0.0198759, 0.0045838, 0.0198568, -0.0152887, 0.0152921
6: -0.0248111, 0.0081205, -0.0247519, 0.0080947, -0.0329058, 0.0328725
7: 0.9579858, 0.9795547, 0.9580316, 0.9795483, -0.0215625, 0.0215231
8: -0.0267998, 0.0114663, -0.0267647, 0.0114225, -0.0382223, 0.0382310
9: -0.0106402, 0.0131507, -0.0106149, 0.0131251, -0.0237653, 0.0237656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0159104
time: 0.77 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0159104
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0090496, 0.0029076, -0.0103616, 0.0033837, -0.0124333, 0.0132692
1: -0.0066940, 0.0001654, -0.0072138, 0.0007357, -0.0074297, 0.0073792
2: 0.0275499, 0.0491039, 0.0267893, 0.0521725, -0.0246226, 0.0223147
3: -0.0038265, 0.0071008, -0.0040008, 0.0085237, -0.0123501, 0.0111015
4: -0.0104100, 0.0062995, -0.0118848, 0.0076634, -0.0180735, 0.0181843
5: 0.0048790, 0.0194960, 0.0037882, 0.0208291, -0.0159501, 0.0157079
6: -0.0236386, 0.0076085, -0.0277527, 0.0094051, -0.0330437, 0.0353612
7: 0.9588920, 0.9794279, 0.9557121, 0.9798731, -0.0209810, 0.0237158
8: -0.0261046, 0.0105991, -0.0285437, 0.0136420, -0.0397466, 0.0391428
9: -0.0101382, 0.0126420, -0.0118996, 0.0144270, -0.0245652, 0.0245416

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0155954
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0156117
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0090496, 0.0029076, -0.0104669, 0.0034219, -0.0124715, 0.0133745
1: -0.0066940, 0.0001654, -0.0072555, 0.0007815, -0.0074755, 0.0074209
2: 0.0275499, 0.0491039, 0.0267282, 0.0524189, -0.0248690, 0.0223757
3: -0.0038265, 0.0071008, -0.0040148, 0.0086379, -0.0124644, 0.0111155
4: -0.0104100, 0.0062995, -0.0120032, 0.0077729, -0.0181829, 0.0183027
5: 0.0048790, 0.0194960, 0.0037006, 0.0209361, -0.0160571, 0.0157954
6: -0.0236386, 0.0076085, -0.0280829, 0.0095493, -0.0331879, 0.0356914
7: 0.9588920, 0.9794279, 0.9554570, 0.9799088, -0.0210167, 0.0239709
8: -0.0261046, 0.0105991, -0.0287395, 0.0138862, -0.0399908, 0.0393386
9: -0.0101382, 0.0126420, -0.0120410, 0.0145703, -0.0247085, 0.0246830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0155954
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0156117
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0092323, 0.0029739, -0.0104099, 0.0034013, -0.0126336, 0.0133838
1: -0.0067664, 0.0002448, -0.0072329, 0.0007567, -0.0075231, 0.0074778
2: 0.0274439, 0.0495314, 0.0267612, 0.0522856, -0.0248417, 0.0227702
3: -0.0038508, 0.0072990, -0.0040072, 0.0085761, -0.0124268, 0.0113062
4: -0.0106154, 0.0064895, -0.0119391, 0.0077137, -0.0183291, 0.0184287
5: 0.0047271, 0.0196817, 0.0037480, 0.0208782, -0.0161511, 0.0159337
6: -0.0242117, 0.0078588, -0.0279043, 0.0094713, -0.0336830, 0.0357630
7: 0.9584491, 0.9794899, 0.9555951, 0.9798895, -0.0214404, 0.0238948
8: -0.0264444, 0.0110229, -0.0286336, 0.0137541, -0.0401985, 0.0396565
9: -0.0103836, 0.0128907, -0.0119645, 0.0144927, -0.0248763, 0.0248552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0156846
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0156914
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0092323, 0.0029739, -0.0105182, 0.0034405, -0.0126729, 0.0134921
1: -0.0067664, 0.0002448, -0.0072758, 0.0008038, -0.0075702, 0.0075206
2: 0.0274439, 0.0495314, 0.0266985, 0.0525387, -0.0250948, 0.0228329
3: -0.0038508, 0.0072990, -0.0040216, 0.0086935, -0.0125442, 0.0113206
4: -0.0106154, 0.0064895, -0.0120608, 0.0078262, -0.0184417, 0.0185503
5: 0.0047271, 0.0196817, 0.0036580, 0.0209882, -0.0162611, 0.0160237
6: -0.0242117, 0.0078588, -0.0282437, 0.0096195, -0.0338312, 0.0361024
7: 0.9584491, 0.9794899, 0.9553328, 0.9799262, -0.0214772, 0.0241571
8: -0.0264444, 0.0110229, -0.0288348, 0.0140051, -0.0404495, 0.0398578
9: -0.0103836, 0.0128907, -0.0121098, 0.0146400, -0.0250236, 0.0250005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A2_B2_A1_A2_B2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156745, upper bound: 0.0156914
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156745, upper bound: 0.0156914
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0092773, 0.0029902, -0.0103936, 0.0033953, -0.0126727, 0.0133839
1: -0.0067843, 0.0002644, -0.0072265, 0.0007496, -0.0075339, 0.0074909
2: 0.0274178, 0.0496367, 0.0267707, 0.0522475, -0.0248296, 0.0228660
3: -0.0038567, 0.0073478, -0.0040051, 0.0085584, -0.0124151, 0.0113529
4: -0.0106661, 0.0065363, -0.0119208, 0.0076967, -0.0183628, 0.0184571
5: 0.0046896, 0.0197275, 0.0037615, 0.0208616, -0.0161720, 0.0159659
6: -0.0243529, 0.0079204, -0.0278531, 0.0094490, -0.0338018, 0.0357735
7: 0.9583400, 0.9795051, 0.9556345, 0.9798839, -0.0215439, 0.0238706
8: -0.0265281, 0.0111273, -0.0286033, 0.0137163, -0.0402443, 0.0397307
9: -0.0104440, 0.0129519, -0.0119426, 0.0144706, -0.0249146, 0.0248945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A2_B2_A2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156508, upper bound: 0.0157093
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156508, upper bound: 0.0157093
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0092773, 0.0029902, -0.0105010, 0.0034343, -0.0127117, 0.0134913
1: -0.0067843, 0.0002644, -0.0072690, 0.0007963, -0.0075805, 0.0075334
2: 0.0274178, 0.0496367, 0.0267084, 0.0524987, -0.0250808, 0.0229283
3: -0.0038567, 0.0073478, -0.0040193, 0.0086749, -0.0125316, 0.0113671
4: -0.0106661, 0.0065363, -0.0120415, 0.0078084, -0.0184744, 0.0185779
5: 0.0046896, 0.0197275, 0.0036723, 0.0209707, -0.0162811, 0.0160552
6: -0.0243529, 0.0079204, -0.0281899, 0.0095960, -0.0339489, 0.0361103
7: 0.9583400, 0.9795051, 0.9553742, 0.9799203, -0.0215803, 0.0241309
8: -0.0265281, 0.0111273, -0.0288029, 0.0139653, -0.0404934, 0.0399303
9: -0.0104440, 0.0129519, -0.0120868, 0.0146167, -0.0250607, 0.0250387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A2_B2_A2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156508, upper bound: 0.0157093
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156508, upper bound: 0.0157093
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0093782, 0.0030268, -0.0104420, 0.0034129, -0.0127911, 0.0134689
1: -0.0068242, 0.0003082, -0.0072456, 0.0007707, -0.0075948, 0.0075539
2: 0.0273594, 0.0498725, 0.0267426, 0.0523607, -0.0250013, 0.0231299
3: -0.0038701, 0.0074571, -0.0040115, 0.0086109, -0.0124810, 0.0114686
4: -0.0107794, 0.0066411, -0.0119752, 0.0077471, -0.0185265, 0.0186163
5: 0.0046058, 0.0198299, 0.0037213, 0.0209108, -0.0163050, 0.0161086
6: -0.0246690, 0.0080585, -0.0280049, 0.0095152, -0.0341842, 0.0360634
7: 0.9580956, 0.9795393, 0.9555173, 0.9799004, -0.0218048, 0.0240220
8: -0.0267155, 0.0113612, -0.0286933, 0.0138285, -0.0405440, 0.0400544
9: -0.0105794, 0.0130891, -0.0120076, 0.0145364, -0.0251158, 0.0250967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A2_B2_A2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156508, upper bound: 0.0157206
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156508, upper bound: 0.0157206
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0093782, 0.0030268, -0.0105518, 0.0034528, -0.0128309, 0.0135787
1: -0.0068242, 0.0003082, -0.0072891, 0.0008184, -0.0076426, 0.0075974
2: 0.0273594, 0.0498725, 0.0266790, 0.0526175, -0.0252581, 0.0231935
3: -0.0038701, 0.0074571, -0.0040261, 0.0087300, -0.0126001, 0.0114832
4: -0.0107794, 0.0066411, -0.0120987, 0.0078612, -0.0186406, 0.0187398
5: 0.0046058, 0.0198299, 0.0036300, 0.0210224, -0.0164165, 0.0161999
6: -0.0246690, 0.0080585, -0.0283492, 0.0096656, -0.0343346, 0.0364077
7: 0.9580956, 0.9795393, 0.9552512, 0.9799376, -0.0218420, 0.0242882
8: -0.0267155, 0.0113612, -0.0288974, 0.0140832, -0.0407987, 0.0402586
9: -0.0105794, 0.0130891, -0.0121550, 0.0146858, -0.0252652, 0.0252441

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A2_B2_A2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156508, upper bound: 0.0157206
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156508, upper bound: 0.0157206
time: 0.81 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.17 seconds
IS_A1_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157243
IS_A1_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157243
IS_A1_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157243
IS_A1_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157243
IS_A1_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0155956, upper bound: 0.0158523
IS_A1_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0155956, upper bound: 0.0158523
IS_A1_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0158523
IS_A1_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0158523
IS_A1_B1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156848, upper bound: 0.0155431
IS_A1_B1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156848, upper bound: 0.0157243
IS_A1_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157243
IS_A1_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157243
IS_A1_B1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156848, upper bound: 0.0157137
IS_A1_B1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156848, upper bound: 0.0158523
IS_A1_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0158523
IS_A1_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0158523
IS_A1_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0155196
IS_A1_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0155394
IS_A1_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0155196
IS_A1_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0155394
IS_A1_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156749, upper bound: 0.0156508
IS_A1_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156749, upper bound: 0.0156508
IS_A1_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156749, upper bound: 0.0156508
IS_A1_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156749, upper bound: 0.0156508
IS_A1_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156846, upper bound: 0.0156701
IS_A1_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156846, upper bound: 0.0156745
IS_A1_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156846, upper bound: 0.0156701
IS_A1_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156846, upper bound: 0.0156745
IS_A1_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156846, upper bound: 0.0157206
IS_A1_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156846, upper bound: 0.0157206
IS_A1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156117, upper bound: 0.0157206
IS_A1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156117, upper bound: 0.0157206
IS_A2_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157939
IS_A2_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157939
IS_A2_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157939
IS_A2_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157939
IS_A2_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0159104
IS_A2_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0159104
IS_A2_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0159104
IS_A2_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0159104
IS_A2_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157939
IS_A2_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157939
IS_A2_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157939
IS_A2_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0157939
IS_A2_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0159104
IS_A2_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0159104
IS_A2_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0159104
IS_A2_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0159104
IS_A2_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0155954
IS_A2_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0156117
IS_A2_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0155954
IS_A2_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0156117
IS_A2_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0156846
IS_A2_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0157206, upper bound: 0.0156914
IS_A2_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156745, upper bound: 0.0156914
IS_A2_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156745, upper bound: 0.0156914
IS_A2_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156508, upper bound: 0.0157093
IS_A2_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156508, upper bound: 0.0157093
IS_A2_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156508, upper bound: 0.0157093
IS_A2_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156508, upper bound: 0.0157093
IS_A2_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156508, upper bound: 0.0157206
IS_A2_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156508, upper bound: 0.0157206
IS_A2_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156508, upper bound: 0.0157206
IS_A2_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 7, lower bound: -0.0156508, upper bound: 0.0157206

## BFS IS instance: IS_A1_B1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0090919, 0.0029230, -0.0090352, 0.0029024, -0.0119943, 0.0119581
1: -0.0067108, 0.0001838, -0.0066883, 0.0001591, -0.0068699, 0.0068721
2: 0.0275253, 0.0492030, 0.0275582, 0.0490703, -0.0215449, 0.0216447
3: -0.0038321, 0.0071467, -0.0038246, 0.0070851, -0.0109172, 0.0109712
4: -0.0104576, 0.0063436, -0.0103939, 0.0062846, -0.0167422, 0.0167374
5: 0.0048438, 0.0195391, 0.0048910, 0.0194814, -0.0146376, 0.0146481
6: -0.0237714, 0.0076665, -0.0235935, 0.0075888, -0.0313602, 0.0312600
7: 0.9587895, 0.9794422, 0.9589269, 0.9794229, -0.0206335, 0.0205154
8: -0.0261834, 0.0106973, -0.0260779, 0.0105657, -0.0367490, 0.0367751
9: -0.0101951, 0.0126997, -0.0101189, 0.0126225, -0.0228175, 0.0228186

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_B1_A1_A1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0158248, upper bound: 0.0156789
time: 0.70 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0158248, upper bound: 0.0158438
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0090919, 0.0029230, -0.0092084, 0.0029652, -0.0120571, 0.0121313
1: -0.0067108, 0.0001838, -0.0067569, 0.0002344, -0.0069452, 0.0069407
2: 0.0275253, 0.0492030, 0.0274578, 0.0494754, -0.0219501, 0.0217452
3: -0.0038321, 0.0071467, -0.0038476, 0.0072730, -0.0111051, 0.0109943
4: -0.0104576, 0.0063436, -0.0105886, 0.0064647, -0.0169223, 0.0169321
5: 0.0048438, 0.0195391, 0.0047469, 0.0196574, -0.0148136, 0.0147921
6: -0.0237714, 0.0076665, -0.0241367, 0.0078260, -0.0315974, 0.0318031
7: 0.9587895, 0.9794422, 0.9585070, 0.9794818, -0.0206923, 0.0209352
8: -0.0261834, 0.0106973, -0.0263999, 0.0109674, -0.0371508, 0.0370972
9: -0.0101951, 0.0126997, -0.0103515, 0.0128581, -0.0230532, 0.0230511

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_B1_A1_A1_B2_A1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0158248, upper bound: 0.0156789
time: 0.99 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B2_A2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0158248, upper bound: 0.0158438
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0103882, 0.0033934, -0.0090352, 0.0029024, -0.0132906, 0.0124285
1: -0.0072243, 0.0007472, -0.0066883, 0.0001591, -0.0073835, 0.0074355
2: 0.0267738, 0.0522347, 0.0275582, 0.0490703, -0.0222964, 0.0246765
3: -0.0040043, 0.0085525, -0.0038246, 0.0070851, -0.0110895, 0.0123770
4: -0.0119147, 0.0076911, -0.0103939, 0.0062846, -0.0181992, 0.0180849
5: 0.0037661, 0.0208561, 0.0048910, 0.0194814, -0.0157153, 0.0159651
6: -0.0278360, 0.0094415, -0.0235935, 0.0075888, -0.0354248, 0.0330350
7: 0.9556477, 0.9798821, 0.9589269, 0.9794229, -0.0237752, 0.0209552
8: -0.0285932, 0.0137036, -0.0260779, 0.0105657, -0.0391589, 0.0397815
9: -0.0119353, 0.0144632, -0.0101189, 0.0126225, -0.0245578, 0.0245821

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_B1_A1_A2_B1_B1

### Relational analysis result of IS_A1_B1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155308, upper bound: 0.0157243
time: 0.85 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2_B1_B2

### Relational analysis result of IS_A1_B1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155308, upper bound: 0.0157243
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0103882, 0.0033934, -0.0092084, 0.0029652, -0.0133534, 0.0126018
1: -0.0072243, 0.0007472, -0.0067569, 0.0002344, -0.0074587, 0.0075042
2: 0.0267738, 0.0522347, 0.0274578, 0.0494754, -0.0227016, 0.0247769
3: -0.0040043, 0.0085525, -0.0038476, 0.0072730, -0.0112774, 0.0124001
4: -0.0119147, 0.0076911, -0.0105886, 0.0064647, -0.0183793, 0.0182796
5: 0.0037661, 0.0208561, 0.0047469, 0.0196574, -0.0158913, 0.0161091
6: -0.0278360, 0.0094415, -0.0241367, 0.0078260, -0.0356621, 0.0335782
7: 0.9556477, 0.9798821, 0.9585070, 0.9794818, -0.0238341, 0.0213751
8: -0.0285932, 0.0137036, -0.0263999, 0.0109674, -0.0395606, 0.0401035
9: -0.0119353, 0.0144632, -0.0103515, 0.0128581, -0.0247934, 0.0248146

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_B1_A1_A2_B2_B1

### Relational analysis result of IS_A1_B1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155308, upper bound: 0.0157243
time: 0.88 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2_B2_B2

### Relational analysis result of IS_A1_B1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155308, upper bound: 0.0157243
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0092068, 0.0029647, -0.0090392, 0.0029038, -0.0121107, 0.0120039
1: -0.0067563, 0.0002338, -0.0066899, 0.0001609, -0.0069172, 0.0069237
2: 0.0274587, 0.0494718, 0.0275559, 0.0490798, -0.0216211, 0.0219159
3: -0.0038474, 0.0072713, -0.0038251, 0.0070896, -0.0109369, 0.0110964
4: -0.0105868, 0.0064630, -0.0103984, 0.0062888, -0.0168756, 0.0168614
5: 0.0047482, 0.0196558, 0.0048876, 0.0194855, -0.0147373, 0.0147682
6: -0.0241318, 0.0078239, -0.0236063, 0.0075944, -0.0317261, 0.0314301
7: 0.9585109, 0.9794813, 0.9589170, 0.9794243, -0.0209134, 0.0205643
8: -0.0263970, 0.0109638, -0.0260855, 0.0105751, -0.0369721, 0.0370493
9: -0.0103494, 0.0128560, -0.0101243, 0.0126280, -0.0229773, 0.0229803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_B1_A2_A1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157279, upper bound: 0.0158657
time: 0.69 seconds

## Relational analysis of IS_A1_B1_B1_A2_A1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157279, upper bound: 0.0160019
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0092068, 0.0029647, -0.0092232, 0.0029706, -0.0121774, 0.0121879
1: -0.0067563, 0.0002338, -0.0067628, 0.0002409, -0.0069972, 0.0069966
2: 0.0274587, 0.0494718, 0.0274492, 0.0495102, -0.0220515, 0.0220226
3: -0.0038474, 0.0072713, -0.0038495, 0.0072891, -0.0111365, 0.0111209
4: -0.0105868, 0.0064630, -0.0106053, 0.0064801, -0.0170669, 0.0170683
5: 0.0047482, 0.0196558, 0.0047346, 0.0196725, -0.0149243, 0.0149212
6: -0.0241318, 0.0078239, -0.0241832, 0.0078464, -0.0319781, 0.0320071
7: 0.9585109, 0.9794813, 0.9584711, 0.9794868, -0.0209759, 0.0210102
8: -0.0263970, 0.0109638, -0.0264275, 0.0110019, -0.0373989, 0.0373913
9: -0.0103494, 0.0128560, -0.0103714, 0.0128783, -0.0232277, 0.0232274

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_B1_A2_A1_B2_B1

### Relational analysis result of IS_A1_B1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156442, upper bound: 0.0160020
time: 0.91 seconds

## Relational analysis of IS_A1_B1_B1_A2_A1_B2_B2

### Relational analysis result of IS_A1_B1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156442, upper bound: 0.0160019
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0103679, 0.0033860, -0.0090804, 0.0029188, -0.0132867, 0.0124664
1: -0.0072163, 0.0007384, -0.0067062, 0.0001788, -0.0073951, 0.0074446
2: 0.0267856, 0.0521872, 0.0275320, 0.0491761, -0.0223905, 0.0246552
3: -0.0040016, 0.0085304, -0.0038306, 0.0071342, -0.0111359, 0.0123610
4: -0.0118918, 0.0076700, -0.0104447, 0.0063316, -0.0182234, 0.0181146
5: 0.0037830, 0.0208354, 0.0048534, 0.0195274, -0.0157444, 0.0159821
6: -0.0277723, 0.0094137, -0.0237353, 0.0076507, -0.0354230, 0.0331490
7: 0.9556970, 0.9798752, 0.9588172, 0.9794384, -0.0237414, 0.0210580
8: -0.0285554, 0.0136565, -0.0261620, 0.0106706, -0.0392260, 0.0398185
9: -0.0119080, 0.0144355, -0.0101796, 0.0126840, -0.0245920, 0.0246151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_B1_A2_A2_B1_B1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155246, upper bound: 0.0158523
time: 0.85 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B1_B2

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155246, upper bound: 0.0158523
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0103679, 0.0033860, -0.0092559, 0.0029825, -0.0133503, 0.0126419
1: -0.0072163, 0.0007384, -0.0067757, 0.0002551, -0.0074713, 0.0075142
2: 0.0267856, 0.0521872, 0.0274303, 0.0495865, -0.0228009, 0.0247569
3: -0.0040016, 0.0085304, -0.0038539, 0.0073245, -0.0113262, 0.0123843
4: -0.0118918, 0.0076700, -0.0106420, 0.0065140, -0.0184059, 0.0183119
5: 0.0037830, 0.0208354, 0.0047075, 0.0197057, -0.0159227, 0.0161280
6: -0.0277723, 0.0094137, -0.0242856, 0.0078910, -0.0356633, 0.0336993
7: 0.9556970, 0.9798752, 0.9583920, 0.9794978, -0.0238008, 0.0214832
8: -0.0285554, 0.0136565, -0.0264882, 0.0110776, -0.0396330, 0.0401447
9: -0.0119080, 0.0144355, -0.0104152, 0.0129227, -0.0248308, 0.0248507

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155246, upper bound: 0.0158523
time: 0.86 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B2

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155246, upper bound: 0.0158523
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0088496, 0.0028350, -0.0093626, 0.0030212, -0.0118708, 0.0121977
1: -0.0066148, 0.0000785, -0.0068180, 0.0003015, -0.0069163, 0.0068965
2: 0.0276658, 0.0486364, 0.0273684, 0.0498361, -0.0221703, 0.0212680
3: -0.0037999, 0.0068840, -0.0038681, 0.0074403, -0.0112402, 0.0107520
4: -0.0101853, 0.0060917, -0.0107619, 0.0066250, -0.0168103, 0.0168536
5: 0.0050452, 0.0192929, 0.0046187, 0.0198141, -0.0147689, 0.0146742
6: -0.0230117, 0.0073348, -0.0246202, 0.0080371, -0.0310489, 0.0319550
7: 0.9593765, 0.9793599, 0.9581333, 0.9795341, -0.0201576, 0.0212266
8: -0.0257330, 0.0101354, -0.0266866, 0.0113251, -0.0370581, 0.0368220
9: -0.0098698, 0.0123700, -0.0105585, 0.0130679, -0.0229378, 0.0229286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_B2_A1_A1_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0158248, upper bound: 0.0156317
time: 0.92 seconds

## Relational analysis of IS_A1_B1_B2_A1_A1_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0158248, upper bound: 0.0156317
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0090996, 0.0029257, -0.0093626, 0.0030212, -0.0121207, 0.0122884
1: -0.0067138, 0.0001871, -0.0068180, 0.0003015, -0.0070153, 0.0070052
2: 0.0275209, 0.0492209, 0.0273684, 0.0498361, -0.0223152, 0.0218525
3: -0.0038331, 0.0071550, -0.0038681, 0.0074403, -0.0112734, 0.0110231
4: -0.0104662, 0.0063515, -0.0107619, 0.0066250, -0.0170912, 0.0171134
5: 0.0048374, 0.0195468, 0.0046187, 0.0198141, -0.0149767, 0.0149281
6: -0.0237954, 0.0076770, -0.0246202, 0.0080371, -0.0318325, 0.0322972
7: 0.9587709, 0.9794449, 0.9581333, 0.9795341, -0.0207632, 0.0213115
8: -0.0261976, 0.0107150, -0.0266866, 0.0113251, -0.0375227, 0.0374016
9: -0.0102054, 0.0127100, -0.0105585, 0.0130679, -0.0232733, 0.0232686

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_B2_A1_A1_A2_B1

### Relational analysis result of IS_A1_B1_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0158248, upper bound: 0.0158438
time: 0.77 seconds

## Relational analysis of IS_A1_B1_B2_A1_A1_A2_B2

### Relational analysis result of IS_A1_B1_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0158248, upper bound: 0.0158438
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0104200, 0.0034049, -0.0092073, 0.0029648, -0.0133848, 0.0126122
1: -0.0072369, 0.0007611, -0.0067565, 0.0002340, -0.0074709, 0.0075176
2: 0.0267554, 0.0523092, 0.0274585, 0.0494729, -0.0227175, 0.0248507
3: -0.0040086, 0.0085870, -0.0038474, 0.0072718, -0.0112804, 0.0124344
4: -0.0119505, 0.0077241, -0.0105873, 0.0064635, -0.0184140, 0.0183114
5: 0.0037396, 0.0208884, 0.0047479, 0.0196563, -0.0159167, 0.0161405
6: -0.0279358, 0.0094851, -0.0241333, 0.0078245, -0.0357603, 0.0336184
7: 0.9555706, 0.9798929, 0.9585097, 0.9794814, -0.0239108, 0.0213832
8: -0.0286523, 0.0137774, -0.0263979, 0.0109649, -0.0396172, 0.0401753
9: -0.0119780, 0.0145064, -0.0103500, 0.0128566, -0.0248346, 0.0248564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_B2_A1_A2_B1_A1

### Relational analysis result of IS_A1_B1_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0155499
time: 0.91 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2_B1_A2

### Relational analysis result of IS_A1_B1_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157243
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0104200, 0.0034049, -0.0093546, 0.0030183, -0.0134383, 0.0127595
1: -0.0072369, 0.0007611, -0.0068149, 0.0002980, -0.0075349, 0.0075759
2: 0.0267554, 0.0523092, 0.0273731, 0.0498174, -0.0230620, 0.0249361
3: -0.0040086, 0.0085870, -0.0038670, 0.0074316, -0.0114401, 0.0124540
4: -0.0119505, 0.0077241, -0.0107529, 0.0066166, -0.0185671, 0.0184770
5: 0.0037396, 0.0208884, 0.0046254, 0.0198060, -0.0160663, 0.0162630
6: -0.0279358, 0.0094851, -0.0245951, 0.0080262, -0.0359620, 0.0340802
7: 0.9555706, 0.9798929, 0.9581528, 0.9795313, -0.0239606, 0.0217401
8: -0.0286523, 0.0137774, -0.0266717, 0.0113065, -0.0399588, 0.0404491
9: -0.0119780, 0.0145064, -0.0105477, 0.0130570, -0.0250350, 0.0250542

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_B2_A1_A2_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0155499
time: 0.91 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2_B2_A2

### Relational analysis result of IS_A1_B1_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157243
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0090405, 0.0029043, -0.0094121, 0.0030392, -0.0120797, 0.0123165
1: -0.0066904, 0.0001615, -0.0068376, 0.0003230, -0.0070134, 0.0069991
2: 0.0275551, 0.0490829, 0.0273397, 0.0499520, -0.0223968, 0.0217432
3: -0.0038253, 0.0070910, -0.0038746, 0.0074940, -0.0113192, 0.0109656
4: -0.0103999, 0.0062902, -0.0108176, 0.0066764, -0.0170763, 0.0171077
5: 0.0048865, 0.0194869, 0.0045776, 0.0198644, -0.0149779, 0.0149093
6: -0.0236104, 0.0075962, -0.0247755, 0.0081050, -0.0317154, 0.0323717
7: 0.9589138, 0.9794248, 0.9580132, 0.9795508, -0.0206370, 0.0214115
8: -0.0260879, 0.0105782, -0.0267787, 0.0114400, -0.0375278, 0.0373568
9: -0.0101261, 0.0126298, -0.0106250, 0.0131353, -0.0232614, 0.0232548

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_B2_A2_A1_A1_B1

### Relational analysis result of IS_A1_B1_B2_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0158249, upper bound: 0.0158186
time: 0.86 seconds

## Relational analysis of IS_A1_B1_B2_A2_A1_A1_B2

### Relational analysis result of IS_A1_B1_B2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0158249, upper bound: 0.0158186
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0092145, 0.0029675, -0.0094121, 0.0030392, -0.0122537, 0.0123796
1: -0.0067594, 0.0002371, -0.0068376, 0.0003230, -0.0070824, 0.0070748
2: 0.0274542, 0.0494898, 0.0273397, 0.0499520, -0.0224977, 0.0221501
3: -0.0038484, 0.0072797, -0.0038746, 0.0074940, -0.0113424, 0.0111543
4: -0.0105955, 0.0064710, -0.0108176, 0.0066764, -0.0172719, 0.0172886
5: 0.0047418, 0.0196637, 0.0045776, 0.0198644, -0.0151226, 0.0150861
6: -0.0241559, 0.0078344, -0.0247755, 0.0081050, -0.0322609, 0.0326099
7: 0.9584922, 0.9794839, 0.9580132, 0.9795508, -0.0210586, 0.0214707
8: -0.0264113, 0.0109817, -0.0267787, 0.0114400, -0.0378513, 0.0377604
9: -0.0103597, 0.0128665, -0.0106250, 0.0131353, -0.0234950, 0.0234915

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_B2_A2_A1_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0158249, upper bound: 0.0160019
time: 0.95 seconds

## Relational analysis of IS_A1_B1_B2_A2_A1_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0158249, upper bound: 0.0158186
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0103997, 0.0033975, -0.0092566, 0.0029827, -0.0133824, 0.0126541
1: -0.0072288, 0.0007522, -0.0067760, 0.0002554, -0.0074843, 0.0075283
2: 0.0267672, 0.0522616, 0.0274299, 0.0495882, -0.0228211, 0.0248317
3: -0.0040059, 0.0085649, -0.0038540, 0.0073253, -0.0113312, 0.0124189
4: -0.0119276, 0.0077030, -0.0106428, 0.0065148, -0.0184423, 0.0183458
5: 0.0037565, 0.0208677, 0.0047069, 0.0197064, -0.0159499, 0.0161609
6: -0.0278721, 0.0094572, -0.0242879, 0.0078920, -0.0357641, 0.0337451
7: 0.9556199, 0.9798860, 0.9583902, 0.9794980, -0.0238780, 0.0214958
8: -0.0286145, 0.0137302, -0.0264896, 0.0110793, -0.0396938, 0.0402198
9: -0.0119507, 0.0144788, -0.0104162, 0.0129237, -0.0248744, 0.0248950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_B2_A2_A2_B1_B1

### Relational analysis result of IS_A1_B1_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156739, upper bound: 0.0158523
time: 0.80 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2_B1_B2

### Relational analysis result of IS_A1_B1_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156739, upper bound: 0.0158523
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0103997, 0.0033975, -0.0094046, 0.0030364, -0.0134361, 0.0128021
1: -0.0072288, 0.0007522, -0.0068347, 0.0003197, -0.0075486, 0.0075869
2: 0.0267672, 0.0522616, 0.0273441, 0.0499343, -0.0231671, 0.0249175
3: -0.0040059, 0.0085649, -0.0038736, 0.0074858, -0.0114917, 0.0124386
4: -0.0119276, 0.0077030, -0.0108091, 0.0066686, -0.0185962, 0.0185121
5: 0.0037565, 0.0208677, 0.0045838, 0.0198568, -0.0161002, 0.0162839
6: -0.0278721, 0.0094572, -0.0247519, 0.0080947, -0.0359667, 0.0342091
7: 0.9556199, 0.9798860, 0.9580316, 0.9795483, -0.0239283, 0.0218544
8: -0.0286145, 0.0137302, -0.0267647, 0.0114225, -0.0400370, 0.0404949
9: -0.0119507, 0.0144788, -0.0106149, 0.0131251, -0.0250757, 0.0250936

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_B2_A2_A2_B2_B1

### Relational analysis result of IS_A1_B1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156739, upper bound: 0.0158523
time: 0.94 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2_B2_B2

### Relational analysis result of IS_A1_B1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156739, upper bound: 0.0158523
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0088496, 0.0028350, -0.0103616, 0.0033837, -0.0122334, 0.0131966
1: -0.0066148, 0.0000785, -0.0072138, 0.0007357, -0.0073505, 0.0072923
2: 0.0276658, 0.0486364, 0.0267893, 0.0521725, -0.0245067, 0.0218471
3: -0.0037999, 0.0068840, -0.0040008, 0.0085237, -0.0123236, 0.0108848
4: -0.0101853, 0.0060917, -0.0118848, 0.0076634, -0.0178487, 0.0179765
5: 0.0050452, 0.0192929, 0.0037882, 0.0208291, -0.0157839, 0.0155047
6: -0.0230117, 0.0073348, -0.0277527, 0.0094051, -0.0324168, 0.0350874
7: 0.9593766, 0.9793600, 0.9557121, 0.9798731, -0.0204965, 0.0236480
8: -0.0257330, 0.0101354, -0.0285437, 0.0136420, -0.0393750, 0.0386792
9: -0.0098698, 0.0123700, -0.0118996, 0.0144270, -0.0242968, 0.0242697

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156539, upper bound: 0.0155230
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156539, upper bound: 0.0155230
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0102285, 0.0033354, -0.0103616, 0.0033837, -0.0136122, 0.0136970
1: -0.0071610, 0.0006778, -0.0072138, 0.0007357, -0.0078967, 0.0078916
2: 0.0268664, 0.0518612, 0.0267893, 0.0521725, -0.0253061, 0.0250720
3: -0.0039831, 0.0083793, -0.0040008, 0.0085237, -0.0125068, 0.0123801
4: -0.0117352, 0.0075251, -0.0118848, 0.0076634, -0.0193986, 0.0194099
5: 0.0038989, 0.0206938, 0.0037882, 0.0208291, -0.0169302, 0.0169056
6: -0.0273353, 0.0092228, -0.0277527, 0.0094051, -0.0367404, 0.0369755
7: 0.9560348, 0.9798278, 0.9557121, 0.9798731, -0.0238382, 0.0241157
8: -0.0282963, 0.0133332, -0.0285437, 0.0136420, -0.0419383, 0.0418770
9: -0.0117209, 0.0142459, -0.0118996, 0.0144270, -0.0261478, 0.0261455

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156539, upper bound: 0.0155456
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156539, upper bound: 0.0155456
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0088496, 0.0028350, -0.0104669, 0.0034219, -0.0122716, 0.0133019
1: -0.0066148, 0.0000785, -0.0072555, 0.0007815, -0.0073963, 0.0073340
2: 0.0276658, 0.0486364, 0.0267282, 0.0524189, -0.0247531, 0.0219082
3: -0.0037999, 0.0068840, -0.0040148, 0.0086379, -0.0124378, 0.0108987
4: -0.0101853, 0.0060917, -0.0120032, 0.0077729, -0.0179582, 0.0180949
5: 0.0050452, 0.0192929, 0.0037006, 0.0209361, -0.0158909, 0.0155923
6: -0.0230117, 0.0073348, -0.0280829, 0.0095493, -0.0325610, 0.0354177
7: 0.9593766, 0.9793600, 0.9554570, 0.9799088, -0.0205322, 0.0239031
8: -0.0257330, 0.0101354, -0.0287395, 0.0138862, -0.0396192, 0.0388750
9: -0.0098698, 0.0123700, -0.0120410, 0.0145703, -0.0244401, 0.0244110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156881, upper bound: 0.0155196
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156881, upper bound: 0.0155196
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0102285, 0.0033354, -0.0104669, 0.0034219, -0.0136504, 0.0138023
1: -0.0071610, 0.0006778, -0.0072555, 0.0007815, -0.0079425, 0.0079333
2: 0.0268664, 0.0518612, 0.0267282, 0.0524189, -0.0255524, 0.0251330
3: -0.0039831, 0.0083793, -0.0040148, 0.0086379, -0.0126210, 0.0123941
4: -0.0117352, 0.0075251, -0.0120032, 0.0077729, -0.0195081, 0.0195283
5: 0.0038989, 0.0206938, 0.0037006, 0.0209361, -0.0170372, 0.0169932
6: -0.0273353, 0.0092228, -0.0280829, 0.0095493, -0.0368846, 0.0373057
7: 0.9560348, 0.9798278, 0.9554570, 0.9799088, -0.0238739, 0.0243708
8: -0.0282963, 0.0133332, -0.0287395, 0.0138862, -0.0421825, 0.0420728
9: -0.0117209, 0.0142459, -0.0120410, 0.0145703, -0.0262912, 0.0262869

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156881, upper bound: 0.0155394
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156881, upper bound: 0.0155394
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0090581, 0.0029107, -0.0103882, 0.0033934, -0.0124515, 0.0132989
1: -0.0066974, 0.0001691, -0.0072243, 0.0007472, -0.0074446, 0.0073934
2: 0.0275449, 0.0491240, 0.0267738, 0.0522347, -0.0246898, 0.0223501
3: -0.0038276, 0.0071101, -0.0040043, 0.0085525, -0.0123801, 0.0111144
4: -0.0104196, 0.0063084, -0.0119147, 0.0076911, -0.0181107, 0.0182231
5: 0.0048719, 0.0195047, 0.0037661, 0.0208561, -0.0159842, 0.0157387
6: -0.0236655, 0.0076202, -0.0278360, 0.0094415, -0.0331070, 0.0354563
7: 0.9588713, 0.9794308, 0.9556477, 0.9798821, -0.0210108, 0.0237831
8: -0.0261206, 0.0106189, -0.0285932, 0.0137036, -0.0398242, 0.0392121
9: -0.0101497, 0.0126537, -0.0119353, 0.0144632, -0.0246129, 0.0245890

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_A2_B1_B1_A1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156704, upper bound: 0.0156430
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B1_A2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156704, upper bound: 0.0156509
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0090581, 0.0029107, -0.0103679, 0.0033860, -0.0124441, 0.0132786
1: -0.0066974, 0.0001691, -0.0072163, 0.0007384, -0.0074358, 0.0073854
2: 0.0275449, 0.0491240, 0.0267856, 0.0521872, -0.0246423, 0.0223383
3: -0.0038276, 0.0071101, -0.0040016, 0.0085304, -0.0123581, 0.0111117
4: -0.0104196, 0.0063084, -0.0118918, 0.0076700, -0.0180896, 0.0182003
5: 0.0048719, 0.0195047, 0.0037830, 0.0208354, -0.0159635, 0.0157218
6: -0.0236655, 0.0076202, -0.0277723, 0.0094137, -0.0330792, 0.0353925
7: 0.9588713, 0.9794308, 0.9556970, 0.9798752, -0.0210039, 0.0237338
8: -0.0261206, 0.0106189, -0.0285554, 0.0136565, -0.0397770, 0.0391743
9: -0.0101497, 0.0126537, -0.0119080, 0.0144355, -0.0245852, 0.0245617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156704, upper bound: 0.0156430
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B2_A2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156704, upper bound: 0.0156509
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0090581, 0.0029107, -0.0104876, 0.0034295, -0.0124876, 0.0133983
1: -0.0066974, 0.0001691, -0.0072637, 0.0007905, -0.0074879, 0.0074328
2: 0.0275449, 0.0491240, 0.0267162, 0.0524673, -0.0249224, 0.0224078
3: -0.0038276, 0.0071101, -0.0040175, 0.0086603, -0.0124880, 0.0111276
4: -0.0104196, 0.0063084, -0.0120265, 0.0077945, -0.0182141, 0.0183349
5: 0.0048719, 0.0195047, 0.0036834, 0.0209571, -0.0160852, 0.0158214
6: -0.0236655, 0.0076202, -0.0281479, 0.0095777, -0.0332432, 0.0357681
7: 0.9588713, 0.9794308, 0.9554067, 0.9799159, -0.0210446, 0.0240241
8: -0.0261206, 0.0106189, -0.0287781, 0.0139343, -0.0400548, 0.0393970
9: -0.0101497, 0.0126537, -0.0120688, 0.0145985, -0.0247482, 0.0247225

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157086, upper bound: 0.0156428
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157086, upper bound: 0.0156508
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0090581, 0.0029107, -0.0104743, 0.0034246, -0.0124827, 0.0133850
1: -0.0066974, 0.0001691, -0.0072584, 0.0007847, -0.0074821, 0.0074275
2: 0.0275449, 0.0491240, 0.0267239, 0.0524361, -0.0248911, 0.0224000
3: -0.0038276, 0.0071101, -0.0040158, 0.0086459, -0.0124735, 0.0111258
4: -0.0104196, 0.0063084, -0.0120114, 0.0077806, -0.0182002, 0.0183199
5: 0.0048719, 0.0195047, 0.0036945, 0.0209435, -0.0160716, 0.0158102
6: -0.0236655, 0.0076202, -0.0281060, 0.0095594, -0.0332249, 0.0357262
7: 0.9588713, 0.9794308, 0.9554392, 0.9799113, -0.0210400, 0.0239916
8: -0.0261206, 0.0106189, -0.0287532, 0.0139033, -0.0400239, 0.0393721
9: -0.0101497, 0.0126537, -0.0120509, 0.0145803, -0.0247300, 0.0247045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157086, upper bound: 0.0156428
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_A2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157086, upper bound: 0.0156508
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0090996, 0.0029257, -0.0104752, 0.0034250, -0.0125245, 0.0134009
1: -0.0067138, 0.0001871, -0.0072588, 0.0007851, -0.0074989, 0.0074459
2: 0.0275209, 0.0492209, 0.0267234, 0.0524383, -0.0249174, 0.0224975
3: -0.0038331, 0.0071550, -0.0040159, 0.0086469, -0.0124800, 0.0111709
4: -0.0104662, 0.0063515, -0.0120125, 0.0077816, -0.0182478, 0.0183640
5: 0.0048374, 0.0195468, 0.0036937, 0.0209445, -0.0161071, 0.0158531
6: -0.0237954, 0.0076770, -0.0281090, 0.0095607, -0.0333560, 0.0357859
7: 0.9587709, 0.9794449, 0.9554368, 0.9799116, -0.0211406, 0.0240080
8: -0.0261976, 0.0107150, -0.0287550, 0.0139055, -0.0401031, 0.0394700
9: -0.0102054, 0.0127100, -0.0120521, 0.0145816, -0.0247869, 0.0247622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156430, upper bound: 0.0156701
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156430, upper bound: 0.0156701
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0103971, 0.0033966, -0.0104752, 0.0034250, -0.0138221, 0.0138718
1: -0.0072278, 0.0007511, -0.0072588, 0.0007851, -0.0080129, 0.0080099
2: 0.0267687, 0.0522556, 0.0267234, 0.0524383, -0.0256696, 0.0255322
3: -0.0040055, 0.0085622, -0.0040159, 0.0086469, -0.0126524, 0.0125781
4: -0.0119247, 0.0077004, -0.0120125, 0.0077816, -0.0197063, 0.0197129
5: 0.0037587, 0.0208651, 0.0036937, 0.0209445, -0.0171858, 0.0171714
6: -0.0278640, 0.0094537, -0.0281090, 0.0095607, -0.0374247, 0.0375627
7: 0.9556262, 0.9798850, 0.9554368, 0.9799116, -0.0242854, 0.0244482
8: -0.0286097, 0.0137243, -0.0287550, 0.0139055, -0.0425152, 0.0424793
9: -0.0119473, 0.0144753, -0.0120521, 0.0145816, -0.0265288, 0.0265274

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156430, upper bound: 0.0156745
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156430, upper bound: 0.0156745
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0090996, 0.0029257, -0.0104851, 0.0034285, -0.0125281, 0.0134108
1: -0.0067138, 0.0001871, -0.0072627, 0.0007894, -0.0075032, 0.0074498
2: 0.0275209, 0.0492209, 0.0267176, 0.0524614, -0.0249405, 0.0225033
3: -0.0038331, 0.0071550, -0.0040172, 0.0086576, -0.0124907, 0.0111722
4: -0.0104662, 0.0063515, -0.0120237, 0.0077919, -0.0182581, 0.0183752
5: 0.0048374, 0.0195468, 0.0036855, 0.0209546, -0.0161171, 0.0158614
6: -0.0237954, 0.0076770, -0.0281401, 0.0095742, -0.0333696, 0.0358170
7: 0.9587709, 0.9794449, 0.9554127, 0.9799150, -0.0211440, 0.0240321
8: -0.0261976, 0.0107150, -0.0287734, 0.0139284, -0.0401260, 0.0394884
9: -0.0102054, 0.0127100, -0.0120654, 0.0145950, -0.0248004, 0.0247755

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156468, upper bound: 0.0156701
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156468, upper bound: 0.0156701
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0103971, 0.0033966, -0.0104851, 0.0034285, -0.0138256, 0.0138817
1: -0.0072278, 0.0007511, -0.0072627, 0.0007894, -0.0080172, 0.0080138
2: 0.0267687, 0.0522556, 0.0267176, 0.0524614, -0.0256928, 0.0255379
3: -0.0040055, 0.0085622, -0.0040172, 0.0086576, -0.0126631, 0.0125794
4: -0.0119247, 0.0077004, -0.0120237, 0.0077919, -0.0197166, 0.0197240
5: 0.0037587, 0.0208651, 0.0036855, 0.0209546, -0.0171959, 0.0171796
6: -0.0278640, 0.0094537, -0.0281401, 0.0095742, -0.0374382, 0.0375938
7: 0.9556262, 0.9798850, 0.9554127, 0.9799150, -0.0242888, 0.0244723
8: -0.0286097, 0.0137243, -0.0287734, 0.0139284, -0.0425381, 0.0424977
9: -0.0119473, 0.0144753, -0.0120654, 0.0145950, -0.0265423, 0.0265407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156468, upper bound: 0.0156745
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156468, upper bound: 0.0156745
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0092145, 0.0029675, -0.0105230, 0.0034423, -0.0126568, 0.0134905
1: -0.0067594, 0.0002371, -0.0072777, 0.0008059, -0.0075652, 0.0075148
2: 0.0274542, 0.0494898, 0.0266957, 0.0525501, -0.0250958, 0.0227941
3: -0.0038484, 0.0072797, -0.0040222, 0.0086987, -0.0125471, 0.0113019
4: -0.0105955, 0.0064710, -0.0120662, 0.0078313, -0.0184267, 0.0185373
5: 0.0047418, 0.0196637, 0.0036540, 0.0209931, -0.0162512, 0.0160097
6: -0.0241559, 0.0078344, -0.0282588, 0.0096261, -0.0337821, 0.0360932
7: 0.9584922, 0.9794839, 0.9553210, 0.9799278, -0.0214357, 0.0241629
8: -0.0264113, 0.0109817, -0.0288438, 0.0140163, -0.0404277, 0.0398255
9: -0.0103597, 0.0128665, -0.0121163, 0.0146466, -0.0250063, 0.0249828

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155954, upper bound: 0.0157206
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155954, upper bound: 0.0157206
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0103766, 0.0033892, -0.0105230, 0.0034423, -0.0138189, 0.0139122
1: -0.0072197, 0.0007422, -0.0072777, 0.0008059, -0.0080256, 0.0080200
2: 0.0267805, 0.0522078, 0.0266957, 0.0525501, -0.0257696, 0.0255121
3: -0.0040028, 0.0085400, -0.0040222, 0.0086987, -0.0127015, 0.0125622
4: -0.0119017, 0.0076791, -0.0120662, 0.0078313, -0.0197330, 0.0197453
5: 0.0037757, 0.0208444, 0.0036540, 0.0209931, -0.0172174, 0.0171904
6: -0.0277999, 0.0094257, -0.0282588, 0.0096261, -0.0374260, 0.0376845
7: 0.9556757, 0.9798781, 0.9553210, 0.9799278, -0.0242521, 0.0245571
8: -0.0285717, 0.0136768, -0.0288438, 0.0140163, -0.0425880, 0.0425206
9: -0.0119198, 0.0144475, -0.0121163, 0.0146466, -0.0265664, 0.0265637

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155954, upper bound: 0.0157206
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155954, upper bound: 0.0157206
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0092317, 0.0029737, -0.0105055, 0.0034359, -0.0126676, 0.0134792
1: -0.0067662, 0.0002446, -0.0072708, 0.0007982, -0.0075644, 0.0075154
2: 0.0274443, 0.0495301, 0.0267058, 0.0525090, -0.0250647, 0.0228242
3: -0.0038507, 0.0072984, -0.0040199, 0.0086797, -0.0125304, 0.0113183
4: -0.0106148, 0.0064889, -0.0120465, 0.0078130, -0.0184278, 0.0185354
5: 0.0047275, 0.0196812, 0.0036685, 0.0209752, -0.0162477, 0.0160126
6: -0.0242099, 0.0078580, -0.0282038, 0.0096021, -0.0338120, 0.0360618
7: 0.9584504, 0.9794897, 0.9553635, 0.9799218, -0.0214714, 0.0241262
8: -0.0264433, 0.0110216, -0.0288112, 0.0139756, -0.0404190, 0.0398328
9: -0.0103828, 0.0128899, -0.0120928, 0.0146227, -0.0250055, 0.0249827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_A2_B2_B1_A1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155994, upper bound: 0.0157206
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B1_A2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155994, upper bound: 0.0157206
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0092317, 0.0029737, -0.0104913, 0.0034308, -0.0126625, 0.0134650
1: -0.0067662, 0.0002446, -0.0072651, 0.0007921, -0.0075582, 0.0075097
2: 0.0274443, 0.0495301, 0.0267141, 0.0524758, -0.0250315, 0.0228160
3: -0.0038507, 0.0072984, -0.0040180, 0.0086643, -0.0125149, 0.0113164
4: -0.0106148, 0.0064889, -0.0120306, 0.0077982, -0.0184131, 0.0185195
5: 0.0047275, 0.0196812, 0.0036804, 0.0209608, -0.0162333, 0.0160008
6: -0.0242099, 0.0078580, -0.0281593, 0.0095827, -0.0337926, 0.0360173
7: 0.9584504, 0.9794897, 0.9553979, 0.9799171, -0.0214667, 0.0240918
8: -0.0264433, 0.0110216, -0.0287848, 0.0139427, -0.0403860, 0.0398064
9: -0.0103828, 0.0128899, -0.0120737, 0.0146034, -0.0249862, 0.0249636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155994, upper bound: 0.0157206
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155994, upper bound: 0.0157206
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0092502, 0.0029804, -0.0090352, 0.0029024, -0.0121526, 0.0120156
1: -0.0067735, 0.0002526, -0.0066883, 0.0001591, -0.0069327, 0.0069409
2: 0.0274336, 0.0495732, 0.0275582, 0.0490703, -0.0216367, 0.0220150
3: -0.0038531, 0.0073184, -0.0038246, 0.0070851, -0.0109383, 0.0111429
4: -0.0106356, 0.0065081, -0.0103939, 0.0062846, -0.0169201, 0.0169020
5: 0.0047122, 0.0196999, 0.0048910, 0.0194814, -0.0147692, 0.0148089
6: -0.0242678, 0.0078833, -0.0235935, 0.0075888, -0.0318566, 0.0314768
7: 0.9584057, 0.9794959, 0.9589269, 0.9794229, -0.0210172, 0.0205690
8: -0.0264777, 0.0110644, -0.0260779, 0.0105657, -0.0370433, 0.0371423
9: -0.0104076, 0.0129150, -0.0101189, 0.0126225, -0.0230301, 0.0230339

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A2_B1_B1_A1_A1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0158184, upper bound: 0.0157822
time: 0.86 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0158184, upper bound: 0.0159308
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0092502, 0.0029804, -0.0092084, 0.0029652, -0.0122154, 0.0121888
1: -0.0067735, 0.0002526, -0.0067569, 0.0002344, -0.0070079, 0.0070095
2: 0.0274336, 0.0495732, 0.0274578, 0.0494754, -0.0220419, 0.0221154
3: -0.0038531, 0.0073184, -0.0038476, 0.0072730, -0.0111262, 0.0111660
4: -0.0106356, 0.0065081, -0.0105886, 0.0064647, -0.0171002, 0.0170967
5: 0.0047122, 0.0196999, 0.0047469, 0.0196574, -0.0149452, 0.0149530
6: -0.0242678, 0.0078833, -0.0241367, 0.0078260, -0.0320938, 0.0320199
7: 0.9584057, 0.9794959, 0.9585070, 0.9794818, -0.0210761, 0.0209889
8: -0.0264777, 0.0110644, -0.0263999, 0.0109674, -0.0374451, 0.0374643
9: -0.0104076, 0.0129150, -0.0103515, 0.0128581, -0.0232657, 0.0232665

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A2_B1_B1_A1_A1_B2_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0158184, upper bound: 0.0157822
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_B2_A2

### Relational analysis result of IS_A2_B1_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0158184, upper bound: 0.0159308
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0104876, 0.0034295, -0.0090352, 0.0029024, -0.0133900, 0.0124646
1: -0.0072637, 0.0007905, -0.0066883, 0.0001591, -0.0074228, 0.0074788
2: 0.0267162, 0.0524673, 0.0275582, 0.0490703, -0.0223541, 0.0249091
3: -0.0040175, 0.0086603, -0.0038246, 0.0070851, -0.0111027, 0.0124849
4: -0.0120265, 0.0077945, -0.0103939, 0.0062846, -0.0183110, 0.0181883
5: 0.0036834, 0.0209571, 0.0048910, 0.0194814, -0.0157980, 0.0160662
6: -0.0281479, 0.0095777, -0.0235935, 0.0075888, -0.0357367, 0.0331712
7: 0.9554067, 0.9799159, 0.9589269, 0.9794229, -0.0240163, 0.0209890
8: -0.0287781, 0.0139343, -0.0260779, 0.0105657, -0.0393437, 0.0400121
9: -0.0120688, 0.0145985, -0.0101189, 0.0126225, -0.0246913, 0.0247174

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A2_B1_B1_A1_A2_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0156697
time: 0.76 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_B1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157939
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0104876, 0.0034295, -0.0092084, 0.0029652, -0.0134529, 0.0126378
1: -0.0072637, 0.0007905, -0.0067569, 0.0002344, -0.0074981, 0.0075474
2: 0.0267162, 0.0524673, 0.0274578, 0.0494754, -0.0227593, 0.0250095
3: -0.0040175, 0.0086603, -0.0038476, 0.0072730, -0.0112906, 0.0125079
4: -0.0120265, 0.0077945, -0.0105886, 0.0064647, -0.0184911, 0.0183830
5: 0.0036834, 0.0209571, 0.0047469, 0.0196574, -0.0159740, 0.0162102
6: -0.0281479, 0.0095777, -0.0241367, 0.0078260, -0.0359739, 0.0337144
7: 0.9554067, 0.9799159, 0.9585070, 0.9794818, -0.0240752, 0.0214089
8: -0.0287781, 0.0139343, -0.0263999, 0.0109674, -0.0397455, 0.0403342
9: -0.0120688, 0.0145985, -0.0103515, 0.0128581, -0.0249269, 0.0249499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A2_B1_B1_A1_A2_B2_B1

### Relational analysis result of IS_A2_B1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155300, upper bound: 0.0157939
time: 0.95 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_B2_B2

### Relational analysis result of IS_A2_B1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155300, upper bound: 0.0157939
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0093525, 0.0030175, -0.0090804, 0.0029188, -0.0122712, 0.0120979
1: -0.0068140, 0.0002971, -0.0067062, 0.0001788, -0.0069928, 0.0070033
2: 0.0273743, 0.0498124, 0.0275320, 0.0491761, -0.0218018, 0.0222804
3: -0.0038667, 0.0074293, -0.0038306, 0.0071342, -0.0110009, 0.0112598
4: -0.0107505, 0.0066144, -0.0104447, 0.0063316, -0.0170821, 0.0170591
5: 0.0046272, 0.0198038, 0.0048534, 0.0195274, -0.0149002, 0.0149504
6: -0.0245884, 0.0080233, -0.0237353, 0.0076507, -0.0322391, 0.0317586
7: 0.9581579, 0.9795305, 0.9588172, 0.9794384, -0.0212805, 0.0207133
8: -0.0266677, 0.0113015, -0.0261620, 0.0106706, -0.0373383, 0.0374635
9: -0.0105449, 0.0130541, -0.0101796, 0.0126840, -0.0232289, 0.0232338

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A2_B1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155225, upper bound: 0.0159104
time: 0.81 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155225, upper bound: 0.0159104
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0104743, 0.0034246, -0.0090804, 0.0029188, -0.0133931, 0.0125050
1: -0.0072584, 0.0007847, -0.0067062, 0.0001788, -0.0074372, 0.0074909
2: 0.0267239, 0.0524361, 0.0275320, 0.0491761, -0.0224522, 0.0249040
3: -0.0040158, 0.0086459, -0.0038306, 0.0071342, -0.0111500, 0.0124764
4: -0.0120114, 0.0077806, -0.0104447, 0.0063316, -0.0183430, 0.0182253
5: 0.0036945, 0.0209435, 0.0048534, 0.0195274, -0.0158329, 0.0160902
6: -0.0281060, 0.0095594, -0.0237353, 0.0076507, -0.0357567, 0.0332947
7: 0.9554392, 0.9799113, 0.9588172, 0.9794384, -0.0239992, 0.0210941
8: -0.0287532, 0.0139033, -0.0261620, 0.0106706, -0.0394238, 0.0400653
9: -0.0120509, 0.0145803, -0.0101796, 0.0126840, -0.0247348, 0.0247599

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A2_B1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155225, upper bound: 0.0159104
time: 0.82 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155225, upper bound: 0.0159104
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0093525, 0.0030175, -0.0092559, 0.0029825, -0.0123349, 0.0122734
1: -0.0068140, 0.0002971, -0.0067757, 0.0002551, -0.0070691, 0.0070728
2: 0.0273743, 0.0498124, 0.0274303, 0.0495865, -0.0222122, 0.0223821
3: -0.0038667, 0.0074293, -0.0038539, 0.0073245, -0.0111912, 0.0112831
4: -0.0107505, 0.0066144, -0.0106420, 0.0065140, -0.0172645, 0.0172563
5: 0.0046272, 0.0198038, 0.0047075, 0.0197057, -0.0150785, 0.0150963
6: -0.0245884, 0.0080233, -0.0242856, 0.0078910, -0.0324794, 0.0323089
7: 0.9581579, 0.9795305, 0.9583920, 0.9794978, -0.0213400, 0.0211385
8: -0.0266677, 0.0113015, -0.0264882, 0.0110776, -0.0377453, 0.0377897
9: -0.0105449, 0.0130541, -0.0104152, 0.0129227, -0.0234676, 0.0234694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155225, upper bound: 0.0159104
time: 0.82 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155225, upper bound: 0.0159104
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0104743, 0.0034246, -0.0092559, 0.0029825, -0.0134568, 0.0126805
1: -0.0072584, 0.0007847, -0.0067757, 0.0002551, -0.0075135, 0.0075604
2: 0.0267239, 0.0524361, 0.0274303, 0.0495865, -0.0228626, 0.0250058
3: -0.0040158, 0.0086459, -0.0038539, 0.0073245, -0.0113403, 0.0124998
4: -0.0120114, 0.0077806, -0.0106420, 0.0065140, -0.0185255, 0.0184225
5: 0.0036945, 0.0209435, 0.0047075, 0.0197057, -0.0160112, 0.0162361
6: -0.0281060, 0.0095594, -0.0242856, 0.0078910, -0.0359970, 0.0338450
7: 0.9554392, 0.9799113, 0.9583920, 0.9794978, -0.0240587, 0.0215193
8: -0.0287532, 0.0139033, -0.0264882, 0.0110776, -0.0398308, 0.0403915
9: -0.0120509, 0.0145803, -0.0104152, 0.0129227, -0.0249736, 0.0249955

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155225, upper bound: 0.0159104
time: 0.82 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155225, upper bound: 0.0159104
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0092829, 0.0029922, -0.0092073, 0.0029648, -0.0122477, 0.0121995
1: -0.0067864, 0.0002668, -0.0067565, 0.0002340, -0.0070204, 0.0070233
2: 0.0274146, 0.0496496, 0.0274585, 0.0494729, -0.0220582, 0.0221912
3: -0.0038575, 0.0073538, -0.0038474, 0.0072718, -0.0111293, 0.0112012
4: -0.0106723, 0.0065421, -0.0105873, 0.0064635, -0.0171358, 0.0171294
5: 0.0046850, 0.0197331, 0.0047479, 0.0196563, -0.0149713, 0.0149852
6: -0.0243702, 0.0079280, -0.0241333, 0.0078245, -0.0321947, 0.0320613
7: 0.9583266, 0.9795070, 0.9585097, 0.9794814, -0.0211548, 0.0209973
8: -0.0265384, 0.0111402, -0.0263979, 0.0109649, -0.0375033, 0.0375381
9: -0.0104515, 0.0129594, -0.0103500, 0.0128566, -0.0233081, 0.0233094

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0156407
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157939
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0105211, 0.0034416, -0.0092073, 0.0029648, -0.0134859, 0.0126489
1: -0.0072770, 0.0008050, -0.0067565, 0.0002340, -0.0075109, 0.0075615
2: 0.0266968, 0.0525456, 0.0274585, 0.0494729, -0.0227761, 0.0250871
3: -0.0040220, 0.0086966, -0.0038474, 0.0072718, -0.0112938, 0.0125441
4: -0.0120641, 0.0078293, -0.0105873, 0.0064635, -0.0185276, 0.0184166
5: 0.0036556, 0.0209911, 0.0047479, 0.0196563, -0.0160007, 0.0162432
6: -0.0282529, 0.0096235, -0.0241333, 0.0078245, -0.0360774, 0.0337568
7: 0.9553257, 0.9799272, 0.9585097, 0.9794814, -0.0241557, 0.0214175
8: -0.0288403, 0.0140119, -0.0263979, 0.0109649, -0.0398052, 0.0404098
9: -0.0121137, 0.0146440, -0.0103500, 0.0128566, -0.0249703, 0.0249940

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0156411
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157939
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0092829, 0.0029922, -0.0093546, 0.0030183, -0.0123012, 0.0123468
1: -0.0067864, 0.0002668, -0.0068149, 0.0002980, -0.0070844, 0.0070817
2: 0.0274146, 0.0496496, 0.0273731, 0.0498174, -0.0224028, 0.0222766
3: -0.0038575, 0.0073538, -0.0038670, 0.0074316, -0.0112890, 0.0112208
4: -0.0106723, 0.0065421, -0.0107529, 0.0066166, -0.0172889, 0.0172950
5: 0.0046850, 0.0197331, 0.0046254, 0.0198060, -0.0151209, 0.0151077
6: -0.0243702, 0.0079280, -0.0245951, 0.0080262, -0.0323964, 0.0325231
7: 0.9583266, 0.9795070, 0.9581528, 0.9795313, -0.0212047, 0.0213542
8: -0.0265384, 0.0111402, -0.0266717, 0.0113065, -0.0378449, 0.0378119
9: -0.0104515, 0.0129594, -0.0105477, 0.0130570, -0.0235085, 0.0235072

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A2_B1_B2_A1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0156407
time: 0.84 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157939
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0105211, 0.0034416, -0.0093546, 0.0030183, -0.0135394, 0.0127962
1: -0.0072770, 0.0008050, -0.0068149, 0.0002980, -0.0075750, 0.0076199
2: 0.0266968, 0.0525456, 0.0273731, 0.0498174, -0.0231206, 0.0251725
3: -0.0040220, 0.0086966, -0.0038670, 0.0074316, -0.0114536, 0.0125636
4: -0.0120641, 0.0078293, -0.0107529, 0.0066166, -0.0186807, 0.0185822
5: 0.0036556, 0.0209911, 0.0046254, 0.0198060, -0.0161504, 0.0163657
6: -0.0282529, 0.0096235, -0.0245951, 0.0080262, -0.0362791, 0.0342186
7: 0.9553257, 0.9799272, 0.9581528, 0.9795313, -0.0242056, 0.0217744
8: -0.0288403, 0.0140119, -0.0266717, 0.0113065, -0.0401468, 0.0406836
9: -0.0121137, 0.0146440, -0.0105477, 0.0130570, -0.0251707, 0.0251917

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A2_B1_B2_A1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0156411
time: 0.88 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157939
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0093853, 0.0030294, -0.0092566, 0.0029827, -0.0123680, 0.0122860
1: -0.0068270, 0.0003113, -0.0067760, 0.0002554, -0.0070824, 0.0070874
2: 0.0273552, 0.0498892, 0.0274299, 0.0495882, -0.0222330, 0.0224594
3: -0.0038711, 0.0074649, -0.0038540, 0.0073253, -0.0111964, 0.0113189
4: -0.0107874, 0.0066486, -0.0106428, 0.0065148, -0.0173022, 0.0172913
5: 0.0045999, 0.0198372, 0.0047069, 0.0197064, -0.0151065, 0.0151303
6: -0.0246915, 0.0080683, -0.0242879, 0.0078920, -0.0325835, 0.0323562
7: 0.9580783, 0.9795417, 0.9583902, 0.9794980, -0.0214197, 0.0211515
8: -0.0267288, 0.0113778, -0.0264896, 0.0110793, -0.0378081, 0.0378673
9: -0.0105890, 0.0130988, -0.0104162, 0.0129237, -0.0235127, 0.0235150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157744
time: 0.86 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0159104
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0105080, 0.0034369, -0.0092566, 0.0029827, -0.0134907, 0.0126935
1: -0.0072718, 0.0007993, -0.0067760, 0.0002554, -0.0075272, 0.0075754
2: 0.0267044, 0.0525149, 0.0274299, 0.0495882, -0.0228839, 0.0250851
3: -0.0040202, 0.0086824, -0.0038540, 0.0073253, -0.0113456, 0.0125364
4: -0.0120493, 0.0078156, -0.0106428, 0.0065148, -0.0185641, 0.0184584
5: 0.0036665, 0.0209778, 0.0047069, 0.0197064, -0.0160400, 0.0162709
6: -0.0282118, 0.0096056, -0.0242879, 0.0078920, -0.0361038, 0.0338935
7: 0.9553574, 0.9799227, 0.9583902, 0.9794980, -0.0241405, 0.0215325
8: -0.0288159, 0.0139815, -0.0264896, 0.0110793, -0.0398951, 0.0404711
9: -0.0120961, 0.0146261, -0.0104162, 0.0129237, -0.0250199, 0.0250423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A2_B1_B2_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157744
time: 0.88 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0159104
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0093853, 0.0030294, -0.0094046, 0.0030364, -0.0124217, 0.0124340
1: -0.0068270, 0.0003113, -0.0068347, 0.0003197, -0.0071467, 0.0071460
2: 0.0273552, 0.0498892, 0.0273441, 0.0499343, -0.0225791, 0.0225452
3: -0.0038711, 0.0074649, -0.0038736, 0.0074858, -0.0113569, 0.0113385
4: -0.0107874, 0.0066486, -0.0108091, 0.0066686, -0.0174560, 0.0174577
5: 0.0045999, 0.0198372, 0.0045838, 0.0198568, -0.0152569, 0.0152533
6: -0.0246915, 0.0080683, -0.0247519, 0.0080947, -0.0327861, 0.0328202
7: 0.9580783, 0.9795417, 0.9580316, 0.9795483, -0.0214700, 0.0215101
8: -0.0267288, 0.0113778, -0.0267647, 0.0114225, -0.0381513, 0.0381425
9: -0.0105890, 0.0130988, -0.0106149, 0.0131251, -0.0237140, 0.0237137

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A2_B1_B2_A2_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157744
time: 0.87 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0159104
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0105080, 0.0034369, -0.0094046, 0.0030364, -0.0135444, 0.0128415
1: -0.0072718, 0.0007993, -0.0068347, 0.0003197, -0.0075915, 0.0076340
2: 0.0267044, 0.0525149, 0.0273441, 0.0499343, -0.0232300, 0.0251708
3: -0.0040202, 0.0086824, -0.0038736, 0.0074858, -0.0115060, 0.0125561
4: -0.0120493, 0.0078156, -0.0108091, 0.0066686, -0.0187180, 0.0186247
5: 0.0036665, 0.0209778, 0.0045838, 0.0198568, -0.0161903, 0.0163940
6: -0.0282118, 0.0096056, -0.0247519, 0.0080947, -0.0363064, 0.0343575
7: 0.9553574, 0.9799227, 0.9580316, 0.9795483, -0.0241908, 0.0218911
8: -0.0288159, 0.0139815, -0.0267647, 0.0114225, -0.0402384, 0.0407462
9: -0.0120961, 0.0146261, -0.0106149, 0.0131251, -0.0252212, 0.0252410

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A2_B1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156739, upper bound: 0.0159104
time: 0.87 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156739, upper bound: 0.0159104
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0090306, 0.0029007, -0.0103616, 0.0033837, -0.0124143, 0.0132623
1: -0.0066865, 0.0001572, -0.0072138, 0.0007357, -0.0074222, 0.0073709
2: 0.0275609, 0.0490596, 0.0267893, 0.0521725, -0.0246117, 0.0222704
3: -0.0038240, 0.0070802, -0.0040008, 0.0085237, -0.0123476, 0.0110810
4: -0.0103887, 0.0062798, -0.0118848, 0.0076634, -0.0180522, 0.0181646
5: 0.0048948, 0.0194768, 0.0037882, 0.0208291, -0.0159343, 0.0156886
6: -0.0235792, 0.0075826, -0.0277527, 0.0094051, -0.0329843, 0.0353353
7: 0.9589379, 0.9794214, 0.9557121, 0.9798731, -0.0209351, 0.0237094
8: -0.0260694, 0.0105551, -0.0285437, 0.0136420, -0.0397114, 0.0390989
9: -0.0101128, 0.0126163, -0.0118996, 0.0144270, -0.0245398, 0.0245159

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A2_B2_A1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156539, upper bound: 0.0155954
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156539, upper bound: 0.0155954
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0103632, 0.0033843, -0.0103616, 0.0033837, -0.0137469, 0.0137459
1: -0.0072144, 0.0007364, -0.0072138, 0.0007357, -0.0079501, 0.0079501
2: 0.0267883, 0.0521762, 0.0267893, 0.0521725, -0.0253842, 0.0253869
3: -0.0040010, 0.0085254, -0.0040008, 0.0085237, -0.0125247, 0.0125262
4: -0.0118866, 0.0076651, -0.0118848, 0.0076634, -0.0195500, 0.0195499
5: 0.0037869, 0.0208307, 0.0037882, 0.0208291, -0.0170422, 0.0170425
6: -0.0277577, 0.0094073, -0.0277527, 0.0094051, -0.0371628, 0.0371599
7: 0.9557084, 0.9798737, 0.9557121, 0.9798731, -0.0241647, 0.0241616
8: -0.0285466, 0.0136456, -0.0285437, 0.0136420, -0.0421886, 0.0421893
9: -0.0119017, 0.0144291, -0.0118996, 0.0144270, -0.0263287, 0.0263287

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A2_B2_A1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156539, upper bound: 0.0156117
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156539, upper bound: 0.0156117
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0090306, 0.0029007, -0.0104669, 0.0034219, -0.0124525, 0.0133676
1: -0.0066865, 0.0001572, -0.0072555, 0.0007815, -0.0074680, 0.0074127
2: 0.0275609, 0.0490596, 0.0267282, 0.0524189, -0.0248580, 0.0223314
3: -0.0038240, 0.0070802, -0.0040148, 0.0086379, -0.0124618, 0.0110950
4: -0.0103887, 0.0062798, -0.0120032, 0.0077729, -0.0181617, 0.0182830
5: 0.0048948, 0.0194768, 0.0037006, 0.0209361, -0.0160413, 0.0157762
6: -0.0235792, 0.0075826, -0.0280829, 0.0095493, -0.0331285, 0.0356655
7: 0.9589379, 0.9794214, 0.9554570, 0.9799088, -0.0209708, 0.0239645
8: -0.0260694, 0.0105551, -0.0287395, 0.0138862, -0.0399556, 0.0392946
9: -0.0101128, 0.0126163, -0.0120410, 0.0145703, -0.0246831, 0.0246572

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A2_B2_A1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156539, upper bound: 0.0155954
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156539, upper bound: 0.0155954
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0103632, 0.0033843, -0.0104669, 0.0034219, -0.0137851, 0.0138512
1: -0.0072144, 0.0007364, -0.0072555, 0.0007815, -0.0079959, 0.0079919
2: 0.0267883, 0.0521762, 0.0267282, 0.0524189, -0.0256305, 0.0254480
3: -0.0040010, 0.0085254, -0.0040148, 0.0086379, -0.0126389, 0.0125402
4: -0.0118866, 0.0076651, -0.0120032, 0.0077729, -0.0196595, 0.0196683
5: 0.0037869, 0.0208307, 0.0037006, 0.0209361, -0.0171492, 0.0171301
6: -0.0277577, 0.0094073, -0.0280829, 0.0095493, -0.0373070, 0.0374902
7: 0.9557084, 0.9798737, 0.9554570, 0.9799088, -0.0242004, 0.0244167
8: -0.0285466, 0.0136456, -0.0287395, 0.0138862, -0.0424328, 0.0423851
9: -0.0119017, 0.0144291, -0.0120410, 0.0145703, -0.0264720, 0.0264701

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A2_B2_A1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156539, upper bound: 0.0156117
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156539, upper bound: 0.0156117
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0092148, 0.0029676, -0.0104099, 0.0034013, -0.0126160, 0.0133775
1: -0.0067595, 0.0002372, -0.0072329, 0.0007567, -0.0075162, 0.0074701
2: 0.0274541, 0.0494904, 0.0267612, 0.0522856, -0.0248315, 0.0227292
3: -0.0038484, 0.0072799, -0.0040072, 0.0085761, -0.0124245, 0.0112872
4: -0.0105957, 0.0064713, -0.0119391, 0.0077137, -0.0183094, 0.0184104
5: 0.0047416, 0.0196639, 0.0037480, 0.0208782, -0.0161365, 0.0159159
6: -0.0241567, 0.0078347, -0.0279043, 0.0094713, -0.0336280, 0.0357390
7: 0.9584916, 0.9794839, 0.9555951, 0.9798895, -0.0213979, 0.0238888
8: -0.0264118, 0.0109823, -0.0286336, 0.0137541, -0.0401659, 0.0396159
9: -0.0103600, 0.0128668, -0.0119645, 0.0144927, -0.0248528, 0.0248313

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A2_B2_A1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156701, upper bound: 0.0156846
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156701, upper bound: 0.0156846
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0104731, 0.0034242, -0.0104099, 0.0034013, -0.0138744, 0.0138341
1: -0.0072579, 0.0007842, -0.0072329, 0.0007567, -0.0080146, 0.0080171
2: 0.0267246, 0.0524333, 0.0267612, 0.0522856, -0.0255610, 0.0256721
3: -0.0040156, 0.0086446, -0.0040072, 0.0085761, -0.0125917, 0.0126518
4: -0.0120101, 0.0077793, -0.0119391, 0.0077137, -0.0197238, 0.0197185
5: 0.0036955, 0.0209423, 0.0037480, 0.0208782, -0.0171827, 0.0171943
6: -0.0281023, 0.0095578, -0.0279043, 0.0094713, -0.0375736, 0.0374620
7: 0.9554421, 0.9799109, 0.9555951, 0.9798895, -0.0244474, 0.0243158
8: -0.0287510, 0.0139006, -0.0286336, 0.0137541, -0.0425051, 0.0425342
9: -0.0120493, 0.0145787, -0.0119645, 0.0144927, -0.0265420, 0.0265431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156701, upper bound: 0.0156914
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156701, upper bound: 0.0156914
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0092323, 0.0029739, -0.0104876, 0.0034295, -0.0126618, 0.0134616
1: -0.0067664, 0.0002448, -0.0072637, 0.0007905, -0.0075569, 0.0075085
2: 0.0274439, 0.0495314, 0.0267162, 0.0524673, -0.0250234, 0.0228152
3: -0.0038508, 0.0072990, -0.0040175, 0.0086603, -0.0125111, 0.0113165
4: -0.0106154, 0.0064895, -0.0120265, 0.0077945, -0.0184099, 0.0185160
5: 0.0047271, 0.0196817, 0.0036834, 0.0209571, -0.0162301, 0.0159984
6: -0.0242117, 0.0078588, -0.0281479, 0.0095777, -0.0337894, 0.0360066
7: 0.9584491, 0.9794899, 0.9554067, 0.9799159, -0.0214668, 0.0240832
8: -0.0264444, 0.0110229, -0.0287781, 0.0139343, -0.0403787, 0.0398010
9: -0.0103836, 0.0128907, -0.0120688, 0.0145985, -0.0249821, 0.0249595

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156701, upper bound: 0.0156846
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156701, upper bound: 0.0156914
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0092323, 0.0029739, -0.0104743, 0.0034246, -0.0126569, 0.0134482
1: -0.0067664, 0.0002448, -0.0072584, 0.0007847, -0.0075511, 0.0075032
2: 0.0274439, 0.0495314, 0.0267239, 0.0524361, -0.0249921, 0.0228074
3: -0.0038508, 0.0072990, -0.0040158, 0.0086459, -0.0124966, 0.0113147
4: -0.0106154, 0.0064895, -0.0120114, 0.0077806, -0.0183960, 0.0185010
5: 0.0047271, 0.0196817, 0.0036945, 0.0209435, -0.0162165, 0.0159872
6: -0.0242117, 0.0078588, -0.0281060, 0.0095594, -0.0337711, 0.0359647
7: 0.9584491, 0.9794899, 0.9554392, 0.9799113, -0.0214622, 0.0240507
8: -0.0264444, 0.0110229, -0.0287532, 0.0139033, -0.0403477, 0.0397762
9: -0.0103836, 0.0128907, -0.0120509, 0.0145803, -0.0249639, 0.0249415

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156701, upper bound: 0.0156846
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156701, upper bound: 0.0156914
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0092773, 0.0029902, -0.0103289, 0.0033718, -0.0126492, 0.0133191
1: -0.0067843, 0.0002644, -0.0072008, 0.0007215, -0.0075057, 0.0074652
2: 0.0274178, 0.0496367, 0.0268082, 0.0520960, -0.0246782, 0.0228285
3: -0.0038567, 0.0073478, -0.0039964, 0.0084882, -0.0123449, 0.0113443
4: -0.0106661, 0.0065363, -0.0118480, 0.0076294, -0.0182955, 0.0183844
5: 0.0046896, 0.0197275, 0.0038154, 0.0207958, -0.0161062, 0.0159121
6: -0.0243529, 0.0079204, -0.0276501, 0.0093603, -0.0337132, 0.0355705
7: 0.9583400, 0.9795051, 0.9557915, 0.9798620, -0.0215220, 0.0237136
8: -0.0265281, 0.0111273, -0.0284829, 0.0135661, -0.0400941, 0.0396103
9: -0.0104440, 0.0129519, -0.0118557, 0.0143825, -0.0248265, 0.0248076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_A1_B1_B1_A1

### Relational analysis result of IS_A2_B2_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156428, upper bound: 0.0157086
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_A1_B1_B1_A2

### Relational analysis result of IS_A2_B2_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156428, upper bound: 0.0157093
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0092773, 0.0029902, -0.0103703, 0.0033869, -0.0126642, 0.0133605
1: -0.0067843, 0.0002644, -0.0072172, 0.0007395, -0.0075237, 0.0074816
2: 0.0274178, 0.0496367, 0.0267842, 0.0521929, -0.0247750, 0.0228525
3: -0.0038567, 0.0073478, -0.0040019, 0.0085331, -0.0123898, 0.0113498
4: -0.0106661, 0.0065363, -0.0118946, 0.0076725, -0.0183385, 0.0184309
5: 0.0046896, 0.0197275, 0.0037810, 0.0208379, -0.0161483, 0.0159465
6: -0.0243529, 0.0079204, -0.0277800, 0.0094170, -0.0337699, 0.0357004
7: 0.9583400, 0.9795051, 0.9556912, 0.9798760, -0.0215360, 0.0238139
8: -0.0265281, 0.0111273, -0.0285599, 0.0136621, -0.0401902, 0.0396872
9: -0.0104440, 0.0129519, -0.0119113, 0.0144388, -0.0248828, 0.0248632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_A1_B1_B2_A1

### Relational analysis result of IS_A2_B2_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156428, upper bound: 0.0157086
time: 0.96 seconds

## Relational analysis of IS_A2_B2_A2_A1_B1_B2_A2

### Relational analysis result of IS_A2_B2_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156428, upper bound: 0.0157093
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0092773, 0.0029902, -0.0104667, 0.0034219, -0.0126992, 0.0134570
1: -0.0067843, 0.0002644, -0.0072554, 0.0007814, -0.0075656, 0.0075198
2: 0.0274178, 0.0496367, 0.0267283, 0.0524185, -0.0250006, 0.0229084
3: -0.0038567, 0.0073478, -0.0040148, 0.0086377, -0.0124944, 0.0113626
4: -0.0106661, 0.0065363, -0.0120030, 0.0077727, -0.0184388, 0.0185393
5: 0.0046896, 0.0197275, 0.0037008, 0.0209359, -0.0162463, 0.0160267
6: -0.0243529, 0.0079204, -0.0280824, 0.0095490, -0.0339019, 0.0360028
7: 0.9583400, 0.9795051, 0.9554574, 0.9799086, -0.0215687, 0.0240477
8: -0.0265281, 0.0111273, -0.0287392, 0.0138858, -0.0404139, 0.0398665
9: -0.0104440, 0.0129519, -0.0120408, 0.0145700, -0.0250140, 0.0249927

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_A1_B2_B1_A1

### Relational analysis result of IS_A2_B2_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156428, upper bound: 0.0157086
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2_B1_A2

### Relational analysis result of IS_A2_B2_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156428, upper bound: 0.0157093
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0092773, 0.0029902, -0.0104772, 0.0034257, -0.0127030, 0.0134675
1: -0.0067843, 0.0002644, -0.0072596, 0.0007860, -0.0075702, 0.0075240
2: 0.0274178, 0.0496367, 0.0267222, 0.0524431, -0.0250252, 0.0229145
3: -0.0038567, 0.0073478, -0.0040162, 0.0086491, -0.0125058, 0.0113640
4: -0.0106661, 0.0065363, -0.0120148, 0.0077837, -0.0184497, 0.0185511
5: 0.0046896, 0.0197275, 0.0036920, 0.0209466, -0.0162570, 0.0160355
6: -0.0243529, 0.0079204, -0.0281153, 0.0095635, -0.0339163, 0.0360357
7: 0.9583400, 0.9795051, 0.9554319, 0.9799123, -0.0215724, 0.0240732
8: -0.0265281, 0.0111273, -0.0287587, 0.0139102, -0.0404383, 0.0398861
9: -0.0104440, 0.0129519, -0.0120549, 0.0145843, -0.0250284, 0.0250068

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_A1_B2_B2_A1

### Relational analysis result of IS_A2_B2_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156428, upper bound: 0.0157086
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2_B2_A2

### Relational analysis result of IS_A2_B2_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156428, upper bound: 0.0157093
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0093782, 0.0030268, -0.0103724, 0.0033877, -0.0127658, 0.0133993
1: -0.0068242, 0.0003082, -0.0072181, 0.0007404, -0.0075646, 0.0075263
2: 0.0273594, 0.0498725, 0.0267830, 0.0521979, -0.0248385, 0.0230895
3: -0.0038701, 0.0074571, -0.0040022, 0.0085354, -0.0124055, 0.0114594
4: -0.0107794, 0.0066411, -0.0118970, 0.0076747, -0.0184541, 0.0185381
5: 0.0046058, 0.0198299, 0.0037792, 0.0208401, -0.0162343, 0.0160507
6: -0.0246690, 0.0080585, -0.0277867, 0.0094199, -0.0340889, 0.0358452
7: 0.9580956, 0.9795393, 0.9556860, 0.9798768, -0.0217812, 0.0238534
8: -0.0267155, 0.0113612, -0.0285639, 0.0136671, -0.0403826, 0.0399251
9: -0.0105794, 0.0130891, -0.0119142, 0.0144417, -0.0250211, 0.0250032

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_A2_B1_B1_A1

### Relational analysis result of IS_A2_B2_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156428, upper bound: 0.0157206
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A2_A2_B1_B1_A2

### Relational analysis result of IS_A2_B2_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156428, upper bound: 0.0157206
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0093782, 0.0030268, -0.0104189, 0.0034045, -0.0127827, 0.0134457
1: -0.0068242, 0.0003082, -0.0072365, 0.0007606, -0.0075848, 0.0075447
2: 0.0273594, 0.0498725, 0.0267560, 0.0523066, -0.0249472, 0.0231165
3: -0.0038701, 0.0074571, -0.0040084, 0.0085858, -0.0124559, 0.0114655
4: -0.0107794, 0.0066411, -0.0119492, 0.0077230, -0.0185024, 0.0185903
5: 0.0046058, 0.0198299, 0.0037405, 0.0208873, -0.0162815, 0.0160894
6: -0.0246690, 0.0080585, -0.0279324, 0.0094836, -0.0341526, 0.0359908
7: 0.9580956, 0.9795393, 0.9555732, 0.9798924, -0.0217969, 0.0239661
8: -0.0267155, 0.0113612, -0.0286503, 0.0137748, -0.0404904, 0.0400115
9: -0.0105794, 0.0130891, -0.0119765, 0.0145050, -0.0250843, 0.0250656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_A2_B1_B2_A1

### Relational analysis result of IS_A2_B2_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156428, upper bound: 0.0157206
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A2_A2_B1_B2_A2

### Relational analysis result of IS_A2_B2_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156428, upper bound: 0.0157206
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0093782, 0.0030268, -0.0105149, 0.0034393, -0.0128175, 0.0135417
1: -0.0068242, 0.0003082, -0.0072745, 0.0008023, -0.0076265, 0.0075827
2: 0.0273594, 0.0498725, 0.0267004, 0.0525310, -0.0251717, 0.0231721
3: -0.0038701, 0.0074571, -0.0040212, 0.0086899, -0.0125600, 0.0114783
4: -0.0107794, 0.0066411, -0.0120571, 0.0078228, -0.0186022, 0.0186982
5: 0.0046058, 0.0198299, 0.0036607, 0.0209848, -0.0163790, 0.0161692
6: -0.0246690, 0.0080585, -0.0282334, 0.0096150, -0.0342840, 0.0362918
7: 0.9580956, 0.9795393, 0.9553407, 0.9799250, -0.0218295, 0.0241986
8: -0.0267155, 0.0113612, -0.0288287, 0.0139975, -0.0407130, 0.0401899
9: -0.0105794, 0.0130891, -0.0121053, 0.0146355, -0.0252149, 0.0251944

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156428, upper bound: 0.0157206
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156428, upper bound: 0.0157206
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0093782, 0.0030268, -0.0105283, 0.0034442, -0.0128224, 0.0135551
1: -0.0068242, 0.0003082, -0.0072798, 0.0008081, -0.0076323, 0.0075880
2: 0.0273594, 0.0498725, 0.0266926, 0.0525624, -0.0252030, 0.0231799
3: -0.0038701, 0.0074571, -0.0040229, 0.0087044, -0.0125746, 0.0114801
4: -0.0107794, 0.0066411, -0.0120721, 0.0078367, -0.0186161, 0.0187133
5: 0.0046058, 0.0198299, 0.0036496, 0.0209984, -0.0163926, 0.0161803
6: -0.0246690, 0.0080585, -0.0282754, 0.0096333, -0.0343023, 0.0363338
7: 0.9580956, 0.9795393, 0.9553082, 0.9799296, -0.0218340, 0.0242311
8: -0.0267155, 0.0113612, -0.0288536, 0.0140285, -0.0407440, 0.0402148
9: -0.0105794, 0.0130891, -0.0121233, 0.0146537, -0.0252331, 0.0252124

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156428, upper bound: 0.0157206
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156428, upper bound: 0.0157206
time: 0.86 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.55 seconds
IS_A1_B1_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0158248, upper bound: 0.0156789
IS_A1_B1_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0158248, upper bound: 0.0158438
IS_A1_B1_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0158248, upper bound: 0.0156789
IS_A1_B1_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0158248, upper bound: 0.0158438
IS_A1_B1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0155308, upper bound: 0.0157243
IS_A1_B1_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0155308, upper bound: 0.0157243
IS_A1_B1_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0155308, upper bound: 0.0157243
IS_A1_B1_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0155308, upper bound: 0.0157243
IS_A1_B1_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0157279, upper bound: 0.0158657
IS_A1_B1_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0157279, upper bound: 0.0160019
IS_A1_B1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156442, upper bound: 0.0160020
IS_A1_B1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156442, upper bound: 0.0160019
IS_A1_B1_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0155246, upper bound: 0.0158523
IS_A1_B1_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0155246, upper bound: 0.0158523
IS_A1_B1_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0155246, upper bound: 0.0158523
IS_A1_B1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0155246, upper bound: 0.0158523
IS_A1_B1_B2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0158248, upper bound: 0.0156317
IS_A1_B1_B2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0158248, upper bound: 0.0156317
IS_A1_B1_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0158248, upper bound: 0.0158438
IS_A1_B1_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0158248, upper bound: 0.0158438
IS_A1_B1_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0155499
IS_A1_B1_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157243
IS_A1_B1_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0155499
IS_A1_B1_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157243
IS_A1_B1_B2_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0158249, upper bound: 0.0158186
IS_A1_B1_B2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0158249, upper bound: 0.0158186
IS_A1_B1_B2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0158249, upper bound: 0.0160019
IS_A1_B1_B2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0158249, upper bound: 0.0158186
IS_A1_B1_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156739, upper bound: 0.0158523
IS_A1_B1_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156739, upper bound: 0.0158523
IS_A1_B1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156739, upper bound: 0.0158523
IS_A1_B1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156739, upper bound: 0.0158523
IS_A1_B2_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156539, upper bound: 0.0155230
IS_A1_B2_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156539, upper bound: 0.0155230
IS_A1_B2_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156539, upper bound: 0.0155456
IS_A1_B2_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156539, upper bound: 0.0155456
IS_A1_B2_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156881, upper bound: 0.0155196
IS_A1_B2_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156881, upper bound: 0.0155196
IS_A1_B2_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156881, upper bound: 0.0155394
IS_A1_B2_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156881, upper bound: 0.0155394
IS_A1_B2_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156704, upper bound: 0.0156430
IS_A1_B2_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156704, upper bound: 0.0156509
IS_A1_B2_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156704, upper bound: 0.0156430
IS_A1_B2_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156704, upper bound: 0.0156509
IS_A1_B2_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0157086, upper bound: 0.0156428
IS_A1_B2_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0157086, upper bound: 0.0156508
IS_A1_B2_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0157086, upper bound: 0.0156428
IS_A1_B2_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0157086, upper bound: 0.0156508
IS_A1_B2_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156430, upper bound: 0.0156701
IS_A1_B2_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156430, upper bound: 0.0156701
IS_A1_B2_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156430, upper bound: 0.0156745
IS_A1_B2_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156430, upper bound: 0.0156745
IS_A1_B2_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156468, upper bound: 0.0156701
IS_A1_B2_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156468, upper bound: 0.0156701
IS_A1_B2_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156468, upper bound: 0.0156745
IS_A1_B2_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156468, upper bound: 0.0156745
IS_A1_B2_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0155954, upper bound: 0.0157206
IS_A1_B2_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0155954, upper bound: 0.0157206
IS_A1_B2_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0155954, upper bound: 0.0157206
IS_A1_B2_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0155954, upper bound: 0.0157206
IS_A1_B2_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0155994, upper bound: 0.0157206
IS_A1_B2_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0155994, upper bound: 0.0157206
IS_A1_B2_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0155994, upper bound: 0.0157206
IS_A1_B2_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0155994, upper bound: 0.0157206
IS_A2_B1_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0158184, upper bound: 0.0157822
IS_A2_B1_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0158184, upper bound: 0.0159308
IS_A2_B1_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0158184, upper bound: 0.0157822
IS_A2_B1_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0158184, upper bound: 0.0159308
IS_A2_B1_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0156697
IS_A2_B1_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157939
IS_A2_B1_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0155300, upper bound: 0.0157939
IS_A2_B1_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0155300, upper bound: 0.0157939
IS_A2_B1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0155225, upper bound: 0.0159104
IS_A2_B1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0155225, upper bound: 0.0159104
IS_A2_B1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0155225, upper bound: 0.0159104
IS_A2_B1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0155225, upper bound: 0.0159104
IS_A2_B1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0155225, upper bound: 0.0159104
IS_A2_B1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0155225, upper bound: 0.0159104
IS_A2_B1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0155225, upper bound: 0.0159104
IS_A2_B1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0155225, upper bound: 0.0159104
IS_A2_B1_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0156407
IS_A2_B1_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157939
IS_A2_B1_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0156411
IS_A2_B1_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157939
IS_A2_B1_B2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0156407
IS_A2_B1_B2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157939
IS_A2_B1_B2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0156411
IS_A2_B1_B2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157939
IS_A2_B1_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157744
IS_A2_B1_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0159104
IS_A2_B1_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157744
IS_A2_B1_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0159104
IS_A2_B1_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0157744
IS_A2_B1_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156455, upper bound: 0.0159104
IS_A2_B1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156739, upper bound: 0.0159104
IS_A2_B1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156739, upper bound: 0.0159104
IS_A2_B2_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156539, upper bound: 0.0155954
IS_A2_B2_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156539, upper bound: 0.0155954
IS_A2_B2_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156539, upper bound: 0.0156117
IS_A2_B2_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156539, upper bound: 0.0156117
IS_A2_B2_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156539, upper bound: 0.0155954
IS_A2_B2_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156539, upper bound: 0.0155954
IS_A2_B2_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156539, upper bound: 0.0156117
IS_A2_B2_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156539, upper bound: 0.0156117
IS_A2_B2_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156701, upper bound: 0.0156846
IS_A2_B2_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156701, upper bound: 0.0156846
IS_A2_B2_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156701, upper bound: 0.0156914
IS_A2_B2_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156701, upper bound: 0.0156914
IS_A2_B2_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156701, upper bound: 0.0156846
IS_A2_B2_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156701, upper bound: 0.0156914
IS_A2_B2_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156701, upper bound: 0.0156846
IS_A2_B2_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156701, upper bound: 0.0156914
IS_A2_B2_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156428, upper bound: 0.0157086
IS_A2_B2_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156428, upper bound: 0.0157093
IS_A2_B2_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156428, upper bound: 0.0157086
IS_A2_B2_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156428, upper bound: 0.0157093
IS_A2_B2_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156428, upper bound: 0.0157086
IS_A2_B2_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156428, upper bound: 0.0157093
IS_A2_B2_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156428, upper bound: 0.0157086
IS_A2_B2_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156428, upper bound: 0.0157093
IS_A2_B2_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156428, upper bound: 0.0157206
IS_A2_B2_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156428, upper bound: 0.0157206
IS_A2_B2_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156428, upper bound: 0.0157206
IS_A2_B2_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156428, upper bound: 0.0157206
IS_A2_B2_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156428, upper bound: 0.0157206
IS_A2_B2_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156428, upper bound: 0.0157206
IS_A2_B2_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156428, upper bound: 0.0157206
IS_A2_B2_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 7, lower bound: -0.0156428, upper bound: 0.0157206

## BFS IS instance: IS_A1_B1_B1_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0088496, 0.0028350, -0.0090352, 0.0029024, -0.0117520, 0.0118702
1: -0.0066148, 0.0000785, -0.0066883, 0.0001591, -0.0067740, 0.0067668
2: 0.0276658, 0.0486364, 0.0275582, 0.0490703, -0.0214045, 0.0210781
3: -0.0037999, 0.0068840, -0.0038246, 0.0070851, -0.0108851, 0.0107085
4: -0.0101853, 0.0060917, -0.0103939, 0.0062846, -0.0164699, 0.0164855
5: 0.0050452, 0.0192929, 0.0048910, 0.0194814, -0.0144362, 0.0144019
6: -0.0230117, 0.0073348, -0.0235935, 0.0075888, -0.0306005, 0.0309283
7: 0.9593765, 0.9793599, 0.9589269, 0.9794229, -0.0200464, 0.0204331
8: -0.0257330, 0.0101354, -0.0260779, 0.0105657, -0.0362987, 0.0362133
9: -0.0098698, 0.0123700, -0.0101189, 0.0126225, -0.0224923, 0.0224890

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_B1_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156719, upper bound: 0.0156958
time: 0.93 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156719, upper bound: 0.0156958
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0090996, 0.0029257, -0.0090352, 0.0029024, -0.0120019, 0.0119609
1: -0.0067138, 0.0001871, -0.0066883, 0.0001591, -0.0068730, 0.0068754
2: 0.0275209, 0.0492209, 0.0275582, 0.0490703, -0.0215494, 0.0216627
3: -0.0038331, 0.0071550, -0.0038246, 0.0070851, -0.0109183, 0.0109795
4: -0.0104662, 0.0063515, -0.0103939, 0.0062846, -0.0167508, 0.0167453
5: 0.0048374, 0.0195468, 0.0048910, 0.0194814, -0.0146440, 0.0146559
6: -0.0237954, 0.0076770, -0.0235935, 0.0075888, -0.0313841, 0.0312705
7: 0.9587709, 0.9794449, 0.9589269, 0.9794229, -0.0206520, 0.0205180
8: -0.0261976, 0.0107150, -0.0260779, 0.0105657, -0.0367633, 0.0367929
9: -0.0102054, 0.0127100, -0.0101189, 0.0126225, -0.0228278, 0.0228289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_B1_A1_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156305, upper bound: 0.0156304
time: 0.90 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155926, upper bound: 0.0156146
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0088496, 0.0028350, -0.0092084, 0.0029652, -0.0118149, 0.0120434
1: -0.0066148, 0.0000785, -0.0067569, 0.0002344, -0.0068492, 0.0068354
2: 0.0276658, 0.0486364, 0.0274578, 0.0494754, -0.0218096, 0.0211785
3: -0.0037999, 0.0068840, -0.0038476, 0.0072730, -0.0110729, 0.0107315
4: -0.0101853, 0.0060917, -0.0105886, 0.0064647, -0.0166500, 0.0166803
5: 0.0050452, 0.0192929, 0.0047469, 0.0196574, -0.0146122, 0.0145460
6: -0.0230117, 0.0073348, -0.0241367, 0.0078260, -0.0308377, 0.0314714
7: 0.9593765, 0.9793599, 0.9585070, 0.9794818, -0.0201053, 0.0208529
8: -0.0257330, 0.0101354, -0.0263999, 0.0109674, -0.0367004, 0.0365353
9: -0.0098698, 0.0123700, -0.0103515, 0.0128581, -0.0227280, 0.0227215

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_B1_A1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157510, upper bound: 0.0156789
time: 0.80 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157510, upper bound: 0.0156789
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0090996, 0.0029257, -0.0092084, 0.0029652, -0.0120648, 0.0121341
1: -0.0067138, 0.0001871, -0.0067569, 0.0002344, -0.0069483, 0.0069441
2: 0.0275209, 0.0492209, 0.0274578, 0.0494754, -0.0219545, 0.0217631
3: -0.0038331, 0.0071550, -0.0038476, 0.0072730, -0.0111061, 0.0110026
4: -0.0104662, 0.0063515, -0.0105886, 0.0064647, -0.0169309, 0.0169401
5: 0.0048374, 0.0195468, 0.0047469, 0.0196574, -0.0148200, 0.0147999
6: -0.0237954, 0.0076770, -0.0241367, 0.0078260, -0.0316214, 0.0318136
7: 0.9587709, 0.9794449, 0.9585070, 0.9794818, -0.0207109, 0.0209379
8: -0.0261976, 0.0107150, -0.0263999, 0.0109674, -0.0371650, 0.0371149
9: -0.0102054, 0.0127100, -0.0103515, 0.0128581, -0.0230635, 0.0230615

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_B1_A1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157510, upper bound: 0.0158438
time: 0.80 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0157510, upper bound: 0.0158438
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0103882, 0.0033934, -0.0088496, 0.0028350, -0.0132232, 0.0122430
1: -0.0072243, 0.0007472, -0.0066148, 0.0000785, -0.0073028, 0.0073621
2: 0.0267738, 0.0522347, 0.0276658, 0.0486364, -0.0218625, 0.0245689
3: -0.0040043, 0.0085525, -0.0037999, 0.0068840, -0.0108883, 0.0123524
4: -0.0119147, 0.0076911, -0.0101853, 0.0060917, -0.0180064, 0.0178764
5: 0.0037661, 0.0208561, 0.0050452, 0.0192929, -0.0155268, 0.0158109
6: -0.0278360, 0.0094415, -0.0230117, 0.0073348, -0.0351708, 0.0324532
7: 0.9556477, 0.9798821, 0.9593766, 0.9793600, -0.0237123, 0.0205055
8: -0.0285932, 0.0137036, -0.0257330, 0.0101354, -0.0387286, 0.0394366
9: -0.0119353, 0.0144632, -0.0098698, 0.0123700, -0.0243053, 0.0243330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_B1_A1_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155308, upper bound: 0.0155953
time: 0.78 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155308, upper bound: 0.0157340
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0103882, 0.0033934, -0.0090405, 0.0029043, -0.0132925, 0.0124339
1: -0.0072243, 0.0007472, -0.0066904, 0.0001615, -0.0073858, 0.0074377
2: 0.0267738, 0.0522347, 0.0275551, 0.0490829, -0.0223090, 0.0246796
3: -0.0040043, 0.0085525, -0.0038253, 0.0070910, -0.0110953, 0.0123778
4: -0.0119147, 0.0076911, -0.0103999, 0.0062902, -0.0182048, 0.0180910
5: 0.0037661, 0.0208561, 0.0048865, 0.0194869, -0.0157208, 0.0159696
6: -0.0278360, 0.0094415, -0.0236104, 0.0075962, -0.0354322, 0.0330519
7: 0.9556477, 0.9798821, 0.9589138, 0.9794248, -0.0237771, 0.0209683
8: -0.0285932, 0.0137036, -0.0260879, 0.0105782, -0.0391713, 0.0397915
9: -0.0119353, 0.0144632, -0.0101261, 0.0126298, -0.0245651, 0.0245893

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_B1_A1_A2_B1_B2_B1

### Relational analysis result of IS_A1_B1_B1_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0150531, upper bound: 0.0153668
time: 0.85 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2_B1_B2_B2

### Relational analysis result of IS_A1_B1_B1_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0149930, upper bound: 0.0152990
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0103882, 0.0033934, -0.0090306, 0.0029007, -0.0132889, 0.0124240
1: -0.0072243, 0.0007472, -0.0066865, 0.0001572, -0.0073815, 0.0074338
2: 0.0267738, 0.0522347, 0.0275609, 0.0490596, -0.0222858, 0.0246739
3: -0.0040043, 0.0085525, -0.0038240, 0.0070802, -0.0110845, 0.0123764
4: -0.0119147, 0.0076911, -0.0103887, 0.0062798, -0.0181945, 0.0180798
5: 0.0037661, 0.0208561, 0.0048948, 0.0194768, -0.0157107, 0.0159613
6: -0.0278360, 0.0094415, -0.0235792, 0.0075826, -0.0354186, 0.0330207
7: 0.9556477, 0.9798821, 0.9589379, 0.9794214, -0.0237737, 0.0209441
8: -0.0285932, 0.0137036, -0.0260694, 0.0105551, -0.0391483, 0.0397730
9: -0.0119353, 0.0144632, -0.0101128, 0.0126163, -0.0245515, 0.0245760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_B1_A1_A2_B2_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155996, upper bound: 0.0155843
time: 0.83 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2_B2_B1_A2

### Relational analysis result of IS_A1_B1_B1_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155996, upper bound: 0.0157243
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0103882, 0.0033934, -0.0092148, 0.0029676, -0.0133557, 0.0126082
1: -0.0072243, 0.0007472, -0.0067595, 0.0002372, -0.0074615, 0.0075067
2: 0.0267738, 0.0522347, 0.0274541, 0.0494904, -0.0227166, 0.0247806
3: -0.0040043, 0.0085525, -0.0038484, 0.0072799, -0.0112843, 0.0124009
4: -0.0119147, 0.0076911, -0.0105957, 0.0064713, -0.0183860, 0.0182868
5: 0.0037661, 0.0208561, 0.0047416, 0.0196639, -0.0158978, 0.0161145
6: -0.0278360, 0.0094415, -0.0241567, 0.0078347, -0.0356708, 0.0335982
7: 0.9556477, 0.9798821, 0.9584916, 0.9794839, -0.0238362, 0.0213905
8: -0.0285932, 0.0137036, -0.0264118, 0.0109823, -0.0395755, 0.0401154
9: -0.0119353, 0.0144632, -0.0103600, 0.0128668, -0.0248021, 0.0248232

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_B1_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_B1_B1_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0150834, upper bound: 0.0153643
time: 0.78 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2_B2_B2_B2

### Relational analysis result of IS_A1_B1_B1_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0150464, upper bound: 0.0152990
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0090405, 0.0029043, -0.0090392, 0.0029038, -0.0119444, 0.0119435
1: -0.0066904, 0.0001615, -0.0066899, 0.0001609, -0.0068514, 0.0068514
2: 0.0275551, 0.0490829, 0.0275559, 0.0490798, -0.0215247, 0.0215270
3: -0.0038253, 0.0070910, -0.0038251, 0.0070896, -0.0109148, 0.0109161
4: -0.0103999, 0.0062902, -0.0103984, 0.0062888, -0.0166887, 0.0166886
5: 0.0048865, 0.0194869, 0.0048876, 0.0194855, -0.0145990, 0.0145993
6: -0.0236104, 0.0075962, -0.0236063, 0.0075944, -0.0312047, 0.0312024
7: 0.9589138, 0.9794248, 0.9589170, 0.9794243, -0.0205105, 0.0205078
8: -0.0260879, 0.0105782, -0.0260855, 0.0105751, -0.0366630, 0.0366636
9: -0.0101261, 0.0126298, -0.0101243, 0.0126280, -0.0227541, 0.0227541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_B1_A2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156442, upper bound: 0.0158657
time: 0.68 seconds

## Relational analysis of IS_A1_B1_B1_A2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156442, upper bound: 0.0158657
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0092145, 0.0029675, -0.0090392, 0.0029038, -0.0121184, 0.0120067
1: -0.0067594, 0.0002371, -0.0066899, 0.0001609, -0.0069203, 0.0069270
2: 0.0274542, 0.0494898, 0.0275559, 0.0490798, -0.0216256, 0.0219339
3: -0.0038484, 0.0072797, -0.0038251, 0.0070896, -0.0109380, 0.0111048
4: -0.0105955, 0.0064710, -0.0103984, 0.0062888, -0.0168842, 0.0168694
5: 0.0047418, 0.0196637, 0.0048876, 0.0194855, -0.0147437, 0.0147761
6: -0.0241559, 0.0078344, -0.0236063, 0.0075944, -0.0317503, 0.0314407
7: 0.9584922, 0.9794839, 0.9589170, 0.9794243, -0.0209321, 0.0205669
8: -0.0264113, 0.0109817, -0.0260855, 0.0105751, -0.0369865, 0.0370672
9: -0.0103597, 0.0128665, -0.0101243, 0.0126280, -0.0229877, 0.0229908

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_B1_A2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156442, upper bound: 0.0160019
time: 0.69 seconds

## Relational analysis of IS_A1_B1_B1_A2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_B1_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156442, upper bound: 0.0160019
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0092068, 0.0029647, -0.0090405, 0.0029043, -0.0121111, 0.0120052
1: -0.0067563, 0.0002338, -0.0066904, 0.0001615, -0.0069178, 0.0069242
2: 0.0274587, 0.0494718, 0.0275551, 0.0490829, -0.0216242, 0.0219166
3: -0.0038474, 0.0072713, -0.0038253, 0.0070910, -0.0109383, 0.0110966
4: -0.0105868, 0.0064630, -0.0103999, 0.0062902, -0.0168770, 0.0168629
5: 0.0047482, 0.0196558, 0.0048865, 0.0194869, -0.0147386, 0.0147693
6: -0.0241318, 0.0078239, -0.0236104, 0.0075962, -0.0317279, 0.0314342
7: 0.9585109, 0.9794813, 0.9589138, 0.9794248, -0.0209139, 0.0205675
8: -0.0263970, 0.0109638, -0.0260879, 0.0105782, -0.0369752, 0.0370517
9: -0.0103494, 0.0128560, -0.0101261, 0.0126298, -0.0229791, 0.0229821

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_B1_A2_A1_B2_B1_A1

### Relational analysis result of IS_A1_B1_B1_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156664, upper bound: 0.0158658
time: 0.86 seconds

## Relational analysis of IS_A1_B1_B1_A2_A1_B2_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156664, upper bound: 0.0160019
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0092068, 0.0029647, -0.0092148, 0.0029676, -0.0121744, 0.0121794
1: -0.0067563, 0.0002338, -0.0067595, 0.0002372, -0.0069935, 0.0069932
2: 0.0274587, 0.0494718, 0.0274541, 0.0494904, -0.0220317, 0.0220177
3: -0.0038474, 0.0072713, -0.0038484, 0.0072799, -0.0111273, 0.0111197
4: -0.0105868, 0.0064630, -0.0105957, 0.0064713, -0.0170581, 0.0170588
5: 0.0047482, 0.0196558, 0.0047416, 0.0196639, -0.0149157, 0.0149142
6: -0.0241318, 0.0078239, -0.0241567, 0.0078347, -0.0319665, 0.0319806
7: 0.9585109, 0.9794813, 0.9584916, 0.9794839, -0.0209730, 0.0209897
8: -0.0263970, 0.0109638, -0.0264118, 0.0109823, -0.0373793, 0.0373756
9: -0.0103494, 0.0128560, -0.0103600, 0.0128668, -0.0232162, 0.0232160

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_B1_A2_A1_B2_B2_A1

### Relational analysis result of IS_A1_B1_B1_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156664, upper bound: 0.0158658
time: 0.90 seconds

## Relational analysis of IS_A1_B1_B1_A2_A1_B2_B2_A2

### Relational analysis result of IS_A1_B1_B1_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156664, upper bound: 0.0160019
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0103679, 0.0033860, -0.0088496, 0.0028350, -0.0132029, 0.0122356
1: -0.0072163, 0.0007384, -0.0066148, 0.0000785, -0.0072948, 0.0073532
2: 0.0267856, 0.0521872, 0.0276658, 0.0486364, -0.0218508, 0.0245214
3: -0.0040016, 0.0085304, -0.0037999, 0.0068840, -0.0108856, 0.0123304
4: -0.0118918, 0.0076700, -0.0101853, 0.0060917, -0.0179836, 0.0178553
5: 0.0037830, 0.0208354, 0.0050452, 0.0192929, -0.0155100, 0.0157902
6: -0.0277723, 0.0094137, -0.0230117, 0.0073348, -0.0351071, 0.0324254
7: 0.9556970, 0.9798752, 0.9593766, 0.9793600, -0.0236630, 0.0204986
8: -0.0285554, 0.0136565, -0.0257330, 0.0101354, -0.0386908, 0.0393895
9: -0.0119080, 0.0144355, -0.0098698, 0.0123700, -0.0242781, 0.0243053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_B1_A2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155246, upper bound: 0.0157489
time: 0.87 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155246, upper bound: 0.0158592
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0103679, 0.0033860, -0.0090405, 0.0029043, -0.0132722, 0.0124265
1: -0.0072163, 0.0007384, -0.0066904, 0.0001615, -0.0073777, 0.0074289
2: 0.0267856, 0.0521872, 0.0275551, 0.0490829, -0.0222973, 0.0246321
3: -0.0040016, 0.0085304, -0.0038253, 0.0070910, -0.0110926, 0.0123557
4: -0.0118918, 0.0076700, -0.0103999, 0.0062902, -0.0181820, 0.0180699
5: 0.0037830, 0.0208354, 0.0048865, 0.0194869, -0.0157039, 0.0159489
6: -0.0277723, 0.0094137, -0.0236104, 0.0075962, -0.0353685, 0.0330241
7: 0.9556970, 0.9798752, 0.9589138, 0.9794248, -0.0237278, 0.0209614
8: -0.0285554, 0.0136565, -0.0260879, 0.0105782, -0.0391335, 0.0397444
9: -0.0119080, 0.0144355, -0.0101261, 0.0126298, -0.0245378, 0.0245616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_B1_A2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155246, upper bound: 0.0157489
time: 0.96 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155246, upper bound: 0.0158592
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0103679, 0.0033860, -0.0090306, 0.0029007, -0.0132686, 0.0124166
1: -0.0072163, 0.0007384, -0.0066865, 0.0001572, -0.0073734, 0.0074249
2: 0.0267856, 0.0521872, 0.0275609, 0.0490596, -0.0222740, 0.0246263
3: -0.0040016, 0.0085304, -0.0038240, 0.0070802, -0.0110818, 0.0123544
4: -0.0118918, 0.0076700, -0.0103887, 0.0062798, -0.0181717, 0.0180587
5: 0.0037830, 0.0208354, 0.0048948, 0.0194768, -0.0156938, 0.0159407
6: -0.0277723, 0.0094137, -0.0235792, 0.0075826, -0.0353549, 0.0329929
7: 0.9556970, 0.9798752, 0.9589379, 0.9794214, -0.0237244, 0.0209373
8: -0.0285554, 0.0136565, -0.0260694, 0.0105551, -0.0391105, 0.0397259
9: -0.0119080, 0.0144355, -0.0101128, 0.0126163, -0.0245243, 0.0245483

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B1_A1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155956, upper bound: 0.0157390
time: 0.82 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155956, upper bound: 0.0158523
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0103679, 0.0033860, -0.0092148, 0.0029676, -0.0133354, 0.0126008
1: -0.0072163, 0.0007384, -0.0067595, 0.0002372, -0.0074535, 0.0074979
2: 0.0267856, 0.0521872, 0.0274541, 0.0494904, -0.0227048, 0.0247331
3: -0.0040016, 0.0085304, -0.0038484, 0.0072799, -0.0112816, 0.0123789
4: -0.0118918, 0.0076700, -0.0105957, 0.0064713, -0.0183631, 0.0182657
5: 0.0037830, 0.0208354, 0.0047416, 0.0196639, -0.0158809, 0.0160938
6: -0.0277723, 0.0094137, -0.0241567, 0.0078347, -0.0356070, 0.0335704
7: 0.9556970, 0.9798752, 0.9584916, 0.9794839, -0.0237869, 0.0213836
8: -0.0285554, 0.0136565, -0.0264118, 0.0109823, -0.0395376, 0.0400683
9: -0.0119080, 0.0144355, -0.0103600, 0.0128668, -0.0247748, 0.0247956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B2_A1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155956, upper bound: 0.0157394
time: 0.90 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B2_A2

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155956, upper bound: 0.0158523
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0088496, 0.0028350, -0.0092073, 0.0029648, -0.0118145, 0.0120423
1: -0.0066148, 0.0000785, -0.0067565, 0.0002340, -0.0068488, 0.0068350
2: 0.0276658, 0.0486364, 0.0274585, 0.0494729, -0.0218071, 0.0211779
3: -0.0037999, 0.0068840, -0.0038474, 0.0072718, -0.0110717, 0.0107314
4: -0.0101853, 0.0060917, -0.0105873, 0.0064635, -0.0166488, 0.0166790
5: 0.0050452, 0.0192929, 0.0047479, 0.0196563, -0.0146111, 0.0145450
6: -0.0230117, 0.0073348, -0.0241333, 0.0078245, -0.0308362, 0.0314680
7: 0.9593765, 0.9793599, 0.9585097, 0.9794814, -0.0201049, 0.0208502
8: -0.0257330, 0.0101354, -0.0263979, 0.0109649, -0.0366979, 0.0365333
9: -0.0098698, 0.0123700, -0.0103500, 0.0128566, -0.0227265, 0.0227200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_B2_A1_A1_A1_B1_B1

### Relational analysis result of IS_A1_B1_B2_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0158869, upper bound: 0.0156317
time: 0.87 seconds

## Relational analysis of IS_A1_B1_B2_A1_A1_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0158869, upper bound: 0.0156317
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0088496, 0.0028350, -0.0093546, 0.0030183, -0.0118679, 0.0121896
1: -0.0066148, 0.0000785, -0.0068149, 0.0002980, -0.0069128, 0.0068934
2: 0.0276658, 0.0486364, 0.0273731, 0.0498174, -0.0221516, 0.0212633
3: -0.0037999, 0.0068840, -0.0038670, 0.0074316, -0.0112315, 0.0107510
4: -0.0101853, 0.0060917, -0.0107529, 0.0066166, -0.0168019, 0.0168446
5: 0.0050452, 0.0192929, 0.0046254, 0.0198060, -0.0147607, 0.0146675
6: -0.0230117, 0.0073348, -0.0245951, 0.0080262, -0.0310379, 0.0319299
7: 0.9593765, 0.9793599, 0.9581528, 0.9795313, -0.0201548, 0.0212072
8: -0.0257330, 0.0101354, -0.0266717, 0.0113065, -0.0370395, 0.0368071
9: -0.0098698, 0.0123700, -0.0105477, 0.0130570, -0.0229268, 0.0229178

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_B2_A1_A1_A1_B2_B1

### Relational analysis result of IS_A1_B1_B2_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0158869, upper bound: 0.0156317
time: 0.75 seconds

## Relational analysis of IS_A1_B1_B2_A1_A1_A1_B2_B2

### Relational analysis result of IS_A1_B1_B2_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0158869, upper bound: 0.0156317
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0090996, 0.0029257, -0.0092073, 0.0029648, -0.0120644, 0.0121330
1: -0.0067138, 0.0001871, -0.0067565, 0.0002340, -0.0069478, 0.0069436
2: 0.0275209, 0.0492209, 0.0274585, 0.0494729, -0.0219520, 0.0217624
3: -0.0038331, 0.0071550, -0.0038474, 0.0072718, -0.0111049, 0.0110024
4: -0.0104662, 0.0063515, -0.0105873, 0.0064635, -0.0169297, 0.0169388
5: 0.0048374, 0.0195468, 0.0047479, 0.0196563, -0.0148189, 0.0147990
6: -0.0237954, 0.0076770, -0.0241333, 0.0078245, -0.0316198, 0.0318102
7: 0.9587709, 0.9794449, 0.9585097, 0.9794814, -0.0207105, 0.0209352
8: -0.0261976, 0.0107150, -0.0263979, 0.0109649, -0.0371625, 0.0371129
9: -0.0102054, 0.0127100, -0.0103500, 0.0128566, -0.0230620, 0.0230600

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_B2_A1_A1_A2_B1_B1

### Relational analysis result of IS_A1_B1_B2_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156840, upper bound: 0.0158438
time: 0.71 seconds

## Relational analysis of IS_A1_B1_B2_A1_A1_A2_B1_B2

### Relational analysis result of IS_A1_B1_B2_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156840, upper bound: 0.0158438
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0090996, 0.0029257, -0.0093546, 0.0030183, -0.0121178, 0.0122803
1: -0.0067138, 0.0001871, -0.0068149, 0.0002980, -0.0070118, 0.0070020
2: 0.0275209, 0.0492209, 0.0273731, 0.0498174, -0.0222965, 0.0218478
3: -0.0038331, 0.0071550, -0.0038670, 0.0074316, -0.0112647, 0.0110220
4: -0.0104662, 0.0063515, -0.0107529, 0.0066166, -0.0170829, 0.0171044
5: 0.0048374, 0.0195468, 0.0046254, 0.0198060, -0.0149685, 0.0149214
6: -0.0237954, 0.0076770, -0.0245951, 0.0080262, -0.0318215, 0.0322721
7: 0.9587709, 0.9794449, 0.9581528, 0.9795313, -0.0207604, 0.0212921
8: -0.0261976, 0.0107150, -0.0266717, 0.0113065, -0.0375041, 0.0373867
9: -0.0102054, 0.0127100, -0.0105477, 0.0130570, -0.0232624, 0.0232577

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_B2_A1_A1_A2_B2_B1

### Relational analysis result of IS_A1_B1_B2_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156840, upper bound: 0.0158438
time: 0.85 seconds

## Relational analysis of IS_A1_B1_B2_A1_A1_A2_B2_B2

### Relational analysis result of IS_A1_B1_B2_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156840, upper bound: 0.0158438
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0102285, 0.0033354, -0.0092073, 0.0029648, -0.0131933, 0.0125427
1: -0.0071610, 0.0006778, -0.0067565, 0.0002340, -0.0073950, 0.0074343
2: 0.0268664, 0.0518612, 0.0274585, 0.0494729, -0.0226065, 0.0244028
3: -0.0039831, 0.0083793, -0.0038474, 0.0072718, -0.0112549, 0.0122267
4: -0.0117352, 0.0075251, -0.0105873, 0.0064635, -0.0181987, 0.0181124
5: 0.0038989, 0.0206938, 0.0047479, 0.0196563, -0.0157574, 0.0159459
6: -0.0273353, 0.0092228, -0.0241333, 0.0078245, -0.0351598, 0.0333561
7: 0.9560348, 0.9798278, 0.9585097, 0.9794814, -0.0234466, 0.0213181
8: -0.0282963, 0.0133332, -0.0263979, 0.0109649, -0.0392612, 0.0397311
9: -0.0117209, 0.0142459, -0.0103500, 0.0128566, -0.0245775, 0.0245959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_B2_A1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155308, upper bound: 0.0155601
time: 1.02 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155308, upper bound: 0.0155601
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0103971, 0.0033966, -0.0092073, 0.0029648, -0.0133619, 0.0126039
1: -0.0072278, 0.0007511, -0.0067565, 0.0002340, -0.0074618, 0.0075076
2: 0.0267687, 0.0522556, 0.0274585, 0.0494729, -0.0227042, 0.0247971
3: -0.0040055, 0.0085622, -0.0038474, 0.0072718, -0.0112773, 0.0124096
4: -0.0119247, 0.0077004, -0.0105873, 0.0064635, -0.0183882, 0.0182877
5: 0.0037587, 0.0208651, 0.0047479, 0.0196563, -0.0158976, 0.0161172
6: -0.0278640, 0.0094537, -0.0241333, 0.0078245, -0.0356885, 0.0335870
7: 0.9556262, 0.9798850, 0.9585097, 0.9794814, -0.0238552, 0.0213754
8: -0.0286097, 0.0137243, -0.0263979, 0.0109649, -0.0395746, 0.0401222
9: -0.0119473, 0.0144753, -0.0103500, 0.0128566, -0.0248039, 0.0248253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_B2_A1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_B2_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155308, upper bound: 0.0157341
time: 0.79 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_B2_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155308, upper bound: 0.0157341
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0102285, 0.0033354, -0.0093546, 0.0030183, -0.0132468, 0.0126900
1: -0.0071610, 0.0006778, -0.0068149, 0.0002980, -0.0074590, 0.0074927
2: 0.0268664, 0.0518612, 0.0273731, 0.0498174, -0.0229510, 0.0244882
3: -0.0039831, 0.0083793, -0.0038670, 0.0074316, -0.0114147, 0.0122463
4: -0.0117352, 0.0075251, -0.0107529, 0.0066166, -0.0183518, 0.0182780
5: 0.0038989, 0.0206938, 0.0046254, 0.0198060, -0.0159071, 0.0160684
6: -0.0273353, 0.0092228, -0.0245951, 0.0080262, -0.0353615, 0.0338179
7: 0.9560348, 0.9798278, 0.9581528, 0.9795313, -0.0234964, 0.0216751
8: -0.0282963, 0.0133332, -0.0266717, 0.0113065, -0.0396028, 0.0400049
9: -0.0117209, 0.0142459, -0.0105477, 0.0130570, -0.0247779, 0.0247936

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_B2_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155996, upper bound: 0.0155499
time: 0.88 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155996, upper bound: 0.0155499
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0103971, 0.0033966, -0.0093546, 0.0030183, -0.0134154, 0.0127512
1: -0.0072278, 0.0007511, -0.0068149, 0.0002980, -0.0075258, 0.0075660
2: 0.0267687, 0.0522556, 0.0273731, 0.0498174, -0.0230487, 0.0248825
3: -0.0040055, 0.0085622, -0.0038670, 0.0074316, -0.0114371, 0.0124292
4: -0.0119247, 0.0077004, -0.0107529, 0.0066166, -0.0185413, 0.0184532
5: 0.0037587, 0.0208651, 0.0046254, 0.0198060, -0.0160473, 0.0162397
6: -0.0278640, 0.0094537, -0.0245951, 0.0080262, -0.0358902, 0.0340488
7: 0.9556262, 0.9798850, 0.9581528, 0.9795313, -0.0239051, 0.0217323
8: -0.0286097, 0.0137243, -0.0266717, 0.0113065, -0.0399162, 0.0403960
9: -0.0119473, 0.0144753, -0.0105477, 0.0130570, -0.0250043, 0.0250230

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_B2_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155996, upper bound: 0.0157243
time: 0.95 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155996, upper bound: 0.0157243
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0090405, 0.0029043, -0.0092566, 0.0029827, -0.0120233, 0.0121609
1: -0.0066904, 0.0001615, -0.0067760, 0.0002554, -0.0069458, 0.0069375
2: 0.0275551, 0.0490829, 0.0274299, 0.0495882, -0.0220331, 0.0216530
3: -0.0038253, 0.0070910, -0.0038540, 0.0073253, -0.0111506, 0.0109450
4: -0.0103999, 0.0062902, -0.0106428, 0.0065148, -0.0169147, 0.0169329
5: 0.0048865, 0.0194869, 0.0047069, 0.0197064, -0.0148199, 0.0147800
6: -0.0236104, 0.0075962, -0.0242879, 0.0078920, -0.0315024, 0.0318841
7: 0.9589138, 0.9794248, 0.9583902, 0.9794980, -0.0205842, 0.0210346
8: -0.0260879, 0.0105782, -0.0264896, 0.0110793, -0.0371672, 0.0370677
9: -0.0101261, 0.0126298, -0.0104162, 0.0129237, -0.0230499, 0.0230460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_B2_A2_A1_A1_B1_B1

### Relational analysis result of IS_A1_B1_B2_A2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0158623, upper bound: 0.0158184
time: 1.00 seconds

## Relational analysis of IS_A1_B1_B2_A2_A1_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2_A2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0158623, upper bound: 0.0158186
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0090405, 0.0029043, -0.0094046, 0.0030364, -0.0120770, 0.0123089
1: -0.0066904, 0.0001615, -0.0068347, 0.0003197, -0.0070102, 0.0069961
2: 0.0275551, 0.0490829, 0.0273441, 0.0499343, -0.0223792, 0.0217388
3: -0.0038253, 0.0070910, -0.0038736, 0.0074858, -0.0113111, 0.0109646
4: -0.0103999, 0.0062902, -0.0108091, 0.0066686, -0.0170685, 0.0170993
5: 0.0048865, 0.0194869, 0.0045838, 0.0198568, -0.0149703, 0.0149031
6: -0.0236104, 0.0075962, -0.0247519, 0.0080947, -0.0317050, 0.0323481
7: 0.9589138, 0.9794248, 0.9580316, 0.9795483, -0.0206345, 0.0213932
8: -0.0260879, 0.0105782, -0.0267647, 0.0114225, -0.0375104, 0.0373428
9: -0.0101261, 0.0126298, -0.0106149, 0.0131251, -0.0232512, 0.0232446

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_B2_A2_A1_A1_B2_B1

### Relational analysis result of IS_A1_B1_B2_A2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0158623, upper bound: 0.0158184
time: 0.74 seconds

## Relational analysis of IS_A1_B1_B2_A2_A1_A1_B2_B2

### Relational analysis result of IS_A1_B1_B2_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0158623, upper bound: 0.0158186
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0092145, 0.0029675, -0.0092566, 0.0029827, -0.0121972, 0.0122241
1: -0.0067594, 0.0002371, -0.0067760, 0.0002554, -0.0070148, 0.0070131
2: 0.0274542, 0.0494898, 0.0274299, 0.0495882, -0.0221340, 0.0220599
3: -0.0038484, 0.0072797, -0.0038540, 0.0073253, -0.0111737, 0.0111337
4: -0.0105955, 0.0064710, -0.0106428, 0.0065148, -0.0171102, 0.0171138
5: 0.0047418, 0.0196637, 0.0047069, 0.0197064, -0.0149646, 0.0149568
6: -0.0241559, 0.0078344, -0.0242879, 0.0078920, -0.0320480, 0.0321223
7: 0.9584922, 0.9794839, 0.9583902, 0.9794980, -0.0210058, 0.0210937
8: -0.0264113, 0.0109817, -0.0264896, 0.0110793, -0.0374906, 0.0374713
9: -0.0103597, 0.0128665, -0.0104162, 0.0129237, -0.0232835, 0.0232827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_B2_A2_A1_A2_B1_B1

### Relational analysis result of IS_A1_B1_B2_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156592, upper bound: 0.0160019
time: 0.90 seconds

## Relational analysis of IS_A1_B1_B2_A2_A1_A2_B1_B2

### Relational analysis result of IS_A1_B1_B2_A2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156592, upper bound: 0.0160019
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0092145, 0.0029675, -0.0094046, 0.0030364, -0.0122510, 0.0123721
1: -0.0067594, 0.0002371, -0.0068347, 0.0003197, -0.0070791, 0.0070718
2: 0.0274542, 0.0494898, 0.0273441, 0.0499343, -0.0224801, 0.0221457
3: -0.0038484, 0.0072797, -0.0038736, 0.0074858, -0.0113342, 0.0111533
4: -0.0105955, 0.0064710, -0.0108091, 0.0066686, -0.0172641, 0.0172801
5: 0.0047418, 0.0196637, 0.0045838, 0.0198568, -0.0151149, 0.0150798
6: -0.0241559, 0.0078344, -0.0247519, 0.0080947, -0.0322506, 0.0325863
7: 0.9584922, 0.9794839, 0.9580316, 0.9795483, -0.0210561, 0.0214523
8: -0.0264113, 0.0109817, -0.0267647, 0.0114225, -0.0378338, 0.0377464
9: -0.0103597, 0.0128665, -0.0106149, 0.0131251, -0.0234848, 0.0234813

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_B2_A2_A1_A2_B2_B1

### Relational analysis result of IS_A1_B1_B2_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156592, upper bound: 0.0160019
time: 0.92 seconds

## Relational analysis of IS_A1_B1_B2_A2_A1_A2_B2_B2

### Relational analysis result of IS_A1_B1_B2_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156592, upper bound: 0.0160020
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0103997, 0.0033975, -0.0090996, 0.0029257, -0.0133254, 0.0124971
1: -0.0072288, 0.0007522, -0.0067138, 0.0001871, -0.0074160, 0.0074661
2: 0.0267672, 0.0522616, 0.0275209, 0.0492209, -0.0224537, 0.0247407
3: -0.0040059, 0.0085649, -0.0038331, 0.0071550, -0.0111608, 0.0123981
4: -0.0119276, 0.0077030, -0.0104662, 0.0063515, -0.0182791, 0.0181693
5: 0.0037565, 0.0208677, 0.0048374, 0.0195468, -0.0157903, 0.0160303
6: -0.0278721, 0.0094572, -0.0237954, 0.0076770, -0.0355491, 0.0332526
7: 0.9556199, 0.9798860, 0.9587709, 0.9794449, -0.0238249, 0.0211151
8: -0.0286145, 0.0137302, -0.0261976, 0.0107150, -0.0393296, 0.0399278
9: -0.0119507, 0.0144788, -0.0102054, 0.0127100, -0.0246607, 0.0246841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_B2_A2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_B2_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155246, upper bound: 0.0157206
time: 0.87 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_B2_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155246, upper bound: 0.0158592
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0103997, 0.0033975, -0.0092145, 0.0029675, -0.0133671, 0.0126121
1: -0.0072288, 0.0007522, -0.0067594, 0.0002371, -0.0074660, 0.0075116
2: 0.0267672, 0.0522616, 0.0274542, 0.0494898, -0.0227226, 0.0248073
3: -0.0040059, 0.0085649, -0.0038484, 0.0072797, -0.0112855, 0.0124133
4: -0.0119276, 0.0077030, -0.0105955, 0.0064710, -0.0183986, 0.0182985
5: 0.0037565, 0.0208677, 0.0047418, 0.0196637, -0.0159071, 0.0161259
6: -0.0278721, 0.0094572, -0.0241559, 0.0078344, -0.0357065, 0.0336131
7: 0.9556199, 0.9798860, 0.9584922, 0.9794839, -0.0238640, 0.0213938
8: -0.0286145, 0.0137302, -0.0264113, 0.0109817, -0.0395962, 0.0401415
9: -0.0119507, 0.0144788, -0.0103597, 0.0128665, -0.0248172, 0.0248385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_B2_A2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155246, upper bound: 0.0157216
time: 0.91 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155246, upper bound: 0.0158592
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0103997, 0.0033975, -0.0092591, 0.0029836, -0.0133833, 0.0126566
1: -0.0072288, 0.0007522, -0.0067770, 0.0002565, -0.0074853, 0.0075293
2: 0.0267672, 0.0522616, 0.0274284, 0.0495940, -0.0228268, 0.0248332
3: -0.0040059, 0.0085649, -0.0038543, 0.0073280, -0.0113339, 0.0124193
4: -0.0119276, 0.0077030, -0.0106456, 0.0065174, -0.0184449, 0.0183486
5: 0.0037565, 0.0208677, 0.0047048, 0.0197090, -0.0159524, 0.0161629
6: -0.0278721, 0.0094572, -0.0242957, 0.0078954, -0.0357675, 0.0337529
7: 0.9556199, 0.9798860, 0.9583842, 0.9794989, -0.0238790, 0.0215018
8: -0.0286145, 0.0137302, -0.0264942, 0.0110850, -0.0396996, 0.0402244
9: -0.0119507, 0.0144788, -0.0104195, 0.0129271, -0.0248778, 0.0248983

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_B2_A2_A2_B2_B1_A1

### Relational analysis result of IS_A1_B1_B2_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155956, upper bound: 0.0157125
time: 0.84 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2_B2_B1_A2

### Relational analysis result of IS_A1_B1_B2_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155956, upper bound: 0.0158523
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0103997, 0.0033975, -0.0093619, 0.0030209, -0.0134206, 0.0127594
1: -0.0072288, 0.0007522, -0.0068178, 0.0003012, -0.0075300, 0.0075700
2: 0.0267672, 0.0522616, 0.0273688, 0.0498345, -0.0230673, 0.0248928
3: -0.0040059, 0.0085649, -0.0038680, 0.0074395, -0.0114454, 0.0124329
4: -0.0119276, 0.0077030, -0.0107611, 0.0066242, -0.0185518, 0.0184641
5: 0.0037565, 0.0208677, 0.0046193, 0.0198134, -0.0160569, 0.0162484
6: -0.0278721, 0.0094572, -0.0246180, 0.0080362, -0.0359083, 0.0340753
7: 0.9556199, 0.9798860, 0.9581350, 0.9795338, -0.0239139, 0.0217510
8: -0.0286145, 0.0137302, -0.0266853, 0.0113235, -0.0399380, 0.0404155
9: -0.0119507, 0.0144788, -0.0105576, 0.0130670, -0.0250177, 0.0250364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_B2_A2_A2_B2_B2_A1

### Relational analysis result of IS_A1_B1_B2_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155956, upper bound: 0.0157137
time: 0.83 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2_B2_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155956, upper bound: 0.0158523
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0088496, 0.0028350, -0.0103289, 0.0033718, -0.0122215, 0.0131639
1: -0.0066148, 0.0000785, -0.0072008, 0.0007215, -0.0073363, 0.0072793
2: 0.0276658, 0.0486364, 0.0268082, 0.0520960, -0.0244302, 0.0218282
3: -0.0037999, 0.0068840, -0.0039964, 0.0084882, -0.0122881, 0.0108804
4: -0.0101853, 0.0060917, -0.0118480, 0.0076294, -0.0178147, 0.0179397
5: 0.0050452, 0.0192929, 0.0038154, 0.0207958, -0.0157506, 0.0154775
6: -0.0230117, 0.0073348, -0.0276501, 0.0093603, -0.0323720, 0.0349849
7: 0.9593766, 0.9793600, 0.9557915, 0.9798620, -0.0204854, 0.0235686
8: -0.0257330, 0.0101354, -0.0284829, 0.0135661, -0.0392991, 0.0386184
9: -0.0098698, 0.0123700, -0.0118557, 0.0143825, -0.0242523, 0.0242258

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0154382, upper bound: 0.0150949
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0153723, upper bound: 0.0150087
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0088496, 0.0028350, -0.0103703, 0.0033869, -0.0122365, 0.0132053
1: -0.0066148, 0.0000785, -0.0072172, 0.0007395, -0.0073543, 0.0072957
2: 0.0276658, 0.0486364, 0.0267842, 0.0521929, -0.0245271, 0.0218522
3: -0.0037999, 0.0068840, -0.0040019, 0.0085331, -0.0123330, 0.0108859
4: -0.0101853, 0.0060917, -0.0118946, 0.0076725, -0.0178578, 0.0179863
5: 0.0050452, 0.0192929, 0.0037810, 0.0208379, -0.0157927, 0.0155120
6: -0.0230117, 0.0073348, -0.0277800, 0.0094170, -0.0324287, 0.0351147
7: 0.9593766, 0.9793600, 0.9556912, 0.9798760, -0.0204994, 0.0236688
8: -0.0257330, 0.0101354, -0.0285599, 0.0136621, -0.0393951, 0.0386953
9: -0.0098698, 0.0123700, -0.0119113, 0.0144388, -0.0243087, 0.0242813

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156171, upper bound: 0.0155246
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156171, upper bound: 0.0155246
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0102285, 0.0033354, -0.0103289, 0.0033718, -0.0136003, 0.0136643
1: -0.0071610, 0.0006778, -0.0072008, 0.0007215, -0.0078825, 0.0078786
2: 0.0268664, 0.0518612, 0.0268082, 0.0520960, -0.0252296, 0.0250530
3: -0.0039831, 0.0083793, -0.0039964, 0.0084882, -0.0124713, 0.0123757
4: -0.0117352, 0.0075251, -0.0118480, 0.0076294, -0.0193646, 0.0193731
5: 0.0038989, 0.0206938, 0.0038154, 0.0207958, -0.0168970, 0.0168784
6: -0.0273353, 0.0092228, -0.0276501, 0.0093603, -0.0366956, 0.0368729
7: 0.9560348, 0.9798278, 0.9557915, 0.9798620, -0.0238271, 0.0240363
8: -0.0282963, 0.0133332, -0.0284829, 0.0135661, -0.0418623, 0.0418162
9: -0.0117209, 0.0142459, -0.0118557, 0.0143825, -0.0261033, 0.0261016

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155481, upper bound: 0.0155456
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155481, upper bound: 0.0155456
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0102285, 0.0033354, -0.0103703, 0.0033869, -0.0136154, 0.0137057
1: -0.0071610, 0.0006778, -0.0072172, 0.0007395, -0.0079005, 0.0078950
2: 0.0268664, 0.0518612, 0.0267842, 0.0521929, -0.0253265, 0.0250770
3: -0.0039831, 0.0083793, -0.0040019, 0.0085331, -0.0125162, 0.0123812
4: -0.0117352, 0.0075251, -0.0118946, 0.0076725, -0.0194076, 0.0194196
5: 0.0038989, 0.0206938, 0.0037810, 0.0208379, -0.0169390, 0.0169129
6: -0.0273353, 0.0092228, -0.0277800, 0.0094170, -0.0367523, 0.0370028
7: 0.9560348, 0.9798278, 0.9556912, 0.9798760, -0.0238411, 0.0241366
8: -0.0282963, 0.0133332, -0.0285599, 0.0136621, -0.0419584, 0.0418931
9: -0.0117209, 0.0142459, -0.0119113, 0.0144388, -0.0261597, 0.0261572

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155481, upper bound: 0.0155456
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B2_A1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155481, upper bound: 0.0155456
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0088496, 0.0028350, -0.0104667, 0.0034219, -0.0122715, 0.0133018
1: -0.0066148, 0.0000785, -0.0072554, 0.0007814, -0.0073962, 0.0073339
2: 0.0276658, 0.0486364, 0.0267283, 0.0524185, -0.0247527, 0.0219081
3: -0.0037999, 0.0068840, -0.0040148, 0.0086377, -0.0124376, 0.0108987
4: -0.0101853, 0.0060917, -0.0120030, 0.0077727, -0.0179580, 0.0180947
5: 0.0050452, 0.0192929, 0.0037008, 0.0209359, -0.0158907, 0.0155921
6: -0.0230117, 0.0073348, -0.0280824, 0.0095490, -0.0325608, 0.0354171
7: 0.9593766, 0.9793600, 0.9554574, 0.9799086, -0.0205321, 0.0239026
8: -0.0257330, 0.0101354, -0.0287392, 0.0138858, -0.0396188, 0.0388746
9: -0.0098698, 0.0123700, -0.0120408, 0.0145700, -0.0244399, 0.0244108

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155348, upper bound: 0.0150953
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0154392, upper bound: 0.0150102
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0088496, 0.0028350, -0.0104772, 0.0034257, -0.0122753, 0.0133123
1: -0.0066148, 0.0000785, -0.0072596, 0.0007860, -0.0074008, 0.0073381
2: 0.0276658, 0.0486364, 0.0267222, 0.0524431, -0.0247773, 0.0219142
3: -0.0037999, 0.0068840, -0.0040162, 0.0086491, -0.0124490, 0.0109001
4: -0.0101853, 0.0060917, -0.0120148, 0.0077837, -0.0179690, 0.0181065
5: 0.0050452, 0.0192929, 0.0036920, 0.0209466, -0.0159014, 0.0156009
6: -0.0230117, 0.0073348, -0.0281153, 0.0095635, -0.0325752, 0.0354501
7: 0.9593766, 0.9793600, 0.9554319, 0.9799123, -0.0205358, 0.0239281
8: -0.0257330, 0.0101354, -0.0287587, 0.0139102, -0.0396432, 0.0388942
9: -0.0098698, 0.0123700, -0.0120549, 0.0145843, -0.0244542, 0.0244249

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A1_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156942, upper bound: 0.0155225
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156942, upper bound: 0.0155225
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0102285, 0.0033354, -0.0104667, 0.0034219, -0.0136503, 0.0138021
1: -0.0071610, 0.0006778, -0.0072554, 0.0007814, -0.0079424, 0.0079332
2: 0.0268664, 0.0518612, 0.0267283, 0.0524185, -0.0255520, 0.0251329
3: -0.0039831, 0.0083793, -0.0040148, 0.0086377, -0.0126208, 0.0123940
4: -0.0117352, 0.0075251, -0.0120030, 0.0077727, -0.0195079, 0.0195280
5: 0.0038989, 0.0206938, 0.0037008, 0.0209359, -0.0170370, 0.0169930
6: -0.0273353, 0.0092228, -0.0280824, 0.0095490, -0.0368843, 0.0373052
7: 0.9560348, 0.9798278, 0.9554574, 0.9799086, -0.0238738, 0.0243704
8: -0.0282963, 0.0133332, -0.0287392, 0.0138858, -0.0421821, 0.0420724
9: -0.0117209, 0.0142459, -0.0120408, 0.0145700, -0.0262909, 0.0262867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A1_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156167, upper bound: 0.0155394
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156167, upper bound: 0.0155394
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0102285, 0.0033354, -0.0104772, 0.0034257, -0.0136542, 0.0138127
1: -0.0071610, 0.0006778, -0.0072596, 0.0007860, -0.0079470, 0.0079374
2: 0.0268664, 0.0518612, 0.0267222, 0.0524431, -0.0255766, 0.0251390
3: -0.0039831, 0.0083793, -0.0040162, 0.0086491, -0.0126322, 0.0123955
4: -0.0117352, 0.0075251, -0.0120148, 0.0077837, -0.0195189, 0.0195399
5: 0.0038989, 0.0206938, 0.0036920, 0.0209466, -0.0170477, 0.0170018
6: -0.0273353, 0.0092228, -0.0281153, 0.0095635, -0.0368988, 0.0373382
7: 0.9560348, 0.9798278, 0.9554319, 0.9799123, -0.0238775, 0.0243959
8: -0.0282963, 0.0133332, -0.0287587, 0.0139102, -0.0422065, 0.0420920
9: -0.0117209, 0.0142459, -0.0120549, 0.0145843, -0.0263052, 0.0263008

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A1_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156167, upper bound: 0.0155394
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156167, upper bound: 0.0155394
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0090405, 0.0029043, -0.0103882, 0.0033934, -0.0124339, 0.0132925
1: -0.0066904, 0.0001615, -0.0072243, 0.0007472, -0.0074377, 0.0073858
2: 0.0275551, 0.0490829, 0.0267738, 0.0522347, -0.0246796, 0.0223090
3: -0.0038253, 0.0070910, -0.0040043, 0.0085525, -0.0123778, 0.0110953
4: -0.0103999, 0.0062902, -0.0119147, 0.0076911, -0.0180910, 0.0182048
5: 0.0048865, 0.0194869, 0.0037661, 0.0208561, -0.0159696, 0.0157208
6: -0.0236104, 0.0075962, -0.0278360, 0.0094415, -0.0330519, 0.0354322
7: 0.9589138, 0.9794248, 0.9556477, 0.9798821, -0.0209683, 0.0237771
8: -0.0260879, 0.0105782, -0.0285932, 0.0137036, -0.0397915, 0.0391713
9: -0.0101261, 0.0126298, -0.0119353, 0.0144632, -0.0245893, 0.0245651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_A2_B1_B1_A1_A1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0152585, upper bound: 0.0151052
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B1_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0151038, upper bound: 0.0150802
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0103340, 0.0033737, -0.0103882, 0.0033934, -0.0137274, 0.0137619
1: -0.0072028, 0.0007237, -0.0072243, 0.0007472, -0.0079501, 0.0079480
2: 0.0268052, 0.0521082, 0.0267738, 0.0522347, -0.0254295, 0.0253343
3: -0.0039971, 0.0084938, -0.0040043, 0.0085525, -0.0125496, 0.0124981
4: -0.0118539, 0.0076348, -0.0119147, 0.0076911, -0.0195449, 0.0195495
5: 0.0038111, 0.0208011, 0.0037661, 0.0208561, -0.0170450, 0.0170350
6: -0.0276664, 0.0093674, -0.0278360, 0.0094415, -0.0371079, 0.0372034
7: 0.9557789, 0.9798637, 0.9556477, 0.9798821, -0.0241032, 0.0242160
8: -0.0284925, 0.0135781, -0.0285932, 0.0137036, -0.0421961, 0.0421713
9: -0.0118626, 0.0143895, -0.0119353, 0.0144632, -0.0263258, 0.0263248

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A1_A2_B1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155432, upper bound: 0.0156509
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155432, upper bound: 0.0156509
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0090405, 0.0029043, -0.0103679, 0.0033860, -0.0124265, 0.0132722
1: -0.0066904, 0.0001615, -0.0072163, 0.0007384, -0.0074289, 0.0073777
2: 0.0275551, 0.0490829, 0.0267856, 0.0521872, -0.0246321, 0.0222973
3: -0.0038253, 0.0070910, -0.0040016, 0.0085304, -0.0123557, 0.0110926
4: -0.0103999, 0.0062902, -0.0118918, 0.0076700, -0.0180699, 0.0181820
5: 0.0048865, 0.0194869, 0.0037830, 0.0208354, -0.0159489, 0.0157039
6: -0.0236104, 0.0075962, -0.0277723, 0.0094137, -0.0330241, 0.0353685
7: 0.9589138, 0.9794248, 0.9556970, 0.9798752, -0.0209614, 0.0237278
8: -0.0260879, 0.0105782, -0.0285554, 0.0136565, -0.0397444, 0.0391335
9: -0.0101261, 0.0126298, -0.0119080, 0.0144355, -0.0245616, 0.0245378

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A1_A2_B1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155541, upper bound: 0.0156430
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155541, upper bound: 0.0156430
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0103340, 0.0033737, -0.0103679, 0.0033860, -0.0137200, 0.0137416
1: -0.0072028, 0.0007237, -0.0072163, 0.0007384, -0.0079413, 0.0079400
2: 0.0268052, 0.0521082, 0.0267856, 0.0521872, -0.0253820, 0.0253225
3: -0.0039971, 0.0084938, -0.0040016, 0.0085304, -0.0125276, 0.0124954
4: -0.0118539, 0.0076348, -0.0118918, 0.0076700, -0.0195238, 0.0195267
5: 0.0038111, 0.0208011, 0.0037830, 0.0208354, -0.0170244, 0.0170181
6: -0.0276664, 0.0093674, -0.0277723, 0.0094137, -0.0370801, 0.0371397
7: 0.9557789, 0.9798637, 0.9556970, 0.9798752, -0.0240963, 0.0241667
8: -0.0284925, 0.0135781, -0.0285554, 0.0136565, -0.0421490, 0.0421335
9: -0.0118626, 0.0143895, -0.0119080, 0.0144355, -0.0262981, 0.0262975

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A1_A2_B1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155541, upper bound: 0.0156509
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155541, upper bound: 0.0156509
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0090405, 0.0029043, -0.0104876, 0.0034295, -0.0124700, 0.0133920
1: -0.0066904, 0.0001615, -0.0072637, 0.0007905, -0.0074809, 0.0074252
2: 0.0275551, 0.0490829, 0.0267162, 0.0524673, -0.0249122, 0.0223667
3: -0.0038253, 0.0070910, -0.0040175, 0.0086603, -0.0124856, 0.0111085
4: -0.0103999, 0.0062902, -0.0120265, 0.0077945, -0.0181944, 0.0183166
5: 0.0048865, 0.0194869, 0.0036834, 0.0209571, -0.0160706, 0.0158035
6: -0.0236104, 0.0075962, -0.0281479, 0.0095777, -0.0331881, 0.0357441
7: 0.9589138, 0.9794248, 0.9554067, 0.9799159, -0.0210021, 0.0240181
8: -0.0260879, 0.0105782, -0.0287781, 0.0139343, -0.0400222, 0.0393562
9: -0.0101261, 0.0126298, -0.0120688, 0.0145985, -0.0247246, 0.0246986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156132, upper bound: 0.0156428
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156132, upper bound: 0.0156428
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0103340, 0.0033737, -0.0104876, 0.0034295, -0.0137635, 0.0138614
1: -0.0072028, 0.0007237, -0.0072637, 0.0007905, -0.0079933, 0.0079874
2: 0.0268052, 0.0521082, 0.0267162, 0.0524673, -0.0256621, 0.0253920
3: -0.0039971, 0.0084938, -0.0040175, 0.0086603, -0.0126575, 0.0125113
4: -0.0118539, 0.0076348, -0.0120265, 0.0077945, -0.0196483, 0.0196613
5: 0.0038111, 0.0208011, 0.0036834, 0.0209571, -0.0171461, 0.0171177
6: -0.0276664, 0.0093674, -0.0281479, 0.0095777, -0.0372441, 0.0375153
7: 0.9557789, 0.9798637, 0.9554067, 0.9799159, -0.0241370, 0.0244570
8: -0.0284925, 0.0135781, -0.0287781, 0.0139343, -0.0424268, 0.0423561
9: -0.0118626, 0.0143895, -0.0120688, 0.0145985, -0.0264611, 0.0264583

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156132, upper bound: 0.0156508
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156132, upper bound: 0.0156508
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0090405, 0.0029043, -0.0104743, 0.0034246, -0.0124652, 0.0133786
1: -0.0066904, 0.0001615, -0.0072584, 0.0007847, -0.0074751, 0.0074199
2: 0.0275551, 0.0490829, 0.0267239, 0.0524361, -0.0248809, 0.0223590
3: -0.0038253, 0.0070910, -0.0040158, 0.0086459, -0.0124711, 0.0111067
4: -0.0103999, 0.0062902, -0.0120114, 0.0077806, -0.0181805, 0.0183016
5: 0.0048865, 0.0194869, 0.0036945, 0.0209435, -0.0160570, 0.0157924
6: -0.0236104, 0.0075962, -0.0281060, 0.0095594, -0.0331698, 0.0357022
7: 0.9589138, 0.9794248, 0.9554392, 0.9799113, -0.0209975, 0.0239856
8: -0.0260879, 0.0105782, -0.0287532, 0.0139033, -0.0399912, 0.0393314
9: -0.0101261, 0.0126298, -0.0120509, 0.0145803, -0.0247064, 0.0246806

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156151, upper bound: 0.0156428
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156151, upper bound: 0.0156428
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0103340, 0.0033737, -0.0104743, 0.0034246, -0.0137587, 0.0138480
1: -0.0072028, 0.0007237, -0.0072584, 0.0007847, -0.0079875, 0.0079821
2: 0.0268052, 0.0521082, 0.0267239, 0.0524361, -0.0256309, 0.0253842
3: -0.0039971, 0.0084938, -0.0040158, 0.0086459, -0.0126430, 0.0125096
4: -0.0118539, 0.0076348, -0.0120114, 0.0077806, -0.0196344, 0.0196462
5: 0.0038111, 0.0208011, 0.0036945, 0.0209435, -0.0171325, 0.0171066
6: -0.0276664, 0.0093674, -0.0281060, 0.0095594, -0.0372258, 0.0374734
7: 0.9557789, 0.9798637, 0.9554392, 0.9799113, -0.0241324, 0.0244246
8: -0.0284925, 0.0135781, -0.0287532, 0.0139033, -0.0423959, 0.0423313
9: -0.0118626, 0.0143895, -0.0120509, 0.0145803, -0.0264429, 0.0264404

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156151, upper bound: 0.0156508
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0156151, upper bound: 0.0156508
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0090996, 0.0029257, -0.0103289, 0.0033718, -0.0124714, 0.0132546
1: -0.0067138, 0.0001871, -0.0072008, 0.0007215, -0.0074353, 0.0073879
2: 0.0275209, 0.0492209, 0.0268082, 0.0520960, -0.0245751, 0.0224127
3: -0.0038331, 0.0071550, -0.0039964, 0.0084882, -0.0123213, 0.0111514
4: -0.0104662, 0.0063515, -0.0118480, 0.0076294, -0.0180957, 0.0181995
5: 0.0048374, 0.0195468, 0.0038154, 0.0207958, -0.0159584, 0.0157315
6: -0.0237954, 0.0076770, -0.0276501, 0.0093603, -0.0331557, 0.0353271
7: 0.9587709, 0.9794449, 0.9557915, 0.9798620, -0.0210910, 0.0236534
8: -0.0261976, 0.0107150, -0.0284829, 0.0135661, -0.0397637, 0.0391979
9: -0.0102054, 0.0127100, -0.0118557, 0.0143825, -0.0245878, 0.0245657

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0154236, upper bound: 0.0151975
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0153715, upper bound: 0.0151489
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0090996, 0.0029257, -0.0104667, 0.0034219, -0.0125214, 0.0133925
1: -0.0067138, 0.0001871, -0.0072554, 0.0007814, -0.0074952, 0.0074426
2: 0.0275209, 0.0492209, 0.0267283, 0.0524185, -0.0248976, 0.0224926
3: -0.0038331, 0.0071550, -0.0040148, 0.0086377, -0.0124708, 0.0111698
4: -0.0104662, 0.0063515, -0.0120030, 0.0077727, -0.0182390, 0.0183545
5: 0.0048374, 0.0195468, 0.0037008, 0.0209359, -0.0160984, 0.0158461
6: -0.0237954, 0.0076770, -0.0280824, 0.0095490, -0.0333444, 0.0357594
7: 0.9587709, 0.9794449, 0.9554574, 0.9799086, -0.0211377, 0.0239874
8: -0.0261976, 0.0107150, -0.0287392, 0.0138858, -0.0400834, 0.0394542
9: -0.0102054, 0.0127100, -0.0120408, 0.0145700, -0.0247754, 0.0247508

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0154236, upper bound: 0.0151996
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0153715, upper bound: 0.0151521
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0103971, 0.0033966, -0.0103289, 0.0033718, -0.0137689, 0.0137255
1: -0.0072278, 0.0007511, -0.0072008, 0.0007215, -0.0079493, 0.0079519
2: 0.0267687, 0.0522556, 0.0268082, 0.0520960, -0.0253274, 0.0254473
3: -0.0040055, 0.0085622, -0.0039964, 0.0084882, -0.0124937, 0.0125586
4: -0.0119247, 0.0077004, -0.0118480, 0.0076294, -0.0195541, 0.0195484
5: 0.0037587, 0.0208651, 0.0038154, 0.0207958, -0.0170372, 0.0170497
6: -0.0278640, 0.0094537, -0.0276501, 0.0093603, -0.0372243, 0.0371038
7: 0.9556262, 0.9798850, 0.9557915, 0.9798620, -0.0242358, 0.0240936
8: -0.0286097, 0.0137243, -0.0284829, 0.0135661, -0.0421758, 0.0422072
9: -0.0119473, 0.0144753, -0.0118557, 0.0143825, -0.0263298, 0.0263310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0152980, upper bound: 0.0151832
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0150802, upper bound: 0.0151038
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0103971, 0.0033966, -0.0104667, 0.0034219, -0.0138190, 0.0138633
1: -0.0072278, 0.0007511, -0.0072554, 0.0007814, -0.0080092, 0.0080065
2: 0.0267687, 0.0522556, 0.0267283, 0.0524185, -0.0256498, 0.0255273
3: -0.0040055, 0.0085622, -0.0040148, 0.0086377, -0.0126432, 0.0125769
4: -0.0119247, 0.0077004, -0.0120030, 0.0077727, -0.0196974, 0.0197033
5: 0.0037587, 0.0208651, 0.0037008, 0.0209359, -0.0171772, 0.0171643
6: -0.0278640, 0.0094537, -0.0280824, 0.0095490, -0.0374130, 0.0375361
7: 0.9556262, 0.9798850, 0.9554574, 0.9799086, -0.0242825, 0.0244276
8: -0.0286097, 0.0137243, -0.0287392, 0.0138858, -0.0424955, 0.0424635
9: -0.0119473, 0.0144753, -0.0120408, 0.0145700, -0.0265173, 0.0265160

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A2_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155269, upper bound: 0.0156745
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0155269, upper bound: 0.0156745
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0090996, 0.0029257, -0.0103703, 0.0033869, -0.0124864, 0.0132960
1: -0.0067138, 0.0001871, -0.0072172, 0.0007395, -0.0074533, 0.0074044
2: 0.0275209, 0.0492209, 0.0267842, 0.0521929, -0.0246720, 0.0224367
3: -0.0038331, 0.0071550, -0.0040019, 0.0085331, -0.0123662, 0.0111569
4: -0.0104662, 0.0063515, -0.0118946, 0.0076725, -0.0181387, 0.0182461
5: 0.0048374, 0.0195468, 0.0037810, 0.0208379, -0.0160005, 0.0157659
6: -0.0237954, 0.0076770, -0.0277800, 0.0094170, -0.0332124, 0.0354569
7: 0.9587709, 0.9794449, 0.9556912, 0.9798760, -0.0211051, 0.0237536
8: -0.0261976, 0.0107150, -0.0285599, 0.0136621, -0.0398597, 0.0392749
9: -0.0102054, 0.0127100, -0.0119113, 0.0144388, -0.0246442, 0.0246213

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0154475, upper bound: 0.0152067
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0153992, upper bound: 0.0151676
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0090996, 0.0029257, -0.0104772, 0.0034257, -0.0125252, 0.0134030
1: -0.0067138, 0.0001871, -0.0072596, 0.0007860, -0.0074998, 0.0074467
2: 0.0275209, 0.0492209, 0.0267222, 0.0524431, -0.0249222, 0.0224987
3: -0.0038331, 0.0071550, -0.0040162, 0.0086491, -0.0124822, 0.0111712
4: -0.0104662, 0.0063515, -0.0120148, 0.0077837, -0.0182499, 0.0183663
5: 0.0048374, 0.0195468, 0.0036920, 0.0209466, -0.0161091, 0.0158548
6: -0.0237954, 0.0076770, -0.0281153, 0.0095635, -0.0333588, 0.0357923
7: 0.9587709, 0.9794449, 0.9554319, 0.9799123, -0.0211414, 0.0240129
8: -0.0261976, 0.0107150, -0.0287587, 0.0139102, -0.0401078, 0.0394738
9: -0.0102054, 0.0127100, -0.0120549, 0.0145843, -0.0247897, 0.0247649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.29 + 596.80 = 600.09 seconds
