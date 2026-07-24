## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 11.027958876


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610)
1: (-5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247)
2: (-6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207)
3: (-7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554)
4: (-7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166)
5: (-6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819)
6: (-6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685)
7: (-7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842)
8: (-7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869)
9: (-6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.38 + 5.79 = 7.17 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -11.1393523, upper bound: 11.1393524

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1258847, upper bound: 11.1249726
time: 12.12 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1392413, upper bound: 11.1392413
time: 3.03 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 15.30 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 15.30
Output dim: 7, lower bound: -11.1258847, upper bound: 11.1249726
IS_B2, status: Status.UNKNOWN, split count: 1, time: 15.30
Output dim: 7, lower bound: -11.1392413, upper bound: 11.1392413

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -6.5408664, 5.5315504, -6.8031778, 5.7365303, -12.2773952, 12.3347282
1: -5.2291765, 4.7948642, -5.4268703, 4.9493256, -10.1785011, 10.2217350
2: -6.7434249, 4.4685254, -6.9864039, 4.5101366, -11.2535610, 11.4549294
3: -7.5769348, 3.9080787, -7.9464793, 3.9219804, -11.4989147, 11.8545570
4: -7.3340139, 5.7357998, -7.6412487, 6.0155678, -13.3495817, 13.3770485
5: -6.3652673, 5.0766630, -6.5650902, 5.2264585, -11.5917244, 11.6417522
6: -5.9306264, 6.3051825, -6.1009660, 6.6026506, -12.5332747, 12.4061489
7: -7.3453012, 4.7119579, -7.6735902, 4.6290693, -11.9743690, 12.3855476
8: -7.3520002, 5.3517675, -7.5839128, 5.5273619, -12.8793621, 12.9356804
9: -5.9191351, 6.0162024, -6.1305461, 6.2312937, -12.1504288, 12.1467485

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=40, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=25, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=246, inp2_unstable=238, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1225680, upper bound: 11.1213891
time: 3.95 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1234755, upper bound: 11.1220705
time: 4.63 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -6.7478456, 5.7016158, -6.6649265, 5.6336174, -12.3814621, 12.3665400
1: -5.4021134, 4.9384117, -5.3328838, 4.8809633, -10.2830772, 10.2712955
2: -6.9570780, 4.6066432, -6.8713760, 4.5503244, -11.5074024, 11.4780197
3: -7.8119092, 4.0386467, -7.7183523, 3.9860766, -11.7979860, 11.7569981
4: -7.5572267, 5.8959913, -7.4682083, 5.8314939, -13.3887196, 13.3641996
5: -6.5690393, 5.2315445, -6.4871206, 5.1695743, -11.7386122, 11.7186651
6: -6.1139588, 6.4829106, -6.0401793, 6.4118648, -12.5258236, 12.5230875
7: -7.5546770, 4.8859076, -7.4717345, 4.8148556, -12.3695316, 12.3576422
8: -7.5906458, 5.5113425, -7.4947567, 5.4467425, -13.0373878, 13.0060978
9: -6.1002841, 6.2026968, -6.0273323, 6.1278629, -12.2281466, 12.2300291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=40, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=31, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=246, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1324266, upper bound: 11.1324713
time: 4.81 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1309096, upper bound: 11.1309096
time: 2.89 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 9.12 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 9.12
Output dim: 7, lower bound: -11.1225680, upper bound: 11.1213891
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 9.12
Output dim: 7, lower bound: -11.1234755, upper bound: 11.1220705
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 9.12
Output dim: 7, lower bound: -11.1324266, upper bound: 11.1324713
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 9.12
Output dim: 7, lower bound: -11.1309096, upper bound: 11.1309096

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -4.5744228, 3.8640525, -6.2507482, 5.2790480, -9.8534708, 10.1148005
1: -3.5252075, 3.4119380, -4.9510698, 4.5656743, -8.0908813, 8.3630075
2: -4.6719408, 3.1021495, -6.4114866, 4.1304226, -8.8023634, 9.5136356
3: -5.2895856, 2.7272158, -7.3036981, 3.6017828, -8.8913689, 10.0309124
4: -5.1002498, 4.1211710, -7.0220742, 5.5625143, -10.6627636, 11.1432457
5: -4.3063946, 3.5411646, -5.9953432, 4.7903504, -9.0967445, 9.5365076
6: -4.1372309, 4.5380106, -5.6040087, 6.1073976, -10.2446289, 10.1420193
7: -5.2028780, 2.9970264, -7.0791755, 4.1575947, -9.3604708, 10.0762024
8: -5.1302977, 3.7843778, -6.9752588, 5.0848684, -10.2151661, 10.7596369
9: -4.1421614, 4.1759424, -5.6395855, 5.7202034, -9.8623648, 9.8155270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=69, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=22, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=235, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1225066, upper bound: 11.1212255
time: 4.53 seconds

## Relational analysis of IS_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1225066, upper bound: 11.1213891
time: 3.93 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -5.5281343, 4.6863813, -6.6220293, 5.5887957, -11.1169281, 11.3084106
1: -4.3680534, 4.0923243, -5.2712431, 4.8230157, -9.1910686, 9.3635645
2: -5.7016325, 3.7813463, -6.7997065, 4.3863392, -10.0879717, 10.5810509
3: -6.4184365, 3.3160992, -7.7338290, 3.8175335, -10.2359695, 11.0499287
4: -6.2030582, 4.9158473, -7.4371614, 5.8665614, -12.0696201, 12.3530064
5: -5.3431625, 4.2918596, -6.3804317, 5.0872831, -10.4304457, 10.6722908
6: -5.0285292, 5.4121289, -5.9373446, 6.4411411, -11.4696693, 11.3494740
7: -6.2777100, 3.8834634, -7.4797716, 4.4821267, -10.7598362, 11.3632355
8: -6.2183871, 4.5599732, -7.3853812, 5.3836389, -11.6020241, 11.9453545
9: -5.0229177, 5.0965190, -5.9711208, 6.0651298, -11.0880470, 11.0676384

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=235, inp2_unstable=236, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1229936, upper bound: 11.1216162
time: 3.91 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1229936, upper bound: 11.1220705
time: 3.76 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -6.4082499, 5.4196887, -6.6649265, 5.6336174, -12.0418663, 12.0846148
1: -5.1161637, 4.7054176, -5.3328838, 4.8809633, -9.9971256, 10.0383015
2: -6.6072059, 4.3798723, -6.8713760, 4.5503244, -11.1575298, 11.2512484
3: -7.4170475, 3.8383527, -7.7183523, 3.9860766, -11.4031239, 11.5567055
4: -7.1759787, 5.6193500, -7.4682083, 5.8314939, -13.0074730, 13.0875587
5: -6.2298455, 4.9709430, -6.4871206, 5.1695743, -11.3994198, 11.4580631
6: -5.8129530, 6.1855145, -6.0401793, 6.4118648, -12.2248173, 12.2256937
7: -7.2034159, 4.6160998, -7.4717345, 4.8148556, -12.0182695, 12.0878334
8: -7.2060909, 5.2446194, -7.4947567, 5.4467425, -12.6528330, 12.7393761
9: -5.7991829, 5.8931599, -6.0273323, 6.1278629, -11.9270458, 11.9204922

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=69, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=246, inp2_unstable=246, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1276412, upper bound: 11.1272514
time: 3.85 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1308100, upper bound: 11.1308075
time: 4.12 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -5.2869511, 4.4547110, -6.2145462, 5.2596622, -10.5466137, 10.6692562
1: -4.1359158, 3.9093082, -4.9531746, 4.5724859, -8.7084017, 8.8624821
2: -5.4387102, 3.5314331, -6.4084034, 4.2483110, -9.6870213, 9.9398365
3: -6.1459117, 3.1254158, -7.1994381, 3.7177644, -9.8636751, 10.3248539
4: -5.8773479, 4.7190485, -6.9666481, 5.4677544, -11.3451023, 11.6856956
5: -5.0242925, 4.0595641, -6.0376511, 4.8225303, -9.8468227, 10.0972157
6: -4.7705789, 5.2180290, -5.6410384, 6.0192318, -10.7898102, 10.8590641
7: -6.0054502, 3.5436971, -7.0066876, 4.4514723, -10.4569225, 10.5503845
8: -5.9268217, 4.3456569, -6.9832215, 5.0948181, -11.0216398, 11.3288784
9: -4.7787194, 4.8431206, -5.6283226, 5.7189217, -10.4976397, 10.4714432

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=82, inp2_unstable=69, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=29, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=232, inp2_unstable=242, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1262215, upper bound: 11.1258713
time: 3.35 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1292490, upper bound: 11.1292490
time: 3.45 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 8.26 seconds
IS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 8.26
Output dim: 7, lower bound: -11.1225066, upper bound: 11.1212255
IS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 8.26
Output dim: 7, lower bound: -11.1225066, upper bound: 11.1213891
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 8.26
Output dim: 7, lower bound: -11.1229936, upper bound: 11.1216162
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 8.26
Output dim: 7, lower bound: -11.1229936, upper bound: 11.1220705
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 8.26
Output dim: 7, lower bound: -11.1276412, upper bound: 11.1272514
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 8.26
Output dim: 7, lower bound: -11.1308100, upper bound: 11.1308075
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 8.26
Output dim: 7, lower bound: -11.1262215, upper bound: 11.1258713
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 8.26
Output dim: 7, lower bound: -11.1292490, upper bound: 11.1292490

## BFS IS instance: IS_B1_A1_B1

### Backsubstitution after applying IS history:
0: -4.5744228, 3.8640525, -5.1864638, 4.3578529, -8.9322758, 9.0505161
1: -3.5252075, 3.4119380, -4.0422354, 3.8235800, -7.3487873, 7.4541731
2: -4.6719408, 3.1021495, -5.2643023, 3.3974533, -8.0693941, 8.3664513
3: -5.2895856, 2.7272158, -6.0598745, 2.9941173, -8.2837029, 8.7870903
4: -5.1002498, 4.1211710, -5.7920265, 4.6691141, -9.7693634, 9.9131975
5: -4.3063946, 3.5411646, -4.8655477, 3.9480324, -8.2544270, 8.4067125
6: -4.1372309, 4.5380106, -4.6559863, 5.1183386, -9.2555695, 9.1939964
7: -5.2028780, 2.9970264, -5.8829145, 3.2477770, -8.4506550, 8.8799410
8: -5.1302977, 3.7843778, -5.7833338, 4.2279506, -9.3582478, 9.5677118
9: -4.1421614, 4.1759424, -4.6677103, 4.6918378, -8.8339996, 8.8436527

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=39, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=69, inp2_unstable=77, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=226, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 107

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0913966, upper bound: 11.0317570
time: 4.35 seconds

## Relational analysis of IS_B1_A1_B1_A2

### Relational analysis result of IS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1225066, upper bound: 11.1212255
time: 3.64 seconds

## BFS IS instance: IS_B1_A1_B2

### Backsubstitution after applying IS history:
0: -4.5744228, 3.8640525, -5.9462008, 5.0301385, -9.6045609, 9.8102522
1: -3.5252075, 3.4119380, -4.6917210, 4.3515415, -7.8767490, 8.1036587
2: -4.6719408, 3.1021495, -6.0979252, 3.9255395, -8.5974808, 9.2000751
3: -5.2895856, 2.7272158, -6.9426851, 3.4317157, -8.7213011, 9.6698980
4: -5.1002498, 4.1211710, -6.6729393, 5.3072672, -10.4075165, 10.7941103
5: -4.3063946, 3.5411646, -5.6887264, 4.5628657, -8.8692608, 9.2298908
6: -4.1372309, 4.5380106, -5.3339663, 5.8307419, -9.9679728, 9.8719769
7: -5.2028780, 2.9970264, -6.7502947, 3.9317126, -9.1345882, 9.7473211
8: -5.1302977, 3.7843778, -6.6413298, 4.8465104, -9.9768085, 10.4257050
9: -4.1421614, 4.1759424, -5.3739576, 5.4398603, -9.5820217, 9.5499001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=39, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=69, inp2_unstable=76, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=20, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=234, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_B1_A1_B2_A1

### Relational analysis result of IS_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0913966, upper bound: 11.0327793
time: 3.48 seconds

## Relational analysis of IS_B1_A1_B2_A2

### Relational analysis result of IS_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1225066, upper bound: 11.1213891
time: 2.94 seconds

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -5.5281343, 4.6863813, -5.1864638, 4.3578529, -9.8859854, 9.8728447
1: -4.3680534, 4.0923243, -4.0422354, 3.8235800, -8.1916304, 8.1345596
2: -5.7016325, 3.7813463, -5.2643023, 3.3974533, -9.0990858, 9.0456486
3: -6.4184365, 3.3160992, -6.0598745, 2.9941173, -9.4125538, 9.3759737
4: -6.2030582, 4.9158473, -5.7920265, 4.6691141, -10.8721724, 10.7078724
5: -5.3431625, 4.2918596, -4.8655477, 3.9480324, -9.2911949, 9.1574068
6: -5.0285292, 5.4121289, -4.6559863, 5.1183386, -10.1468668, 10.0681152
7: -6.2777100, 3.8834634, -5.8829145, 3.2477770, -9.5254869, 9.7663774
8: -6.2183871, 4.5599732, -5.7833338, 4.2279506, -10.4463367, 10.3433075
9: -5.0229177, 5.0965190, -4.6677103, 4.6918378, -9.7147551, 9.7642288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=39, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=77, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=235, inp2_unstable=226, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 190

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1193433, upper bound: 11.0869943
time: 3.25 seconds

## Relational analysis of IS_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1193780, upper bound: 11.0869746
time: 3.46 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5.5281343, 4.6863813, -5.9462008, 5.0301385, -10.5582705, 10.6325817
1: -4.3680534, 4.0923243, -4.6917210, 4.3515415, -8.7195930, 8.7840433
2: -5.7016325, 3.7813463, -6.0979252, 3.9255395, -9.6271725, 9.8792706
3: -6.4184365, 3.3160992, -6.9426851, 3.4317157, -9.8501520, 10.2587843
4: -6.2030582, 4.9158473, -6.6729393, 5.3072672, -11.5103254, 11.5887861
5: -5.3431625, 4.2918596, -5.6887264, 4.5628657, -9.9060287, 9.9805851
6: -5.0285292, 5.4121289, -5.3339663, 5.8307419, -10.8592710, 10.7460938
7: -6.2777100, 3.8834634, -6.7502947, 3.9317126, -10.2094231, 10.6337585
8: -6.2183871, 4.5599732, -6.6413298, 4.8465104, -11.0648975, 11.2013035
9: -5.0229177, 5.0965190, -5.3739576, 5.4398603, -10.4627762, 10.4704762

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=39, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=76, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=20, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=235, inp2_unstable=234, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 62

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1074207, upper bound: 11.0375426
time: 3.37 seconds

## Relational analysis of IS_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1229936, upper bound: 11.1216162
time: 17.05 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5.7361183, 4.8533425, -4.6506619, 3.9273903, -9.6635084, 9.5040045
1: -4.5460401, 4.2412977, -3.5918355, 3.4686661, -8.0147047, 7.8331332
2: -5.9129109, 3.9205291, -4.7561903, 3.1554313, -9.0683422, 8.6767197
3: -6.6575689, 3.4389141, -5.3846388, 2.7772379, -9.4348049, 8.8235531
4: -6.4318986, 5.0808926, -5.1896172, 4.1842041, -10.6161022, 10.2705078
5: -5.5463138, 4.4385128, -4.3880510, 3.5961809, -9.1424942, 8.8265638
6: -5.2169018, 5.5958843, -4.2093425, 4.6101899, -9.8270912, 9.8052273
7: -6.4992638, 4.0417275, -5.2906771, 3.0606380, -9.5599022, 9.3324051
8: -6.4490533, 4.7176037, -5.2177114, 3.8465910, -10.2956419, 9.9353151
9: -5.1999531, 5.2824326, -4.2104707, 4.2491322, -9.4490852, 9.4929028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=39, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=69, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=18, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=238, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 62

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1271331, upper bound: 11.1269069
time: 3.52 seconds

## Relational analysis of IS_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1271331, upper bound: 11.1272514
time: 3.90 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6.1834564, 5.2336950, -5.6321340, 4.7739940, -10.9574490, 10.8658295
1: -4.9278798, 4.5514851, -4.4582376, 4.1670780, -9.0949554, 9.0097227
2: -6.3775539, 4.2291851, -5.8096819, 3.8521299, -10.2296839, 10.0388670
3: -7.1649151, 3.7052929, -6.5404949, 3.3813059, -10.5462208, 10.2457876
4: -6.9289365, 5.4374080, -6.3185225, 4.9957252, -11.9246616, 11.7559299
5: -6.0066900, 4.7957606, -5.4498758, 4.3682470, -10.3749371, 10.2456360
6: -5.6159530, 5.9896526, -5.1249552, 5.5032926, -11.1192446, 11.1146078
7: -6.9719386, 4.4332876, -6.3899922, 3.9725924, -10.9445305, 10.8232803
8: -6.9527240, 5.0687447, -6.3383703, 4.6399903, -11.5927143, 11.4071150
9: -5.6000385, 5.6910501, -5.1139555, 5.1916995, -10.7917385, 10.8050060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=39, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=69, inp2_unstable=68, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=26, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=242, inp2_unstable=237, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1283849, upper bound: 11.1286259
time: 3.45 seconds

## Relational analysis of IS_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1283849, upper bound: 11.1308074
time: 3.48 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.8096061, 4.0427065, -4.3444147, 3.6630158, -8.4726219, 8.3871212
1: -3.7159479, 3.5654666, -3.3237481, 3.2465346, -6.9624825, 6.8892145
2: -4.9107666, 3.2039785, -4.4240656, 2.9369316, -7.8476954, 7.6280441
3: -5.5733905, 2.8373909, -5.0061455, 2.5933790, -8.1667690, 7.8435364
4: -5.3228307, 4.3135777, -4.8212271, 3.9263148, -9.2491455, 9.1348038
5: -4.5113540, 3.6820767, -4.0586762, 3.3646750, -7.8760290, 7.7407513
6: -4.3267069, 4.7709532, -3.9242003, 4.3213425, -8.6480494, 8.6951532
7: -5.4571705, 3.1074793, -4.9383984, 2.8050406, -8.2622108, 8.0458775
8: -5.3873525, 3.9582891, -4.8682251, 3.6020455, -8.9893970, 8.8265142
9: -4.3369613, 4.3820362, -3.9307845, 3.9556148, -8.2925749, 8.3128204

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=39, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=81, inp2_unstable=69, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1250220, upper bound: 11.1250220
time: 2.33 seconds

## Relational analysis of IS_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1250220, upper bound: 11.1258713
time: 2.34 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5.1298699, 4.3214297, -5.2550688, 4.4531546, -9.5830231, 9.5764980
1: -3.9969282, 3.7944558, -4.1278057, 3.8999507, -7.8968782, 7.9222612
2: -5.2671633, 3.4259973, -5.4177375, 3.5898154, -8.8569775, 8.8437347
3: -5.9567332, 3.0314963, -6.0902853, 3.1582687, -9.1150017, 9.1217804
4: -5.6948948, 4.5846710, -5.8840542, 4.6870537, -10.3819485, 10.4687252
5: -4.8580666, 3.9385295, -5.0547428, 4.0843000, -8.9423656, 8.9932709
6: -4.6244974, 5.0709348, -4.7750130, 5.1669002, -9.7913961, 9.8459473
7: -5.8256841, 3.4092956, -5.9776363, 3.6566114, -9.4822960, 9.3869324
8: -5.7508888, 4.2178221, -5.9084744, 4.3449755, -10.0958614, 10.1262960
9: -4.6351109, 4.6919498, -4.7770643, 4.8386016, -9.4737110, 9.4690142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=39, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=82, inp2_unstable=68, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=23, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=231, inp2_unstable=234, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1258713, upper bound: 11.1262215
time: 3.21 seconds

## Relational analysis of IS_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1258713, upper bound: 11.1292490
time: 3.73 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 18.90 seconds
IS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 7, lower bound: -11.0913966, upper bound: 11.0317570
IS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 7, lower bound: -11.1225066, upper bound: 11.1212255
IS_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 7, lower bound: -11.0913966, upper bound: 11.0327793
IS_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 7, lower bound: -11.1225066, upper bound: 11.1213891
IS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 7, lower bound: -11.1193433, upper bound: 11.0869943
IS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 7, lower bound: -11.1193780, upper bound: 11.0869746
IS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 7, lower bound: -11.1074207, upper bound: 11.0375426
IS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 7, lower bound: -11.1229936, upper bound: 11.1216162
IS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 7, lower bound: -11.1271331, upper bound: 11.1269069
IS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 7, lower bound: -11.1271331, upper bound: 11.1272514
IS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 7, lower bound: -11.1283849, upper bound: 11.1286259
IS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 7, lower bound: -11.1283849, upper bound: 11.1308074
IS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 7, lower bound: -11.1250220, upper bound: 11.1250220
IS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 7, lower bound: -11.1250220, upper bound: 11.1258713
IS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 7, lower bound: -11.1258713, upper bound: 11.1262215
IS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 7, lower bound: -11.1258713, upper bound: 11.1292490

## BFS IS instance: IS_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.6957765, 3.9242623, -4.9910803, 4.1865163, -8.8822927, 8.9153423
1: -3.6004856, 3.4749079, -3.8721504, 3.6833088, -7.2837944, 7.3470583
2: -4.7387257, 3.0926251, -5.0500469, 3.2704675, -8.0091934, 8.1426716
3: -5.4384098, 2.7352300, -5.8131685, 2.8859425, -8.3243523, 8.5483990
4: -5.1805401, 4.2105713, -5.5563722, 4.4989576, -9.6794977, 9.7669430
5: -4.3308239, 3.5488086, -4.6531296, 3.7965858, -8.1274099, 8.2019386
6: -4.2085271, 4.6330347, -4.4796081, 4.9300327, -9.1385593, 9.1126423
7: -5.2811546, 2.8557394, -5.6546888, 3.0932393, -8.3743935, 8.5104284
8: -5.2385731, 3.8500896, -5.5587187, 4.0756321, -9.3142052, 9.4088078
9: -4.1920543, 4.2246346, -4.4857845, 4.5032077, -8.6952620, 8.7104187

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=77, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=225, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 195

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_B1_A1_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9076481, upper bound: 10.8803418
time: 3.10 seconds

## Relational analysis of IS_B1_A1_B1_A1_B2

### Relational analysis result of IS_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.8994532, upper bound: 10.8733505
time: 3.75 seconds

## BFS IS instance: IS_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.4072294, 3.7170155, -5.1864638, 4.3578529, -8.7650824, 8.9034796
1: -3.3769495, 3.2878823, -4.0422354, 3.8235800, -7.2005296, 7.3301177
2: -4.4849176, 2.9860685, -5.2643023, 3.3974533, -7.8823709, 8.2503710
3: -5.0772867, 2.6246707, -6.0598745, 2.9941173, -8.0714035, 8.6845455
4: -4.8955069, 3.9768665, -5.7920265, 4.6691141, -9.5646210, 9.7688932
5: -4.1209269, 3.4117231, -4.8655477, 3.9480324, -8.0689592, 8.2772713
6: -3.9807222, 4.3750391, -4.6559863, 5.1183386, -9.0990610, 9.0310249
7: -5.0045075, 2.8537393, -5.8829145, 3.2477770, -8.2522850, 8.7366543
8: -4.9366379, 3.6500447, -5.7833338, 4.2279506, -9.1645889, 9.4333782
9: -3.9865377, 4.0115771, -4.6677103, 4.6918378, -8.6783752, 8.6792870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=77, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=227, inp2_unstable=226, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 195

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_B1_A1_B1_A2_A1

### Relational analysis result of IS_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1168698, upper bound: 11.0060424
time: 4.50 seconds

## Relational analysis of IS_B1_A1_B1_A2_A2

### Relational analysis result of IS_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1164730, upper bound: 11.0057913
time: 3.23 seconds

## BFS IS instance: IS_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.6957765, 3.9242623, -5.7106733, 4.8304787, -9.5262547, 9.6349354
1: -3.6004856, 3.4749079, -4.4902420, 4.1892529, -7.7897387, 7.9651475
2: -4.7387257, 3.0926251, -5.8485131, 3.7705231, -8.5092487, 8.9411383
3: -5.4384098, 2.7352300, -6.6568880, 3.3013361, -8.7397461, 9.3921185
4: -5.1805401, 4.2105713, -6.4034963, 5.1084132, -10.2889538, 10.6140671
5: -4.3308239, 3.5488086, -5.4422760, 4.3800688, -8.7108908, 8.9910851
6: -4.2085271, 4.6330347, -5.1251931, 5.6120968, -9.8206234, 9.7582245
7: -5.2811546, 2.8557394, -6.4884343, 3.7428422, -9.0239964, 9.3441734
8: -5.2385731, 3.8500896, -6.3776250, 4.6627674, -9.9013405, 10.2277145
9: -4.1920543, 4.2246346, -5.1638660, 5.2168851, -9.4089394, 9.3885002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=19, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=233, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_B1_A1_B2_A1_B1

### Relational analysis result of IS_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9459074, upper bound: 10.9110006
time: 4.25 seconds

## Relational analysis of IS_B1_A1_B2_A1_B2

### Relational analysis result of IS_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9428805, upper bound: 10.9048650
time: 2.66 seconds

## BFS IS instance: IS_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.4072294, 3.7170155, -5.9462008, 5.0301385, -9.4373684, 9.6632128
1: -3.3769495, 3.2878823, -4.6917210, 4.3515415, -7.7284908, 7.9796019
2: -4.4849176, 2.9860685, -6.0979252, 3.9255395, -8.4104576, 9.0839930
3: -5.0772867, 2.6246707, -6.9426851, 3.4317157, -8.5090027, 9.5673542
4: -4.8955069, 3.9768665, -6.6729393, 5.3072672, -10.2027740, 10.6498060
5: -4.1209269, 3.4117231, -5.6887264, 4.5628657, -8.6837921, 9.1004496
6: -3.9807222, 4.3750391, -5.3339663, 5.8307419, -9.8114643, 9.7090034
7: -5.0045075, 2.8537393, -6.7502947, 3.9317126, -8.9362202, 9.6040344
8: -4.9366379, 3.6500447, -6.6413298, 4.8465104, -9.7831478, 10.2913742
9: -3.9865377, 4.0115771, -5.3739576, 5.4398603, -9.4263983, 9.3855343

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=76, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=20, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=227, inp2_unstable=234, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_B1_A1_B2_A2_B1

### Relational analysis result of IS_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1187829, upper bound: 11.1029968
time: 12.45 seconds

## Relational analysis of IS_B1_A1_B2_A2_B2

### Relational analysis result of IS_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1188033, upper bound: 11.0941458
time: 3.13 seconds

## BFS IS instance: IS_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -4.2643251, 3.6176028, -4.9442382, 4.1477203, -8.4120445, 8.5618410
1: -3.2729497, 3.1902621, -3.8339763, 3.6507661, -6.9237156, 7.0242386
2: -4.3465343, 2.9136219, -5.0002880, 3.2426953, -7.5892296, 7.9139099
3: -4.8863459, 2.5611954, -5.7538104, 2.8608918, -7.7472377, 8.3150063
4: -4.7385626, 3.8574886, -5.5037842, 4.4599276, -9.1984901, 9.3612728
5: -3.9810562, 3.3332303, -4.6010432, 3.7626524, -7.7437086, 7.9342718
6: -3.8573647, 4.2425499, -4.4400868, 4.8867931, -8.7441578, 8.6826363
7: -4.8548388, 2.7836242, -5.6022453, 3.0568135, -7.9116526, 8.3858700
8: -4.7798905, 3.5581045, -5.5068407, 4.0415030, -8.8213921, 9.0649452
9: -3.8749232, 3.8916588, -4.4435339, 4.4605536, -8.3354769, 8.3351927

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=77, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=226, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 107

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_B1_A2_B1_A1_A1

### Relational analysis result of IS_B1_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -11.0135655, upper bound: 10.9362424
time: 4.06 seconds

## Relational analysis of IS_B1_A2_B1_A1_A2

### Relational analysis result of IS_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1193433, upper bound: 11.0869943
time: 3.51 seconds

## BFS IS instance: IS_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -4.8042078, 4.0689430, -4.8386889, 4.0582848, -8.8624926, 8.9076319
1: -3.7387209, 3.5740337, -3.7442331, 3.5751555, -7.3138762, 7.3182669
2: -4.9197574, 3.2469783, -4.8879538, 3.1801243, -8.0998821, 8.1349316
3: -5.5713234, 2.8543987, -5.6208000, 2.8043032, -8.3756266, 8.4751987
4: -5.3726263, 4.3245196, -5.3806238, 4.3700671, -9.7426929, 9.7051430
5: -4.5353093, 3.7222204, -4.4956808, 3.6870666, -8.2223759, 8.2179012
6: -4.3413472, 4.7579288, -4.3474541, 4.7861338, -9.1274805, 9.1053829
7: -5.4715266, 3.1585081, -5.4824762, 2.9912453, -8.4627724, 8.6409845
8: -5.3923268, 3.9723761, -5.3890414, 3.9604700, -9.3527946, 9.3614178
9: -4.3584714, 4.3918290, -4.3509970, 4.3637633, -8.7222347, 8.7428265

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=76, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_B1_A2_B1_A2_A1

### Relational analysis result of IS_B1_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9597565, upper bound: 10.9017391
time: 4.04 seconds

## Relational analysis of IS_B1_A2_B1_A2_A2

### Relational analysis result of IS_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1193779, upper bound: 11.0869746
time: 3.97 seconds

## BFS IS instance: IS_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5.7513757, 4.8261790, -5.7106733, 4.8304787, -10.5818539, 10.5368519
1: -4.5284433, 4.2284870, -4.4902420, 4.1892529, -8.7176952, 8.7187271
2: -5.8862128, 3.8242960, -5.8485131, 3.7705231, -9.6567335, 9.6728067
3: -6.6836076, 3.3746753, -6.6568880, 3.3013361, -9.9849434, 10.0315628
4: -6.3953648, 5.0885973, -6.4034963, 5.1084132, -11.5037785, 11.4920940
5: -5.4553452, 4.3688307, -5.4422760, 4.3800688, -9.8354130, 9.8111067
6: -5.1739559, 5.6166935, -5.1251931, 5.6120968, -10.7860527, 10.7418842
7: -6.4849014, 3.8154230, -6.4884343, 3.7428422, -10.2277431, 10.3038578
8: -6.4363947, 4.7033520, -6.3776250, 4.6627674, -11.0991602, 11.0809765
9: -5.1632404, 5.2408319, -5.1638660, 5.2168851, -10.3801250, 10.4046974

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=73, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=19, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=234, inp2_unstable=233, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 195

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_B1_A2_B2_A1_B1

### Relational analysis result of IS_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9315759, upper bound: 10.8938927
time: 3.71 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2

### Relational analysis result of IS_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9232114, upper bound: 10.8852125
time: 3.07 seconds

## BFS IS instance: IS_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -5.3496671, 4.5350680, -5.9462008, 5.0301385, -10.3798056, 10.4812651
1: -4.2112093, 3.9647894, -4.6917210, 4.3515415, -8.5627508, 8.6565104
2: -5.5114331, 3.6602042, -6.0979252, 3.9255395, -9.4369717, 9.7581291
3: -6.1953759, 3.2098646, -6.9426851, 3.4317157, -9.6270895, 10.1525497
4: -5.9929481, 4.7649536, -6.6729393, 5.3072672, -11.3002148, 11.4378920
5: -5.1525245, 4.1538281, -5.6887264, 4.5628657, -9.7153902, 9.8425541
6: -4.8635740, 5.2468381, -5.3339663, 5.8307419, -10.6943159, 10.5808048
7: -6.0765209, 3.7344501, -6.7502947, 3.9317126, -10.0082340, 10.4847450
8: -6.0151997, 4.4192753, -6.6413298, 4.8465104, -10.8617096, 11.0606031
9: -4.8619461, 4.9255872, -5.3739576, 5.4398603, -10.3018064, 10.2995453

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=76, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=20, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=234, inp2_unstable=234, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 62

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_B1_A2_B2_A2_A1

### Relational analysis result of IS_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1175657, upper bound: 11.0082028
time: 3.19 seconds

## Relational analysis of IS_B1_A2_B2_A2_A2

### Relational analysis result of IS_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1172932, upper bound: 11.0080611
time: 4.00 seconds

## BFS IS instance: IS_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.4639344, 3.7627633, -4.6506619, 3.9273903, -8.3913250, 8.4134254
1: -3.4279914, 3.3329320, -3.5918355, 3.4686661, -6.8966575, 6.9247675
2: -4.5545049, 3.0240276, -4.7561903, 3.1554313, -7.7099361, 7.7802181
3: -5.1531153, 2.6677372, -5.3846388, 2.7772379, -7.9303532, 8.0523758
4: -4.9622030, 4.0243893, -5.1896172, 4.1842041, -9.1464071, 9.2140064
5: -4.1852784, 3.4526145, -4.3880510, 3.5961809, -7.7814593, 7.8406658
6: -4.0353599, 4.4334331, -4.2093425, 4.6101899, -8.6455498, 8.6427755
7: -5.0743918, 2.9091270, -5.2906771, 3.0606380, -8.1350298, 8.1998043
8: -5.0045600, 3.6973698, -5.2177114, 3.8465910, -8.8511505, 8.9150810
9: -4.0385876, 4.0698185, -4.2104707, 4.2491322, -8.2877197, 8.2802887

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=69, inp2_unstable=69, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=227, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_B2_A1_B1_A1_A1

### Relational analysis result of IS_B2_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -11.0039508, upper bound: 10.9813865
time: 3.42 seconds

## Relational analysis of IS_B2_A1_B1_A1_A2

### Relational analysis result of IS_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1271221, upper bound: 11.1268908
time: 3.57 seconds

## BFS IS instance: IS_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5.4039345, 4.5828538, -4.6506619, 3.9273903, -9.3313246, 9.2335157
1: -4.2620926, 4.0085993, -3.5918355, 3.4686661, -7.7307577, 7.6004348
2: -5.5749097, 3.6982975, -4.7561903, 3.1554313, -8.7303400, 8.4544878
3: -6.2691259, 3.2524133, -5.3846388, 2.7772379, -9.0463619, 8.6370525
4: -6.0545640, 4.8072472, -5.1896172, 4.1842041, -10.2387676, 9.9968643
5: -5.2149382, 4.1953735, -4.3880510, 3.5961809, -8.8111191, 8.5834236
6: -4.9176188, 5.3008947, -4.2093425, 4.6101899, -9.5278091, 9.5102367
7: -6.1430016, 3.7921386, -5.2906771, 3.0606380, -9.2036400, 9.0828142
8: -6.0835600, 4.4624348, -5.2177114, 3.8465910, -9.9301500, 9.6801462
9: -4.9121599, 4.9810410, -4.2104707, 4.2491322, -9.1612921, 9.1915112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=69, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=18, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=235, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 62

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_B2_A1_B1_A2_A1

### Relational analysis result of IS_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1245937, upper bound: 11.1245630
time: 3.87 seconds

## Relational analysis of IS_B2_A1_B1_A2_A2

### Relational analysis result of IS_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1248072, upper bound: 11.1246574
time: 3.64 seconds

## BFS IS instance: IS_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.4639344, 3.7627633, -5.6321340, 4.7739940, -9.2379284, 9.3948975
1: -3.4279914, 3.3329320, -4.4582376, 4.1670780, -7.5950694, 7.7911692
2: -4.5545049, 3.0240276, -5.8096819, 3.8521299, -8.4066353, 8.8337078
3: -5.1531153, 2.6677372, -6.5404949, 3.3813059, -8.5344210, 9.2082310
4: -4.9622030, 4.0243893, -6.3185225, 4.9957252, -9.9579277, 10.3429117
5: -4.1852784, 3.4526145, -5.4498758, 4.3682470, -8.5535259, 8.9024906
6: -4.0353599, 4.4334331, -5.1249552, 5.5032926, -9.5386524, 9.5583878
7: -5.0743918, 2.9091270, -6.3899922, 3.9725924, -9.0469837, 9.2991190
8: -5.0045600, 3.6973698, -6.3383703, 4.6399903, -9.6445503, 10.0357399
9: -4.0385876, 4.0698185, -5.1139555, 5.1916995, -9.2302876, 9.1837740

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=69, inp2_unstable=68, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=26, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=227, inp2_unstable=237, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_B2_A1_B2_A1_B1

### Relational analysis result of IS_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1243888, upper bound: 11.1258433
time: 3.60 seconds

## Relational analysis of IS_B2_A1_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1248072, upper bound: 11.1262509
time: 4.19 seconds

## BFS IS instance: IS_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5.4039345, 4.5828538, -5.6321340, 4.7739940, -10.1779270, 10.2149878
1: -4.2620926, 4.0085993, -4.4582376, 4.1670780, -8.4291706, 8.4668360
2: -5.5749097, 3.6982975, -5.8096819, 3.8521299, -9.4270391, 9.5079784
3: -6.2691259, 3.2524133, -6.5404949, 3.3813059, -9.6504316, 9.7929068
4: -6.0545640, 4.8072472, -6.3185225, 4.9957252, -11.0502872, 11.1257696
5: -5.2149382, 4.1953735, -5.4498758, 4.3682470, -9.5831852, 9.6452494
6: -4.9176188, 5.3008947, -5.1249552, 5.5032926, -10.4209108, 10.4258490
7: -6.1430016, 3.7921386, -6.3899922, 3.9725924, -10.1155939, 10.1821289
8: -6.0835600, 4.4624348, -6.3383703, 4.6399903, -10.7235508, 10.8008041
9: -4.9121599, 4.9810410, -5.1139555, 5.1916995, -10.1038589, 10.0949965

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=68, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=235, inp2_unstable=237, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 107

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_B2_A1_B2_A2_B1

### Relational analysis result of IS_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1261489, upper bound: 11.1299233
time: 3.55 seconds

## Relational analysis of IS_B2_A1_B2_A2_B2

### Relational analysis result of IS_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1261489, upper bound: 11.1299875
time: 4.32 seconds

## BFS IS instance: IS_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -3.8584981, 3.2181478, -4.3444147, 3.6630158, -7.5215139, 7.5625625
1: -2.8772225, 2.8652768, -3.3237481, 3.2465346, -6.1237574, 6.1890249
2: -3.8263783, 2.5537817, -4.4240656, 2.9369316, -6.7633100, 6.9778471
3: -4.3763485, 2.2742977, -5.0061455, 2.5933790, -6.9697275, 7.2804432
4: -4.1753302, 3.4981897, -4.8212271, 3.9263148, -8.1016445, 8.3194170
5: -3.5086632, 2.9527082, -4.0586762, 3.3646750, -6.8733382, 7.0113845
6: -3.4602578, 3.8353443, -3.9242003, 4.3213425, -7.7816000, 7.7595444
7: -4.3166809, 2.5465417, -4.9383984, 2.8050406, -7.1217213, 7.4849401
8: -4.2752895, 3.1826899, -4.8682251, 3.6020455, -7.8773351, 8.0509148
9: -3.4437122, 3.4383764, -3.9307845, 3.9556148, -7.3993273, 7.3691607

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=82, inp2_unstable=69, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=15, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=215, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_B2_A2_B1_A1_B1

### Relational analysis result of IS_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1240400, upper bound: 11.1241512
time: 3.80 seconds

## Relational analysis of IS_B2_A2_B1_A1_B2

### Relational analysis result of IS_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1238903, upper bound: 11.1238903
time: 3.91 seconds

## BFS IS instance: IS_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -4.5584850, 3.8303990, -4.3444147, 3.6630158, -8.2215004, 8.1748142
1: -3.5020332, 3.3843598, -3.3237481, 3.2465346, -6.7485676, 6.7081079
2: -4.6381016, 3.0409908, -4.4240656, 2.9369316, -7.5750332, 7.4650564
3: -5.2599850, 2.6985364, -5.0061455, 2.5933790, -7.8533640, 7.7046819
4: -5.0237656, 4.0985851, -4.8212271, 3.9263148, -8.9500809, 8.9198122
5: -4.2589765, 3.4976873, -4.0586762, 3.3646750, -7.6236515, 7.5563636
6: -4.0969558, 4.5319109, -3.9242003, 4.3213425, -8.4182987, 8.4561110
7: -5.1691523, 2.9202616, -4.9383984, 2.8050406, -7.9741926, 7.8586597
8: -5.1048861, 3.7618594, -4.8682251, 3.6020455, -8.7069321, 8.6300850
9: -4.1117144, 4.1464911, -3.9307845, 3.9556148, -8.0673294, 8.0772753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=81, inp2_unstable=69, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 188

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_B2_A2_B1_A2_A1

### Relational analysis result of IS_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1241512, upper bound: 11.1248668
time: 3.64 seconds

## Relational analysis of IS_B2_A2_B1_A2_A2

### Relational analysis result of IS_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1238903, upper bound: 11.1238903
time: 3.11 seconds

## BFS IS instance: IS_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -3.8584981, 3.2181478, -5.2550688, 4.4531546, -8.3116531, 8.4732151
1: -2.8772225, 2.8652768, -4.1278057, 3.8999507, -6.7771730, 6.9930825
2: -3.8263783, 2.5537817, -5.4177375, 3.5898154, -7.4161940, 7.9715195
3: -4.3763485, 2.2742977, -6.0902853, 3.1582687, -7.5346174, 8.3645821
4: -4.1753302, 3.4981897, -5.8840542, 4.6870537, -8.8623838, 9.3822441
5: -3.5086632, 2.9527082, -5.0547428, 4.0843000, -7.5929632, 8.0074511
6: -3.4602578, 3.8353443, -4.7750130, 5.1669002, -8.6271582, 8.6103573
7: -4.3166809, 2.5465417, -5.9776363, 3.6566114, -7.9732924, 8.5241776
8: -4.2752895, 3.1826899, -5.9084744, 4.3449755, -8.6202650, 9.0911636
9: -3.4437122, 3.4383764, -4.7770643, 4.8386016, -8.2823143, 8.2154398

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=82, inp2_unstable=68, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=23, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=215, inp2_unstable=234, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_B2_A2_B2_A1_B1

### Relational analysis result of IS_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1241491, upper bound: 11.1253344
time: 2.49 seconds

## Relational analysis of IS_B2_A2_B2_A1_B2

### Relational analysis result of IS_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1241622, upper bound: 11.1253464
time: 2.79 seconds

## BFS IS instance: IS_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4.5584850, 3.8303990, -5.2550688, 4.4531546, -9.0116396, 9.0854673
1: -3.5020332, 3.3843598, -4.1278057, 3.8999507, -7.4019842, 7.5121655
2: -4.6381016, 3.0409908, -5.4177375, 3.5898154, -8.2279167, 8.4587278
3: -5.2599850, 2.6985364, -6.0902853, 3.1582687, -8.4182539, 8.7888203
4: -5.0237656, 4.0985851, -5.8840542, 4.6870537, -9.7108173, 9.9826393
5: -4.2589765, 3.4976873, -5.0547428, 4.0843000, -8.3432770, 8.5524302
6: -4.0969558, 4.5319109, -4.7750130, 5.1669002, -9.2638559, 9.3069229
7: -5.1691523, 2.9202616, -5.9776363, 3.6566114, -8.8257627, 8.8978977
8: -5.1048861, 3.7618594, -5.9084744, 4.3449755, -9.4498615, 9.6703339
9: -4.1117144, 4.1464911, -4.7770643, 4.8386016, -8.9503155, 8.9235554

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=81, inp2_unstable=68, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=23, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=234, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 62

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_B2_A2_B2_A2_B1

### Relational analysis result of IS_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1220729, upper bound: 11.1266882
time: 3.71 seconds

## Relational analysis of IS_B2_A2_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1224174, upper bound: 11.1269788
time: 3.59 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 10.88 seconds
IS_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 5, time: 10.88
Output dim: 7, lower bound: -10.9076481, upper bound: 10.8803418
IS_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 5, time: 10.88
Output dim: 7, lower bound: -10.8994532, upper bound: 10.8733505
IS_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 10.88
Output dim: 7, lower bound: -11.1168698, upper bound: 11.0060424
IS_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 10.88
Output dim: 7, lower bound: -11.1164730, upper bound: 11.0057913
IS_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 10.88
Output dim: 7, lower bound: -10.9459074, upper bound: 10.9110006
IS_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 10.88
Output dim: 7, lower bound: -10.9428805, upper bound: 10.9048650
IS_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 10.88
Output dim: 7, lower bound: -11.1187829, upper bound: 11.1029968
IS_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 10.88
Output dim: 7, lower bound: -11.1188033, upper bound: 11.0941458
IS_B1_A2_B1_A1_A1, status: Status.VERIFIED, split count: 5, time: 10.88
Output dim: 7, lower bound: -11.0135655, upper bound: 10.9362424
IS_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 10.88
Output dim: 7, lower bound: -11.1193433, upper bound: 11.0869943
IS_B1_A2_B1_A2_A1, status: Status.VERIFIED, split count: 5, time: 10.88
Output dim: 7, lower bound: -10.9597565, upper bound: 10.9017391
IS_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 10.88
Output dim: 7, lower bound: -11.1193779, upper bound: 11.0869746
IS_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 10.88
Output dim: 7, lower bound: -10.9315759, upper bound: 10.8938927
IS_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 10.88
Output dim: 7, lower bound: -10.9232114, upper bound: 10.8852125
IS_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 10.88
Output dim: 7, lower bound: -11.1175657, upper bound: 11.0082028
IS_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 10.88
Output dim: 7, lower bound: -11.1172932, upper bound: 11.0080611
IS_B2_A1_B1_A1_A1, status: Status.VERIFIED, split count: 5, time: 10.88
Output dim: 7, lower bound: -11.0039508, upper bound: 10.9813865
IS_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 10.88
Output dim: 7, lower bound: -11.1271221, upper bound: 11.1268908
IS_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 10.88
Output dim: 7, lower bound: -11.1245937, upper bound: 11.1245630
IS_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 10.88
Output dim: 7, lower bound: -11.1248072, upper bound: 11.1246574
IS_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 10.88
Output dim: 7, lower bound: -11.1243888, upper bound: 11.1258433
IS_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 10.88
Output dim: 7, lower bound: -11.1248072, upper bound: 11.1262509
IS_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 10.88
Output dim: 7, lower bound: -11.1261489, upper bound: 11.1299233
IS_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 10.88
Output dim: 7, lower bound: -11.1261489, upper bound: 11.1299875
IS_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 10.88
Output dim: 7, lower bound: -11.1240400, upper bound: 11.1241512
IS_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 10.88
Output dim: 7, lower bound: -11.1238903, upper bound: 11.1238903
IS_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 10.88
Output dim: 7, lower bound: -11.1241512, upper bound: 11.1248668
IS_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 10.88
Output dim: 7, lower bound: -11.1238903, upper bound: 11.1238903
IS_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 10.88
Output dim: 7, lower bound: -11.1241491, upper bound: 11.1253344
IS_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 10.88
Output dim: 7, lower bound: -11.1241622, upper bound: 11.1253464
IS_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 10.88
Output dim: 7, lower bound: -11.1220729, upper bound: 11.1266882
IS_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 10.88
Output dim: 7, lower bound: -11.1224174, upper bound: 11.1269788

## BFS IS instance: IS_B1_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -3.9492617, 3.3225079, -5.1096673, 4.2909670, -8.2402287, 8.4321747
1: -2.9739110, 2.9505847, -3.9767737, 3.7683315, -6.7422428, 6.9273586
2: -3.9727244, 2.6676846, -5.1801553, 3.3482504, -7.3209748, 7.8478398
3: -4.5059576, 2.3508911, -5.9644470, 2.9513569, -7.4573145, 8.3153381
4: -4.3417745, 3.5873179, -5.6998339, 4.6034484, -8.9452229, 9.2871513
5: -3.6429276, 3.0598085, -4.7842011, 3.8893666, -7.5322943, 7.8440094
6: -3.5634487, 3.9306087, -4.5871773, 5.0449390, -8.6083879, 8.5177860
7: -4.4625702, 2.5388021, -5.7944584, 3.1894038, -7.6519737, 8.3332605
8: -4.4096813, 3.2768176, -5.6959181, 4.1682644, -8.5779457, 8.9727354
9: -3.5624352, 3.5667450, -4.5976634, 4.6180534, -8.1804886, 8.1644087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=77, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=10, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=219, inp2_unstable=226, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B1_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_B1_A1_B1_A2_A1_B1

### Relational analysis result of IS_B1_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1164730, upper bound: 11.0057913
time: 4.79 seconds

## Relational analysis of IS_B1_A1_B1_A2_A1_B2

### Relational analysis result of IS_B1_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1164730, upper bound: 11.0057913
time: 4.02 seconds

## BFS IS instance: IS_B1_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -4.3264685, 3.6196613, -5.0355129, 4.2248430, -8.5513115, 8.6551743
1: -3.2899902, 3.2147391, -3.9133272, 3.7148876, -7.0048780, 7.1280661
2: -4.3557467, 2.8670359, -5.0983210, 3.2996349, -7.6553817, 7.9653568
3: -4.9862804, 2.5450547, -5.8723111, 2.9100783, -7.8963585, 8.4173660
4: -4.7601867, 3.9001796, -5.6099405, 4.5395184, -9.2997055, 9.5101204
5: -3.9925668, 3.2992568, -4.7032948, 3.8304002, -7.8229671, 8.0025520
6: -3.8892434, 4.2864451, -4.5207281, 4.9744492, -8.8636923, 8.8071728
7: -4.8717508, 2.6595528, -5.7081633, 3.1304951, -8.0022459, 8.3677158
8: -4.8308635, 3.5596609, -5.6119642, 4.1104703, -8.9413338, 9.1716251
9: -3.8794930, 3.8948932, -4.5288715, 4.5465961, -8.4260893, 8.4237652

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=74, inp2_unstable=77, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=222, inp2_unstable=225, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_B1_A1_B1_A2_A2_B1

### Relational analysis result of IS_B1_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1164730, upper bound: 11.0057913
time: 4.02 seconds

## Relational analysis of IS_B1_A1_B1_A2_A2_B2

### Relational analysis result of IS_B1_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1164730, upper bound: 11.0057913
time: 4.55 seconds

## BFS IS instance: IS_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.1163635, 3.4715874, -4.8701849, 4.1056938, -8.2220573, 8.3417721
1: -3.1242783, 3.0781796, -3.7816434, 3.5987210, -6.7229996, 6.8598232
2: -4.1654334, 2.7844100, -4.9293094, 3.2163463, -7.3817797, 7.7137194
3: -4.7178478, 2.4531202, -5.6432462, 2.8272502, -7.5450983, 8.0963669
4: -4.5470562, 3.7328842, -5.4284754, 4.3971071, -8.9441633, 9.1613598
5: -3.8115289, 3.1894405, -4.5370197, 3.7347803, -7.5463095, 7.7264605
6: -3.7168541, 4.0965734, -4.3780646, 4.8139448, -8.5307989, 8.4746380
7: -4.6673040, 2.5984330, -5.5224190, 3.0373206, -7.7046247, 8.1208515
8: -4.6032705, 3.4200051, -5.4259911, 3.9999204, -8.6031914, 8.8459959
9: -3.7169430, 3.7309437, -4.3953648, 4.4036007, -8.1205435, 8.1263084

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=38, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=74, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=221, inp2_unstable=225, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 62

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1119949, upper bound: 10.9386156
time: 3.16 seconds

## Relational analysis of IS_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1114607, upper bound: 10.9382242
time: 3.04 seconds

## BFS IS instance: IS_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -3.9987974, 3.3731961, -5.2997408, 4.4669905, -8.4657879, 8.6729374
1: -3.0228441, 2.9932909, -4.1513953, 3.9037690, -6.9266129, 7.1446862
2: -4.0377803, 2.7095742, -5.3876290, 3.4789457, -7.5167260, 8.0972033
3: -4.5717058, 2.3868377, -6.1883206, 3.0593064, -7.6310120, 8.5751581
4: -4.4091339, 3.6337261, -5.9335942, 4.7705784, -9.1797123, 9.5673199
5: -3.6953335, 3.1049125, -4.9820724, 4.0454836, -7.7408171, 8.0869846
6: -3.6131332, 3.9832540, -4.7610006, 5.2258520, -8.8389854, 8.7442551
7: -4.5327301, 2.5688994, -6.0186887, 3.3360651, -7.8687954, 8.5875883
8: -4.4706769, 3.3269331, -5.9123864, 4.3297362, -8.8004131, 9.2393198
9: -3.6131639, 3.6211057, -4.7840595, 4.8038530, -8.4170170, 8.4051647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=38, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=79, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=220, inp2_unstable=227, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 62

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_B1_A1_B2_A2_B2_B1

### Relational analysis result of IS_B1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1151638, upper bound: 11.0887778
time: 5.56 seconds

## Relational analysis of IS_B1_A1_B2_A2_B2_B2

### Relational analysis result of IS_B1_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1110199, upper bound: 10.9217986
time: 4.30 seconds

## BFS IS instance: IS_B1_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -4.1006732, 3.4765601, -4.9442382, 4.1477203, -8.2483921, 8.4207983
1: -3.1271696, 3.0710406, -3.8339763, 3.6507661, -6.7779360, 6.9050169
2: -4.1642666, 2.8017714, -5.0002880, 3.2426953, -7.4069614, 7.8020592
3: -4.6820593, 2.4623525, -5.7538104, 2.8608918, -7.5429511, 8.2161627
4: -4.5397091, 3.7169251, -5.5037842, 4.4599276, -8.9996367, 9.2207088
5: -3.8016036, 3.2045059, -4.6010432, 3.7626524, -7.5642557, 7.8055487
6: -3.7080681, 4.0847898, -4.4400868, 4.8867931, -8.5948610, 8.5248766
7: -4.6635189, 2.6373219, -5.6022453, 3.0568135, -7.7203321, 8.2395668
8: -4.5919671, 3.4288514, -5.5068407, 4.0415030, -8.6334686, 8.9356918
9: -3.7229123, 3.7322321, -4.4435339, 4.4605536, -8.1834660, 8.1757660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=77, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=224, inp2_unstable=224, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 195

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B1_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_B1_A2_B1_A1_A2_A1

### Relational analysis result of IS_B1_A2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1177881, upper bound: 11.0641470
time: 4.48 seconds

## Relational analysis of IS_B1_A2_B1_A1_A2_A2

### Relational analysis result of IS_B1_A2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1176609, upper bound: 11.0640514
time: 3.44 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 19.29 seconds
IS_B1_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 19.29
Output dim: 7, lower bound: -11.1164730, upper bound: 11.0057913
IS_B1_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 19.29
Output dim: 7, lower bound: -11.1164730, upper bound: 11.0057913
IS_B1_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 19.29
Output dim: 7, lower bound: -11.1164730, upper bound: 11.0057913
IS_B1_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 19.29
Output dim: 7, lower bound: -11.1164730, upper bound: 11.0057913
IS_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 19.29
Output dim: 7, lower bound: -11.1119949, upper bound: 10.9386156
IS_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 19.29
Output dim: 7, lower bound: -11.1114607, upper bound: 10.9382242
IS_B1_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 19.29
Output dim: 7, lower bound: -11.1151638, upper bound: 11.0887778
IS_B1_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 19.29
Output dim: 7, lower bound: -11.1110199, upper bound: 10.9217986
IS_B1_A2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 19.29
Output dim: 7, lower bound: -11.1177881, upper bound: 11.0641470
IS_B1_A2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 19.29
Output dim: 7, lower bound: -11.1176609, upper bound: 11.0640514
IS_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 7, lower bound: -11.1193779, upper bound: 11.0869746
IS_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 7, lower bound: -11.1175657, upper bound: 11.0082028
IS_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 7, lower bound: -11.1172932, upper bound: 11.0080611
IS_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 7, lower bound: -11.1271221, upper bound: 11.1268908
IS_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 7, lower bound: -11.1245937, upper bound: 11.1245630
IS_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 7, lower bound: -11.1248072, upper bound: 11.1246574
IS_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 7, lower bound: -11.1243888, upper bound: 11.1258433
IS_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 7, lower bound: -11.1248072, upper bound: 11.1262509
IS_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 7, lower bound: -11.1261489, upper bound: 11.1299233
IS_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 7, lower bound: -11.1261489, upper bound: 11.1299875
IS_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 7, lower bound: -11.1240400, upper bound: 11.1241512
IS_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 7, lower bound: -11.1238903, upper bound: 11.1238903
IS_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 7, lower bound: -11.1241512, upper bound: 11.1248668
IS_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 7, lower bound: -11.1238903, upper bound: 11.1238903
IS_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 7, lower bound: -11.1241491, upper bound: 11.1253344
IS_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 7, lower bound: -11.1241622, upper bound: 11.1253464
IS_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 7, lower bound: -11.1220729, upper bound: 11.1266882
IS_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 7, lower bound: -11.1224174, upper bound: 11.1269788

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 7.17 + 593.69 = 600.86 seconds
