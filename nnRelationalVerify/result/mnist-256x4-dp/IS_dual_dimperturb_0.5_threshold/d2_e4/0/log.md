## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.25792804


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.1381726, 0.1261228, -0.1381726, 0.1261228, -0.2642954, 0.2642954)
1: (-0.1470754, 0.1889696, -0.1470754, 0.1889696, -0.3360450, 0.3360450)
2: (-0.1507361, 0.2591240, -0.1507361, 0.2591240, -0.4098601, 0.4098601)
3: (-0.1104658, 0.1236263, -0.1104658, 0.1236263, -0.2340922, 0.2340922)
4: (-0.1971372, 0.1653046, -0.1971372, 0.1653046, -0.3624418, 0.3624418)
5: (-0.1622983, 0.1999135, -0.1622983, 0.1999135, -0.3622118, 0.3622118)
6: (0.6790259, 1.0546613, 0.6790259, 1.0546613, -0.3756354, 0.3756354)
7: (-0.2114900, 0.1940268, -0.2114900, 0.1940268, -0.4055167, 0.4055167)
8: (-0.1435570, 0.2282303, -0.1435570, 0.2282303, -0.3717873, 0.3717873)
9: (-0.1827184, 0.1620893, -0.1827184, 0.1620893, -0.3448077, 0.3448077)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.49 + 2.25 = 3.74 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
time: 1.07 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
time: 1.28 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.51 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.51
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.51
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0712045, 0.0725610, -0.1081061, 0.1021021, -0.1733065, 0.1806671
1: -0.0844986, 0.1312806, -0.1190118, 0.1630038, -0.2475024, 0.2502923
2: -0.1007596, 0.1786014, -0.1283232, 0.2228584, -0.3236180, 0.3069246
3: -0.0708348, 0.0620699, -0.0926927, 0.0959756, -0.1668104, 0.1547625
4: -0.1221487, 0.1117384, -0.1632845, 0.1412819, -0.2634305, 0.2750229
5: -0.1115529, 0.1166123, -0.1395371, 0.1625557, -0.2741086, 0.2561494
6: 0.7749881, 1.0336611, 0.7220621, 1.0451236, -0.2701355, 0.3115990
7: -0.1422768, 0.1182976, -0.1804501, 0.1597862, -0.3020630, 0.2987477
8: -0.0913423, 0.1506976, -0.1201202, 0.1934592, -0.2848015, 0.2708178
9: -0.1068000, 0.0917523, -0.1485575, 0.1305454, -0.2373455, 0.2403098

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
time: 1.35 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
time: 1.05 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0991842, 0.0949741, -0.1272322, 0.1173822, -0.2165664, 0.2222063
1: -0.1106841, 0.1552988, -0.1368637, 0.1795214, -0.2902054, 0.2921626
2: -0.1216723, 0.2120970, -0.1425806, 0.2459277, -0.3676000, 0.3546776
3: -0.0874186, 0.0877706, -0.1039986, 0.1135650, -0.2009835, 0.1917692
4: -0.1532391, 0.1341533, -0.1848190, 0.1565635, -0.3098026, 0.3189722
5: -0.1327830, 0.1514700, -0.1540160, 0.1863200, -0.3191030, 0.3054860
6: 0.7348324, 1.0422931, 0.6946857, 1.0511907, -0.3163583, 0.3476074
7: -0.1712392, 0.1496258, -0.2001955, 0.1815675, -0.3528067, 0.3498213
8: -0.1131653, 0.1831414, -0.1350289, 0.2155778, -0.3287431, 0.3181704
9: -0.1384208, 0.1211848, -0.1702880, 0.1506113, -0.2890320, 0.2914729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
time: 1.34 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
time: 1.12 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.01 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.01
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.01
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.01
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.01
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0571393, 0.0597526, -0.0589163, 0.0611979, -0.1183371, 0.1186690
1: -0.0704349, 0.1178787, -0.0722054, 0.1195017, -0.1899366, 0.1900842
2: -0.0902225, 0.1608056, -0.0916368, 0.1628040, -0.2530265, 0.2524424
3: -0.0628082, 0.0458978, -0.0635900, 0.0481653, -0.1109734, 0.1094878
4: -0.1065680, 0.1007225, -0.1083132, 0.1021769, -0.2087449, 0.2090357
5: -0.0988089, 0.0999936, -0.1005433, 0.1019107, -0.2007197, 0.2005369
6: 0.7975610, 1.0292323, 0.7946948, 1.0297748, -0.2322137, 0.2345375
7: -0.1274486, 0.1012553, -0.1292613, 0.1032565, -0.2307052, 0.2305165
8: -0.0809118, 0.1302527, -0.0819736, 0.1331193, -0.2140311, 0.2122264
9: -0.0887184, 0.0819145, -0.0909409, 0.0828072, -0.1715255, 0.1728554

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 163

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
time: 1.27 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0546061, 0.0583510, -0.0729955, 0.0739972, -0.1286032, 0.1313464
1: -0.0681690, 0.1155652, -0.0861765, 0.1328146, -0.2009836, 0.2017417
2: -0.0882065, 0.1581783, -0.1020996, 0.1807395, -0.2689461, 0.2602779
3: -0.0620072, 0.0426656, -0.0718975, 0.0637142, -0.1257215, 0.1145631
4: -0.1040803, 0.0986895, -0.1241290, 0.1131746, -0.2172549, 0.2228185
5: -0.0963365, 0.0977755, -0.1129131, 0.1188458, -0.2151824, 0.2106886
6: 0.8013092, 1.0284593, 0.7724150, 1.0342077, -0.2328985, 0.2560443
7: -0.1251311, 0.0984025, -0.1441326, 0.1202900, -0.2454212, 0.2425351
8: -0.0796378, 0.1261666, -0.0927396, 0.1527763, -0.2324141, 0.2189062
9: -0.0856738, 0.0806420, -0.1088200, 0.0936382, -0.1793119, 0.1894620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 163

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
time: 1.22 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
time: 1.09 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0811050, 0.0804997, -0.0754343, 0.0759525, -0.1570575, 0.1559340
1: -0.0937736, 0.1397602, -0.0884611, 0.1349032, -0.2286768, 0.2282213
2: -0.1081670, 0.1904201, -0.1039243, 0.1836504, -0.2918174, 0.2943444
3: -0.0767089, 0.0711600, -0.0733443, 0.0659535, -0.1426623, 0.1445043
4: -0.1330951, 0.1196778, -0.1268253, 0.1151303, -0.2482254, 0.2465031
5: -0.1190716, 0.1289590, -0.1147651, 0.1218871, -0.2409587, 0.2437241
6: 0.7607647, 1.0366832, 0.7689116, 1.0349523, -0.2741876, 0.2677716
7: -0.1525355, 0.1293118, -0.1466596, 0.1230030, -0.2755384, 0.2759714
8: -0.0990661, 0.1621892, -0.0946420, 0.1556072, -0.2546733, 0.2568312
9: -0.1179665, 0.1021775, -0.1115705, 0.0962063, -0.2141728, 0.2137480

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 163

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
time: 1.30 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
time: 1.10 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0784703, 0.0783871, -0.0922960, 0.0894709, -0.1679412, 0.1706831
1: -0.0913054, 0.1375034, -0.1042547, 0.1493502, -0.2406556, 0.2417581
2: -0.1061957, 0.1872748, -0.1165376, 0.2037886, -0.3099843, 0.3038124
3: -0.0751457, 0.0687409, -0.0833467, 0.0814359, -0.1565816, 0.1520877
4: -0.1301821, 0.1175650, -0.1454834, 0.1286497, -0.2588318, 0.2630484
5: -0.1170708, 0.1256733, -0.1275684, 0.1429113, -0.2599820, 0.2532417
6: 0.7645498, 1.0358789, 0.7446920, 1.0401081, -0.2755583, 0.2911869
7: -0.1498055, 0.1263806, -0.1641282, 0.1417812, -0.2915867, 0.2905087
8: -0.0970106, 0.1591311, -0.1077960, 0.1751752, -0.2721859, 0.2669271
9: -0.1149947, 0.0994032, -0.1305945, 0.1139584, -0.2289532, 0.2299977

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 163

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
time: 1.04 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
time: 1.09 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.56 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0464263, 0.0533241, -0.0516923, 0.0567389, -0.1031651, 0.1050165
1: -0.0605005, 0.1072680, -0.0655626, 0.1129042, -0.1734047, 0.1728306
2: -0.0816050, 0.1487553, -0.0858878, 0.1551563, -0.2367612, 0.2346431
3: -0.0591348, 0.0318472, -0.0610860, 0.0389478, -0.0980825, 0.0929332
4: -0.0951583, 0.0917626, -0.1012190, 0.0963509, -0.1915092, 0.1929816
5: -0.0883153, 0.0898203, -0.0934928, 0.0952242, -0.1835395, 0.1833131
6: 0.8143866, 1.0256869, 0.8056204, 1.0275702, -0.2131836, 0.2200664
7: -0.1168192, 0.0886127, -0.1224654, 0.0951212, -0.2119403, 0.2110780
8: -0.0750687, 0.1138738, -0.0781724, 0.1214667, -0.1965354, 0.1920462
9: -0.0761315, 0.0760781, -0.0821717, 0.0791783, -0.1553098, 0.1582498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2837133, upper bound: 0.2782578
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0490539, 0.0551551, -0.0554944, 0.0588425, -0.1078964, 0.1106495
1: -0.0631156, 0.1102901, -0.0689636, 0.1163765, -0.1794922, 0.1792537
2: -0.0837653, 0.1521875, -0.0889135, 0.1590997, -0.2428651, 0.2411010
3: -0.0601810, 0.0354870, -0.0622881, 0.0437991, -0.1039802, 0.0977751
4: -0.0984080, 0.0941439, -0.1049528, 0.0994023, -0.1978103, 0.1990967
5: -0.0909085, 0.0927179, -0.0972035, 0.0985534, -0.1894619, 0.1899214
6: 0.8097653, 1.0266968, 0.7999948, 1.0287305, -0.2189652, 0.2267020
7: -0.1198466, 0.0920071, -0.1259439, 0.0994028, -0.2192494, 0.2179509
8: -0.0767330, 0.1174338, -0.0800846, 0.1275996, -0.2043326, 0.1975183
9: -0.0790722, 0.0777404, -0.0867414, 0.0810882, -0.1601604, 0.1644818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2837133, upper bound: 0.2781328
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
time: 1.30 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0441048, 0.0517065, -0.0640288, 0.0661508, -0.1102556, 0.1157353
1: -0.0581902, 0.1045977, -0.0775427, 0.1242757, -0.1824659, 0.1821404
2: -0.0796963, 0.1457229, -0.0956255, 0.1689843, -0.2486806, 0.2413484
3: -0.0582104, 0.0286312, -0.0663004, 0.0545605, -0.1127710, 0.0949317
4: -0.0922872, 0.0896588, -0.1134317, 0.1063322, -0.1986194, 0.2030905
5: -0.0860242, 0.0872604, -0.1054349, 0.1080780, -0.1941021, 0.1926953
6: 0.8184696, 1.0247947, 0.7860694, 1.0313044, -0.2128348, 0.2387253
7: -0.1141443, 0.0856140, -0.1347282, 0.1091989, -0.2233432, 0.2203422
8: -0.0735984, 0.1107284, -0.0853624, 0.1412041, -0.2148025, 0.1960909
9: -0.0735333, 0.0746094, -0.0975754, 0.0853250, -0.1588583, 0.1721848

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2740120, upper bound: 0.2839397
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
time: 1.20 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0470120, 0.0537323, -0.0682279, 0.0701026, -0.1171146, 0.1219602
1: -0.0610834, 0.1079416, -0.0816802, 0.1286549, -0.1897383, 0.1896218
2: -0.0820865, 0.1495205, -0.0985536, 0.1749415, -0.2570280, 0.2480741
3: -0.0593680, 0.0326584, -0.0690297, 0.0592550, -0.1186231, 0.1016881
4: -0.0958827, 0.0922934, -0.1187590, 0.1093825, -0.2052652, 0.2110524
5: -0.0888934, 0.0904662, -0.1092246, 0.1129372, -0.2018306, 0.1996908
6: 0.8133565, 1.0259120, 0.7793328, 1.0327252, -0.2193688, 0.2465793
7: -0.1174939, 0.0893694, -0.1391649, 0.1148869, -0.2323808, 0.2285343
8: -0.0754397, 0.1146673, -0.0889539, 0.1471390, -0.2225787, 0.2036212
9: -0.0767870, 0.0764486, -0.1033421, 0.0886565, -0.1654435, 0.1797907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2738452, upper bound: 0.2837474
time: 1.22 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
time: 1.23 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0707357, 0.0721852, -0.0659189, 0.0678482, -0.1385839, 0.1381041
1: -0.0840595, 0.1308791, -0.0793484, 0.1262469, -0.2103065, 0.2102275
2: -0.1004089, 0.1780420, -0.0969435, 0.1715854, -0.2719942, 0.2749855
3: -0.0705568, 0.0616395, -0.0674394, 0.0566736, -0.1272304, 0.1290789
4: -0.1216307, 0.1113624, -0.1156505, 0.1077052, -0.2293359, 0.2270130
5: -0.1111970, 0.1160278, -0.1070894, 0.1102652, -0.2214622, 0.2231172
6: 0.7756612, 1.0335181, 0.7830372, 1.0318670, -0.2562058, 0.2504809
7: -0.1417912, 0.1177762, -0.1366161, 0.1117591, -0.2535503, 0.2543923
8: -0.0909767, 0.1501536, -0.0867801, 0.1438756, -0.2348522, 0.2369337
9: -0.1062714, 0.0912588, -0.1001711, 0.0864422, -0.1927136, 0.1914299

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2863269, upper bound: 0.2798574
time: 1.27 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
time: 1.26 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0685894, 0.0704555, -0.0706998, 0.0721563, -0.1407458, 0.1411553
1: -0.0820453, 0.1290317, -0.0840258, 0.1308485, -0.2128938, 0.2130575
2: -0.0988056, 0.1754670, -0.1003821, 0.1779990, -0.2768046, 0.2758491
3: -0.0692787, 0.0596591, -0.0705355, 0.0616064, -0.1308850, 0.1301947
4: -0.1192457, 0.1096450, -0.1215908, 0.1113337, -0.2305794, 0.2312359
5: -0.1095589, 0.1133555, -0.1111697, 0.1159830, -0.2255418, 0.2245252
6: 0.7787531, 1.0328596, 0.7757130, 1.0335070, -0.2547539, 0.2571466
7: -0.1395640, 0.1153766, -0.1417540, 0.1177362, -0.2573002, 0.2571306
8: -0.0892943, 0.1476499, -0.0909486, 0.1501118, -0.2394061, 0.2385986
9: -0.1038386, 0.0890033, -0.1062308, 0.0912208, -0.1950594, 0.1952341

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2863270, upper bound: 0.2798574
time: 1.08 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
time: 1.25 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0682767, 0.0701503, -0.0792305, 0.0789967, -0.1472735, 0.1493808
1: -0.0817295, 0.1287058, -0.0920175, 0.1381548, -0.2198844, 0.2207233
2: -0.0985876, 0.1750126, -0.1067645, 0.1881824, -0.2867700, 0.2817771
3: -0.0690633, 0.0593096, -0.0755967, 0.0694390, -0.1385023, 0.1349063
4: -0.1188248, 0.1094179, -0.1310225, 0.1181747, -0.2369995, 0.2404404
5: -0.1092698, 0.1129937, -0.1176482, 0.1266213, -0.2358911, 0.2306420
6: 0.7792546, 1.0327436, 0.7634575, 1.0361110, -0.2568564, 0.2692860
7: -0.1392188, 0.1149531, -0.1505932, 0.1272263, -0.2664451, 0.2655463
8: -0.0890000, 0.1472079, -0.0976037, 0.1600135, -0.2490135, 0.2448116
9: -0.1034092, 0.0887034, -0.1158523, 0.1002037, -0.2036129, 0.2045557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2770779, upper bound: 0.2858459
time: 0.99 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
time: 1.11 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0736451, 0.0745181, -0.0797709, 0.0794301, -0.1530751, 0.1542889
1: -0.0867851, 0.1333710, -0.0925237, 0.1386176, -0.2254027, 0.2258947
2: -0.1025857, 0.1815149, -0.1071688, 0.1888275, -0.2914132, 0.2886837
3: -0.0722829, 0.0643107, -0.0759173, 0.0699351, -0.1422180, 0.1402281
4: -0.1248472, 0.1136957, -0.1316200, 0.1186080, -0.2434552, 0.2453157
5: -0.1134064, 0.1196560, -0.1180585, 0.1272952, -0.2407016, 0.2377145
6: 0.7714818, 1.0344062, 0.7626814, 1.0362759, -0.2647941, 0.2717248
7: -0.1448059, 0.1210127, -0.1511531, 0.1278274, -0.2726333, 0.2721658
8: -0.0932465, 0.1535304, -0.0980254, 0.1606406, -0.2538871, 0.2515557
9: -0.1095527, 0.0943224, -0.1164618, 0.1007726, -0.2103253, 0.2107842

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2770779, upper bound: 0.2858459
time: 1.09 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
time: 1.32 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.86 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 6, lower bound: -0.2837133, upper bound: 0.2782578
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 6, lower bound: -0.2837133, upper bound: 0.2781328
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
IS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 6, lower bound: -0.2740120, upper bound: 0.2839397
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
IS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 6, lower bound: -0.2738452, upper bound: 0.2837474
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 6, lower bound: -0.2863269, upper bound: 0.2798574
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 6, lower bound: -0.2863270, upper bound: 0.2798574
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 6, lower bound: -0.2770779, upper bound: 0.2858459
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 6, lower bound: -0.2770779, upper bound: 0.2858459
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.86
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2945554

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0313263, 0.0345683, -0.0221404, 0.0257803, -0.0571067, 0.0567086
1: -0.0451436, 0.0787837, -0.0353709, 0.0670153, -0.1121589, 0.1141547
2: -0.0625182, 0.1179906, -0.0478597, 0.0962220, -0.1587402, 0.1658503
3: -0.0506370, 0.0091284, -0.0421599, 0.0066634, -0.0573003, 0.0512883
4: -0.0692913, 0.0718333, -0.0539762, 0.0590767, -0.1283680, 0.1258095
5: -0.0641746, 0.0739667, -0.0519411, 0.0627815, -0.1269561, 0.1259078
6: 0.8539361, 1.0177271, 0.8813641, 1.0123494, -0.1584132, 0.1363630
7: -0.0933086, 0.0645669, -0.0816994, 0.0432052, -0.1365138, 0.1462663
8: -0.0621594, 0.0841159, -0.0529443, 0.0664357, -0.1285951, 0.1370602
9: -0.0498291, 0.0646280, -0.0377053, 0.0560460, -0.1058751, 0.1023333

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2755403, upper bound: 0.2661805
time: 1.26 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2754779, upper bound: 0.2661306
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0464263, 0.0533241, -0.0423517, 0.0502944, -0.0967206, 0.0956758
1: -0.0605005, 0.1072680, -0.0563323, 0.1023230, -0.1628234, 0.1636002
2: -0.0816050, 0.1487553, -0.0781192, 0.1431479, -0.2247529, 0.2268745
3: -0.0591348, 0.0318472, -0.0574846, 0.0258833, -0.0850181, 0.0893318
4: -0.0951583, 0.0917626, -0.0898982, 0.0879541, -0.1831124, 0.1816608
5: -0.0883153, 0.0898203, -0.0840243, 0.0852737, -0.1735890, 0.1738446
6: 0.8143866, 1.0256869, 0.8219622, 1.0241063, -0.2097197, 0.2037246
7: -0.1168192, 0.0886127, -0.1118093, 0.0833582, -0.2001774, 0.2004220
8: -0.0750687, 0.1138738, -0.0724417, 0.1080761, -0.1831448, 0.1863154
9: -0.0761315, 0.0760781, -0.0713697, 0.0733273, -0.1494588, 0.1474478

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2926339, upper bound: 0.2943939
time: 1.24 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2926339, upper bound: 0.2940131
time: 1.08 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0324400, 0.0367648, -0.0231233, 0.0264391, -0.0588792, 0.0598882
1: -0.0461872, 0.0805513, -0.0364235, 0.0682710, -0.1144582, 0.1169749
2: -0.0642346, 0.1213179, -0.0494355, 0.0983374, -0.1625720, 0.1707534
3: -0.0515934, 0.0095974, -0.0430613, 0.0069281, -0.0585215, 0.0526587
4: -0.0718275, 0.0733482, -0.0554403, 0.0604507, -0.1322782, 0.1287886
5: -0.0662730, 0.0751612, -0.0531054, 0.0639862, -0.1302593, 0.1282666
6: 0.8502467, 1.0184903, 0.8785602, 1.0129186, -0.1626719, 0.1399301
7: -0.0947659, 0.0670646, -0.0829367, 0.0454944, -0.1402603, 0.1500013
8: -0.0632165, 0.0871665, -0.0539369, 0.0681027, -0.1313193, 0.1411034
9: -0.0522280, 0.0655445, -0.0388243, 0.0569704, -0.1091984, 0.1043688

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2755403, upper bound: 0.2660753
time: 1.44 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2754699, upper bound: 0.2659926
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0490539, 0.0551551, -0.0451032, 0.0524022, -0.1014561, 0.1002583
1: -0.0631156, 0.1102901, -0.0591838, 0.1057460, -0.1688617, 0.1694738
2: -0.0837653, 0.1521875, -0.0805172, 0.1470271, -0.2307924, 0.2327047
3: -0.0601810, 0.0354870, -0.0586080, 0.0300142, -0.0901953, 0.0940949
4: -0.0984080, 0.0941439, -0.0935220, 0.0905635, -0.1889715, 0.1876659
5: -0.0909085, 0.0927179, -0.0870095, 0.0883613, -0.1792698, 0.1797274
6: 0.8097653, 1.0266968, 0.8167137, 1.0251782, -0.2154129, 0.2099831
7: -0.1198466, 0.0920071, -0.1152946, 0.0869036, -0.2067503, 0.2073017
8: -0.0767330, 0.1174338, -0.0742308, 0.1120812, -0.1888142, 0.1916645
9: -0.0790722, 0.0777404, -0.0746507, 0.0752411, -0.1543133, 0.1523910

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2935109, upper bound: 0.2940150
time: 1.30 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2926339, upper bound: 0.2939715
time: 1.26 seconds

## BFS IS instance: IS_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0188647, 0.0235848, -0.0369562, 0.0448955, -0.0637602, 0.0605410
1: -0.0318630, 0.0628309, -0.0504490, 0.0936352, -0.1254982, 0.1132798
2: -0.0426081, 0.0891721, -0.0725550, 0.1336343, -0.1762423, 0.1617271
3: -0.0391560, 0.0057811, -0.0551338, 0.0160626, -0.0552186, 0.0609149
4: -0.0490965, 0.0544976, -0.0812840, 0.0821258, -0.1312223, 0.1357816
5: -0.0480612, 0.0587664, -0.0763781, 0.0795829, -0.1276440, 0.1351445
6: 0.8907083, 1.0104531, 0.8344601, 1.0214895, -0.1307812, 0.1759930
7: -0.0775758, 0.0355760, -0.1034876, 0.0764832, -0.1540590, 0.1390636
8: -0.0496365, 0.0608799, -0.0686821, 0.0984583, -0.1480947, 0.1295620
9: -0.0339761, 0.0529654, -0.0634628, 0.0689370, -0.1029132, 0.1164282

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=39, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2764628
time: 1.86 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2763133
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0373631, 0.0455192, -0.0640288, 0.0661508, -0.1035139, 0.1095480
1: -0.0509409, 0.0946388, -0.0775427, 0.1242757, -0.1752166, 0.1721815
2: -0.0731978, 0.1345791, -0.0956255, 0.1689843, -0.2421821, 0.2302046
3: -0.0554054, 0.0168710, -0.0663004, 0.0545605, -0.1099659, 0.0831714
4: -0.0820379, 0.0827991, -0.1134317, 0.1063322, -0.1883701, 0.1962308
5: -0.0772614, 0.0799221, -0.1054349, 0.1080780, -0.1853393, 0.1853569
6: 0.8332493, 1.0217918, 0.7860694, 1.0313044, -0.1980550, 0.2357224
7: -0.1041567, 0.0772773, -0.1347282, 0.1091989, -0.2133556, 0.2120055
8: -0.0691164, 0.0993244, -0.0853624, 0.1412041, -0.2103205, 0.1846868
9: -0.0643762, 0.0691973, -0.0975754, 0.0853250, -0.1497012, 0.1667727

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=39, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2926339, upper bound: 0.2945554
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2926339, upper bound: 0.2942497
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0202905, 0.0245404, -0.0387305, 0.0469487, -0.0672392, 0.0632709
1: -0.0333898, 0.0646522, -0.0524456, 0.0969393, -0.1303292, 0.1170978
2: -0.0448939, 0.0922407, -0.0746711, 0.1370545, -0.1819484, 0.1669119
3: -0.0404635, 0.0061651, -0.0560278, 0.0193793, -0.0598428, 0.0621930
4: -0.0512204, 0.0564907, -0.0842506, 0.0843424, -0.1355628, 0.1407413
5: -0.0497500, 0.0605140, -0.0792861, 0.0813389, -0.1310889, 0.1398001
6: 0.8866411, 1.0112784, 0.8300059, 1.0224847, -0.1358436, 0.1812725
7: -0.0793707, 0.0388968, -0.1062775, 0.0790978, -0.1584685, 0.1451742
8: -0.0510763, 0.0632981, -0.0701119, 0.1018018, -0.1528781, 0.1334101
9: -0.0355993, 0.0543063, -0.0664699, 0.0702899, -0.1058893, 0.1207762

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=39, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2602165, upper bound: 0.2757568
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2602165, upper bound: 0.2757155
time: 1.22 seconds

## BFS IS instance: IS_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0390983, 0.0472886, -0.0682279, 0.0701026, -0.1092008, 0.1155165
1: -0.0528403, 0.0974860, -0.0816802, 0.1286549, -0.1814952, 0.1791662
2: -0.0750213, 0.1376734, -0.0985536, 0.1749415, -0.2499628, 0.2362270
3: -0.0561758, 0.0200397, -0.0690297, 0.0592550, -0.1154308, 0.0890694
4: -0.0848241, 0.0847091, -0.1187590, 0.1093825, -0.1942066, 0.2034682
5: -0.0797673, 0.0817384, -0.1092246, 0.1129372, -0.1927045, 0.1909630
6: 0.8291891, 1.0226493, 0.7793328, 1.0327252, -0.2035361, 0.2433165
7: -0.1068392, 0.0795304, -0.1391649, 0.1148869, -0.2217261, 0.2186954
8: -0.0703485, 0.1024390, -0.0889539, 0.1471390, -0.2174875, 0.1913929
9: -0.0669675, 0.0705983, -0.1033421, 0.0886565, -0.1556240, 0.1739404

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=39, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2926339, upper bound: 0.2945239
time: 1.17 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2926339, upper bound: 0.2941940
time: 1.07 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0398051, 0.0479416, -0.0257905, 0.0282269, -0.0680320, 0.0737321
1: -0.0535990, 0.0985368, -0.0392799, 0.0716782, -0.1252772, 0.1378168
2: -0.0756944, 0.1388628, -0.0537116, 0.1040777, -0.1797721, 0.1925744
3: -0.0564602, 0.0213094, -0.0455072, 0.0076464, -0.0641066, 0.0668166
4: -0.0859265, 0.0854142, -0.0594136, 0.0641793, -0.1501058, 0.1448277
5: -0.0806921, 0.0825065, -0.0562647, 0.0672555, -0.1479476, 0.1387712
6: 0.8276190, 1.0229658, 0.8709517, 1.0144626, -0.1868436, 0.1520141
7: -0.1079190, 0.0803621, -0.0862944, 0.0517065, -0.1596255, 0.1666565
8: -0.0708033, 0.1036637, -0.0566303, 0.0726265, -0.1434298, 0.1602940
9: -0.0679239, 0.0711912, -0.0418607, 0.0594787, -0.1274027, 0.1130519

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2788394, upper bound: 0.2674065
time: 1.08 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2788088, upper bound: 0.2673942
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0707357, 0.0721852, -0.0538814, 0.0579501, -0.1286858, 0.1260667
1: -0.0840595, 0.1308791, -0.0675207, 0.1149033, -0.1989629, 0.1983998
2: -0.1004089, 0.1780420, -0.0876299, 0.1574266, -0.2578355, 0.2656719
3: -0.0705568, 0.0616395, -0.0617781, 0.0417409, -0.1122977, 0.1234176
4: -0.1216307, 0.1113624, -0.1033687, 0.0981078, -0.2197385, 0.2147311
5: -0.1111970, 0.1160278, -0.0956294, 0.0971410, -0.2083379, 0.2116572
6: 0.7756612, 1.0335181, 0.8023814, 1.0282382, -0.2525769, 0.2311367
7: -0.1417912, 0.1177762, -0.1244681, 0.0975865, -0.2393777, 0.2422443
8: -0.0909767, 0.1501536, -0.0792733, 0.1249979, -0.2159746, 0.2294269
9: -0.1062714, 0.0912588, -0.0848028, 0.0802779, -0.1865493, 0.1760616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2944311, upper bound: 0.2945554
time: 1.06 seconds

## Relational analysis of IS_A2_B1_A1_B2_B2

### Relational analysis result of IS_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2944311, upper bound: 0.2942308
time: 1.13 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0389877, 0.0471864, -0.0268156, 0.0289139, -0.0679017, 0.0740020
1: -0.0527217, 0.0973218, -0.0403777, 0.0729877, -0.1257094, 0.1376995
2: -0.0749162, 0.1374874, -0.0553550, 0.1062839, -0.1812000, 0.1928424
3: -0.0561314, 0.0198412, -0.0464472, 0.0079225, -0.0640539, 0.0662885
4: -0.0846518, 0.0845990, -0.0609406, 0.0656122, -0.1502640, 0.1455395
5: -0.0796227, 0.0816184, -0.0574789, 0.0685119, -0.1481346, 0.1390972
6: 0.8294344, 1.0225999, 0.8680276, 1.0150563, -0.1856219, 0.1545724
7: -0.1066704, 0.0794006, -0.0875847, 0.0540939, -0.1607643, 0.1669853
8: -0.0702774, 0.1022475, -0.0576654, 0.0743651, -0.1446425, 0.1599129
9: -0.0668180, 0.0705057, -0.0430277, 0.0604427, -0.1272607, 0.1135334

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2788462, upper bound: 0.2674067
time: 1.24 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2788088, upper bound: 0.2673942
time: 1.47 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0685894, 0.0704555, -0.0577372, 0.0600835, -0.1286729, 0.1281927
1: -0.0820453, 0.1290317, -0.0709697, 0.1184248, -0.2004701, 0.2000015
2: -0.0988056, 0.1754670, -0.0906984, 0.1614258, -0.2602313, 0.2661654
3: -0.0692787, 0.0596591, -0.0629972, 0.0466607, -0.1159393, 0.1226564
4: -0.1192457, 0.1096450, -0.1071552, 0.1012024, -0.2204481, 0.2168002
5: -0.1095589, 0.1133555, -0.0993924, 0.1005172, -0.2100761, 0.2127480
6: 0.7787531, 1.0328596, 0.7966764, 1.0294149, -0.2506618, 0.2361832
7: -0.1395640, 0.1153766, -0.1279957, 0.1019287, -0.2414927, 0.2433723
8: -0.0892943, 0.1476499, -0.0812125, 0.1312174, -0.2205117, 0.2288624
9: -0.1038386, 0.0890033, -0.0894371, 0.0822148, -0.1860534, 0.1784404

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A2_B2_B1

### Relational analysis result of IS_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2944494, upper bound: 0.2945554
time: 1.17 seconds

## Relational analysis of IS_A2_B1_A2_B2_B2

### Relational analysis result of IS_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2944494, upper bound: 0.2942308
time: 1.13 seconds

## BFS IS instance: IS_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0250423, 0.0277254, -0.0437125, 0.0514199, -0.0764622, 0.0714379
1: -0.0384786, 0.0707224, -0.0577808, 0.1041343, -0.1426129, 0.1285032
2: -0.0525120, 0.1024673, -0.0793582, 0.1451979, -0.1977099, 0.1818255
3: -0.0448210, 0.0074449, -0.0580467, 0.0280716, -0.0728927, 0.0654916
4: -0.0582989, 0.0631333, -0.0917983, 0.0892861, -0.1475851, 0.1549316
5: -0.0553784, 0.0663383, -0.0856184, 0.0868085, -0.1421869, 0.1519567
6: 0.8730862, 1.0140295, 0.8191929, 1.0246519, -0.1515657, 0.1948366
7: -0.0853524, 0.0499638, -0.1136705, 0.0850875, -0.1704399, 0.1636343
8: -0.0558746, 0.0713574, -0.0733380, 0.1101870, -0.1660616, 0.1446954
9: -0.0410089, 0.0587750, -0.0730791, 0.0743492, -0.1153581, 0.1318542

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_B1_A1_B1

### Relational analysis result of IS_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2786061
time: 1.10 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2

### Relational analysis result of IS_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2784927
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0561087, 0.0591824, -0.0792305, 0.0789967, -0.1351055, 0.1384129
1: -0.0695130, 0.1169376, -0.0920175, 0.1381548, -0.2076679, 0.2089551
2: -0.0894024, 0.1597367, -0.1067645, 0.1881824, -0.2775848, 0.2665012
3: -0.0624824, 0.0445828, -0.0755967, 0.0694390, -0.1319214, 0.1201795
4: -0.1055560, 0.0998954, -0.1310225, 0.1181747, -0.2237307, 0.2309179
5: -0.0978030, 0.0990912, -0.1176482, 0.1266213, -0.2244243, 0.2167395
6: 0.7990859, 1.0289179, 0.7634575, 1.0361110, -0.2370251, 0.2654604
7: -0.1265058, 0.1000947, -0.1505932, 0.1272263, -0.2537321, 0.2506879
8: -0.0803934, 0.1285905, -0.0976037, 0.1600135, -0.2404069, 0.2261942
9: -0.0874798, 0.0813968, -0.1158523, 0.1002037, -0.1876834, 0.1972491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_B1_A2_A1

### Relational analysis result of IS_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2944311
time: 1.35 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2

### Relational analysis result of IS_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2944494, upper bound: 0.2944311
time: 1.55 seconds

## BFS IS instance: IS_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0265597, 0.0287424, -0.0438995, 0.0515582, -0.0781179, 0.0726419
1: -0.0401036, 0.0726607, -0.0579784, 0.1043567, -0.1444603, 0.1306392
2: -0.0549447, 0.1057331, -0.0795214, 0.1454498, -0.2003945, 0.1852545
3: -0.0462126, 0.0078536, -0.0581257, 0.0283406, -0.0745531, 0.0659793
4: -0.0605593, 0.0652545, -0.0920318, 0.0894660, -0.1500253, 0.1572863
5: -0.0571757, 0.0681982, -0.0858143, 0.0870264, -0.1442021, 0.1540126
6: 0.8687577, 1.0149080, 0.8188438, 1.0247188, -0.1559611, 0.1960641
7: -0.0872626, 0.0534979, -0.1138991, 0.0853410, -0.1726037, 0.1673970
8: -0.0574070, 0.0739310, -0.0734636, 0.1104464, -0.1678534, 0.1473947
9: -0.0427364, 0.0602021, -0.0732976, 0.0744748, -0.1172111, 0.1334997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_B2_A1_B1

### Relational analysis result of IS_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2770677, upper bound: 0.2858251
time: 0.98 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2

### Relational analysis result of IS_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2770677, upper bound: 0.2856936
time: 1.13 seconds

## BFS IS instance: IS_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0600298, 0.0622983, -0.0797709, 0.0794301, -0.1394599, 0.1420692
1: -0.0733912, 0.1205187, -0.0925237, 0.1386176, -0.2120088, 0.2130424
2: -0.0925229, 0.1641217, -0.1071688, 0.1888275, -0.2813504, 0.2712906
3: -0.0641726, 0.0495859, -0.0759173, 0.0699351, -0.1341076, 0.1255033
4: -0.1094066, 0.1031001, -0.1316200, 0.1186080, -0.2280146, 0.2347201
5: -0.1016299, 0.1032643, -0.1180585, 0.1272952, -0.2289251, 0.2213229
6: 0.7927988, 1.0301145, 0.7626814, 1.0362759, -0.2434770, 0.2674332
7: -0.1304758, 0.1045105, -0.1511531, 0.1278274, -0.2583033, 0.2556636
8: -0.0827099, 0.1349155, -0.0980254, 0.1606406, -0.2433506, 0.2329408
9: -0.0923701, 0.0833665, -0.1164618, 0.1007726, -0.1931427, 0.1998284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_B2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2944494
time: 1.41 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2944494, upper bound: 0.2944494
time: 1.30 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.38 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 6, lower bound: -0.2755403, upper bound: 0.2661805
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 6, lower bound: -0.2754779, upper bound: 0.2661306
IS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 6, lower bound: -0.2926339, upper bound: 0.2943939
IS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 6, lower bound: -0.2926339, upper bound: 0.2940131
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 6, lower bound: -0.2755403, upper bound: 0.2660753
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 6, lower bound: -0.2754699, upper bound: 0.2659926
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 6, lower bound: -0.2935109, upper bound: 0.2940150
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 6, lower bound: -0.2926339, upper bound: 0.2939715
IS_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2764628
IS_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2763133
IS_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 6, lower bound: -0.2926339, upper bound: 0.2945554
IS_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 6, lower bound: -0.2926339, upper bound: 0.2942497
IS_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 6, lower bound: -0.2602165, upper bound: 0.2757568
IS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 6, lower bound: -0.2602165, upper bound: 0.2757155
IS_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 6, lower bound: -0.2926339, upper bound: 0.2945239
IS_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 6, lower bound: -0.2926339, upper bound: 0.2941940
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 6, lower bound: -0.2788394, upper bound: 0.2674065
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 6, lower bound: -0.2788088, upper bound: 0.2673942
IS_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 6, lower bound: -0.2944311, upper bound: 0.2945554
IS_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 6, lower bound: -0.2944311, upper bound: 0.2942308
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 6, lower bound: -0.2788462, upper bound: 0.2674067
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 6, lower bound: -0.2788088, upper bound: 0.2673942
IS_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 6, lower bound: -0.2944494, upper bound: 0.2945554
IS_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 6, lower bound: -0.2944494, upper bound: 0.2942308
IS_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2786061
IS_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2784927
IS_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2944311
IS_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 6, lower bound: -0.2944494, upper bound: 0.2944311
IS_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 6, lower bound: -0.2770677, upper bound: 0.2858251
IS_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 6, lower bound: -0.2770677, upper bound: 0.2856936
IS_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2944494
IS_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.38
Output dim: 6, lower bound: -0.2944494, upper bound: 0.2944494

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0259017, 0.0283014, -0.0211835, 0.0251389, -0.0510407, 0.0494849
1: -0.0393990, 0.0718202, -0.0343462, 0.0657930, -0.1051920, 0.1061663
2: -0.0538898, 0.1043171, -0.0463256, 0.0941627, -0.1480525, 0.1506426
3: -0.0456092, 0.0076764, -0.0412824, 0.0064056, -0.0520148, 0.0489588
4: -0.0595792, 0.0643347, -0.0525507, 0.0577391, -0.1173183, 0.1168854
5: -0.0563963, 0.0673917, -0.0508078, 0.0616086, -0.1180050, 0.1181995
6: 0.8706347, 1.0145271, 0.8840936, 1.0117955, -0.1411608, 0.1304335
7: -0.0864343, 0.0519654, -0.0804949, 0.0409766, -0.1274109, 0.1324603
8: -0.0567425, 0.0728150, -0.0519781, 0.0648128, -0.1215553, 0.1247931
9: -0.0419873, 0.0595833, -0.0366160, 0.0551462, -0.0971334, 0.0961992

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2755403, upper bound: 0.2661805
time: 1.41 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2754987, upper bound: 0.2661474
time: 1.34 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0250220, 0.0277117, -0.0182845, 0.0231959, -0.0482178, 0.0459963
1: -0.0384569, 0.0706964, -0.0312416, 0.0620896, -0.1005465, 0.1019380
2: -0.0524794, 0.1024237, -0.0416778, 0.0879234, -0.1404029, 0.1441016
3: -0.0448024, 0.0074394, -0.0386239, 0.0056248, -0.0504273, 0.0460633
4: -0.0582687, 0.0631049, -0.0482322, 0.0536865, -0.1119552, 0.1113371
5: -0.0553543, 0.0663135, -0.0473739, 0.0580553, -0.1134096, 0.1136874
6: 0.8731441, 1.0140178, 0.8923634, 1.0101168, -0.1369727, 0.1216545
7: -0.0853269, 0.0499165, -0.0768455, 0.0342247, -0.1195515, 0.1267619
8: -0.0558542, 0.0713230, -0.0490506, 0.0598959, -0.1157501, 0.1203736
9: -0.0409858, 0.0587560, -0.0333156, 0.0524198, -0.0934055, 0.0920716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753169, upper bound: 0.2660903
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2754173, upper bound: 0.2660903
time: 1.25 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0431134, 0.0509768, -0.0328953, 0.0376117, -0.0807251, 0.0838721
1: -0.0571479, 0.1034210, -0.0465897, 0.0819143, -0.1390622, 0.1500106
2: -0.0788353, 0.1443908, -0.0651002, 0.1226009, -0.2014361, 0.2094910
3: -0.0577934, 0.0272100, -0.0519622, 0.0101923, -0.0679857, 0.0791722
4: -0.0910502, 0.0887097, -0.0728054, 0.0742626, -0.1653128, 0.1615151
5: -0.0849908, 0.0861105, -0.0672984, 0.0756218, -0.1606126, 0.1534090
6: 0.8203115, 1.0244371, 0.8486023, 1.0187846, -0.1984731, 0.1758348
7: -0.1129376, 0.0842753, -0.0956744, 0.0680277, -0.1809653, 0.1799497
8: -0.0729351, 0.1093559, -0.0637821, 0.0883426, -0.1612777, 0.1731380
9: -0.0723790, 0.0739468, -0.0533853, 0.0658979, -0.1382769, 0.1273321

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2762160
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2943456
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0371464, 0.0451870, -0.0331091, 0.0380095, -0.0751559, 0.0782961
1: -0.0506788, 0.0941043, -0.0467787, 0.0825544, -0.1332332, 0.1408830
2: -0.0728554, 0.1340758, -0.0655068, 0.1232034, -0.1960588, 0.1995826
3: -0.0552607, 0.0164405, -0.0521354, 0.0104717, -0.0657324, 0.0685759
4: -0.0816363, 0.0824405, -0.0732647, 0.0746921, -0.1563283, 0.1557051
5: -0.0767909, 0.0797414, -0.0677801, 0.0758381, -0.1526290, 0.1475214
6: 0.8338944, 1.0216308, 0.8478298, 1.0189228, -0.1850284, 0.1738009
7: -0.1038003, 0.0768542, -0.0961011, 0.0684800, -0.1722802, 0.1729553
8: -0.0688851, 0.0988630, -0.0640477, 0.0888951, -0.1577802, 0.1629106
9: -0.0638897, 0.0690586, -0.0539288, 0.0660639, -0.1299535, 0.1229875

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2759654
time: 1.35 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2939427
time: 1.40 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0267768, 0.0288879, -0.0221751, 0.0258036, -0.0525804, 0.0510630
1: -0.0403361, 0.0729380, -0.0354081, 0.0670596, -0.1073957, 0.1083461
2: -0.0552927, 0.1062004, -0.0479153, 0.0962967, -0.1515894, 0.1541156
3: -0.0464117, 0.0079120, -0.0421917, 0.0066727, -0.0530844, 0.0501037
4: -0.0608828, 0.0655579, -0.0540278, 0.0591252, -0.1200080, 0.1195857
5: -0.0574329, 0.0684643, -0.0519822, 0.0628240, -0.1202569, 0.1204466
6: 0.8681384, 1.0150337, 0.8812650, 1.0123696, -0.1442312, 0.1337687
7: -0.0875359, 0.0540035, -0.0817431, 0.0432860, -0.1308219, 0.1357466
8: -0.0576262, 0.0742992, -0.0529793, 0.0664945, -0.1241207, 0.1272786
9: -0.0429835, 0.0604062, -0.0377448, 0.0560786, -0.0990621, 0.0981510

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A2_B1_A1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2755403, upper bound: 0.2660753
time: 1.32 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2754986, upper bound: 0.2660518
time: 1.26 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0260173, 0.0283789, -0.0192272, 0.0238277, -0.0498451, 0.0476061
1: -0.0395228, 0.0719679, -0.0322511, 0.0632939, -0.1028166, 0.1042191
2: -0.0540752, 0.1045659, -0.0431892, 0.0899523, -0.1440275, 0.1477550
3: -0.0457152, 0.0077075, -0.0394884, 0.0058787, -0.0515940, 0.0471959
4: -0.0597514, 0.0644963, -0.0496365, 0.0550043, -0.1147557, 0.1141328
5: -0.0565333, 0.0675334, -0.0484905, 0.0592107, -0.1157440, 0.1160240
6: 0.8703048, 1.0145941, 0.8896741, 1.0106627, -0.1403579, 0.1249200
7: -0.0865798, 0.0522347, -0.0780321, 0.0364203, -0.1230001, 0.1302668
8: -0.0568593, 0.0730112, -0.0500025, 0.0614947, -0.1183540, 0.1230137
9: -0.0421189, 0.0596920, -0.0343888, 0.0533063, -0.0954252, 0.0940808

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A2_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2754660, upper bound: 0.2659926
time: 1.24 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2754145, upper bound: 0.2659520
time: 1.38 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0355787, 0.0426040, -0.0419201, 0.0498955, -0.0854742, 0.0845240
1: -0.0489617, 0.0899478, -0.0558690, 0.1016812, -0.1506429, 0.1458167
2: -0.0702022, 0.1301631, -0.0777082, 0.1424216, -0.2126237, 0.2078712
3: -0.0541360, 0.0136986, -0.0573110, 0.0251081, -0.0792441, 0.0710095
4: -0.0785694, 0.0796520, -0.0892251, 0.0875235, -0.1660929, 0.1688771
5: -0.0733425, 0.0783367, -0.0834595, 0.0848047, -0.1581472, 0.1617961
6: 0.8389094, 1.0205189, 0.8229213, 1.0239130, -0.1850036, 0.1975976
7: -0.1010295, 0.0737042, -0.1111499, 0.0828504, -0.1838799, 0.1848541
8: -0.0671156, 0.0952758, -0.0721640, 0.1073283, -0.1744438, 0.1674398
9: -0.0602069, 0.0679809, -0.0707856, 0.0729653, -0.1331721, 0.1387665

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A2_B2_A1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2934943, upper bound: 0.2938857
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2933704, upper bound: 0.2938765
time: 1.24 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0348873, 0.0413177, -0.0361421, 0.0436474, -0.0785347, 0.0774598
1: -0.0483505, 0.0878778, -0.0494645, 0.0916268, -0.1399773, 0.1373423
2: -0.0688876, 0.1282146, -0.0712687, 0.1317437, -0.2006312, 0.1994832
3: -0.0535759, 0.0127951, -0.0545903, 0.0144448, -0.0680207, 0.0673854
4: -0.0770842, 0.0782633, -0.0797754, 0.0807784, -0.1578626, 0.1580387
5: -0.0717852, 0.0776371, -0.0746104, 0.0789041, -0.1506893, 0.1522476
6: 0.8414068, 1.0200721, 0.8368834, 1.0208843, -0.1794775, 0.1831887
7: -0.0996497, 0.0722416, -0.1021488, 0.0748937, -0.1745434, 0.1743904
8: -0.0662566, 0.0934894, -0.0678129, 0.0967249, -0.1629816, 0.1613023
9: -0.0584492, 0.0674442, -0.0616349, 0.0684163, -0.1268655, 0.1290791

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2923237, upper bound: 0.2939110
time: 1.49 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2923237, upper bound: 0.2938361
time: 1.27 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0178949, 0.0229347, -0.0303160, 0.0324272, -0.0503221, 0.0532507
1: -0.0308244, 0.0615919, -0.0441263, 0.0774666, -0.1082910, 0.1057182
2: -0.0410532, 0.0870849, -0.0609668, 0.1147474, -0.1558006, 0.1480517
3: -0.0382666, 0.0055199, -0.0497047, 0.0088652, -0.0471318, 0.0552246
4: -0.0476518, 0.0531419, -0.0668813, 0.0705054, -0.1181572, 0.1200231
5: -0.0469124, 0.0575777, -0.0622412, 0.0728023, -0.1197148, 0.1198189
6: 0.8934748, 1.0098912, 0.8574324, 1.0170829, -0.1236082, 0.1524587
7: -0.0763550, 0.0333173, -0.0920443, 0.0622463, -0.1386013, 0.1253616
8: -0.0486571, 0.0592351, -0.0612002, 0.0812285, -0.1298856, 0.1204352
9: -0.0328720, 0.0520534, -0.0477193, 0.0637346, -0.0966067, 0.0997727

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=46, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2762224
time: 1.20 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2763835
time: 1.56 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0153115, 0.0212032, -0.0301401, 0.0320308, -0.0473422, 0.0513433
1: -0.0280577, 0.0582918, -0.0439380, 0.0772402, -0.1052979, 0.1022297
2: -0.0369115, 0.0815249, -0.0606849, 0.1141469, -0.1510584, 0.1422098
3: -0.0358975, 0.0048241, -0.0495321, 0.0088179, -0.0447154, 0.0543562
4: -0.0438034, 0.0495305, -0.0664459, 0.0702596, -0.1140630, 0.1159764
5: -0.0438525, 0.0544112, -0.0618858, 0.0725868, -0.1164393, 0.1162970
6: 0.9008440, 1.0083957, 0.8580797, 1.0169811, -0.1161371, 0.1503160
7: -0.0731029, 0.0273005, -0.0918103, 0.0618368, -0.1349397, 0.1191108
8: -0.0460483, 0.0548534, -0.0610226, 0.0807091, -0.1267574, 0.1158760
9: -0.0299310, 0.0496238, -0.0473504, 0.0635692, -0.0935003, 0.0969742

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2763133
time: 1.20 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2762773
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0358669, 0.0431403, -0.0424444, 0.0503801, -0.0862470, 0.0855848
1: -0.0492165, 0.0908107, -0.0564318, 0.1024607, -0.1516771, 0.1472425
2: -0.0707502, 0.1309755, -0.0782074, 0.1433040, -0.2140542, 0.2091830
3: -0.0543695, 0.0140752, -0.0575219, 0.0260499, -0.0804194, 0.0715971
4: -0.0791886, 0.0802309, -0.0900429, 0.0880465, -0.1672351, 0.1702738
5: -0.0739918, 0.0786284, -0.0841456, 0.0853745, -0.1593663, 0.1627740
6: 0.8378680, 1.0207053, 0.8217565, 1.0241479, -0.1862798, 0.1989489
7: -0.1016048, 0.0743140, -0.1119509, 0.0834673, -0.1850721, 0.1862649
8: -0.0674736, 0.0960206, -0.0725013, 0.1082368, -0.1757105, 0.1685220
9: -0.0609397, 0.0682047, -0.0714951, 0.0734051, -0.1343448, 0.1396998

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2775888, upper bound: 0.2699899
time: 1.29 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2775888, upper bound: 0.2945140
time: 1.28 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0329028, 0.0376256, -0.0419026, 0.0498794, -0.0827822, 0.0795282
1: -0.0465963, 0.0819367, -0.0558502, 0.1016553, -0.1482515, 0.1377869
2: -0.0651144, 0.1226220, -0.0776915, 0.1423922, -0.2075066, 0.2003136
3: -0.0519683, 0.0102021, -0.0573040, 0.0250767, -0.0770450, 0.0675060
4: -0.0728214, 0.0742776, -0.0891978, 0.0875061, -0.1603275, 0.1634755
5: -0.0673153, 0.0756294, -0.0834366, 0.0847857, -0.1521010, 0.1590660
6: 0.8485754, 1.0187895, 0.8229600, 1.0239053, -0.1753299, 0.1958296
7: -0.0956893, 0.0680434, -0.1111232, 0.0828298, -0.1785191, 0.1791667
8: -0.0637914, 0.0883620, -0.0721527, 0.1072980, -0.1710894, 0.1605147
9: -0.0534043, 0.0659037, -0.0707620, 0.0729506, -0.1263549, 0.1366657

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_A2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2923237, upper bound: 0.2941596
time: 1.28 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2923237, upper bound: 0.2940040
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0192539, 0.0238456, -0.0311400, 0.0341967, -0.0534506, 0.0549856
1: -0.0322797, 0.0633279, -0.0449671, 0.0785381, -0.1108178, 0.1082949
2: -0.0432319, 0.0900097, -0.0622438, 0.1174278, -0.1606597, 0.1522535
3: -0.0395128, 0.0058859, -0.0504752, 0.0090814, -0.0485942, 0.0563611
4: -0.0496762, 0.0550416, -0.0688624, 0.0716029, -0.1212792, 0.1239040
5: -0.0485221, 0.0592434, -0.0638366, 0.0737647, -0.1222868, 0.1230800
6: 0.8895983, 1.0106782, 0.8545427, 1.0175982, -0.1279999, 0.1561354
7: -0.0780657, 0.0364824, -0.0930893, 0.0641446, -0.1422103, 0.1295717
8: -0.0500294, 0.0615399, -0.0619930, 0.0836000, -0.1336294, 0.1235329
9: -0.0344191, 0.0533314, -0.0494415, 0.0644730, -0.0988921, 0.1027729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2602165, upper bound: 0.2757568
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2602165, upper bound: 0.2757116
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0165906, 0.0220605, -0.0317323, 0.0353776, -0.0519682, 0.0537928
1: -0.0294275, 0.0599257, -0.0455281, 0.0793191, -0.1087466, 0.1054538
2: -0.0389621, 0.0842778, -0.0631160, 0.1192166, -0.1581787, 0.1473938
3: -0.0370705, 0.0051686, -0.0509894, 0.0092307, -0.0463012, 0.0561580
4: -0.0457088, 0.0513185, -0.0702259, 0.0723353, -0.1180442, 0.1215444
5: -0.0453675, 0.0559790, -0.0649110, 0.0744068, -0.1197743, 0.1208900
6: 0.8971956, 1.0091360, 0.8526143, 1.0180086, -0.1208130, 0.1565216
7: -0.0747130, 0.0302795, -0.0937866, 0.0654872, -0.1402003, 0.1240661
8: -0.0473400, 0.0570228, -0.0625220, 0.0852400, -0.1325800, 0.1195448
9: -0.0313872, 0.0508267, -0.0506735, 0.0649657, -0.0963529, 0.1015001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2602165, upper bound: 0.2757155
time: 1.14 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2602165, upper bound: 0.2756804
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0369381, 0.0448677, -0.0447473, 0.0521542, -0.0890923, 0.0896150
1: -0.0504270, 0.0935905, -0.0588297, 0.1053368, -0.1557638, 0.1524202
2: -0.0725264, 0.1335922, -0.0802246, 0.1465622, -0.2190886, 0.2138169
3: -0.0551217, 0.0160266, -0.0584663, 0.0295215, -0.0846432, 0.0744928
4: -0.0812504, 0.0820958, -0.0930818, 0.0902411, -0.1714916, 0.1751776
5: -0.0763387, 0.0795678, -0.0866584, 0.0879689, -0.1643077, 0.1662262
6: 0.8345143, 1.0214759, 0.8173395, 1.0250417, -0.1905274, 0.2041364
7: -0.1034578, 0.0764476, -0.1148847, 0.0864441, -0.1899019, 0.1913323
8: -0.0686628, 0.0984197, -0.0740054, 0.1115991, -0.1802618, 0.1724251
9: -0.0634221, 0.0689255, -0.0742525, 0.0750159, -0.1384380, 0.1431780

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=17, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2926339, upper bound: 0.2943045
time: 1.18 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2923237, upper bound: 0.2942958
time: 1.26 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0338339, 0.0393578, -0.0462428, 0.0531963, -0.0870301, 0.0856006
1: -0.0474193, 0.0847242, -0.0603179, 0.1070568, -0.1544761, 0.1450421
2: -0.0668847, 0.1252459, -0.0814541, 0.1485156, -0.2154004, 0.2067000
3: -0.0527225, 0.0114187, -0.0590617, 0.0315930, -0.0843155, 0.0704804
4: -0.0748214, 0.0761477, -0.0949313, 0.0915964, -0.1664178, 0.1710790
5: -0.0694125, 0.0765714, -0.0881342, 0.0896179, -0.1590304, 0.1647056
6: 0.8452120, 1.0193912, 0.8147093, 1.0256165, -0.1804045, 0.2046819
7: -0.0975475, 0.0700131, -0.1166077, 0.0883757, -0.1859233, 0.1866208
8: -0.0649480, 0.0907676, -0.0749525, 0.1136251, -0.1785732, 0.1657201
9: -0.0557713, 0.0666265, -0.0759261, 0.0759619, -0.1317332, 0.1425526

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=17, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2923237, upper bound: 0.2941162
time: 1.23 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2923237, upper bound: 0.2939759
time: 1.13 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0318665, 0.0356452, -0.0248076, 0.0275681, -0.0594346, 0.0604528
1: -0.0456553, 0.0794961, -0.0382273, 0.0704226, -0.1160779, 0.1177234
2: -0.0633137, 0.1196220, -0.0521359, 0.1019623, -0.1652760, 0.1717578
3: -0.0511059, 0.0092646, -0.0446059, 0.0073817, -0.0584876, 0.0538704
4: -0.0705349, 0.0725013, -0.0579494, 0.0628053, -0.1333402, 0.1304507
5: -0.0651545, 0.0745524, -0.0551005, 0.0660507, -0.1312052, 0.1296529
6: 0.8521776, 1.0181015, 0.8737556, 1.0138937, -0.1617162, 0.1443458
7: -0.0939446, 0.0657916, -0.0850570, 0.0494173, -0.1433620, 0.1508486
8: -0.0626419, 0.0856116, -0.0556377, 0.0709595, -0.1336014, 0.1412493
9: -0.0509526, 0.0650774, -0.0407418, 0.0585544, -0.1095070, 0.1058192

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2757875, upper bound: 0.2615459
time: 1.14 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2757875, upper bound: 0.2673921
time: 1.28 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0308873, 0.0336929, -0.0217277, 0.0255037, -0.0563910, 0.0554205
1: -0.0447276, 0.0782048, -0.0349289, 0.0664881, -0.1112157, 0.1131337
2: -0.0618717, 0.1166646, -0.0471980, 0.0953338, -0.1572055, 0.1638626
3: -0.0502558, 0.0090177, -0.0417814, 0.0065522, -0.0568080, 0.0507991
4: -0.0682807, 0.0712904, -0.0533613, 0.0584997, -0.1267804, 0.1246518
5: -0.0633781, 0.0734907, -0.0514523, 0.0622756, -0.1256537, 0.1249429
6: 0.8553656, 1.0174233, 0.8825415, 1.0121104, -0.1567447, 0.1348818
7: -0.0927918, 0.0635716, -0.0811799, 0.0422439, -0.1350356, 0.1447515
8: -0.0617672, 0.0829002, -0.0525275, 0.0657356, -0.1275029, 0.1354277
9: -0.0489159, 0.0642627, -0.0372354, 0.0556579, -0.1045738, 0.1014982

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2786367, upper bound: 0.2673906
time: 1.79 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2787397, upper bound: 0.2673906
time: 1.23 seconds

## BFS IS instance: IS_A2_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0652591, 0.0672161, -0.0371023, 0.0451194, -0.1103785, 0.1043184
1: -0.0786906, 0.1255587, -0.0506255, 0.0939955, -0.1726861, 0.1761842
2: -0.0964834, 0.1706382, -0.0727857, 0.1339735, -0.2304569, 0.2434239
3: -0.0669983, 0.0559359, -0.0552313, 0.0163528, -0.0833511, 0.1111672
4: -0.1147890, 0.1072258, -0.0815546, 0.0823674, -0.1971564, 0.1887805
5: -0.1064869, 0.1095017, -0.0766951, 0.0797046, -0.1861916, 0.1861968
6: 0.7840958, 1.0316331, 0.8340255, 1.0215979, -0.2375021, 0.1976076
7: -0.1359039, 0.1108654, -0.1037278, 0.0767682, -0.2126721, 0.2145932
8: -0.0861886, 0.1429430, -0.0688380, 0.0987692, -0.1849578, 0.2117810
9: -0.0992649, 0.0858665, -0.0637907, 0.0690305, -0.1682954, 0.1496572

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2785825
time: 1.19 seconds

## Relational analysis of IS_A2_B1_A1_B2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2945554
time: 1.20 seconds

## BFS IS instance: IS_A2_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0541445, 0.0580956, -0.0371184, 0.0451441, -0.0992886, 0.0952140
1: -0.0677561, 0.1151438, -0.0506450, 0.0940351, -0.1617912, 0.1657889
2: -0.0878393, 0.1576996, -0.0728112, 0.1340108, -0.2218501, 0.2305107
3: -0.0618613, 0.0420767, -0.0552421, 0.0163848, -0.0782461, 0.0973188
4: -0.1036271, 0.0983191, -0.0815845, 0.0823941, -0.1860212, 0.1799035
5: -0.0958861, 0.0973714, -0.0767301, 0.0797180, -0.1756042, 0.1741015
6: 0.8019920, 1.0283185, 0.8339776, 1.0216100, -0.2196180, 0.1943409
7: -0.1247089, 0.0978827, -0.1037542, 0.0767997, -0.2015086, 0.2016369
8: -0.0794057, 0.1254222, -0.0688552, 0.0988035, -0.1782092, 0.1942774
9: -0.0851191, 0.0804101, -0.0638268, 0.0690408, -0.1541598, 0.1442369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2782676
time: 1.06 seconds

## Relational analysis of IS_A2_B1_A1_B2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2942308
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0314631, 0.0348409, -0.0258371, 0.0282581, -0.0597212, 0.0606780
1: -0.0452731, 0.0789641, -0.0393297, 0.0717377, -0.1170108, 0.1182938
2: -0.0627195, 0.1184036, -0.0537862, 0.1041779, -0.1668975, 0.1721898
3: -0.0507557, 0.0091628, -0.0455500, 0.0076590, -0.0584146, 0.0547128
4: -0.0696061, 0.0720024, -0.0594829, 0.0642444, -0.1338505, 0.1314854
5: -0.0644227, 0.0741149, -0.0563198, 0.0673125, -0.1317352, 0.1304348
6: 0.8534910, 1.0178219, 0.8708188, 1.0144897, -0.1609986, 0.1470031
7: -0.0934696, 0.0648770, -0.0863530, 0.0518149, -0.1452845, 0.1512300
8: -0.0622816, 0.0844945, -0.0566773, 0.0727054, -0.1349870, 0.1411718
9: -0.0501136, 0.0647418, -0.0419137, 0.0595225, -0.1096361, 0.1066555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2757892, upper bound: 0.2615459
time: 1.17 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2757892, upper bound: 0.2673956
time: 1.22 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0304422, 0.0327116, -0.0226925, 0.0261504, -0.0565926, 0.0554041
1: -0.0442614, 0.0776292, -0.0359622, 0.0677206, -0.1119820, 0.1135914
2: -0.0611691, 0.1151782, -0.0487448, 0.0974103, -0.1585794, 0.1639230
3: -0.0498286, 0.0088993, -0.0426662, 0.0068120, -0.0566406, 0.0515655
4: -0.0671936, 0.0706819, -0.0547986, 0.0598485, -0.1270421, 0.1254804
5: -0.0624962, 0.0729570, -0.0525951, 0.0634582, -0.1259544, 0.1255522
6: 0.8569677, 1.0171559, 0.8797892, 1.0126690, -0.1557012, 0.1373667
7: -0.0922123, 0.0625403, -0.0823944, 0.0444911, -0.1367034, 0.1449347
8: -0.0613276, 0.0816013, -0.0535019, 0.0673721, -0.1286997, 0.1351032
9: -0.0479840, 0.0638534, -0.0383338, 0.0565652, -0.1045492, 0.1021872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2758206, upper bound: 0.2615459
time: 1.25 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2758206, upper bound: 0.2673760
time: 1.21 seconds

## BFS IS instance: IS_A2_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0635229, 0.0657130, -0.0388697, 0.0470774, -0.1106003, 0.1045827
1: -0.0770708, 0.1237483, -0.0525950, 0.0971462, -0.1742170, 0.1763433
2: -0.0952729, 0.1683044, -0.0748037, 0.1372888, -0.2325617, 0.2431082
3: -0.0660135, 0.0539951, -0.0560839, 0.0196292, -0.0856427, 0.1100789
4: -0.1128738, 0.1059648, -0.0844677, 0.0844812, -0.1973550, 0.1904325
5: -0.1050024, 0.1074926, -0.0794682, 0.0814901, -0.1864925, 0.1869607
6: 0.7868809, 1.0311691, 0.8296967, 1.0225471, -0.2356663, 0.2014724
7: -0.1342448, 0.1085136, -0.1064900, 0.0792617, -0.2135065, 0.2150036
8: -0.0850228, 0.1404894, -0.0702015, 0.1020430, -0.1870658, 0.2106908
9: -0.0968808, 0.0851024, -0.0666583, 0.0704066, -0.1672874, 0.1517606

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2783924
time: 1.15 seconds

## Relational analysis of IS_A2_B1_A2_B2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2945554
time: 1.26 seconds

## BFS IS instance: IS_A2_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0523243, 0.0570885, -0.0387520, 0.0469688, -0.0992930, 0.0958406
1: -0.0661279, 0.1134814, -0.0524688, 0.0969713, -0.1630993, 0.1659502
2: -0.0863907, 0.1558117, -0.0746917, 0.1370908, -0.2234815, 0.2305034
3: -0.0612858, 0.0397542, -0.0560365, 0.0194180, -0.0807039, 0.0957907
4: -0.1018396, 0.0968581, -0.0842843, 0.0843639, -0.1862035, 0.1811424
5: -0.0941097, 0.0957776, -0.0793143, 0.0813623, -0.1754719, 0.1750919
6: 0.8046853, 1.0277630, 0.8299578, 1.0224942, -0.2178089, 0.1978052
7: -0.1230435, 0.0958330, -0.1063103, 0.0791233, -0.2021668, 0.2021433
8: -0.0784903, 0.1224861, -0.0701258, 0.1018392, -0.1803295, 0.1926119
9: -0.0829313, 0.0794958, -0.0664992, 0.0703080, -0.1532393, 0.1459949

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2781647
time: 1.09 seconds

## Relational analysis of IS_A2_B1_A2_B2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2942308
time: 1.42 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0239603, 0.0270001, -0.0336164, 0.0389533, -0.0629136, 0.0606165
1: -0.0373199, 0.0693401, -0.0472271, 0.0840730, -0.1213929, 0.1165672
2: -0.0507773, 0.1001388, -0.0664712, 0.1246331, -0.1754104, 0.1666100
3: -0.0438288, 0.0071535, -0.0525464, 0.0111345, -0.0549633, 0.0596999
4: -0.0566871, 0.0616207, -0.0743543, 0.0757108, -0.1323980, 0.1359751
5: -0.0540968, 0.0650121, -0.0689226, 0.0763513, -0.1304481, 0.1339347
6: 0.8761728, 1.0134032, 0.8459976, 1.0192506, -0.1430779, 0.1674055
7: -0.0839903, 0.0474438, -0.0971134, 0.0695531, -0.1535434, 0.1445572
8: -0.0547821, 0.0695223, -0.0646779, 0.0902058, -0.1449879, 0.1342001
9: -0.0397771, 0.0577575, -0.0552184, 0.0664577, -0.1062348, 0.1129759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=46, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2784293
time: 0.96 seconds

## Relational analysis of IS_A2_B2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2785324
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0213382, 0.0252426, -0.0329970, 0.0378009, -0.0591391, 0.0582396
1: -0.0345118, 0.0659905, -0.0466795, 0.0822187, -0.1167305, 0.1126700
2: -0.0465735, 0.0944954, -0.0652935, 0.1228874, -0.1694609, 0.1597890
3: -0.0414242, 0.0064473, -0.0520446, 0.0103251, -0.0517493, 0.0584919
4: -0.0527811, 0.0579553, -0.0730238, 0.0744668, -0.1272479, 0.1309791
5: -0.0509910, 0.0617981, -0.0675275, 0.0757247, -0.1267156, 0.1293256
6: 0.8836525, 1.0118849, 0.8482351, 1.0188506, -0.1351981, 0.1636499
7: -0.0806895, 0.0413368, -0.0958773, 0.0682427, -0.1489322, 0.1372141
8: -0.0521342, 0.0650750, -0.0639083, 0.0886054, -0.1407396, 0.1289833
9: -0.0367920, 0.0552915, -0.0536437, 0.0659769, -0.1027689, 0.1089353

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_B1_A1_B2_B1

### Relational analysis result of IS_A2_B2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2784922
time: 1.31 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2_B2

### Relational analysis result of IS_A2_B2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2784056
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0373123, 0.0454413, -0.0730361, 0.0740299, -0.1113422, 0.1184775
1: -0.0508795, 0.0945136, -0.0862146, 0.1328496, -0.1837291, 0.1807282
2: -0.0731175, 0.1344611, -0.1021301, 0.1807881, -0.2539056, 0.2365912
3: -0.0553715, 0.0167702, -0.0719217, 0.0637516, -0.1191231, 0.0886919
4: -0.0819438, 0.0827150, -0.1241740, 0.1132073, -0.1951512, 0.2068890
5: -0.0771511, 0.0798797, -0.1129441, 0.1188965, -0.1960476, 0.1928238
6: 0.8334003, 1.0217539, 0.7723564, 1.0342203, -0.2008200, 0.2493975
7: -0.1040731, 0.0771781, -0.1441748, 0.1203354, -0.2244085, 0.2213530
8: -0.0690622, 0.0992163, -0.0927714, 0.1528237, -0.2218858, 0.1919876
9: -0.0642622, 0.0691648, -0.1088660, 0.0936810, -0.1579432, 0.1780308

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_B1_A2_A1_B1

### Relational analysis result of IS_A2_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2926339
time: 1.27 seconds

## Relational analysis of IS_A2_B2_B1_A2_A1_B2

### Relational analysis result of IS_A2_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2944311
time: 1.40 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0372334, 0.0453204, -0.0603521, 0.0626167, -0.0998501, 0.1056725
1: -0.0507840, 0.0943190, -0.0737343, 0.1208129, -0.1715970, 0.1680533
2: -0.0729929, 0.1342779, -0.0927793, 0.1645030, -0.2374960, 0.2270573
3: -0.0553188, 0.0166133, -0.0643411, 0.0499971, -0.1053160, 0.0809544
4: -0.0817976, 0.0825845, -0.1097231, 0.1033672, -0.1851648, 0.1923076
5: -0.0769799, 0.0798139, -0.1019445, 0.1036560, -0.1806358, 0.1817584
6: 0.8336352, 1.0216953, 0.7922501, 1.0302129, -0.1965777, 0.2294452
7: -0.1039434, 0.0770242, -0.1308272, 0.1048734, -0.2088168, 0.2078514
8: -0.0689780, 0.0990483, -0.0829230, 0.1354352, -0.2044132, 0.1819713
9: -0.0640851, 0.0691144, -0.0927837, 0.0835283, -0.1476134, 0.1618981

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2942497, upper bound: 0.2926339
time: 1.15 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2942497, upper bound: 0.2944311
time: 1.51 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0257759, 0.0282171, -0.0386202, 0.0468469, -0.0726228, 0.0668373
1: -0.0392643, 0.0716596, -0.0523272, 0.0967754, -0.1360397, 0.1239868
2: -0.0536882, 0.1040463, -0.0745662, 0.1368689, -0.1905571, 0.1786125
3: -0.0454938, 0.0076425, -0.0559835, 0.0191812, -0.0646750, 0.0636260
4: -0.0593918, 0.0641589, -0.0840785, 0.0842324, -0.1436243, 0.1482374
5: -0.0562474, 0.0672376, -0.0791418, 0.0812189, -0.1374663, 0.1463794
6: 0.8709934, 1.0144541, 0.8302509, 1.0224354, -0.1514420, 0.1842033
7: -0.0862760, 0.0516724, -0.1061089, 0.0789681, -0.1652441, 0.1577813
8: -0.0566155, 0.0726018, -0.0700410, 0.1016107, -0.1582262, 0.1426428
9: -0.0418441, 0.0594650, -0.0663207, 0.0701974, -0.1120414, 0.1257857

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_B2_A1_B1_B1

### Relational analysis result of IS_A2_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2786180
time: 1.10 seconds

## Relational analysis of IS_A2_B2_B2_A1_B1_B2

### Relational analysis result of IS_A2_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2784940
time: 1.26 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0260228, 0.0283826, -0.0526577, 0.0580374, -0.0840602, 0.0810403
1: -0.0395287, 0.0719749, -0.0672321, 0.1147827, -0.1543114, 0.1392070
2: -0.0540840, 0.1045776, -0.0871660, 0.1572504, -0.2113344, 0.1917436
3: -0.0457203, 0.0077090, -0.0618280, 0.0409366, -0.0866569, 0.0695370
4: -0.0597596, 0.0645040, -0.1029693, 0.0978923, -0.1576519, 0.1674733
5: -0.0565399, 0.0675402, -0.0949905, 0.0972306, -0.1537704, 0.1625306
6: 0.8702891, 1.0145972, 0.8024907, 1.0278592, -0.1575701, 0.2121065
7: -0.0865868, 0.0522476, -0.1246124, 0.0972166, -0.1838034, 0.1768599
8: -0.0568648, 0.0730205, -0.0793526, 0.1225974, -0.1794622, 0.1523731
9: -0.0421252, 0.0596972, -0.0835335, 0.0803571, -0.1224823, 0.1432307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2785338
time: 1.12 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2784057
time: 1.28 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0391839, 0.0473677, -0.0735336, 0.0744288, -0.1136127, 0.1209013
1: -0.0529322, 0.0976134, -0.0866807, 0.1332757, -0.1862079, 0.1842941
2: -0.0751029, 0.1378175, -0.1025023, 0.1813819, -0.2564847, 0.2403198
3: -0.0562103, 0.0201936, -0.0722168, 0.0642084, -0.1204187, 0.0924104
4: -0.0849577, 0.0847946, -0.1247240, 0.1136062, -0.1985639, 0.2095186
5: -0.0798793, 0.0818315, -0.1133218, 0.1195170, -0.1993963, 0.1951533
6: 0.8289988, 1.0226878, 0.7716417, 1.0343720, -0.2053732, 0.2510461
7: -0.1069700, 0.0796312, -0.1446904, 0.1208887, -0.2278587, 0.2243217
8: -0.0704037, 0.1025874, -0.0931595, 0.1534011, -0.2238047, 0.1957469
9: -0.0670834, 0.0706702, -0.1094270, 0.0942049, -0.1612883, 0.1800972

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_B2_A2_A1_B1

### Relational analysis result of IS_A2_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2944494
time: 1.33 seconds

## Relational analysis of IS_A2_B2_B2_A2_A1_B2

### Relational analysis result of IS_A2_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2942633
time: 1.29 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0407435, 0.0488087, -0.0608745, 0.0631328, -0.1038763, 0.1096832
1: -0.0546062, 0.0999321, -0.0742905, 0.1212899, -0.1758962, 0.1742226
2: -0.0765879, 0.1404420, -0.0931950, 0.1651212, -0.2417091, 0.2336370
3: -0.0568377, 0.0229949, -0.0646144, 0.0506637, -0.1075013, 0.0876093
4: -0.0873901, 0.0863502, -0.1102360, 0.1038001, -0.1911903, 0.1965862
5: -0.0819201, 0.0835262, -0.1024542, 0.1042909, -0.1862110, 0.1859804
6: 0.8255345, 1.0233861, 0.7913609, 1.0303724, -0.2048379, 0.2320253
7: -0.1093526, 0.0814662, -0.1313969, 0.1054616, -0.2148142, 0.2128631
8: -0.0714071, 0.1052898, -0.0832683, 0.1362778, -0.2076849, 0.1885581
9: -0.0691938, 0.0719784, -0.0934541, 0.0837908, -0.1529846, 0.1654325

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_B2_A2_A2_B1

### Relational analysis result of IS_A2_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2941940, upper bound: 0.2926339
time: 1.15 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_B2

### Relational analysis result of IS_A2_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2941940, upper bound: 0.2944494
time: 1.25 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.98 seconds
IS_A1_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2755403, upper bound: 0.2661805
IS_A1_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2754987, upper bound: 0.2661474
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2753169, upper bound: 0.2660903
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2754173, upper bound: 0.2660903
IS_A1_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2762160
IS_A1_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2943456
IS_A1_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2759654
IS_A1_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2939427
IS_A1_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2755403, upper bound: 0.2660753
IS_A1_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2754986, upper bound: 0.2660518
IS_A1_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2754660, upper bound: 0.2659926
IS_A1_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2754145, upper bound: 0.2659520
IS_A1_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2934943, upper bound: 0.2938857
IS_A1_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2933704, upper bound: 0.2938765
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2923237, upper bound: 0.2939110
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2923237, upper bound: 0.2938361
IS_A1_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2762224
IS_A1_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2763835
IS_A1_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2763133
IS_A1_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2762773
IS_A1_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2775888, upper bound: 0.2699899
IS_A1_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2775888, upper bound: 0.2945140
IS_A1_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2923237, upper bound: 0.2941596
IS_A1_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2923237, upper bound: 0.2940040
IS_A1_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2602165, upper bound: 0.2757568
IS_A1_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2602165, upper bound: 0.2757116
IS_A1_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2602165, upper bound: 0.2757155
IS_A1_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2602165, upper bound: 0.2756804
IS_A1_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2926339, upper bound: 0.2943045
IS_A1_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2923237, upper bound: 0.2942958
IS_A1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2923237, upper bound: 0.2941162
IS_A1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2923237, upper bound: 0.2939759
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2757875, upper bound: 0.2615459
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2757875, upper bound: 0.2673921
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2786367, upper bound: 0.2673906
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2787397, upper bound: 0.2673906
IS_A2_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2785825
IS_A2_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2945554
IS_A2_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2782676
IS_A2_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2942308
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2757892, upper bound: 0.2615459
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2757892, upper bound: 0.2673956
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2758206, upper bound: 0.2615459
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2758206, upper bound: 0.2673760
IS_A2_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2783924
IS_A2_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2945554
IS_A2_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2781647
IS_A2_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2942308
IS_A2_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2784293
IS_A2_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2785324
IS_A2_B2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2784922
IS_A2_B2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2784056
IS_A2_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2926339
IS_A2_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2944311
IS_A2_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2942497, upper bound: 0.2926339
IS_A2_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2942497, upper bound: 0.2944311
IS_A2_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2786180
IS_A2_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2784940
IS_A2_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2785338
IS_A2_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2784057
IS_A2_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2944494
IS_A2_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2945554, upper bound: 0.2942633
IS_A2_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2941940, upper bound: 0.2926339
IS_A2_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 6, lower bound: -0.2941940, upper bound: 0.2944494

## BFS IS instance: IS_A1_B1_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0238771, 0.0269443, -0.0204082, 0.0246193, -0.0484964, 0.0473525
1: -0.0372307, 0.0692339, -0.0335159, 0.0648025, -0.1020333, 0.1027498
2: -0.0506439, 0.0999596, -0.0450826, 0.0924939, -0.1431378, 0.1450422
3: -0.0437525, 0.0071311, -0.0405714, 0.0061968, -0.0499493, 0.0477025
4: -0.0565632, 0.0615044, -0.0513958, 0.0566552, -0.1132184, 0.1129002
5: -0.0539982, 0.0649101, -0.0498894, 0.0606583, -0.1146565, 0.1147995
6: 0.8764101, 1.0133548, 0.8863054, 1.0113465, -0.1349363, 0.1270494
7: -0.0838856, 0.0472499, -0.0795188, 0.0391708, -0.1230564, 0.1267688
8: -0.0546980, 0.0693812, -0.0511951, 0.0634977, -0.1181957, 0.1205762
9: -0.0396824, 0.0576792, -0.0357332, 0.0544170, -0.0940993, 0.0934125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2651796, upper bound: 0.2569876
time: 1.20 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2613599, upper bound: 0.2502546
time: 1.29 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0284730, 0.0300248, -0.0206555, 0.0247850, -0.0532580, 0.0506803
1: -0.0421526, 0.0751048, -0.0337807, 0.0651184, -0.1072710, 0.1088855
2: -0.0580121, 0.1098508, -0.0454790, 0.0930262, -0.1510383, 0.1553298
3: -0.0479671, 0.0083689, -0.0407982, 0.0062634, -0.0542306, 0.0491670
4: -0.0634095, 0.0679290, -0.0517641, 0.0570009, -0.1204104, 0.1196931
5: -0.0594420, 0.0705434, -0.0501823, 0.0609614, -0.1204033, 0.1207257
6: 0.8632999, 1.0160160, 0.8855999, 1.0114895, -0.1481896, 0.1304161
7: -0.0896711, 0.0579539, -0.0798301, 0.0397468, -0.1294179, 0.1377841
8: -0.0593390, 0.0771761, -0.0514448, 0.0639172, -0.1232562, 0.1286209
9: -0.0449145, 0.0620014, -0.0360148, 0.0546495, -0.0995640, 0.0980162

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2647452, upper bound: 0.2569295
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2607098, upper bound: 0.2501556
time: 1.17 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0242942, 0.0272239, -0.0162280, 0.0218175, -0.0461116, 0.0434519
1: -0.0376774, 0.0697666, -0.0290393, 0.0594626, -0.0971399, 0.0988059
2: -0.0513125, 0.1008573, -0.0383809, 0.0834974, -0.1348100, 0.1392381
3: -0.0441350, 0.0072434, -0.0367380, 0.0050710, -0.0492059, 0.0439814
4: -0.0571845, 0.0620874, -0.0451687, 0.0508117, -0.1079962, 0.1072562
5: -0.0544922, 0.0654213, -0.0449381, 0.0555346, -0.1100268, 0.1103594
6: 0.8752204, 1.0135964, 0.8982297, 1.0089263, -0.1337059, 0.1153667
7: -0.0844106, 0.0482213, -0.0742566, 0.0294352, -0.1138458, 0.1224780
8: -0.0551192, 0.0700885, -0.0469739, 0.0564079, -0.1115271, 0.1170624
9: -0.0401572, 0.0580715, -0.0309744, 0.0504857, -0.0906429, 0.0890459

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=44, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2621067, upper bound: 0.2504352
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2601264, upper bound: 0.2500037
time: 2.96 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0245139, 0.0273712, -0.0209577, 0.0249876, -0.0495015, 0.0483289
1: -0.0379127, 0.0700473, -0.0341044, 0.0655046, -0.1034173, 0.1041517
2: -0.0516649, 0.1013302, -0.0459636, 0.0936767, -0.1453416, 0.1472938
3: -0.0443365, 0.0073026, -0.0410753, 0.0063448, -0.0506813, 0.0483780
4: -0.0575118, 0.0623946, -0.0522143, 0.0574235, -0.1149353, 0.1146090
5: -0.0547525, 0.0656907, -0.0505403, 0.0613319, -0.1160844, 0.1162310
6: 0.8745933, 1.0137237, 0.8847377, 1.0116646, -0.1370713, 0.1289860
7: -0.0846872, 0.0487332, -0.0802106, 0.0404507, -0.1251380, 0.1289438
8: -0.0553411, 0.0704612, -0.0517500, 0.0644298, -0.1197708, 0.1222113
9: -0.0404074, 0.0582781, -0.0363589, 0.0549338, -0.0953411, 0.0946370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2621030, upper bound: 0.2503715
time: 1.27 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2601684, upper bound: 0.2499768
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0186451, 0.0234376, -0.0328953, 0.0376117, -0.0562568, 0.0563329
1: -0.0316278, 0.0625503, -0.0465897, 0.0819143, -0.1135421, 0.1091400
2: -0.0422560, 0.0886996, -0.0651002, 0.1226009, -0.1648569, 0.1537998
3: -0.0389546, 0.0057220, -0.0519622, 0.0101923, -0.0491469, 0.0576842
4: -0.0487694, 0.0541906, -0.0728054, 0.0742626, -0.1230320, 0.1269960
5: -0.0478011, 0.0584973, -0.0672984, 0.0756218, -0.1234229, 0.1257957
6: 0.8913347, 1.0103257, 0.8486023, 1.0187846, -0.1274500, 0.1617234
7: -0.0772994, 0.0350645, -0.0956744, 0.0680277, -0.1453271, 0.1307389
8: -0.0494147, 0.0605075, -0.0637821, 0.0883426, -0.1377574, 0.1242895
9: -0.0337261, 0.0527589, -0.0533853, 0.0658979, -0.0996240, 0.1061442

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2761200
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2761858
time: 2.45 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0369322, 0.0448586, -0.0328953, 0.0376117, -0.0745439, 0.0777539
1: -0.0504199, 0.0935759, -0.0465897, 0.0819143, -0.1323341, 0.1401655
2: -0.0725170, 0.1335784, -0.0651002, 0.1226009, -0.1951179, 0.1986787
3: -0.0551177, 0.0160148, -0.0519622, 0.0101923, -0.0653100, 0.0679770
4: -0.0812394, 0.0820860, -0.0728054, 0.0742626, -0.1555020, 0.1548913
5: -0.0763258, 0.0795628, -0.0672984, 0.0756218, -0.1519476, 0.1468612
6: 0.8345318, 1.0214716, 0.8486023, 1.0187846, -0.1842529, 0.1728693
7: -0.1034480, 0.0764361, -0.0956744, 0.0680277, -0.1714757, 0.1721105
8: -0.0686564, 0.0984070, -0.0637821, 0.0883426, -0.1569991, 0.1621890
9: -0.0634088, 0.0689217, -0.0533853, 0.0658979, -0.1293066, 0.1223070

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2942374
time: 1.34 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2942078
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0159641, 0.0216406, -0.0331091, 0.0380095, -0.0539736, 0.0547497
1: -0.0287566, 0.0591254, -0.0467787, 0.0825544, -0.1113110, 0.1059041
2: -0.0379577, 0.0829295, -0.0655068, 0.1232034, -0.1611611, 0.1484362
3: -0.0364960, 0.0049999, -0.0521354, 0.0104717, -0.0469677, 0.0571353
4: -0.0447756, 0.0504427, -0.0732647, 0.0746921, -0.1194677, 0.1237074
5: -0.0446255, 0.0552111, -0.0677801, 0.0758381, -0.1204636, 0.1229911
6: 0.8989826, 1.0087732, 0.8478298, 1.0189228, -0.1199402, 0.1609434
7: -0.0739244, 0.0288204, -0.0961011, 0.0684800, -0.1424044, 0.1249215
8: -0.0467074, 0.0559603, -0.0640477, 0.0888951, -0.1356025, 0.1200079
9: -0.0306740, 0.0502376, -0.0539288, 0.0660639, -0.0967378, 0.1041664

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2758563
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2759379
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0337052, 0.0391186, -0.0331091, 0.0380095, -0.0717148, 0.0722277
1: -0.0473056, 0.0843391, -0.0467787, 0.0825544, -0.1298600, 0.1311178
2: -0.0666401, 0.1248835, -0.0655068, 0.1232034, -0.1898435, 0.1903902
3: -0.0526183, 0.0112506, -0.0521354, 0.0104717, -0.0630900, 0.0633860
4: -0.0745452, 0.0758893, -0.0732647, 0.0746921, -0.1492372, 0.1491539
5: -0.0691228, 0.0764413, -0.0677801, 0.0758381, -0.1449609, 0.1442213
6: 0.8456767, 1.0193081, 0.8478298, 1.0189228, -0.1732461, 0.1714783
7: -0.0972908, 0.0697410, -0.0961011, 0.0684800, -0.1657708, 0.1658421
8: -0.0647882, 0.0904353, -0.0640477, 0.0888951, -0.1536833, 0.1544830
9: -0.0554443, 0.0665266, -0.0539288, 0.0660639, -0.1215082, 0.1204555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2938506
time: 1.17 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2937822
time: 1.22 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0246943, 0.0274921, -0.0213852, 0.0252741, -0.0499684, 0.0488773
1: -0.0381059, 0.0702778, -0.0345621, 0.0660506, -0.1041565, 0.1048399
2: -0.0519541, 0.1017184, -0.0466488, 0.0945966, -0.1465507, 0.1483673
3: -0.0445019, 0.0073512, -0.0414673, 0.0064599, -0.0509619, 0.0488185
4: -0.0577806, 0.0626468, -0.0528511, 0.0580210, -0.1158015, 0.1154979
5: -0.0549662, 0.0659118, -0.0510466, 0.0618558, -0.1168220, 0.1169584
6: 0.8740789, 1.0138280, 0.8835185, 1.0119121, -0.1378332, 0.1303095
7: -0.0849144, 0.0491533, -0.0807487, 0.0414463, -0.1263606, 0.1299021
8: -0.0555233, 0.0707672, -0.0521817, 0.0651547, -0.1206780, 0.1229489
9: -0.0406127, 0.0584478, -0.0368455, 0.0553358, -0.0959485, 0.0952933

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A2_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2651796, upper bound: 0.2569796
time: 1.35 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2613599, upper bound: 0.2502005
time: 1.22 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0291902, 0.0305055, -0.0216448, 0.0254482, -0.0546383, 0.0521503
1: -0.0429206, 0.0760210, -0.0348402, 0.0663822, -0.1093028, 0.1108612
2: -0.0591619, 0.1113942, -0.0470651, 0.0951553, -0.1543172, 0.1584594
3: -0.0486248, 0.0085620, -0.0417054, 0.0065299, -0.0551547, 0.0502674
4: -0.0644778, 0.0689316, -0.0532379, 0.0583839, -0.1228617, 0.1221695
5: -0.0602915, 0.0714224, -0.0513541, 0.0621740, -0.1224655, 0.1227765
6: 0.8612542, 1.0164310, 0.8827778, 1.0120624, -0.1508082, 0.1336532
7: -0.0905739, 0.0596242, -0.0810755, 0.0420509, -0.1326248, 0.1406998
8: -0.0600633, 0.0783924, -0.0524438, 0.0655951, -0.1256584, 0.1308363
9: -0.0457309, 0.0626759, -0.0371411, 0.0555799, -0.1013109, 0.0998169

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2647452, upper bound: 0.2569232
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2607098, upper bound: 0.2500947
time: 1.42 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0240030, 0.0270287, -0.0184391, 0.0232995, -0.0473025, 0.0454678
1: -0.0373656, 0.0693946, -0.0314071, 0.0622871, -0.0996526, 0.1008018
2: -0.0508457, 0.1002306, -0.0419257, 0.0882561, -0.1391019, 0.1421562
3: -0.0438679, 0.0071650, -0.0387657, 0.0056665, -0.0495344, 0.0459306
4: -0.0567507, 0.0616804, -0.0484625, 0.0539026, -0.1106533, 0.1101429
5: -0.0541473, 0.0650644, -0.0475570, 0.0582447, -0.1123920, 0.1126214
6: 0.8760509, 1.0134277, 0.8919224, 1.0102062, -0.1341553, 0.1215054
7: -0.0840440, 0.0475432, -0.0770401, 0.0345847, -0.1186288, 0.1245832
8: -0.0548251, 0.0695947, -0.0492067, 0.0601580, -0.1149831, 0.1188013
9: -0.0398257, 0.0577976, -0.0334916, 0.0525651, -0.0923908, 0.0912892

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=46, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2643643, upper bound: 0.2563229
time: 1.25 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2607601, upper bound: 0.2500020
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0285184, 0.0300552, -0.0186967, 0.0234722, -0.0519906, 0.0487519
1: -0.0422012, 0.0751628, -0.0316830, 0.0626162, -0.1048174, 0.1068459
2: -0.0580849, 0.1099485, -0.0423387, 0.0888105, -0.1468954, 0.1522872
3: -0.0480088, 0.0083811, -0.0390019, 0.0057359, -0.0537446, 0.0473830
4: -0.0634771, 0.0679925, -0.0488462, 0.0542627, -0.1177398, 0.1168388
5: -0.0594958, 0.0705990, -0.0478622, 0.0585605, -0.1180563, 0.1184612
6: 0.8631703, 1.0160420, 0.8911875, 1.0103555, -0.1471851, 0.1248545
7: -0.0897283, 0.0580597, -0.0773643, 0.0351848, -0.1249131, 0.1354241
8: -0.0593849, 0.0772531, -0.0494668, 0.0605950, -0.1199799, 0.1267199
9: -0.0449662, 0.0620441, -0.0337849, 0.0528074, -0.0977736, 0.0958290

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=46, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2641339, upper bound: 0.2562194
time: 1.22 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2601684, upper bound: 0.2498814
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0330076, 0.0378206, -0.0399121, 0.0480405, -0.0810480, 0.0777327
1: -0.0466889, 0.0822504, -0.0537139, 0.0986961, -0.1453851, 0.1359642
2: -0.0653137, 0.1229173, -0.0757964, 0.1390428, -0.2043565, 0.1987137
3: -0.0520532, 0.0103390, -0.0565032, 0.0215016, -0.0735548, 0.0668422
4: -0.0730466, 0.0744881, -0.0860935, 0.0855209, -0.1585675, 0.1605816
5: -0.0675513, 0.0757354, -0.0808322, 0.0826228, -0.1501742, 0.1565677
6: 0.8481967, 1.0188571, 0.8273811, 1.0230138, -0.1748171, 0.1914760
7: -0.0958985, 0.0682652, -0.1080826, 0.0804880, -0.1763866, 0.1763478
8: -0.0639215, 0.0886328, -0.0708721, 0.1038492, -0.1677708, 0.1595049
9: -0.0536707, 0.0659851, -0.0680687, 0.0712811, -0.1249518, 0.1340538

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_A1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2664985, upper bound: 0.2773514
time: 1.29 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2664985, upper bound: 0.2653318
time: 1.24 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0381923, 0.0474665, -0.0405830, 0.0486603, -0.0868527, 0.0880495
1: -0.0512721, 0.0977723, -0.0544339, 0.0996933, -0.1509654, 0.1522062
2: -0.0751715, 0.1375288, -0.0764350, 0.1401718, -0.2153432, 0.2139639
3: -0.0562533, 0.0171137, -0.0567731, 0.0227064, -0.0789597, 0.0738868
4: -0.0841836, 0.0849013, -0.0871398, 0.0861899, -0.1703735, 0.1720411
5: -0.0792295, 0.0809810, -0.0817100, 0.0833518, -0.1625813, 0.1626910
6: 0.8294684, 1.0222081, 0.8258911, 1.0233144, -0.1938460, 0.1963170
7: -0.1062454, 0.0792332, -0.1091073, 0.0812773, -0.1875227, 0.1883405
8: -0.0703624, 0.1020288, -0.0713037, 0.1050116, -0.1753741, 0.1733325
9: -0.0668512, 0.0700098, -0.0689765, 0.0718437, -0.1386949, 0.1389863

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_A2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2664985, upper bound: 0.2773834
time: 1.40 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2664985, upper bound: 0.2937951
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0339340, 0.0395443, -0.0335553, 0.0388397, -0.0727737, 0.0730996
1: -0.0475079, 0.0850242, -0.0471731, 0.0838902, -0.1313981, 0.1321973
2: -0.0670753, 0.1255284, -0.0663551, 0.1244609, -0.1915362, 0.1918834
3: -0.0528037, 0.0115496, -0.0524969, 0.0110547, -0.0638584, 0.0640465
4: -0.0750367, 0.0763489, -0.0742231, 0.0755882, -0.1506249, 0.1505721
5: -0.0696383, 0.0766728, -0.0687851, 0.0762896, -0.1459278, 0.1454579
6: 0.8448499, 1.0194560, 0.8462182, 1.0192113, -0.1743613, 0.1732379
7: -0.0977475, 0.0702251, -0.0969916, 0.0694239, -0.1671714, 0.1672167
8: -0.0650725, 0.0910266, -0.0646019, 0.0900480, -0.1551205, 0.1556285
9: -0.0560261, 0.0667043, -0.0550632, 0.0664102, -0.1224363, 0.1217675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2602165, upper bound: 0.2753916
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2602165, upper bound: 0.2938309
time: 1.28 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0342384, 0.0401106, -0.0389651, 0.0479752, -0.0822136, 0.0790757
1: -0.0477769, 0.0859353, -0.0528780, 0.0985909, -0.1463678, 0.1388133
2: -0.0676538, 0.1263860, -0.0757290, 0.1382995, -0.2059533, 0.2021150
3: -0.0530503, 0.0119473, -0.0564748, 0.0200545, -0.0731048, 0.0684221
4: -0.0756905, 0.0769601, -0.0850066, 0.0854503, -0.1611408, 0.1619668
5: -0.0703237, 0.0769807, -0.0807397, 0.0812576, -0.1515814, 0.1577204
6: 0.8437506, 1.0196526, 0.8284808, 1.0229820, -0.1792314, 0.1911718
7: -0.0983548, 0.0708689, -0.1067911, 0.0804048, -0.1787596, 0.1776600
8: -0.0654506, 0.0918130, -0.0708266, 0.1027353, -0.1681859, 0.1626396
9: -0.0567998, 0.0669405, -0.0679731, 0.0702221, -0.1270218, 0.1349136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2602165, upper bound: 0.2753428
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2602165, upper bound: 0.2937682
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0159266, 0.0216155, -0.0295675, 0.0307585, -0.0466851, 0.0511829
1: -0.0287165, 0.0590775, -0.0433247, 0.0765031, -0.1052196, 0.1024022
2: -0.0378976, 0.0828488, -0.0597669, 0.1122064, -0.1501041, 0.1426157
3: -0.0364616, 0.0049898, -0.0489708, 0.0086637, -0.0451253, 0.0539607
4: -0.0447198, 0.0503904, -0.0650399, 0.0694591, -0.1141789, 0.1154303
5: -0.0445810, 0.0551652, -0.0607384, 0.0718849, -0.1164659, 0.1159036
6: 0.8990893, 1.0087515, 0.8601776, 1.0166496, -0.1175603, 0.1485739
7: -0.0738773, 0.0287331, -0.0910490, 0.0605031, -0.1343804, 0.1197820
8: -0.0466695, 0.0558967, -0.0604443, 0.0790324, -0.1257019, 0.1163411
9: -0.0306313, 0.0502023, -0.0461605, 0.0630307, -0.0936620, 0.0963628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2437880, upper bound: 0.2628746
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2437880, upper bound: 0.2613030
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0208356, 0.0249058, -0.0298050, 0.0312755, -0.0521111, 0.0547108
1: -0.0339735, 0.0653485, -0.0435791, 0.0768087, -0.1107823, 0.1089275
2: -0.0457677, 0.0934137, -0.0601476, 0.1130028, -0.1587705, 0.1535613
3: -0.0409633, 0.0063119, -0.0492032, 0.0087276, -0.0496909, 0.0555151
4: -0.0520324, 0.0572526, -0.0656165, 0.0697911, -0.1218235, 0.1228691
5: -0.0503956, 0.0611821, -0.0612087, 0.0721760, -0.1225716, 0.1223908
6: 0.8850862, 1.0115938, 0.8593130, 1.0167871, -0.1317009, 0.1522808
7: -0.0800568, 0.0401662, -0.0913642, 0.0610563, -0.1411131, 0.1315304
8: -0.0516267, 0.0642226, -0.0606842, 0.0797194, -0.1313461, 0.1249067
9: -0.0362198, 0.0548189, -0.0466476, 0.0632541, -0.0994739, 0.1014665

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2428060, upper bound: 0.2628705
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2428060, upper bound: 0.2613279
time: 1.23 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0145797, 0.0207126, -0.0281340, 0.0297976, -0.0443773, 0.0488467
1: -0.0272741, 0.0573570, -0.0417895, 0.0746718, -0.1019459, 0.0991465
2: -0.0357383, 0.0799500, -0.0574686, 0.1091212, -0.1448595, 0.1374186
3: -0.0352264, 0.0046270, -0.0476563, 0.0082776, -0.0435040, 0.0522833
4: -0.0427133, 0.0485076, -0.0629045, 0.0674552, -0.1101685, 0.1114121
5: -0.0429857, 0.0535142, -0.0590404, 0.0701278, -0.1131135, 0.1125547
6: 0.9029318, 1.0079720, 0.8642669, 1.0158195, -0.1128877, 0.1437051
7: -0.0721817, 0.0255961, -0.0892444, 0.0571644, -0.1293461, 0.1148406
8: -0.0453094, 0.0536123, -0.0589967, 0.0766011, -0.1219105, 0.1126090
9: -0.0290979, 0.0489356, -0.0445286, 0.0616826, -0.0907805, 0.0934642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2428060, upper bound: 0.2627289
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2428060, upper bound: 0.2614633
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0148062, 0.0208645, -0.0328634, 0.0381686, -0.0529748, 0.0537279
1: -0.0275167, 0.0576464, -0.0468543, 0.0807465, -0.1082632, 0.1045007
2: -0.0361015, 0.0804376, -0.0650508, 0.1234444, -0.1595459, 0.1454884
3: -0.0354342, 0.0046881, -0.0522047, 0.0095513, -0.0449855, 0.0568928
4: -0.0430508, 0.0488242, -0.0731865, 0.0740664, -0.1171172, 0.1220107
5: -0.0432540, 0.0537919, -0.0673885, 0.0759247, -0.1191787, 0.1211804
6: 0.9022853, 1.0081030, 0.8480569, 1.0185578, -0.1162724, 0.1600461
7: -0.0724668, 0.0261238, -0.0954348, 0.0681792, -0.1406461, 0.1215585
8: -0.0455381, 0.0539965, -0.0637725, 0.0887520, -0.1342902, 0.1177690
9: -0.0293558, 0.0491487, -0.0530618, 0.0661303, -0.0954861, 0.1022105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2428060, upper bound: 0.2624050
time: 2.07 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2428060, upper bound: 0.2607516
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0358669, 0.0431403, -0.0188591, 0.0235810, -0.0594479, 0.0619994
1: -0.0492165, 0.0908107, -0.0318569, 0.0628235, -0.1120400, 0.1226675
2: -0.0707502, 0.1309755, -0.0425990, 0.0891599, -0.1599101, 0.1735745
3: -0.0543695, 0.0140752, -0.0391507, 0.0057796, -0.0601491, 0.0532259
4: -0.0791886, 0.0802309, -0.0490881, 0.0544896, -0.1336783, 0.1293190
5: -0.0739918, 0.0786284, -0.0480544, 0.0587595, -0.1327512, 0.1266828
6: 0.8378680, 1.0207053, 0.8907244, 1.0104494, -0.1725814, 0.1299809
7: -0.1016048, 0.0743140, -0.0775687, 0.0355628, -0.1371676, 0.1518826
8: -0.0674736, 0.0960206, -0.0496307, 0.0608703, -0.1283440, 0.1456513
9: -0.0609397, 0.0682047, -0.0339697, 0.0529600, -0.1138997, 0.1021744

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_A2_B1_B1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2774529, upper bound: 0.2699544
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2775443, upper bound: 0.2699544
time: 2.15 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0358669, 0.0431403, -0.0363547, 0.0439733, -0.0798402, 0.0794950
1: -0.0492165, 0.0908107, -0.0497216, 0.0921511, -0.1413675, 0.1405323
2: -0.0707502, 0.1309755, -0.0716045, 0.1322374, -0.2029876, 0.2025800
3: -0.0543695, 0.0140752, -0.0547322, 0.0148672, -0.0692367, 0.0688074
4: -0.0791886, 0.0802309, -0.0801692, 0.0811301, -0.1603188, 0.1604001
5: -0.0739918, 0.0786284, -0.0750719, 0.0790813, -0.1530731, 0.1537003
6: 0.8378680, 1.0207053, 0.8362510, 1.0210427, -0.1831747, 0.1844543
7: -0.1016048, 0.0743140, -0.1024983, 0.0753086, -0.1769134, 0.1768122
8: -0.0674736, 0.0960206, -0.0680398, 0.0971774, -0.1646511, 0.1640605
9: -0.0609397, 0.0682047, -0.0621121, 0.0685522, -0.1294919, 0.1303168

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_A2_B1_B2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2774529, upper bound: 0.2943560
time: 1.52 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2775443, upper bound: 0.2942349
time: 1.46 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0320052, 0.0359218, -0.0371835, 0.0452441, -0.0772493, 0.0731054
1: -0.0457867, 0.0796790, -0.0507238, 0.0941959, -0.1399826, 0.1304028
2: -0.0635179, 0.1200409, -0.0729142, 0.1341622, -0.1976801, 0.1929551
3: -0.0512263, 0.0092996, -0.0552855, 0.0165142, -0.0677405, 0.0645851
4: -0.0708542, 0.0726729, -0.0817053, 0.0825019, -0.1533561, 0.1543781
5: -0.0654061, 0.0747028, -0.0768716, 0.0797724, -0.1451785, 0.1515744
6: 0.8517257, 1.0181974, 0.8337836, 1.0216584, -0.1699327, 0.1844138
7: -0.0941080, 0.0661061, -0.1038614, 0.0769269, -0.1710349, 0.1699674
8: -0.0627658, 0.0859957, -0.0689247, 0.0989423, -0.1617081, 0.1549204
9: -0.0512412, 0.0651928, -0.0639732, 0.0690824, -0.1203237, 0.1291659

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2752349, upper bound: 0.2647037
time: 1.15 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2752349, upper bound: 0.2940443
time: 1.28 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0322655, 0.0364400, -0.0480327, 0.0555427, -0.0878082, 0.0844727
1: -0.0460329, 0.0800287, -0.0624296, 0.1107687, -0.1568016, 0.1424583
2: -0.0639028, 0.1208260, -0.0835283, 0.1527070, -0.2166098, 0.2043543
3: -0.0514520, 0.0093693, -0.0597699, 0.0360870, -0.0875390, 0.0691393
4: -0.0714525, 0.0729977, -0.0987582, 0.0936200, -0.1650725, 0.1717558
5: -0.0658798, 0.0749846, -0.0914575, 0.0914467, -0.1573265, 0.1664421
6: 0.8508772, 1.0183777, 0.8093441, 1.0266501, -0.1757729, 0.2090336
7: -0.0944175, 0.0666953, -0.1204877, 0.0900417, -0.1844592, 0.1871830
8: -0.0629997, 0.0867154, -0.0760965, 0.1179190, -0.1809187, 0.1628119
9: -0.0517841, 0.0654090, -0.0790563, 0.0780924, -0.1298765, 0.1444653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=17, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753510, upper bound: 0.2647037
time: 1.19 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753510, upper bound: 0.2939128
time: 1.76 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0184891, 0.0233330, -0.0290697, 0.0304248, -0.0489139, 0.0524026
1: -0.0314607, 0.0623510, -0.0427916, 0.0758671, -0.1073278, 0.1051426
2: -0.0420058, 0.0883637, -0.0589688, 0.1111350, -0.1531408, 0.1473324
3: -0.0388115, 0.0056799, -0.0485144, 0.0085296, -0.0473411, 0.0541943
4: -0.0485369, 0.0539725, -0.0642984, 0.0687632, -0.1173001, 0.1182709
5: -0.0476162, 0.0583060, -0.0601488, 0.0712748, -0.1188910, 0.1184548
6: 0.8917798, 1.0102353, 0.8615977, 1.0163614, -0.1245816, 0.1486377
7: -0.0771029, 0.0347012, -0.0904223, 0.0593437, -0.1364466, 0.1251235
8: -0.0492572, 0.0602428, -0.0599416, 0.0781881, -0.1274453, 0.1201844
9: -0.0335485, 0.0526121, -0.0455938, 0.0625626, -0.0961111, 0.0982060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=46, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2534686, upper bound: 0.2654453
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_A2

### Relational analysis result of IS_A1_B2_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2427043, upper bound: 0.2614323
time: 1.18 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0187369, 0.0234991, -0.0343362, 0.0405687, -0.0593056, 0.0578352
1: -0.0317260, 0.0626675, -0.0479946, 0.0827524, -0.1144785, 0.1106621
2: -0.0424031, 0.0888970, -0.0669500, 0.1270801, -0.1694832, 0.1558470
3: -0.0390387, 0.0057467, -0.0532498, 0.0098872, -0.0489260, 0.0589964
4: -0.0489061, 0.0543189, -0.0762195, 0.0755550, -0.1244611, 0.1305383
5: -0.0479097, 0.0586097, -0.0696341, 0.0772299, -0.1251396, 0.1282438
6: 0.8910731, 1.0103787, 0.8441377, 1.0198119, -0.1287388, 0.1662410
7: -0.0774149, 0.0352783, -0.0968520, 0.0713899, -0.1488048, 0.1321303
8: -0.0495074, 0.0606631, -0.0648478, 0.0924492, -0.1419566, 0.1255109
9: -0.0338306, 0.0528452, -0.0560889, 0.0671317, -0.1009622, 0.1089341

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2534234, upper bound: 0.2649901
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_A2

### Relational analysis result of IS_A1_B2_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2427043, upper bound: 0.2606757
time: 1.22 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0158282, 0.0215495, -0.0295831, 0.0307753, -0.0466035, 0.0511325
1: -0.0286111, 0.0589518, -0.0433414, 0.0765230, -0.1051341, 0.1022932
2: -0.0377398, 0.0826369, -0.0597919, 0.1122451, -0.1499849, 0.1424288
3: -0.0363713, 0.0049633, -0.0489854, 0.0086679, -0.0450392, 0.0539487
4: -0.0445731, 0.0502528, -0.0650672, 0.0694809, -0.1140540, 0.1153200
5: -0.0444645, 0.0550445, -0.0607602, 0.0719040, -0.1163685, 0.1158047
6: 0.8993703, 1.0086948, 0.8601298, 1.0166585, -0.1172882, 0.1485650
7: -0.0737533, 0.0285038, -0.0910689, 0.0605395, -0.1342928, 0.1195727
8: -0.0465701, 0.0557297, -0.0604601, 0.0790640, -0.1256341, 0.1161898
9: -0.0305192, 0.0501097, -0.0461822, 0.0630454, -0.0935646, 0.0962919

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A2_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2529066, upper bound: 0.2647521
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2427043, upper bound: 0.2608559
time: 1.43 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0160739, 0.0217142, -0.0347639, 0.0414214, -0.0574953, 0.0564780
1: -0.0288742, 0.0592657, -0.0483998, 0.0833164, -0.1121906, 0.1076655
2: -0.0381338, 0.0831657, -0.0675798, 0.1283717, -0.1665055, 0.1507455
3: -0.0365967, 0.0050294, -0.0536211, 0.0099951, -0.0465917, 0.0586505
4: -0.0449391, 0.0505962, -0.0772040, 0.0760839, -0.1210230, 0.1278003
5: -0.0447555, 0.0553457, -0.0704099, 0.0776936, -0.1224491, 0.1257555
6: 0.8986695, 1.0088369, 0.8427452, 1.0201080, -0.1214385, 0.1660917
7: -0.0740626, 0.0290761, -0.0973556, 0.0723595, -0.1464221, 0.1264317
8: -0.0468182, 0.0561465, -0.0652298, 0.0936335, -0.1404517, 0.1213763
9: -0.0307989, 0.0503408, -0.0569785, 0.0674875, -0.0982864, 0.1073193

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2528600, upper bound: 0.2643859
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2427043, upper bound: 0.2602371
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0342388, 0.0401112, -0.0424466, 0.0503821, -0.0846208, 0.0825578
1: -0.0477772, 0.0859364, -0.0564341, 0.1024640, -0.1502412, 0.1423704
2: -0.0676545, 0.1263871, -0.0782095, 0.1433076, -0.2109621, 0.2045966
3: -0.0530506, 0.0119477, -0.0575228, 0.0260538, -0.0791043, 0.0694706
4: -0.0756912, 0.0769608, -0.0900462, 0.0880487, -0.1637399, 0.1670071
5: -0.0703245, 0.0769811, -0.0841485, 0.0853769, -0.1557013, 0.1611295
6: 0.8437494, 1.0196531, 0.8217515, 1.0241487, -0.1803993, 0.1979015
7: -0.0983556, 0.0708696, -0.1119542, 0.0834699, -0.1818254, 0.1828239
8: -0.0654510, 0.0918139, -0.0725027, 0.1082405, -0.1736916, 0.1643166
9: -0.0568006, 0.0669408, -0.0714981, 0.0734069, -0.1302075, 0.1384389

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2775566, upper bound: 0.2696017
time: 2.18 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2775566, upper bound: 0.2942321
time: 1.38 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0401445, 0.0497832, -0.0430843, 0.0509551, -0.0910996, 0.0928675
1: -0.0543041, 0.1015006, -0.0571171, 0.1033863, -0.1576905, 0.1586177
2: -0.0775924, 0.1410383, -0.0788098, 0.1443514, -0.2219438, 0.2198481
3: -0.0572621, 0.0223983, -0.0577811, 0.0271681, -0.0844302, 0.0801794
4: -0.0871921, 0.0874023, -0.0910138, 0.0886817, -0.1758738, 0.1784161
5: -0.0833005, 0.0822409, -0.0849602, 0.0860766, -0.1693770, 0.1672011
6: 0.8249701, 1.0238587, 0.8203660, 1.0244266, -0.1994565, 0.2034927
7: -0.1087307, 0.0827074, -0.1129020, 0.0842357, -0.1929664, 0.1956094
8: -0.0720858, 0.1052462, -0.0729155, 0.1093154, -0.1814012, 0.1781617
9: -0.0706212, 0.0709765, -0.0723449, 0.0739273, -0.1445485, 0.1433214

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2775258, upper bound: 0.2695683
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2775258, upper bound: 0.2942050
time: 1.74 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0328783, 0.0375802, -0.0403983, 0.0484896, -0.0813679, 0.0779785
1: -0.0465747, 0.0818635, -0.0542356, 0.0994189, -0.1459936, 0.1360991
2: -0.0650680, 0.1225531, -0.0762592, 0.1398609, -0.2049288, 0.1988123
3: -0.0519485, 0.0101701, -0.0566988, 0.0223748, -0.0743233, 0.0668689
4: -0.0727690, 0.0742285, -0.0868517, 0.0860058, -0.1587748, 0.1610802
5: -0.0672603, 0.0756047, -0.0814683, 0.0831510, -0.1504113, 0.1570730
6: 0.8486635, 1.0187738, 0.8263015, 1.0232315, -0.1745680, 0.1924723
7: -0.0956405, 0.0679918, -0.1088252, 0.0810600, -0.1767005, 0.1768170
8: -0.0637610, 0.0882989, -0.0711850, 0.1046915, -0.1684525, 0.1594839
9: -0.0533422, 0.0658848, -0.0687266, 0.0716888, -0.1250310, 0.1346113

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_A2_B2_B1_B1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2752299, upper bound: 0.2645355
time: 1.39 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B1_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2752299, upper bound: 0.2940230
time: 3.46 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0331886, 0.0381573, -0.0542217, 0.0587561, -0.0919446, 0.0923790
1: -0.0468489, 0.0827923, -0.0682587, 0.1162339, -0.1630827, 0.1510509
2: -0.0656577, 0.1234274, -0.0880141, 0.1589376, -0.2245954, 0.2114415
3: -0.0521998, 0.0105755, -0.0622387, 0.0426456, -0.0948454, 0.0728142
4: -0.0734353, 0.0748516, -0.1047993, 0.0988272, -0.1722624, 0.1796508
5: -0.0679590, 0.0759185, -0.0960085, 0.0984165, -0.1663755, 0.1719270
6: 0.8475428, 1.0189742, 0.8006763, 1.0286829, -0.1811401, 0.2182978
7: -0.0962597, 0.0686479, -0.1258009, 0.0986824, -0.1949421, 0.1944488
8: -0.0641464, 0.0891004, -0.0800060, 0.1244352, -0.1885816, 0.1691064
9: -0.0541308, 0.0661256, -0.0848556, 0.0810097, -0.1351405, 0.1509811

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753504, upper bound: 0.2645355
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753504, upper bound: 0.2939018
time: 1.56 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0318665, 0.0356452, -0.0149645, 0.0209706, -0.0528372, 0.0506097
1: -0.0456553, 0.0794961, -0.0276862, 0.0578485, -0.1035038, 0.1071823
2: -0.0633137, 0.1196220, -0.0363552, 0.0807782, -0.1440919, 0.1559772
3: -0.0511059, 0.0092646, -0.0355793, 0.0047307, -0.0558366, 0.0448439
4: -0.0705349, 0.0725013, -0.0432866, 0.0490455, -0.1195804, 0.1157879
5: -0.0651545, 0.0745524, -0.0434415, 0.0539859, -0.1191404, 0.1179939
6: 0.8521776, 1.0181015, 0.9018339, 1.0081947, -0.1560171, 0.1162676
7: -0.0939446, 0.0657916, -0.0726661, 0.0264924, -0.1204371, 0.1384577
8: -0.0626419, 0.0856116, -0.0456980, 0.0542650, -0.1169069, 0.1313096
9: -0.0509526, 0.0650774, -0.0295361, 0.0492975, -0.1002502, 0.0946134

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2756073, upper bound: 0.2615459
time: 1.41 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2757210, upper bound: 0.2615459
time: 1.29 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0318665, 0.0356452, -0.0200646, 0.0243890, -0.0562555, 0.0557098
1: -0.0456553, 0.0794961, -0.0331479, 0.0643635, -0.1100188, 0.1126439
2: -0.0633137, 0.1196220, -0.0445317, 0.0917544, -0.1550681, 0.1641536
3: -0.0511059, 0.0092646, -0.0402562, 0.0061043, -0.0572102, 0.0495208
4: -0.0705349, 0.0725013, -0.0508838, 0.0561749, -0.1267097, 0.1233851
5: -0.0651545, 0.0745524, -0.0494823, 0.0602371, -0.1253916, 0.1240347
6: 0.8521776, 1.0181015, 0.8872857, 1.0111475, -0.1589699, 0.1308157
7: -0.0939446, 0.0657916, -0.0790863, 0.0383705, -0.1323152, 0.1448779
8: -0.0626419, 0.0856116, -0.0508481, 0.0629149, -0.1255568, 0.1364597
9: -0.0509526, 0.0650774, -0.0353421, 0.0540938, -0.1050464, 0.1004195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2756073, upper bound: 0.2673872
time: 1.19 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2757210, upper bound: 0.2673870
time: 1.33 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0301480, 0.0320486, -0.0196679, 0.0241231, -0.0542711, 0.0517165
1: -0.0439464, 0.0772505, -0.0327230, 0.0638568, -0.1078032, 0.1099735
2: -0.0606976, 0.1141739, -0.0438956, 0.0909006, -0.1515982, 0.1580696
3: -0.0495399, 0.0088200, -0.0398925, 0.0059974, -0.0555373, 0.0487125
4: -0.0664655, 0.0702707, -0.0502929, 0.0556203, -0.1220858, 0.1205636
5: -0.0619019, 0.0725965, -0.0490125, 0.0597508, -0.1216527, 0.1216090
6: 0.8580504, 1.0169859, 0.8884173, 1.0109178, -0.1528674, 0.1285686
7: -0.0918208, 0.0618553, -0.0785868, 0.0374465, -0.1292673, 0.1404421
8: -0.0610306, 0.0807325, -0.0504475, 0.0622421, -0.1232726, 0.1311800
9: -0.0473671, 0.0635768, -0.0348905, 0.0537207, -0.1010878, 0.0984672

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=44, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2756650, upper bound: 0.2615459
time: 1.07 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2756650, upper bound: 0.2673659
time: 1.32 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0303676, 0.0325436, -0.0244922, 0.0273566, -0.0577242, 0.0570358
1: -0.0441816, 0.0775332, -0.0378895, 0.0700196, -0.1142011, 0.1154227
2: -0.0610496, 0.1149237, -0.0516300, 0.1012834, -0.1623331, 0.1665537
3: -0.0497554, 0.0088792, -0.0443166, 0.0072967, -0.0570521, 0.0531957
4: -0.0670091, 0.0705777, -0.0574795, 0.0623643, -0.1293734, 0.1280572
5: -0.0623456, 0.0728657, -0.0547268, 0.0656641, -0.1280097, 0.1275925
6: 0.8572422, 1.0171127, 0.8746554, 1.0137111, -0.1564689, 0.1424574
7: -0.0921131, 0.0623667, -0.0846599, 0.0486825, -0.1407956, 0.1470266
8: -0.0612523, 0.0813811, -0.0553192, 0.0704244, -0.1316767, 0.1367003
9: -0.0478277, 0.0637832, -0.0403826, 0.0582577, -0.1060853, 0.1041659

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2757528, upper bound: 0.2615459
time: 1.36 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2757528, upper bound: 0.2673658
time: 1.55 seconds

## BFS IS instance: IS_A2_B1_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0244103, 0.0273017, -0.0371023, 0.0451194, -0.0695297, 0.0644040
1: -0.0378018, 0.0699150, -0.0506255, 0.0939955, -0.1317972, 0.1205405
2: -0.0514987, 0.1011071, -0.0727857, 0.1339735, -0.1854722, 0.1738929
3: -0.0442415, 0.0072747, -0.0552313, 0.0163528, -0.0605943, 0.0625060
4: -0.0573575, 0.0622497, -0.0815546, 0.0823674, -0.1397249, 0.1438044
5: -0.0546298, 0.0655636, -0.0766951, 0.0797046, -0.1343344, 0.1422588
6: 0.8748890, 1.0136635, 0.8340255, 1.0215979, -0.1467088, 0.1796380
7: -0.0845568, 0.0484918, -0.1037278, 0.0767682, -0.1613250, 0.1522195
8: -0.0552364, 0.0702855, -0.0688380, 0.0987692, -0.1540056, 0.1391235
9: -0.0402894, 0.0581807, -0.0637907, 0.0690305, -0.1093199, 0.1219714

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1_B2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2643854, upper bound: 0.2754962
time: 1.29 seconds

## Relational analysis of IS_A2_B1_A1_B2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2643854, upper bound: 0.2784516
time: 1.38 seconds

## BFS IS instance: IS_A2_B1_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0536559, 0.0578252, -0.0371023, 0.0451194, -0.0987753, 0.0949274
1: -0.0673190, 0.1146975, -0.0506255, 0.0939955, -0.1613145, 0.1653231
2: -0.0874504, 0.1571927, -0.0727857, 0.1339735, -0.2214239, 0.2299785
3: -0.0617068, 0.0414532, -0.0552313, 0.0163528, -0.0780596, 0.0966845
4: -0.1031472, 0.0979268, -0.0815546, 0.0823674, -0.1855146, 0.1794815
5: -0.0954092, 0.0969435, -0.0766951, 0.0797046, -0.1751138, 0.1736387
6: 0.8027151, 1.0281693, 0.8340255, 1.0215979, -0.2188827, 0.1941438
7: -0.1242618, 0.0973324, -0.1037278, 0.0767682, -0.2010300, 0.2010601
8: -0.0791600, 0.1246339, -0.0688380, 0.0987692, -0.1779292, 0.1934719
9: -0.0845317, 0.0801646, -0.0637907, 0.0690305, -0.1535622, 0.1439553

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A1_B2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2945554
time: 1.21 seconds

## Relational analysis of IS_A2_B1_A1_B2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2714759
time: 1.25 seconds

## BFS IS instance: IS_A2_B1_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0216976, 0.0254836, -0.0371184, 0.0451441, -0.0668417, 0.0626019
1: -0.0348967, 0.0664496, -0.0506450, 0.0940351, -0.1289318, 0.1170947
2: -0.0471497, 0.0952690, -0.0728112, 0.1340108, -0.1811606, 0.1680801
3: -0.0417538, 0.0065441, -0.0552421, 0.0163848, -0.0581386, 0.0617861
4: -0.0533165, 0.0584577, -0.0815845, 0.0823941, -0.1357106, 0.1400422
5: -0.0514167, 0.0622387, -0.0767301, 0.0797180, -0.1311347, 0.1389688
6: 0.8826272, 1.0120931, 0.8339776, 1.0216100, -0.1389828, 0.1781155
7: -0.0811420, 0.0421739, -0.1037542, 0.0767997, -0.1579417, 0.1459281
8: -0.0524971, 0.0656846, -0.0688552, 0.0988035, -0.1513006, 0.1345398
9: -0.0372012, 0.0556295, -0.0638268, 0.0690408, -0.1062419, 0.1194564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1_B2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2639159, upper bound: 0.2752510
time: 1.34 seconds

## Relational analysis of IS_A2_B1_A1_B2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2639159, upper bound: 0.2781760
time: 1.37 seconds

## BFS IS instance: IS_A2_B1_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0445936, 0.0520471, -0.0371184, 0.0451441, -0.0897377, 0.0891655
1: -0.0586767, 0.1051601, -0.0506450, 0.0940351, -0.1527118, 0.1558051
2: -0.0800982, 0.1463615, -0.0728112, 0.1340108, -0.2141090, 0.2191727
3: -0.0584051, 0.0293085, -0.0552421, 0.0163848, -0.0747898, 0.0845506
4: -0.0928918, 0.0901018, -0.0815845, 0.0823941, -0.1752859, 0.1716862
5: -0.0865067, 0.0877993, -0.0767301, 0.0797180, -0.1662248, 0.1645294
6: 0.8176098, 1.0249825, 0.8339776, 1.0216100, -0.2040002, 0.1910049
7: -0.1147076, 0.0862454, -0.1037542, 0.0767997, -0.1915073, 0.1899996
8: -0.0739080, 0.1113908, -0.0688552, 0.0988035, -0.1727115, 0.1802460
9: -0.0740805, 0.0749187, -0.0638268, 0.0690408, -0.1431212, 0.1387455

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A1_B2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2942308
time: 1.13 seconds

## Relational analysis of IS_A2_B1_A1_B2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2941128
time: 1.14 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0314631, 0.0348409, -0.0159444, 0.0216274, -0.0530905, 0.0507853
1: -0.0452731, 0.0789641, -0.0287356, 0.0591003, -0.1043734, 0.1076997
2: -0.0627195, 0.1184036, -0.0379262, 0.0828872, -0.1456067, 0.1563298
3: -0.0507557, 0.0091628, -0.0364780, 0.0049946, -0.0557503, 0.0456408
4: -0.0696061, 0.0720024, -0.0447463, 0.0504153, -0.1200214, 0.1167488
5: -0.0644227, 0.0741149, -0.0446022, 0.0551870, -0.1196096, 0.1187171
6: 0.8534910, 1.0178219, 0.8990387, 1.0087620, -0.1552710, 0.1187832
7: -0.0934696, 0.0648770, -0.0738997, 0.0287747, -0.1222443, 0.1387767
8: -0.0622816, 0.0844945, -0.0466875, 0.0559270, -0.1182085, 0.1311820
9: -0.0501136, 0.0647418, -0.0306516, 0.0502191, -0.1003326, 0.0953933

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2756088, upper bound: 0.2615459
time: 1.22 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2757287, upper bound: 0.2615459
time: 1.18 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0314631, 0.0348409, -0.0210984, 0.0250819, -0.0565450, 0.0559392
1: -0.0452731, 0.0789641, -0.0342550, 0.0656842, -0.1109573, 0.1132190
2: -0.0627195, 0.1184036, -0.0461890, 0.0939793, -0.1566989, 0.1645926
3: -0.0507557, 0.0091628, -0.0412043, 0.0063827, -0.0571384, 0.0503671
4: -0.0696061, 0.0720024, -0.0524238, 0.0576200, -0.1272261, 0.1244263
5: -0.0644227, 0.0741149, -0.0507068, 0.0615042, -0.1259269, 0.1248218
6: 0.8534910, 1.0178219, 0.8843366, 1.0117462, -0.1582552, 0.1334853
7: -0.0934696, 0.0648770, -0.0803876, 0.0407782, -0.1342478, 0.1452646
8: -0.0622816, 0.0844945, -0.0518920, 0.0646683, -0.1269498, 0.1363865
9: -0.0501136, 0.0647418, -0.0365190, 0.0550660, -0.1051796, 0.1012607

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2756088, upper bound: 0.2673926
time: 1.26 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2757287, upper bound: 0.2673926
time: 1.25 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0304422, 0.0327116, -0.0130117, 0.0196617, -0.0501039, 0.0457233
1: -0.0442614, 0.0776292, -0.0255948, 0.0553538, -0.0996153, 0.1032240
2: -0.0611691, 0.1151782, -0.0332244, 0.0765754, -0.1377445, 0.1484026
3: -0.0498286, 0.0088993, -0.0337885, 0.0042047, -0.0540333, 0.0426877
4: -0.0671936, 0.0706819, -0.0403775, 0.0463155, -0.1135091, 0.1110594
5: -0.0624962, 0.0729570, -0.0411283, 0.0515923, -0.1140885, 0.1140854
6: 0.8569677, 1.0171559, 0.9074047, 1.0070641, -0.1500964, 0.1097512
7: -0.0922123, 0.0625403, -0.0702078, 0.0219441, -0.1141565, 0.1327481
8: -0.0613276, 0.0816013, -0.0437260, 0.0509528, -0.1122804, 0.1253273
9: -0.0479840, 0.0638534, -0.0273128, 0.0474610, -0.0954450, 0.0911662

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2756650, upper bound: 0.2615459
time: 1.14 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2757528, upper bound: 0.2615459
time: 1.56 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0304422, 0.0327116, -0.0179893, 0.0229980, -0.0534402, 0.0507009
1: -0.0442614, 0.0776292, -0.0309254, 0.0617125, -0.1059739, 0.1085546
2: -0.0611691, 0.1151782, -0.0412045, 0.0872880, -0.1484572, 0.1563827
3: -0.0498286, 0.0088993, -0.0383531, 0.0055453, -0.0553739, 0.0472524
4: -0.0671936, 0.0706819, -0.0477924, 0.0532738, -0.1204674, 0.1184742
5: -0.0624962, 0.0729570, -0.0470242, 0.0576934, -0.1201896, 0.1199812
6: 0.8569677, 1.0171559, 0.8932055, 1.0099459, -0.1529781, 0.1239504
7: -0.0922123, 0.0625403, -0.0764738, 0.0335371, -0.1257494, 0.1390141
8: -0.0613276, 0.0816013, -0.0487524, 0.0593951, -0.1207227, 0.1303537
9: -0.0479840, 0.0638534, -0.0329795, 0.0521421, -0.1001260, 0.0968329

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2758206, upper bound: 0.2673738
time: 1.51 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2757528, upper bound: 0.2673693
time: 1.28 seconds

## BFS IS instance: IS_A2_B1_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0243238, 0.0272438, -0.0388697, 0.0470774, -0.0714012, 0.0661135
1: -0.0377092, 0.0698044, -0.0525950, 0.0971462, -0.1348553, 0.1223995
2: -0.0513601, 0.1009210, -0.0748037, 0.1372888, -0.1886489, 0.1757248
3: -0.0441622, 0.0072514, -0.0560839, 0.0196292, -0.0637913, 0.0633352
4: -0.0572286, 0.0621288, -0.0844677, 0.0844812, -0.1417098, 0.1465966
5: -0.0545274, 0.0654576, -0.0794682, 0.0814901, -0.1360175, 0.1449258
6: 0.8751360, 1.0136135, 0.8296967, 1.0225471, -0.1474111, 0.1839168
7: -0.0844479, 0.0482904, -0.1064900, 0.0792617, -0.1637096, 0.1547804
8: -0.0551491, 0.0701388, -0.0702015, 0.1020430, -0.1571921, 0.1403403
9: -0.0401909, 0.0580993, -0.0666583, 0.0704066, -0.1105976, 0.1247576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A2_B2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2643747, upper bound: 0.2752556
time: 1.18 seconds

## Relational analysis of IS_A2_B1_A2_B2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2643747, upper bound: 0.2712650
time: 1.35 seconds

## BFS IS instance: IS_A2_B1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0518118, 0.0568050, -0.0388697, 0.0470774, -0.0988892, 0.0956747
1: -0.0656695, 0.1130134, -0.0525950, 0.0971462, -0.1628157, 0.1656084
2: -0.0859829, 0.1552802, -0.0748037, 0.1372888, -0.2232717, 0.2300839
3: -0.0611238, 0.0391003, -0.0560839, 0.0196292, -0.0807530, 0.0951842
4: -0.1013364, 0.0964468, -0.0844677, 0.0844812, -0.1858176, 0.1809145
5: -0.0936096, 0.0953288, -0.0794682, 0.0814901, -0.1750997, 0.1747969
6: 0.8054436, 1.0276066, 0.8296967, 1.0225471, -0.2171035, 0.1979100
7: -0.1225747, 0.0952558, -0.1064900, 0.0792617, -0.2018363, 0.2017458
8: -0.0782325, 0.1216595, -0.0702015, 0.1020430, -0.1802756, 0.1918609
9: -0.0823154, 0.0792383, -0.0666583, 0.0704066, -0.1527221, 0.1458965

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A2_B2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2945554
time: 1.19 seconds

## Relational analysis of IS_A2_B1_A2_B2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2945554
time: 1.22 seconds

## BFS IS instance: IS_A2_B1_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0215660, 0.0253954, -0.0387520, 0.0469688, -0.0685348, 0.0641474
1: -0.0347558, 0.0662816, -0.0524688, 0.0969713, -0.1317272, 0.1187504
2: -0.0469388, 0.0949858, -0.0746917, 0.1370908, -0.1840297, 0.1696775
3: -0.0416332, 0.0065086, -0.0560365, 0.0194180, -0.0610512, 0.0625452
4: -0.0531206, 0.0582738, -0.0842843, 0.0843639, -0.1374845, 0.1425580
5: -0.0512608, 0.0620775, -0.0793143, 0.0813623, -0.1326231, 0.1413918
6: 0.8830025, 1.0120169, 0.8299578, 1.0224942, -0.1394917, 0.1820590
7: -0.0809764, 0.0418675, -0.1063103, 0.0791233, -0.1600997, 0.1481779
8: -0.0523643, 0.0654615, -0.0701258, 0.1018392, -0.1542035, 0.1355874
9: -0.0370514, 0.0555058, -0.0664992, 0.0703080, -0.1073594, 0.1220050

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A2_B2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2639114, upper bound: 0.2750662
time: 1.21 seconds

## Relational analysis of IS_A2_B1_A2_B2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2639114, upper bound: 0.2781353
time: 1.20 seconds

## BFS IS instance: IS_A2_B1_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0430200, 0.0509076, -0.0387520, 0.0469688, -0.0899888, 0.0896597
1: -0.0570492, 0.1033098, -0.0524688, 0.0969713, -0.1540205, 0.1557786
2: -0.0787537, 0.1442648, -0.0746917, 0.1370908, -0.2158445, 0.2189565
3: -0.0577539, 0.0270756, -0.0560365, 0.0194180, -0.0771719, 0.0831122
4: -0.0909335, 0.0886198, -0.0842843, 0.0843639, -0.1752974, 0.1729041
5: -0.0848929, 0.0860016, -0.0793143, 0.0813623, -0.1662551, 0.1653159
6: 0.8204860, 1.0244035, 0.8299578, 1.0224942, -0.2020082, 0.1944456
7: -0.1128233, 0.0841485, -0.1063103, 0.0791233, -0.1919466, 0.1904589
8: -0.0728723, 0.1092262, -0.0701258, 0.1018392, -0.1747116, 0.1793520
9: -0.0722698, 0.0738840, -0.0664992, 0.0703080, -0.1425778, 0.1403832

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A2_B2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2942308
time: 1.21 seconds

## Relational analysis of IS_A2_B1_A2_B2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2670359
time: 1.16 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0219517, 0.0256538, -0.0327095, 0.0372661, -0.0592178, 0.0583634
1: -0.0351688, 0.0667742, -0.0464255, 0.0813582, -0.1165270, 0.1131997
2: -0.0475570, 0.0958159, -0.0647470, 0.1220774, -0.1696344, 0.1605629
3: -0.0419868, 0.0066125, -0.0518117, 0.0099496, -0.0519364, 0.0584242
4: -0.0536950, 0.0588128, -0.0724064, 0.0738895, -0.1275845, 0.1312192
5: -0.0517176, 0.0625501, -0.0668800, 0.0754339, -0.1271515, 0.1294301
6: 0.8819025, 1.0122401, 0.8492733, 1.0186646, -0.1367621, 0.1629668
7: -0.0814618, 0.0427656, -0.0953038, 0.0676346, -0.1490965, 0.1380694
8: -0.0527537, 0.0661155, -0.0635513, 0.0878627, -0.1406165, 0.1296668
9: -0.0374904, 0.0558685, -0.0529131, 0.0657538, -0.1032442, 0.1087816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=46, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2647862, upper bound: 0.2753541
time: 2.18 seconds

## Relational analysis of IS_A2_B2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2647862, upper bound: 0.2783910
time: 1.41 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0273472, 0.0292702, -0.0329994, 0.0378055, -0.0651527, 0.0622697
1: -0.0409469, 0.0736667, -0.0466817, 0.0822261, -0.1231730, 0.1203485
2: -0.0562072, 0.1074279, -0.0652983, 0.1228944, -0.1791016, 0.1727262
3: -0.0469347, 0.0080657, -0.0520466, 0.0103284, -0.0572631, 0.0601123
4: -0.0617324, 0.0663553, -0.0730291, 0.0744718, -0.1362042, 0.1393844
5: -0.0581085, 0.0691634, -0.0675330, 0.0757272, -0.1338357, 0.1366964
6: 0.8665113, 1.0153642, 0.8482261, 1.0188520, -0.1523407, 0.1671380
7: -0.0882539, 0.0553319, -0.0958823, 0.0682480, -0.1565019, 0.1512142
8: -0.0582022, 0.0752666, -0.0639115, 0.0886118, -0.1468140, 0.1391781
9: -0.0436328, 0.0609427, -0.0536501, 0.0659787, -0.1096116, 0.1145927

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=46, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2647862, upper bound: 0.2754520
time: 1.48 seconds

## Relational analysis of IS_A2_B2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2647862, upper bound: 0.2785061
time: 1.34 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0205864, 0.0247387, -0.0307532, 0.0334125, -0.0539989, 0.0554919
1: -0.0337067, 0.0650302, -0.0445944, 0.0780295, -0.1117363, 0.1096246
2: -0.0453683, 0.0928775, -0.0616677, 0.1162400, -0.1616083, 0.1545452
3: -0.0407348, 0.0062448, -0.0501337, 0.0089830, -0.0497178, 0.0563785
4: -0.0516612, 0.0569043, -0.0679634, 0.0711165, -0.1227777, 0.1248677
5: -0.0501005, 0.0608767, -0.0631246, 0.0733382, -0.1234387, 0.1240012
6: 0.8857971, 1.0114497, 0.8558233, 1.0173360, -0.1315389, 0.1556264
7: -0.0797431, 0.0395860, -0.0926262, 0.0632645, -0.1430076, 0.1322121
8: -0.0513750, 0.0638000, -0.0616416, 0.0825197, -0.1338948, 0.1254416
9: -0.0359361, 0.0545845, -0.0486362, 0.0641458, -0.1000819, 0.1032207

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_B1_A1_B2_B1_B1

### Relational analysis result of IS_A2_B2_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2647157, upper bound: 0.2753838
time: 1.14 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2_B1_B2

### Relational analysis result of IS_A2_B2_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2647157, upper bound: 0.2784628
time: 1.30 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0208144, 0.0248916, -0.0366168, 0.0445354, -0.0653498, 0.0615084
1: -0.0339509, 0.0653215, -0.0498794, 0.0930557, -0.1270067, 0.1152009
2: -0.0457339, 0.0933682, -0.0721760, 0.1330888, -0.1788227, 0.1655442
3: -0.0409439, 0.0063062, -0.0549770, 0.0150550, -0.0559990, 0.0612832
4: -0.0520009, 0.0572231, -0.0807994, 0.0817370, -0.1337379, 0.1380225
5: -0.0503706, 0.0611562, -0.0756808, 0.0793870, -0.1297576, 0.1368370
6: 0.8851466, 1.0115817, 0.8351592, 1.0211900, -0.1360434, 0.1764224
7: -0.0800302, 0.0401170, -0.1031014, 0.0759003, -0.1559305, 0.1432183
8: -0.0516053, 0.0641867, -0.0684053, 0.0979581, -0.1495634, 0.1325920
9: -0.0361958, 0.0547990, -0.0628460, 0.0687869, -0.1049826, 0.1176450

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_B1_A1_B2_B2_B1

### Relational analysis result of IS_A2_B2_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2647037, upper bound: 0.2753510
time: 1.08 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2_B2_B2

### Relational analysis result of IS_A2_B2_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2647037, upper bound: 0.2783753
time: 1.39 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0373123, 0.0454413, -0.0365268, 0.0442372, -0.0815495, 0.0819682
1: -0.0508795, 0.0945136, -0.0499297, 0.0925759, -0.1434554, 0.1444433
2: -0.0731175, 0.1344611, -0.0718766, 0.1326371, -0.2057547, 0.2063377
3: -0.0553715, 0.0167702, -0.0548471, 0.0152094, -0.0705808, 0.0716173
4: -0.0819438, 0.0827150, -0.0804882, 0.0814151, -0.1633590, 0.1632033
5: -0.0771511, 0.0798797, -0.0754457, 0.0792249, -0.1563760, 0.1553255
6: 0.8334003, 1.0217539, 0.8357384, 1.0211705, -0.1877702, 0.1860155
7: -0.1040731, 0.0771781, -0.1027815, 0.0756448, -0.1797179, 0.1799597
8: -0.0690622, 0.0992163, -0.0682237, 0.0975440, -0.1666062, 0.1674400
9: -0.0642622, 0.0691648, -0.0624987, 0.0686624, -0.1329246, 0.1316635

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_B1_A2_A1_B1_B1

### Relational analysis result of IS_A2_B2_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2764628, upper bound: 0.2607699
time: 1.14 seconds

## Relational analysis of IS_A2_B2_B1_A2_A1_B1_B2

### Relational analysis result of IS_A2_B2_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2764628, upper bound: 0.2607699
time: 2.36 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0373123, 0.0454413, -0.0532072, 0.0575769, -0.0948892, 0.0986485
1: -0.0508795, 0.0945136, -0.0669177, 0.1142876, -0.1651671, 0.1614313
2: -0.0731175, 0.1344611, -0.0870932, 0.1567273, -0.2298448, 0.2215543
3: -0.0553715, 0.0167702, -0.0615649, 0.0408807, -0.0962521, 0.0783351
4: -0.0819438, 0.0827150, -0.1027066, 0.0975667, -0.1795105, 0.1854216
5: -0.0771511, 0.0798797, -0.0949712, 0.0965507, -0.1737018, 0.1748510
6: 0.8334003, 1.0217539, 0.8033792, 1.0280327, -0.1946324, 0.2183747
7: -0.1040731, 0.0771781, -0.1238513, 0.0968270, -0.2009002, 0.2010294
8: -0.0690622, 0.0992163, -0.0789342, 0.1239102, -0.1929724, 0.1781505
9: -0.0642622, 0.0691648, -0.0839923, 0.0799393, -0.1442015, 0.1531571

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_B1_A2_A1_B2_B1

### Relational analysis result of IS_A2_B2_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2764628, upper bound: 0.2653970
time: 1.67 seconds

## Relational analysis of IS_A2_B2_B1_A2_A1_B2_B2

### Relational analysis result of IS_A2_B2_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2764628, upper bound: 0.2944311
time: 1.36 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0372334, 0.0453204, -0.0333623, 0.0384805, -0.0757139, 0.0786827
1: -0.0507840, 0.0943190, -0.0470024, 0.0833123, -0.1340963, 0.1413214
2: -0.0729929, 0.1342779, -0.0659881, 0.1239169, -0.1969099, 0.2002660
3: -0.0553188, 0.0166133, -0.0523405, 0.0108025, -0.0661213, 0.0689538
4: -0.0817976, 0.0825845, -0.0738085, 0.0752005, -0.1569981, 0.1563930
5: -0.0769799, 0.0798139, -0.0683503, 0.0760943, -0.1530741, 0.1481642
6: 0.8336352, 1.0216953, 0.8469154, 1.0190865, -0.1854513, 0.1747799
7: -0.1039434, 0.0770242, -0.0966064, 0.0690155, -0.1729589, 0.1736305
8: -0.0689780, 0.0990483, -0.0643622, 0.0895492, -0.1585272, 0.1634105
9: -0.0640851, 0.0691144, -0.0545724, 0.0662604, -0.1303455, 0.1236868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_B1_A2_A2_B1_B1

### Relational analysis result of IS_A2_B2_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2763133, upper bound: 0.2607699
time: 1.08 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_B1_B2

### Relational analysis result of IS_A2_B2_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2763133, upper bound: 0.2926339
time: 1.23 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0372334, 0.0453204, -0.0439174, 0.0515715, -0.0888048, 0.0892379
1: -0.0507840, 0.0943190, -0.0579973, 0.1043780, -0.1551620, 0.1523163
2: -0.0729929, 0.1342779, -0.0795370, 0.1454739, -0.2184669, 0.2138149
3: -0.0553188, 0.0166133, -0.0581333, 0.0283663, -0.0836851, 0.0747465
4: -0.0817976, 0.0825845, -0.0920542, 0.0894832, -0.1712808, 0.1746386
5: -0.0769799, 0.0798139, -0.0858330, 0.0870473, -0.1640271, 0.1656469
6: 0.8336352, 1.0216953, 0.8188104, 1.0247252, -0.1910900, 0.2028849
7: -0.1039434, 0.0770242, -0.1139211, 0.0853653, -0.1893087, 0.1909452
8: -0.0689780, 0.0990483, -0.0734757, 0.1104713, -0.1794492, 0.1725240
9: -0.0640851, 0.0691144, -0.0733185, 0.0744868, -0.1385719, 0.1424329

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_B1_A2_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2763133, upper bound: 0.2653970
time: 1.67 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_B2_B2

### Relational analysis result of IS_A2_B2_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2763133, upper bound: 0.2944311
time: 1.22 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0246926, 0.0274910, -0.0313061, 0.0345278, -0.0592204, 0.0587970
1: -0.0381041, 0.0702756, -0.0451244, 0.0787570, -0.1168612, 0.1154000
2: -0.0519514, 0.1017147, -0.0624884, 0.1179293, -0.1698807, 0.1642031
3: -0.0445004, 0.0073507, -0.0506194, 0.0091233, -0.0536236, 0.0579701
4: -0.0577780, 0.0626444, -0.0692447, 0.0718083, -0.1295863, 0.1318891
5: -0.0549642, 0.0659097, -0.0641378, 0.0739447, -0.1289089, 0.1300475
6: 0.8740839, 1.0138268, 0.8540020, 1.0177132, -0.1436293, 0.1598248
7: -0.0849122, 0.0491494, -0.0932848, 0.0645210, -0.1494332, 0.1424341
8: -0.0555215, 0.0707643, -0.0621413, 0.0840598, -0.1395813, 0.1329056
9: -0.0406108, 0.0584462, -0.0497870, 0.0646111, -0.1052219, 0.1082331

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_B2_A1_B1_B1_B1

### Relational analysis result of IS_A2_B2_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2646626, upper bound: 0.2754858
time: 1.02 seconds

## Relational analysis of IS_A2_B2_B2_A1_B1_B1_B2

### Relational analysis result of IS_A2_B2_B2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2646626, upper bound: 0.2785968
time: 1.35 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0220464, 0.0257173, -0.0310085, 0.0339346, -0.0559810, 0.0567258
1: -0.0352703, 0.0668953, -0.0448425, 0.0783647, -0.1136350, 0.1117378
2: -0.0477089, 0.0960197, -0.0620503, 0.1170309, -0.1647398, 0.1580700
3: -0.0420737, 0.0066380, -0.0503611, 0.0090482, -0.0511219, 0.0569991
4: -0.0538361, 0.0589453, -0.0685598, 0.0714404, -0.1252765, 0.1275052
5: -0.0518298, 0.0626663, -0.0635981, 0.0736222, -0.1254520, 0.1262644
6: 0.8816322, 1.0122949, 0.8549706, 1.0175072, -0.1358750, 0.1573243
7: -0.0815811, 0.0429863, -0.0929345, 0.0638465, -0.1454276, 0.1359209
8: -0.0528494, 0.0662762, -0.0618756, 0.0832360, -0.1360854, 0.1281518
9: -0.0375983, 0.0559576, -0.0491682, 0.0643637, -0.1019620, 0.1051258

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_B2_A1_B1_B2_B1

### Relational analysis result of IS_A2_B2_B2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2645623, upper bound: 0.2753768
time: 1.36 seconds

## Relational analysis of IS_A2_B2_B2_A1_B1_B2_B2

### Relational analysis result of IS_A2_B2_B2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2645623, upper bound: 0.2784628
time: 1.25 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0249371, 0.0276549, -0.0373615, 0.0455167, -0.0704538, 0.0650163
1: -0.0383660, 0.0705880, -0.0509390, 0.0946350, -0.1330010, 0.1215269
2: -0.0523433, 0.1022410, -0.0731953, 0.1345753, -0.1869186, 0.1754363
3: -0.0447246, 0.0074166, -0.0554043, 0.0168679, -0.0615925, 0.0628209
4: -0.0581422, 0.0629862, -0.0820349, 0.0827965, -0.1409387, 0.1450212
5: -0.0552538, 0.0662094, -0.0772579, 0.0799207, -0.1351745, 0.1434673
6: 0.8733864, 1.0139685, 0.8332539, 1.0217906, -0.1484042, 0.1807145
7: -0.0852200, 0.0497188, -0.1041541, 0.0772742, -0.1624942, 0.1538728
8: -0.0557685, 0.0711790, -0.0691147, 0.0993210, -0.1550895, 0.1402937
9: -0.0408891, 0.0586761, -0.0643726, 0.0691963, -0.1100854, 0.1230488

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=46, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_B1

### Relational analysis result of IS_A2_B2_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2646519, upper bound: 0.2754520
time: 1.70 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_B2

### Relational analysis result of IS_A2_B2_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2646519, upper bound: 0.2785061
time: 1.37 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0222871, 0.0258787, -0.0368133, 0.0446763, -0.0669634, 0.0626920
1: -0.0355281, 0.0672028, -0.0502761, 0.0932826, -0.1288107, 0.1174789
2: -0.0480949, 0.0965378, -0.0723292, 0.1333023, -0.1813972, 0.1688670
3: -0.0422945, 0.0067029, -0.0550384, 0.0157786, -0.0580731, 0.0617412
4: -0.0541948, 0.0592818, -0.0810191, 0.0818892, -0.1360840, 0.1403008
5: -0.0521149, 0.0629613, -0.0760677, 0.0794637, -0.1315786, 0.1390290
6: 0.8809455, 1.0124344, 0.8348857, 1.0213832, -0.1404377, 0.1775487
7: -0.0818842, 0.0435469, -0.1032526, 0.0762040, -0.1580882, 0.1467995
8: -0.0530925, 0.0666845, -0.0685295, 0.0981539, -0.1512464, 0.1352140
9: -0.0378723, 0.0561840, -0.0631418, 0.0688457, -0.1067180, 0.1193258

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_B2_A1_B2_B2_B1

### Relational analysis result of IS_A2_B2_B2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2645355, upper bound: 0.2753504
time: 1.47 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_B2_B2

### Relational analysis result of IS_A2_B2_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2645355, upper bound: 0.2783753
time: 1.49 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0374015, 0.0455783, -0.0629426, 0.0651767, -0.1025783, 0.1085209
1: -0.0509874, 0.0947337, -0.0764929, 0.1231790, -0.1741664, 0.1712266
2: -0.0732586, 0.1346685, -0.0948410, 0.1675687, -0.2408273, 0.2295094
3: -0.0554311, 0.0169475, -0.0656965, 0.0533027, -0.1087338, 0.0826440
4: -0.0821092, 0.0828627, -0.1122672, 0.1055149, -0.1876241, 0.1951299
5: -0.0773449, 0.0799541, -0.1044728, 0.1068049, -0.1841498, 0.1844269
6: 0.8331347, 1.0218204, 0.7878389, 1.0310034, -0.1978686, 0.2339815
7: -0.1042199, 0.0773524, -0.1336530, 0.1077907, -0.2120106, 0.2110054
8: -0.0691575, 0.0994064, -0.0846359, 0.1396139, -0.2087714, 0.1840422
9: -0.0644626, 0.0692219, -0.0961086, 0.0848298, -0.1492924, 0.1653305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_B2_A2_A1_B1_B1

### Relational analysis result of IS_A2_B2_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2943045, upper bound: 0.2926339
time: 1.18 seconds

## Relational analysis of IS_A2_B2_B2_A2_A1_B1_B2

### Relational analysis result of IS_A2_B2_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2943045, upper bound: 0.2944494
time: 1.27 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0378819, 0.0461649, -0.0883359, 0.0862978, -0.1241797, 0.1345008
1: -0.0515349, 0.0956779, -0.1005475, 0.1459530, -0.1974878, 0.1962254
2: -0.0738632, 0.1356268, -0.1135769, 0.1990516, -0.2729149, 0.2492037
3: -0.0556865, 0.0178551, -0.0809989, 0.0777988, -0.1334853, 0.0988540
4: -0.0829273, 0.0834961, -0.1410894, 0.1254763, -0.2084036, 0.2245855
5: -0.0781758, 0.0804168, -0.1245629, 0.1379763, -0.2161521, 0.2049797
6: 0.8318905, 1.0221046, 0.7503770, 1.0388904, -0.2069999, 0.2717276
7: -0.1049812, 0.0780995, -0.1600277, 0.1373555, -0.2423367, 0.2381272
8: -0.0695661, 0.1003316, -0.1047071, 0.1705821, -0.2401481, 0.2050387
9: -0.0653219, 0.0695781, -0.1261217, 0.1097915, -0.1751133, 0.1956998

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_B2_A2_A1_B2_B1

### Relational analysis result of IS_A2_B2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2942958, upper bound: 0.2923237
time: 1.17 seconds

## Relational analysis of IS_A2_B2_B2_A2_A1_B2_B2

### Relational analysis result of IS_A2_B2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2942958, upper bound: 0.2942633
time: 1.40 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0407435, 0.0488087, -0.0345624, 0.0407132, -0.0814567, 0.0833711
1: -0.0546062, 0.0999321, -0.0480633, 0.0869050, -0.1415112, 0.1479954
2: -0.0765879, 0.1404420, -0.0682698, 0.1272990, -0.2038869, 0.2087118
3: -0.0568377, 0.0229949, -0.0533127, 0.0123705, -0.0692082, 0.0763077
4: -0.0873901, 0.0863502, -0.0763863, 0.0776108, -0.1650009, 0.1627365
5: -0.0819201, 0.0835262, -0.0710533, 0.0773084, -0.1592286, 0.1545795
6: 0.8255345, 1.0233861, 0.8425806, 1.0198621, -0.1943276, 0.1808056
7: -0.1093526, 0.0814662, -0.0990012, 0.0715543, -0.1809069, 0.1804675
8: -0.0714071, 0.1052898, -0.0658530, 0.0926499, -0.1640570, 0.1711428
9: -0.0691938, 0.0719784, -0.0576232, 0.0671920, -0.1363858, 0.1296016

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_B2_A2_A2_B1_A1

### Relational analysis result of IS_A2_B2_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2941162, upper bound: 0.2923237
time: 1.15 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_B1_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2939759, upper bound: 0.2923237
time: 1.33 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0407435, 0.0488087, -0.0433092, 0.0511214, -0.0918649, 0.0921179
1: -0.0546062, 0.0999321, -0.0573547, 0.1036540, -0.1582602, 0.1572868
2: -0.0765879, 0.1404420, -0.0790061, 0.1446543, -0.2212422, 0.2194480
3: -0.0568377, 0.0229949, -0.0578761, 0.0274915, -0.0843292, 0.0808711
4: -0.0873901, 0.0863502, -0.0912945, 0.0888980, -0.1762881, 0.1776447
5: -0.0819201, 0.0835262, -0.0851957, 0.0863386, -0.1682587, 0.1687219
6: 0.8255345, 1.0233861, 0.8199461, 1.0245074, -0.1989729, 0.2034401
7: -0.1093526, 0.0814662, -0.1131771, 0.0845405, -0.1938931, 0.1946433
8: -0.0714071, 0.1052898, -0.0730667, 0.1096273, -0.1810344, 0.1783564
9: -0.0691938, 0.0719784, -0.0726076, 0.0740783, -0.1432721, 0.1445860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_B2_A2_A2_B2_B1

### Relational analysis result of IS_A2_B2_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2757155, upper bound: 0.2653970
time: 1.33 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2757155, upper bound: 0.2944494
time: 1.29 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.25 seconds
IS_A1_B1_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2651796, upper bound: 0.2569876
IS_A1_B1_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2613599, upper bound: 0.2502546
IS_A1_B1_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2647452, upper bound: 0.2569295
IS_A1_B1_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2607098, upper bound: 0.2501556
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2621067, upper bound: 0.2504352
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2601264, upper bound: 0.2500037
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2621030, upper bound: 0.2503715
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2601684, upper bound: 0.2499768
IS_A1_B1_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2761200
IS_A1_B1_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2761858
IS_A1_B1_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2942374
IS_A1_B1_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2942078
IS_A1_B1_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2758563
IS_A1_B1_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2759379
IS_A1_B1_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2938506
IS_A1_B1_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2937822
IS_A1_B1_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2651796, upper bound: 0.2569796
IS_A1_B1_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2613599, upper bound: 0.2502005
IS_A1_B1_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2647452, upper bound: 0.2569232
IS_A1_B1_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2607098, upper bound: 0.2500947
IS_A1_B1_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2643643, upper bound: 0.2563229
IS_A1_B1_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2607601, upper bound: 0.2500020
IS_A1_B1_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2641339, upper bound: 0.2562194
IS_A1_B1_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2601684, upper bound: 0.2498814
IS_A1_B1_A2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2664985, upper bound: 0.2773514
IS_A1_B1_A2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2664985, upper bound: 0.2653318
IS_A1_B1_A2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2664985, upper bound: 0.2773834
IS_A1_B1_A2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2664985, upper bound: 0.2937951
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2602165, upper bound: 0.2753916
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2602165, upper bound: 0.2938309
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2602165, upper bound: 0.2753428
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2602165, upper bound: 0.2937682
IS_A1_B2_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2437880, upper bound: 0.2628746
IS_A1_B2_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2437880, upper bound: 0.2613030
IS_A1_B2_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2428060, upper bound: 0.2628705
IS_A1_B2_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2428060, upper bound: 0.2613279
IS_A1_B2_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2428060, upper bound: 0.2627289
IS_A1_B2_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2428060, upper bound: 0.2614633
IS_A1_B2_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2428060, upper bound: 0.2624050
IS_A1_B2_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2428060, upper bound: 0.2607516
IS_A1_B2_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2774529, upper bound: 0.2699544
IS_A1_B2_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2775443, upper bound: 0.2699544
IS_A1_B2_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2774529, upper bound: 0.2943560
IS_A1_B2_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2775443, upper bound: 0.2942349
IS_A1_B2_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2752349, upper bound: 0.2647037
IS_A1_B2_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2752349, upper bound: 0.2940443
IS_A1_B2_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2753510, upper bound: 0.2647037
IS_A1_B2_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2753510, upper bound: 0.2939128
IS_A1_B2_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2534686, upper bound: 0.2654453
IS_A1_B2_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2427043, upper bound: 0.2614323
IS_A1_B2_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2534234, upper bound: 0.2649901
IS_A1_B2_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2427043, upper bound: 0.2606757
IS_A1_B2_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2529066, upper bound: 0.2647521
IS_A1_B2_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2427043, upper bound: 0.2608559
IS_A1_B2_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2528600, upper bound: 0.2643859
IS_A1_B2_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2427043, upper bound: 0.2602371
IS_A1_B2_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2775566, upper bound: 0.2696017
IS_A1_B2_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2775566, upper bound: 0.2942321
IS_A1_B2_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2775258, upper bound: 0.2695683
IS_A1_B2_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2775258, upper bound: 0.2942050
IS_A1_B2_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2752299, upper bound: 0.2645355
IS_A1_B2_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2752299, upper bound: 0.2940230
IS_A1_B2_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2753504, upper bound: 0.2645355
IS_A1_B2_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2753504, upper bound: 0.2939018
IS_A2_B1_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2756073, upper bound: 0.2615459
IS_A2_B1_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2757210, upper bound: 0.2615459
IS_A2_B1_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2756073, upper bound: 0.2673872
IS_A2_B1_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2757210, upper bound: 0.2673870
IS_A2_B1_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2756650, upper bound: 0.2615459
IS_A2_B1_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2756650, upper bound: 0.2673659
IS_A2_B1_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2757528, upper bound: 0.2615459
IS_A2_B1_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2757528, upper bound: 0.2673658
IS_A2_B1_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2643854, upper bound: 0.2754962
IS_A2_B1_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2643854, upper bound: 0.2784516
IS_A2_B1_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2945554
IS_A2_B1_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2714759
IS_A2_B1_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2639159, upper bound: 0.2752510
IS_A2_B1_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2639159, upper bound: 0.2781760
IS_A2_B1_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2942308
IS_A2_B1_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2941128
IS_A2_B1_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2756088, upper bound: 0.2615459
IS_A2_B1_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2757287, upper bound: 0.2615459
IS_A2_B1_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2756088, upper bound: 0.2673926
IS_A2_B1_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2757287, upper bound: 0.2673926
IS_A2_B1_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2756650, upper bound: 0.2615459
IS_A2_B1_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2757528, upper bound: 0.2615459
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2758206, upper bound: 0.2673738
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2757528, upper bound: 0.2673693
IS_A2_B1_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2643747, upper bound: 0.2752556
IS_A2_B1_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2643747, upper bound: 0.2712650
IS_A2_B1_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2945554
IS_A2_B1_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2945554
IS_A2_B1_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2639114, upper bound: 0.2750662
IS_A2_B1_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2639114, upper bound: 0.2781353
IS_A2_B1_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2942308
IS_A2_B1_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2670359
IS_A2_B2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2647862, upper bound: 0.2753541
IS_A2_B2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2647862, upper bound: 0.2783910
IS_A2_B2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2647862, upper bound: 0.2754520
IS_A2_B2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2647862, upper bound: 0.2785061
IS_A2_B2_B1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2647157, upper bound: 0.2753838
IS_A2_B2_B1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2647157, upper bound: 0.2784628
IS_A2_B2_B1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2647037, upper bound: 0.2753510
IS_A2_B2_B1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2647037, upper bound: 0.2783753
IS_A2_B2_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2764628, upper bound: 0.2607699
IS_A2_B2_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2764628, upper bound: 0.2607699
IS_A2_B2_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2764628, upper bound: 0.2653970
IS_A2_B2_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2764628, upper bound: 0.2944311
IS_A2_B2_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2763133, upper bound: 0.2607699
IS_A2_B2_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2763133, upper bound: 0.2926339
IS_A2_B2_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2763133, upper bound: 0.2653970
IS_A2_B2_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2763133, upper bound: 0.2944311
IS_A2_B2_B2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2646626, upper bound: 0.2754858
IS_A2_B2_B2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2646626, upper bound: 0.2785968
IS_A2_B2_B2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2645623, upper bound: 0.2753768
IS_A2_B2_B2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2645623, upper bound: 0.2784628
IS_A2_B2_B2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2646519, upper bound: 0.2754520
IS_A2_B2_B2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2646519, upper bound: 0.2785061
IS_A2_B2_B2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2645355, upper bound: 0.2753504
IS_A2_B2_B2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2645355, upper bound: 0.2783753
IS_A2_B2_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2943045, upper bound: 0.2926339
IS_A2_B2_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2943045, upper bound: 0.2944494
IS_A2_B2_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2942958, upper bound: 0.2923237
IS_A2_B2_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2942958, upper bound: 0.2942633
IS_A2_B2_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2941162, upper bound: 0.2923237
IS_A2_B2_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2939759, upper bound: 0.2923237
IS_A2_B2_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2757155, upper bound: 0.2653970
IS_A2_B2_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 6, lower bound: -0.2757155, upper bound: 0.2944494

## BFS IS instance: IS_A1_B1_A1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0232999, 0.0265575, -0.0162997, 0.0218655, -0.0451654, 0.0428572
1: -0.0366126, 0.0684965, -0.0291160, 0.0595541, -0.0961668, 0.0976125
2: -0.0497186, 0.0987175, -0.0384957, 0.0836516, -0.1333702, 0.1372132
3: -0.0432232, 0.0069756, -0.0368037, 0.0050903, -0.0483135, 0.0437793
4: -0.0557034, 0.0606976, -0.0452755, 0.0509119, -0.1066153, 0.1059730
5: -0.0533145, 0.0642026, -0.0450229, 0.0556224, -0.1089369, 0.1092256
6: 0.8780565, 1.0130206, 0.8980253, 1.0089678, -0.1309112, 0.1149954
7: -0.0831590, 0.0459057, -0.0743468, 0.0296020, -0.1127610, 0.1202525
8: -0.0541152, 0.0684022, -0.0470462, 0.0565294, -0.1106446, 0.1154484
9: -0.0390253, 0.0571365, -0.0310560, 0.0505532, -0.0895784, 0.0881924

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1_A1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2399134, upper bound: 0.2316366
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2382189, upper bound: 0.2306075
time: 1.22 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0204285, 0.0246330, -0.0161658, 0.0217758, -0.0422043, 0.0407988
1: -0.0335377, 0.0648285, -0.0289726, 0.0593831, -0.0929208, 0.0938012
2: -0.0451153, 0.0925378, -0.0382811, 0.0833636, -0.1284788, 0.1308189
3: -0.0405901, 0.0062023, -0.0366810, 0.0050542, -0.0456443, 0.0428833
4: -0.0514261, 0.0566837, -0.0450761, 0.0507247, -0.1021508, 0.1017598
5: -0.0499135, 0.0606832, -0.0448644, 0.0554584, -0.1053719, 0.1055476
6: 0.8862474, 1.0113581, 0.8984073, 1.0088902, -0.1226428, 0.1129509
7: -0.0795444, 0.0392183, -0.0741784, 0.0292902, -0.1088346, 0.1133967
8: -0.0512156, 0.0635323, -0.0469110, 0.0563024, -0.1075181, 0.1104433
9: -0.0357565, 0.0544361, -0.0309036, 0.0504273, -0.0861837, 0.0853397

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A1_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A1_B1_A1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2613394, upper bound: 0.2502546
time: 1.27 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2613394, upper bound: 0.2502546
time: 1.22 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0278638, 0.0296165, -0.0165514, 0.0220342, -0.0498980, 0.0461679
1: -0.0415001, 0.0743266, -0.0293856, 0.0598757, -0.1013758, 0.1037121
2: -0.0570354, 0.1085396, -0.0388993, 0.0841935, -0.1412289, 0.1474389
3: -0.0474084, 0.0082048, -0.0370346, 0.0051581, -0.0525665, 0.0452394
4: -0.0625020, 0.0670774, -0.0456505, 0.0512638, -0.1137657, 0.1127279
5: -0.0587204, 0.0697967, -0.0453211, 0.0559310, -0.1146513, 0.1151178
6: 0.8650377, 1.0156630, 0.8973072, 1.0091133, -0.1440756, 0.1183558
7: -0.0889042, 0.0565351, -0.0746638, 0.0301882, -0.1190924, 0.1311988
8: -0.0587238, 0.0761428, -0.0473004, 0.0569564, -0.1156802, 0.1234432
9: -0.0442210, 0.0614284, -0.0313426, 0.0507899, -0.0950109, 0.0927710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1_A1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2389544, upper bound: 0.2312473
time: 1.25 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2364909, upper bound: 0.2293703
time: 1.30 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0251691, 0.0278103, -0.0164290, 0.0219523, -0.0471213, 0.0442394
1: -0.0386144, 0.0708843, -0.0292546, 0.0597194, -0.0983338, 0.1001389
2: -0.0527153, 0.1027403, -0.0387032, 0.0839302, -0.1366455, 0.1414435
3: -0.0449373, 0.0074791, -0.0369224, 0.0051251, -0.0500625, 0.0444014
4: -0.0584879, 0.0633105, -0.0454682, 0.0510928, -0.1095806, 0.1087787
5: -0.0555286, 0.0664938, -0.0451762, 0.0557810, -0.1113096, 0.1116700
6: 0.8727244, 1.0141028, 0.8976561, 1.0090425, -0.1363181, 0.1164467
7: -0.0855121, 0.0502591, -0.0745097, 0.0299034, -0.1154155, 0.1247688
8: -0.0560027, 0.0715725, -0.0471769, 0.0567489, -0.1127516, 0.1187494
9: -0.0411533, 0.0588943, -0.0312033, 0.0506748, -0.0918281, 0.0900977

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=46, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A1_B1_A1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2607048, upper bound: 0.2501556
time: 1.20 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2607048, upper bound: 0.2501556
time: 1.22 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0204070, 0.0246185, -0.0156302, 0.0214168, -0.0418238, 0.0402487
1: -0.0335146, 0.0648010, -0.0283990, 0.0586989, -0.0922135, 0.0932000
2: -0.0450806, 0.0924914, -0.0374224, 0.0822109, -0.1272915, 0.1299139
3: -0.0405703, 0.0061965, -0.0361898, 0.0049100, -0.0454803, 0.0423863
4: -0.0513939, 0.0566535, -0.0442782, 0.0499760, -0.1013699, 0.1009317
5: -0.0498880, 0.0606568, -0.0442300, 0.0548018, -0.1046898, 0.1048868
6: 0.8863088, 1.0113459, 0.8999351, 1.0085800, -0.1222711, 0.1114107
7: -0.0795173, 0.0391681, -0.0735041, 0.0280428, -0.1075601, 0.1126721
8: -0.0511939, 0.0634957, -0.0463702, 0.0553940, -0.1065879, 0.1098659
9: -0.0357319, 0.0544158, -0.0302938, 0.0499235, -0.0856555, 0.0847097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=44, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2437508, upper bound: 0.2379629
time: 1.51 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2609584, upper bound: 0.2447353
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2609584, upper bound: 0.2504352
time: 1.46 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0198706, 0.0242590, -0.0128231, 0.0195353, -0.0394059, 0.0370821
1: -0.0329402, 0.0641158, -0.0253928, 0.0551130, -0.0880532, 0.0895086
2: -0.0442207, 0.0913371, -0.0329220, 0.0761694, -0.1203902, 0.1242591
3: -0.0400784, 0.0060520, -0.0336155, 0.0041539, -0.0442324, 0.0396676
4: -0.0505950, 0.0559038, -0.0400965, 0.0460519, -0.0966469, 0.0960003
5: -0.0492527, 0.0599994, -0.0409049, 0.0513611, -0.1006138, 0.1009043
6: 0.8878387, 1.0110353, 0.9079426, 1.0069548, -0.1191161, 0.1030927
7: -0.0788421, 0.0379189, -0.0699704, 0.0215048, -0.1003470, 0.1078892
8: -0.0506523, 0.0625860, -0.0435355, 0.0506329, -0.1012852, 0.1061215
9: -0.0351213, 0.0539114, -0.0270981, 0.0472836, -0.0824049, 0.0810095

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=44, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2390274, upper bound: 0.2346807
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2592050, upper bound: 0.2447353
time: 1.36 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2592050, upper bound: 0.2500037
time: 1.39 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0206289, 0.0247672, -0.0203670, 0.0245917, -0.0452205, 0.0451342
1: -0.0337522, 0.0650844, -0.0334717, 0.0647499, -0.0985021, 0.0985562
2: -0.0454363, 0.0929689, -0.0450165, 0.0924053, -0.1378416, 0.1379853
3: -0.0407738, 0.0062563, -0.0405336, 0.0061857, -0.0469595, 0.0467898
4: -0.0517245, 0.0569638, -0.0513344, 0.0565976, -0.1083221, 0.1082981
5: -0.0501508, 0.0609288, -0.0498406, 0.0606078, -0.1107586, 0.1107693
6: 0.8856758, 1.0114743, 0.8864230, 1.0113227, -0.1256469, 0.1250513
7: -0.0797966, 0.0396848, -0.0794669, 0.0390748, -0.1188714, 0.1191518
8: -0.0514179, 0.0638720, -0.0511535, 0.0634278, -0.1148458, 0.1150255
9: -0.0359845, 0.0546245, -0.0356863, 0.0543782, -0.0903627, 0.0903108

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2436970, upper bound: 0.2379203
time: 1.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2609527, upper bound: 0.2439026
time: 1.22 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2609527, upper bound: 0.2503715
time: 1.29 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0200947, 0.0244091, -0.0174801, 0.0226567, -0.0427514, 0.0418892
1: -0.0331801, 0.0644020, -0.0303801, 0.0610620, -0.0942421, 0.0947822
2: -0.0445799, 0.0918192, -0.0403882, 0.0861922, -0.1307721, 0.1322074
3: -0.0402839, 0.0061124, -0.0378862, 0.0054082, -0.0456921, 0.0439986
4: -0.0509287, 0.0562169, -0.0470339, 0.0525620, -0.1034907, 0.1032508
5: -0.0495180, 0.0602740, -0.0464211, 0.0570693, -0.1065873, 0.1066951
6: 0.8871998, 1.0111649, 0.8946580, 1.0096511, -0.1224513, 0.1165069
7: -0.0791242, 0.0384406, -0.0758328, 0.0323512, -0.1114753, 0.1142735
8: -0.0508785, 0.0629660, -0.0482382, 0.0585315, -0.1094100, 0.1112042
9: -0.0353764, 0.0541221, -0.0323998, 0.0516632, -0.0870396, 0.0865219

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=44, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2390796, upper bound: 0.2346807
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2592205, upper bound: 0.2439026
time: 1.35 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2592205, upper bound: 0.2499768
time: 1.31 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 11.59 seconds
IS_A1_B1_A1_B1_A1_A1_B1_A1, status: Status.VERIFIED, split count: 8, time: 11.59
Output dim: 6, lower bound: -0.2399134, upper bound: 0.2316366
IS_A1_B1_A1_B1_A1_A1_B1_A2, status: Status.VERIFIED, split count: 8, time: 11.59
Output dim: 6, lower bound: -0.2382189, upper bound: 0.2306075
IS_A1_B1_A1_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 11.59
Output dim: 6, lower bound: -0.2613394, upper bound: 0.2502546
IS_A1_B1_A1_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 11.59
Output dim: 6, lower bound: -0.2613394, upper bound: 0.2502546
IS_A1_B1_A1_B1_A1_A2_B1_A1, status: Status.VERIFIED, split count: 8, time: 11.59
Output dim: 6, lower bound: -0.2389544, upper bound: 0.2312473
IS_A1_B1_A1_B1_A1_A2_B1_A2, status: Status.VERIFIED, split count: 8, time: 11.59
Output dim: 6, lower bound: -0.2364909, upper bound: 0.2293703
IS_A1_B1_A1_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 11.59
Output dim: 6, lower bound: -0.2607048, upper bound: 0.2501556
IS_A1_B1_A1_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 11.59
Output dim: 6, lower bound: -0.2607048, upper bound: 0.2501556
IS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 11.59
Output dim: 6, lower bound: -0.2609584, upper bound: 0.2447353
IS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 11.59
Output dim: 6, lower bound: -0.2609584, upper bound: 0.2504352
IS_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 11.59
Output dim: 6, lower bound: -0.2592050, upper bound: 0.2447353
IS_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 11.59
Output dim: 6, lower bound: -0.2592050, upper bound: 0.2500037
IS_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 11.59
Output dim: 6, lower bound: -0.2609527, upper bound: 0.2439026
IS_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 11.59
Output dim: 6, lower bound: -0.2609527, upper bound: 0.2503715
IS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 11.59
Output dim: 6, lower bound: -0.2592205, upper bound: 0.2439026
IS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 11.59
Output dim: 6, lower bound: -0.2592205, upper bound: 0.2499768
IS_A1_B1_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2761200
IS_A1_B1_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2761858
IS_A1_B1_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2942374
IS_A1_B1_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2942078
IS_A1_B1_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2758563
IS_A1_B1_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2759379
IS_A1_B1_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2938506
IS_A1_B1_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2607699, upper bound: 0.2937822
IS_A1_B1_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2651796, upper bound: 0.2569796
IS_A1_B1_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2613599, upper bound: 0.2502005
IS_A1_B1_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2647452, upper bound: 0.2569232
IS_A1_B1_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2607098, upper bound: 0.2500947
IS_A1_B1_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2643643, upper bound: 0.2563229
IS_A1_B1_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2607601, upper bound: 0.2500020
IS_A1_B1_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2641339, upper bound: 0.2562194
IS_A1_B1_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2601684, upper bound: 0.2498814
IS_A1_B1_A2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2664985, upper bound: 0.2773514
IS_A1_B1_A2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2664985, upper bound: 0.2653318
IS_A1_B1_A2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2664985, upper bound: 0.2773834
IS_A1_B1_A2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2664985, upper bound: 0.2937951
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2602165, upper bound: 0.2753916
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2602165, upper bound: 0.2938309
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2602165, upper bound: 0.2753428
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2602165, upper bound: 0.2937682
IS_A1_B2_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2437880, upper bound: 0.2628746
IS_A1_B2_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2437880, upper bound: 0.2613030
IS_A1_B2_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2428060, upper bound: 0.2628705
IS_A1_B2_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2428060, upper bound: 0.2613279
IS_A1_B2_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2428060, upper bound: 0.2627289
IS_A1_B2_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2428060, upper bound: 0.2614633
IS_A1_B2_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2428060, upper bound: 0.2624050
IS_A1_B2_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2428060, upper bound: 0.2607516
IS_A1_B2_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2774529, upper bound: 0.2699544
IS_A1_B2_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2775443, upper bound: 0.2699544
IS_A1_B2_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2774529, upper bound: 0.2943560
IS_A1_B2_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2775443, upper bound: 0.2942349
IS_A1_B2_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2752349, upper bound: 0.2647037
IS_A1_B2_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2752349, upper bound: 0.2940443
IS_A1_B2_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2753510, upper bound: 0.2647037
IS_A1_B2_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2753510, upper bound: 0.2939128
IS_A1_B2_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2534686, upper bound: 0.2654453
IS_A1_B2_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2427043, upper bound: 0.2614323
IS_A1_B2_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2534234, upper bound: 0.2649901
IS_A1_B2_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2427043, upper bound: 0.2606757
IS_A1_B2_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2529066, upper bound: 0.2647521
IS_A1_B2_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2427043, upper bound: 0.2608559
IS_A1_B2_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2528600, upper bound: 0.2643859
IS_A1_B2_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2427043, upper bound: 0.2602371
IS_A1_B2_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2775566, upper bound: 0.2696017
IS_A1_B2_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2775566, upper bound: 0.2942321
IS_A1_B2_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2775258, upper bound: 0.2695683
IS_A1_B2_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2775258, upper bound: 0.2942050
IS_A1_B2_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2752299, upper bound: 0.2645355
IS_A1_B2_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2752299, upper bound: 0.2940230
IS_A1_B2_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2753504, upper bound: 0.2645355
IS_A1_B2_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2753504, upper bound: 0.2939018
IS_A2_B1_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2756073, upper bound: 0.2615459
IS_A2_B1_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2757210, upper bound: 0.2615459
IS_A2_B1_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2756073, upper bound: 0.2673872
IS_A2_B1_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2757210, upper bound: 0.2673870
IS_A2_B1_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2756650, upper bound: 0.2615459
IS_A2_B1_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2756650, upper bound: 0.2673659
IS_A2_B1_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2757528, upper bound: 0.2615459
IS_A2_B1_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2757528, upper bound: 0.2673658
IS_A2_B1_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2643854, upper bound: 0.2754962
IS_A2_B1_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2643854, upper bound: 0.2784516
IS_A2_B1_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2945554
IS_A2_B1_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2714759
IS_A2_B1_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2639159, upper bound: 0.2752510
IS_A2_B1_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2639159, upper bound: 0.2781760
IS_A2_B1_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2942308
IS_A2_B1_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2941128
IS_A2_B1_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2756088, upper bound: 0.2615459
IS_A2_B1_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2757287, upper bound: 0.2615459
IS_A2_B1_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2756088, upper bound: 0.2673926
IS_A2_B1_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2757287, upper bound: 0.2673926
IS_A2_B1_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2756650, upper bound: 0.2615459
IS_A2_B1_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2757528, upper bound: 0.2615459
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2758206, upper bound: 0.2673738
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2757528, upper bound: 0.2673693
IS_A2_B1_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2643747, upper bound: 0.2752556
IS_A2_B1_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2643747, upper bound: 0.2712650
IS_A2_B1_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2945554
IS_A2_B1_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2945554
IS_A2_B1_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2639114, upper bound: 0.2750662
IS_A2_B1_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2639114, upper bound: 0.2781353
IS_A2_B1_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2942308
IS_A2_B1_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2653970, upper bound: 0.2670359
IS_A2_B2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2647862, upper bound: 0.2753541
IS_A2_B2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2647862, upper bound: 0.2783910
IS_A2_B2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2647862, upper bound: 0.2754520
IS_A2_B2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2647862, upper bound: 0.2785061
IS_A2_B2_B1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2647157, upper bound: 0.2753838
IS_A2_B2_B1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2647157, upper bound: 0.2784628
IS_A2_B2_B1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2647037, upper bound: 0.2753510
IS_A2_B2_B1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2647037, upper bound: 0.2783753
IS_A2_B2_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2764628, upper bound: 0.2607699
IS_A2_B2_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2764628, upper bound: 0.2607699
IS_A2_B2_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2764628, upper bound: 0.2653970
IS_A2_B2_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2764628, upper bound: 0.2944311
IS_A2_B2_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2763133, upper bound: 0.2607699
IS_A2_B2_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2763133, upper bound: 0.2926339
IS_A2_B2_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2763133, upper bound: 0.2653970
IS_A2_B2_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2763133, upper bound: 0.2944311
IS_A2_B2_B2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2646626, upper bound: 0.2754858
IS_A2_B2_B2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2646626, upper bound: 0.2785968
IS_A2_B2_B2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2645623, upper bound: 0.2753768
IS_A2_B2_B2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2645623, upper bound: 0.2784628
IS_A2_B2_B2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2646519, upper bound: 0.2754520
IS_A2_B2_B2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2646519, upper bound: 0.2785061
IS_A2_B2_B2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2645355, upper bound: 0.2753504
IS_A2_B2_B2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2645355, upper bound: 0.2783753
IS_A2_B2_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2943045, upper bound: 0.2926339
IS_A2_B2_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2943045, upper bound: 0.2944494
IS_A2_B2_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2942958, upper bound: 0.2923237
IS_A2_B2_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2942958, upper bound: 0.2942633
IS_A2_B2_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2941162, upper bound: 0.2923237
IS_A2_B2_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2939759, upper bound: 0.2923237
IS_A2_B2_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2757155, upper bound: 0.2653970
IS_A2_B2_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 11.59
Output dim: 6, lower bound: -0.2757155, upper bound: 0.2944494

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.74 + 596.51 = 600.24 seconds
