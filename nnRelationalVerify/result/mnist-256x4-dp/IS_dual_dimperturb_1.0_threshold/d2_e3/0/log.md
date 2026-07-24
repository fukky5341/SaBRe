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
Threshold: 12.3086572218


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819)
1: (-6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910)
2: (-7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517)
3: (-8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391)
4: (-8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165)
5: (-7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348)
6: (-6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664)
7: (-7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242)
8: (-10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918)
9: (-6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.34 + 5.25 = 6.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -12.3209782, upper bound: 12.3209782

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207862, upper bound: 12.3208048
time: 5.06 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207612, upper bound: 12.3207615
time: 3.75 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.94 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8.94
Output dim: 8, lower bound: -12.3207862, upper bound: 12.3208048
IS_A2, status: Status.UNKNOWN, split count: 1, time: 8.94
Output dim: 8, lower bound: -12.3207612, upper bound: 12.3207615

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -6.9145050, 5.6603789, -7.0235176, 5.7524662, -12.6669693, 12.6838970
1: -6.0908675, 5.0896811, -6.1864815, 5.1664095, -11.2572765, 11.2761631
2: -7.7820916, 5.0865397, -7.9090343, 5.1656170, -12.9477081, 12.9955730
3: -8.3309956, 4.1600413, -8.4604549, 4.2214842, -12.5524797, 12.6204967
4: -8.2233934, 5.6603284, -8.3454752, 5.7502418, -13.9736347, 14.0058031
5: -7.0241399, 5.5968919, -7.1336541, 5.6869812, -12.7111206, 12.7305460
6: -6.2827740, 6.3367300, -6.3902140, 6.4361529, -12.7189274, 12.7269440
7: -7.0178199, 6.8376164, -7.1255641, 6.9407606, -13.9585800, 13.9631805
8: -10.2260571, 4.4168253, -10.3838263, 4.4998660, -14.7259235, 14.8006516
9: -6.1759648, 6.2343335, -6.2737536, 6.3332019, -12.5091658, 12.5080862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=32, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=58, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207272, upper bound: 12.3207536
time: 3.22 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207233, upper bound: 12.3207524
time: 4.87 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -7.1849499, 5.8615174, -6.8787990, 5.6300454, -12.8149948, 12.7403164
1: -6.3643727, 5.2857423, -6.0596447, 5.0643029, -11.4286747, 11.3453865
2: -8.1115856, 5.2582517, -7.7406712, 5.0602894, -13.1718740, 12.9989223
3: -8.6973381, 4.3120017, -8.2887440, 4.1400671, -12.8374052, 12.6007462
4: -8.5822849, 5.8753724, -8.1833420, 5.6311707, -14.2134542, 14.0587139
5: -7.2986765, 5.7973976, -6.9882469, 5.5677390, -12.8664141, 12.7856445
6: -6.5070028, 6.5626340, -6.2478952, 6.3048759, -12.8118782, 12.8105297
7: -7.3194623, 7.1339025, -6.9826350, 6.8038430, -14.1233053, 14.1165361
8: -10.6341257, 4.4952536, -10.1745062, 4.3894138, -15.0235395, 14.6697598
9: -6.4205723, 6.4776726, -6.1437764, 6.2016516, -12.6222229, 12.6214466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=32, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=57, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207017, upper bound: 12.3206967
time: 3.33 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206962, upper bound: 12.3206962
time: 3.98 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 8.64 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 8.64
Output dim: 8, lower bound: -12.3207272, upper bound: 12.3207536
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 8.64
Output dim: 8, lower bound: -12.3207233, upper bound: 12.3207524
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 8.64
Output dim: 8, lower bound: -12.3207017, upper bound: 12.3206967
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 8.64
Output dim: 8, lower bound: -12.3206962, upper bound: 12.3206962

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -6.9145050, 5.6603789, -6.8240829, 5.5771694, -12.4916744, 12.4844608
1: -6.0908675, 5.0896811, -6.0102949, 5.0247502, -11.1156178, 11.0999746
2: -7.7820916, 5.0865397, -7.6687365, 5.0140209, -12.7961121, 12.7552757
3: -8.3309956, 4.1600413, -8.2226067, 4.1078248, -12.4388199, 12.3826485
4: -8.2233934, 5.6603284, -8.1181736, 5.5820365, -13.8054276, 13.7785015
5: -7.0241399, 5.5968919, -6.9425273, 5.5226936, -12.5468330, 12.5394173
6: -6.2827740, 6.3367300, -6.2002292, 6.2560167, -12.5387907, 12.5369587
7: -7.0178199, 6.8376164, -6.9225917, 6.7518883, -13.7697067, 13.7602081
8: -10.2260571, 4.4168253, -10.0793161, 4.3257952, -14.5518522, 14.4961414
9: -6.1759648, 6.2343335, -6.0891914, 6.1457334, -12.3216982, 12.3235226

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=58, inp2_unstable=57, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207232, upper bound: 12.3207515
time: 6.56 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207232, upper bound: 12.3207520
time: 3.82 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -6.8078103, 5.5680652, -8.4467106, 6.8592362, -13.6670465, 14.0147762
1: -5.9956288, 5.0137358, -7.5243611, 6.1799479, -12.1755772, 12.5380974
2: -7.6541023, 5.0066438, -9.5779924, 6.1184621, -13.7725630, 14.5846367
3: -8.2020836, 4.0994778, -10.2551126, 5.0135622, -13.2156448, 14.3545904
4: -8.1001415, 5.5710988, -10.0881882, 6.8794599, -14.9795990, 15.6592865
5: -6.9205308, 5.5093536, -8.5949430, 6.7904534, -13.7109842, 14.1042967
6: -6.1810999, 6.2402654, -7.6619344, 7.6517954, -13.8328953, 13.9021988
7: -6.9087396, 6.7348790, -8.6093159, 8.3841286, -15.2928677, 15.3441944
8: -10.0644226, 4.3301220, -12.4761162, 5.1991315, -15.2635536, 16.8062363
9: -6.0773873, 6.1346779, -7.5354133, 7.6007175, -13.6781044, 13.6700916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=233, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207233, upper bound: 12.3207511
time: 6.37 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207233, upper bound: 12.3207520
time: 4.71 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -7.1849499, 5.8615174, -6.6955628, 5.4689059, -12.6538563, 12.5570803
1: -6.3643727, 5.2857423, -5.8986597, 4.9339314, -11.2983036, 11.1843996
2: -8.1115856, 5.2582517, -7.5189228, 4.9232278, -13.0348129, 12.7771740
3: -8.6973381, 4.3120017, -8.0691023, 4.0361290, -12.7334671, 12.3811035
4: -8.5822849, 5.8753724, -7.9749317, 5.4769645, -14.0592489, 13.8503027
5: -7.2986765, 5.7973976, -6.8105984, 5.4171162, -12.7157907, 12.6079960
6: -6.5070028, 6.5626340, -6.0734529, 6.1383696, -12.6453724, 12.6360874
7: -7.3194623, 7.1339025, -6.7975497, 6.6310749, -13.9505367, 13.9314518
8: -10.6341257, 4.4952536, -9.8902149, 4.2298002, -14.8639240, 14.3854675
9: -6.4205723, 6.4776726, -5.9757700, 6.0280290, -12.4486008, 12.4534416

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206969, upper bound: 12.3206963
time: 4.79 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206969, upper bound: 12.3206962
time: 3.47 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -7.0784059, 5.7708116, -8.2395277, 6.6895952, -13.7680006, 14.0103378
1: -6.2691326, 5.2105460, -7.3392797, 6.0339198, -12.3030529, 12.5498257
2: -7.9840326, 5.1831970, -9.3364954, 5.9747138, -13.9587450, 14.5196924
3: -8.5667133, 4.2523489, -10.0040522, 4.8984470, -13.4651604, 14.2564011
4: -8.4597397, 5.7881103, -9.8497505, 6.7124295, -15.1721687, 15.6378613
5: -7.1927581, 5.7112880, -8.3840866, 6.6240535, -13.8168116, 14.0953741
6: -6.4068899, 6.4656825, -7.4657502, 7.4683976, -13.8752880, 13.9314327
7: -7.2125812, 7.0310521, -8.4029026, 8.1848221, -15.3974037, 15.4339523
8: -10.4725103, 4.4151015, -12.1716795, 5.0607557, -15.5332661, 16.5867805
9: -6.3247399, 6.3780279, -7.3522134, 7.4131165, -13.7378540, 13.7302408

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=231, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206969, upper bound: 12.3206963
time: 4.27 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206969, upper bound: 12.3206969
time: 4.87 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 10.46 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 10.46
Output dim: 8, lower bound: -12.3207232, upper bound: 12.3207515
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 10.46
Output dim: 8, lower bound: -12.3207232, upper bound: 12.3207520
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 10.46
Output dim: 8, lower bound: -12.3207233, upper bound: 12.3207511
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 10.46
Output dim: 8, lower bound: -12.3207233, upper bound: 12.3207520
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 10.46
Output dim: 8, lower bound: -12.3206969, upper bound: 12.3206963
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 10.46
Output dim: 8, lower bound: -12.3206969, upper bound: 12.3206962
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 10.46
Output dim: 8, lower bound: -12.3206969, upper bound: 12.3206963
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 10.46
Output dim: 8, lower bound: -12.3206969, upper bound: 12.3206969

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -6.7293825, 5.4974775, -6.8240829, 5.5771694, -12.3065519, 12.3215590
1: -5.9278474, 4.9581099, -6.0102949, 5.0247502, -10.9525967, 10.9684038
2: -7.5583863, 4.9472923, -7.6687365, 5.0140209, -12.5724068, 12.6160278
3: -8.1094646, 4.0548248, -8.2226067, 4.1078248, -12.2172890, 12.2774315
4: -8.0126896, 5.5044532, -8.1181736, 5.5820365, -13.5947227, 13.6226263
5: -6.8454456, 5.4445453, -6.9425273, 5.5226936, -12.3681393, 12.3870707
6: -6.1067233, 6.1687536, -6.2002292, 6.2560167, -12.3627377, 12.3689823
7: -6.8303452, 6.6627579, -6.9225917, 6.7518883, -13.5822315, 13.5853500
8: -9.9405537, 4.2551603, -10.0793161, 4.3257952, -14.2663488, 14.3344765
9: -6.0056019, 6.0593967, -6.0891914, 6.1457334, -12.1513348, 12.1485872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=57, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207265, upper bound: 12.3207531
time: 3.65 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207265, upper bound: 12.3207535
time: 3.93 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.2756357, 6.7197776, -6.8240829, 5.5771694, -13.8528051, 13.5438576
1: -7.3706841, 6.0592847, -6.0102949, 5.0247502, -12.3954325, 12.0695782
2: -9.3782692, 5.9999976, -7.6687365, 5.0140209, -14.3922882, 13.6687326
3: -10.0468102, 4.9184432, -8.2226067, 4.1078248, -14.1546345, 13.1410484
4: -9.8902740, 6.7410755, -8.1181736, 5.5820365, -15.4723082, 14.8592491
5: -8.4211397, 6.6531434, -6.9425273, 5.5226936, -13.9438324, 13.5956688
6: -7.5003910, 7.5006127, -6.2002292, 6.2560167, -13.7564068, 13.7008419
7: -8.4379387, 8.2188835, -6.9225917, 6.7518883, -15.1898270, 15.1414738
8: -12.2246685, 5.0867276, -10.0793161, 4.3257952, -16.5504646, 15.1660442
9: -7.3840685, 7.4459705, -6.0891914, 6.1457334, -13.5298014, 13.5351601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=57, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=231, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207265, upper bound: 12.3207536
time: 5.36 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207265, upper bound: 12.3207531
time: 4.83 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -6.7293825, 5.4974775, -8.4467106, 6.8592362, -13.5886192, 13.9441872
1: -5.9278474, 4.9581099, -7.5243611, 6.1799479, -12.1077957, 12.4824696
2: -7.5583863, 4.9472923, -9.5779924, 6.1184621, -13.6768465, 14.5252848
3: -8.1094646, 4.0548248, -10.2551126, 5.0135622, -13.1230268, 14.3099375
4: -8.0126896, 5.5044532, -10.0881882, 6.8794599, -14.8921471, 15.5926418
5: -6.8454456, 5.4445453, -8.5949430, 6.7904534, -13.6358976, 14.0394859
6: -6.1067233, 6.1687536, -7.6619344, 7.6517954, -13.7585144, 13.8306866
7: -6.8303452, 6.6627579, -8.6093159, 8.3841286, -15.2144728, 15.2720737
8: -9.9405537, 4.2551603, -12.4761162, 5.1991315, -15.1396847, 16.7312737
9: -6.0056019, 6.0593967, -7.5354133, 7.6007175, -13.6063194, 13.5948095

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=233, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207232, upper bound: 12.3207511
time: 5.38 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207232, upper bound: 12.3207511
time: 5.14 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.2702541, 6.7197776, -8.4467106, 6.8592362, -15.1294899, 15.1664858
1: -7.3678489, 6.0592847, -7.5243611, 6.1799479, -13.5477962, 13.5836449
2: -9.3772926, 5.9999976, -9.5779924, 6.1184621, -15.4957542, 15.5779896
3: -10.0460329, 4.9184432, -10.2551126, 5.0135622, -15.0595951, 15.1735544
4: -9.8897123, 6.7410755, -10.0881882, 6.8794599, -16.7691727, 16.8292637
5: -8.4211397, 6.6518397, -8.5949430, 6.7904534, -15.2115927, 15.2467823
6: -7.4996166, 7.5006127, -7.6619344, 7.6517954, -15.1514111, 15.1625462
7: -8.4379387, 8.2173624, -8.6093159, 8.3841286, -16.8220673, 16.8266792
8: -12.2246685, 5.0834618, -12.4761162, 5.1991315, -17.4237995, 17.5595741
9: -7.3840685, 7.4438233, -7.5354133, 7.6007175, -14.9847860, 14.9792356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=231, inp2_unstable=233, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205700, upper bound: 12.3206087
time: 4.96 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205681, upper bound: 12.3206090
time: 4.31 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.9895306, 5.6913362, -6.6955628, 5.4689059, -12.4584370, 12.3868990
1: -6.1919088, 5.1479669, -5.8986597, 4.9339314, -11.1258383, 11.0466261
2: -7.8762584, 5.1190448, -7.5189228, 4.9232278, -12.7994833, 12.6379662
3: -8.4608793, 4.2022519, -8.0691023, 4.0361290, -12.4970083, 12.2713547
4: -8.3618374, 5.7133770, -7.9749317, 5.4769645, -13.8388014, 13.6883078
5: -7.1066108, 5.6386104, -6.8105984, 5.4171162, -12.5237246, 12.4492083
6: -6.3234072, 6.3838353, -6.0734529, 6.1383696, -12.4617748, 12.4572887
7: -7.1255670, 6.9493318, -6.7975497, 6.6310749, -13.7566414, 13.7468815
8: -10.3340893, 4.3345008, -9.8902149, 4.2298002, -14.5638866, 14.2247152
9: -6.2449045, 6.2926207, -5.9757700, 6.0280290, -12.2729321, 12.2683907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207003, upper bound: 12.3206980
time: 3.82 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207010, upper bound: 12.3206979
time: 2.82 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.5698814, 6.9411693, -6.6955628, 5.4689059, -14.0387859, 13.6367321
1: -7.6660314, 6.2734976, -5.8986597, 4.9339314, -12.5999613, 12.1721554
2: -9.7376099, 6.1946559, -7.5189228, 4.9232278, -14.6608372, 13.7135792
3: -10.4423542, 5.0843115, -8.0691023, 4.0361290, -14.4784832, 13.1534138
4: -10.2801332, 6.9778852, -7.9749317, 5.4769645, -15.7570972, 14.9528151
5: -8.7151575, 6.8729825, -6.8105984, 5.4171162, -14.1322718, 13.6835804
6: -7.7491550, 7.7519722, -6.0734529, 6.1383696, -13.8875246, 13.8254242
7: -8.7687416, 8.5399113, -6.7975497, 6.6310749, -15.3998165, 15.3374577
8: -12.6675615, 5.1925240, -9.8902149, 4.2298002, -16.8973598, 15.0827389
9: -7.6537256, 7.7102900, -5.9757700, 6.0280290, -13.6817551, 13.6860600

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=232, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207003, upper bound: 12.3206980
time: 4.16 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207010, upper bound: 12.3206974
time: 3.49 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.9895306, 5.6913362, -8.2395277, 6.6895952, -13.6791258, 13.9308624
1: -6.1919088, 5.1479669, -7.3392797, 6.0339198, -12.2258272, 12.4872465
2: -7.8762584, 5.1190448, -9.3364954, 5.9747138, -13.8509712, 14.4555397
3: -8.4608793, 4.2022519, -10.0040522, 4.8984470, -13.3593235, 14.2063046
4: -8.3618374, 5.7133770, -9.8497505, 6.7124295, -15.0742664, 15.5631256
5: -7.1066108, 5.6386104, -8.3840866, 6.6240535, -13.7306643, 14.0226955
6: -6.3234072, 6.3838353, -7.4657502, 7.4683976, -13.7918034, 13.8495855
7: -7.1255670, 6.9493318, -8.4029026, 8.1848221, -15.3103886, 15.3522320
8: -10.3340893, 4.3345008, -12.1716795, 5.0607557, -15.3948441, 16.5061798
9: -6.2449045, 6.2926207, -7.3522134, 7.4131165, -13.6580181, 13.6448336

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=231, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205409, upper bound: 12.3205413
time: 3.07 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205413, upper bound: 12.3205414
time: 3.24 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.5698814, 6.9411693, -8.2395277, 6.6895952, -15.2594767, 15.1806965
1: -7.6660314, 6.2734976, -7.3392797, 6.0339198, -13.6999512, 13.6127758
2: -9.7376099, 6.1946559, -9.3364954, 5.9747138, -15.7123222, 15.5311499
3: -10.4423542, 5.0843115, -10.0040522, 4.8984470, -15.3408012, 15.0883636
4: -10.2801332, 6.9778852, -9.8497505, 6.7124295, -16.9925632, 16.8276348
5: -8.7151575, 6.8729825, -8.3840866, 6.6240535, -15.3392105, 15.2570686
6: -7.7491550, 7.7519722, -7.4657502, 7.4683976, -15.2175503, 15.2177200
7: -8.7687416, 8.5399113, -8.4029026, 8.1848221, -16.9535637, 16.9428120
8: -12.6675615, 5.1925240, -12.1716795, 5.0607557, -17.7283154, 17.3642044
9: -7.6537256, 7.7102900, -7.3522134, 7.4131165, -15.0668421, 15.0625038

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=232, inp2_unstable=231, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205409, upper bound: 12.3205418
time: 4.10 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205413, upper bound: 12.3205413
time: 3.71 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 9.15 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 9.15
Output dim: 8, lower bound: -12.3207265, upper bound: 12.3207531
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 9.15
Output dim: 8, lower bound: -12.3207265, upper bound: 12.3207535
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 9.15
Output dim: 8, lower bound: -12.3207265, upper bound: 12.3207536
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 9.15
Output dim: 8, lower bound: -12.3207265, upper bound: 12.3207531
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 9.15
Output dim: 8, lower bound: -12.3207232, upper bound: 12.3207511
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 9.15
Output dim: 8, lower bound: -12.3207232, upper bound: 12.3207511
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 9.15
Output dim: 8, lower bound: -12.3205700, upper bound: 12.3206087
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 9.15
Output dim: 8, lower bound: -12.3205681, upper bound: 12.3206090
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 9.15
Output dim: 8, lower bound: -12.3207003, upper bound: 12.3206980
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 9.15
Output dim: 8, lower bound: -12.3207010, upper bound: 12.3206979
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 9.15
Output dim: 8, lower bound: -12.3207003, upper bound: 12.3206980
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 9.15
Output dim: 8, lower bound: -12.3207010, upper bound: 12.3206974
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 9.15
Output dim: 8, lower bound: -12.3205409, upper bound: 12.3205413
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 9.15
Output dim: 8, lower bound: -12.3205413, upper bound: 12.3205414
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 9.15
Output dim: 8, lower bound: -12.3205409, upper bound: 12.3205418
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 9.15
Output dim: 8, lower bound: -12.3205413, upper bound: 12.3205413

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.7293825, 5.4974775, -6.7293825, 5.4974775, -12.2268600, 12.2268600
1: -5.9278474, 4.9581099, -5.9278474, 4.9581099, -10.8859558, 10.8859558
2: -7.5583863, 4.9472923, -7.5583863, 4.9472923, -12.5056782, 12.5056782
3: -8.1094646, 4.0548248, -8.1094646, 4.0548248, -12.1642895, 12.1642895
4: -8.0126896, 5.5044532, -8.0126896, 5.5044532, -13.5171413, 13.5171413
5: -6.8454456, 5.4445453, -6.8454456, 5.4445453, -12.2899904, 12.2899904
6: -6.1067233, 6.1687536, -6.1067233, 6.1687536, -12.2754745, 12.2754745
7: -6.8303452, 6.6627579, -6.8303452, 6.6627579, -13.4931030, 13.4931030
8: -9.9405537, 4.2551603, -9.9405537, 4.2551603, -14.1957130, 14.1957121
9: -6.0056019, 6.0593967, -6.0056019, 6.0593967, -12.0649986, 12.0649986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3202294, upper bound: 12.3201494
time: 5.18 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205371, upper bound: 12.3204886
time: 2.90 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206927, upper bound: 12.3207281
time: 15.74 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.7293825, 5.4974775, -6.9895306, 5.6913362, -12.4207191, 12.4870071
1: -5.9278474, 4.9581099, -6.1919088, 5.1479669, -11.0758142, 11.1500158
2: -7.5583863, 4.9472923, -7.8762584, 5.1190448, -12.6774292, 12.8235512
3: -8.1094646, 4.0548248, -8.4608793, 4.2022519, -12.3117161, 12.5157042
4: -8.0126896, 5.5044532, -8.3618374, 5.7133770, -13.7260637, 13.8662901
5: -6.8454456, 5.4445453, -7.1066108, 5.6386104, -12.4840555, 12.5511551
6: -6.1067233, 6.1687536, -6.3234072, 6.3838353, -12.4905567, 12.4921589
7: -6.8303452, 6.6627579, -7.1255670, 6.9493318, -13.7796764, 13.7883244
8: -9.9405537, 4.2551603, -10.3340893, 4.3345008, -14.2750549, 14.5892477
9: -6.0056019, 6.0593967, -6.2449045, 6.2926207, -12.2982225, 12.3042994

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205371, upper bound: 12.3204886
time: 3.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206927, upper bound: 12.3207284
time: 5.27 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.2756357, 6.7197776, -6.7293825, 5.4974775, -13.7731133, 13.4491577
1: -7.3706841, 6.0592847, -5.9278474, 4.9581099, -12.3287907, 11.9871311
2: -9.3782692, 5.9999976, -7.5583863, 4.9472923, -14.3255615, 13.5583830
3: -10.0468102, 4.9184432, -8.1094646, 4.0548248, -14.1016350, 13.0279064
4: -9.8902740, 6.7410755, -8.0126896, 5.5044532, -15.3947258, 14.7537632
5: -8.4211397, 6.6531434, -6.8454456, 5.4445453, -13.8656836, 13.4985886
6: -7.5003910, 7.5006127, -6.1067233, 6.1687536, -13.6691437, 13.6073341
7: -8.4379387, 8.2188835, -6.8303452, 6.6627579, -15.1006966, 15.0492268
8: -12.2246685, 5.0867276, -9.9405537, 4.2551603, -16.4798279, 15.0272808
9: -7.3840685, 7.4459705, -6.0056019, 6.0593967, -13.4434643, 13.4515724

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=231, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205854, upper bound: 12.3206086
time: 3.37 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205853, upper bound: 12.3206131
time: 3.14 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.2756357, 6.7197776, -6.9895306, 5.6913362, -13.9669724, 13.7093067
1: -7.3706841, 6.0592847, -6.1919088, 5.1479669, -12.5186491, 12.2511911
2: -9.3782692, 5.9999976, -7.8762584, 5.1190448, -14.4973125, 13.8762550
3: -10.0468102, 4.9184432, -8.4608793, 4.2022519, -14.2490616, 13.3793201
4: -9.8902740, 6.7410755, -8.3618374, 5.7133770, -15.6036482, 15.1029119
5: -8.4211397, 6.6531434, -7.1066108, 5.6386104, -14.0597467, 13.7597513
6: -7.5003910, 7.5006127, -6.3234072, 6.3838353, -13.8842258, 13.8240175
7: -8.4379387, 8.2188835, -7.1255670, 6.9493318, -15.3872690, 15.3444490
8: -12.2246685, 5.0867276, -10.3340893, 4.3345008, -16.5591698, 15.4208164
9: -7.3840685, 7.4459705, -6.2449045, 6.2926207, -13.6766891, 13.6908731

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=231, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205854, upper bound: 12.3206086
time: 3.67 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205853, upper bound: 12.3206125
time: 4.02 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6.7293825, 5.4974775, -8.3527031, 6.7804961, -13.5098782, 13.8501797
1: -5.9278474, 4.9581099, -7.4427881, 6.1141105, -12.0419579, 12.4008980
2: -7.5583863, 4.9472923, -9.4691372, 6.0522199, -13.6106043, 14.4164286
3: -8.1094646, 4.0548248, -10.1436157, 4.9613180, -13.0707817, 14.1984406
4: -8.0126896, 5.5044532, -9.9841671, 6.8027225, -14.8154078, 15.4886169
5: -6.8454456, 5.4445453, -8.4992323, 6.7130294, -13.5584755, 13.9437771
6: -6.1067233, 6.1687536, -7.5694661, 7.5665636, -13.6732836, 13.7382202
7: -6.8303452, 6.6627579, -8.5182629, 8.2965355, -15.1268797, 15.1810188
8: -9.9405537, 4.2551603, -12.3382273, 5.1280508, -15.0686045, 16.5933857
9: -6.0056019, 6.0593967, -7.4529228, 7.5151081, -13.5207100, 13.5123177

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=231, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205720, upper bound: 12.3206164
time: 4.55 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205698, upper bound: 12.3206157
time: 6.02 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6.7293825, 5.4974775, -8.6048441, 6.9687753, -13.6981583, 14.1023216
1: -5.9278474, 4.9581099, -7.6986752, 6.2983317, -12.2261782, 12.6567822
2: -7.5583863, 4.9472923, -9.7788553, 6.2183223, -13.7767067, 14.7261467
3: -8.1094646, 4.0548248, -10.4863005, 5.1037650, -13.2132301, 14.5411243
4: -8.0126896, 5.5044532, -10.3225985, 7.0058084, -15.0184956, 15.8270512
5: -6.8454456, 5.4445453, -8.7506447, 6.9001713, -13.7456160, 14.1951885
6: -6.1067233, 6.1687536, -7.7805314, 7.7822499, -13.8889713, 13.9492836
7: -6.8303452, 6.6627579, -8.8051319, 8.5751514, -15.4054966, 15.4678898
8: -9.9405537, 4.2551603, -12.7189760, 5.2114010, -15.1519547, 16.9741364
9: -6.0056019, 6.0593967, -7.6848869, 7.7416420, -13.7472439, 13.7442827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=232, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205174, upper bound: 12.3204662
time: 5.40 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206783, upper bound: 12.3207186
time: 3.43 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.9055090, 6.4269357, -6.5307431, 5.3273740, -13.2328815, 12.9576778
1: -7.0315676, 5.8005180, -5.7598782, 4.8238630, -11.8554287, 11.5603962
2: -8.9490738, 5.7501454, -7.3339195, 4.8094854, -13.7585573, 13.0840635
3: -9.5932074, 4.7137523, -7.8829160, 3.9413977, -13.5346050, 12.5966682
4: -9.4530334, 6.4487362, -7.7990632, 5.3482261, -14.8012600, 14.2477970
5: -8.0470133, 6.3627543, -6.6346936, 5.2755542, -13.3225651, 12.9974480
6: -7.1650515, 7.1807961, -5.9102049, 5.9731164, -13.1381664, 13.0910006
7: -8.0627937, 7.8537760, -6.6444540, 6.4782176, -14.5410099, 14.4982300
8: -11.6877422, 4.8716173, -9.6615982, 4.1045771, -15.7923193, 14.5332136
9: -7.0586753, 7.1171422, -5.8306189, 5.8863440, -12.9450188, 12.9477596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=57, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=231, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205638, upper bound: 12.3206008
time: 3.35 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205638, upper bound: 12.3206065
time: 5.09 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.9818082, 6.4874291, -7.7105088, 6.2758937, -14.2577009, 14.1979370
1: -7.1024251, 5.8545938, -6.8419647, 5.6589756, -12.7613993, 12.6965580
2: -9.0376244, 5.8017921, -8.7103539, 5.6195402, -14.6571636, 14.5121460
3: -9.6877823, 4.7565031, -9.3404226, 4.6019154, -14.2896967, 14.0969257
4: -9.5453701, 6.5081463, -9.2030525, 6.2869210, -15.8322906, 15.7111979
5: -8.1264687, 6.4221129, -7.8467627, 6.2111130, -14.3375816, 14.2688751
6: -7.2324882, 7.2463474, -6.9894147, 7.0163298, -14.2488165, 14.2357607
7: -8.1413965, 7.9304767, -7.8471293, 7.6470318, -15.7884274, 15.7776012
8: -11.7989225, 4.9121246, -11.4015112, 4.7855611, -16.5844803, 16.3136368
9: -7.1262326, 7.1844625, -6.8802962, 6.9447989, -14.0710316, 14.0647583

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=231, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 85

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205668, upper bound: 12.3206008
time: 3.36 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205668, upper bound: 12.3206004
time: 8.05 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.9895306, 5.6913362, -6.7292461, 5.4973006, -12.4868317, 12.4205818
1: -6.1919088, 5.1479669, -5.9277387, 4.9580154, -11.1499224, 11.0757055
2: -7.8762584, 5.1190448, -7.5581732, 4.9471788, -12.8234367, 12.6772175
3: -8.4608793, 4.2022519, -8.1093235, 4.0547276, -12.5156069, 12.3115749
4: -8.3618374, 5.7133770, -8.0125427, 5.5043306, -13.8661680, 13.7259178
5: -7.1066108, 5.6386104, -6.8453302, 5.4444189, -12.5510292, 12.4839392
6: -6.3234072, 6.3838353, -6.1066017, 6.1685882, -12.4919949, 12.4904366
7: -7.1255670, 6.9493318, -6.8302069, 6.6626287, -13.7881956, 13.7795372
8: -10.3340893, 4.3345008, -9.9402676, 4.2549224, -14.5890121, 14.2747688
9: -6.2449045, 6.2926207, -6.0054545, 6.0592327, -12.3041344, 12.2980747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203360, upper bound: 12.3204809
time: 4.25 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2

### Relational analysis result of IS_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206682, upper bound: 12.3206673
time: 5.56 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.9895306, 5.6913362, -6.9888253, 5.6913362, -12.6808662, 12.6801605
1: -6.1919088, 5.1479669, -6.1915340, 5.1479669, -11.3398752, 11.3395004
2: -7.8762584, 5.1190448, -7.8761301, 5.1190448, -12.9953032, 12.9951744
3: -8.4608793, 4.2022519, -8.4607792, 4.2022519, -12.6631317, 12.6630306
4: -8.3618374, 5.7133770, -8.3617630, 5.7133770, -14.0752134, 14.0751400
5: -7.1066108, 5.6386104, -7.1066108, 5.6384387, -12.7450495, 12.7452202
6: -6.3234072, 6.3838353, -6.3233042, 6.3838353, -12.7072411, 12.7071400
7: -7.1255670, 6.9493318, -7.1255670, 6.9491315, -14.0746984, 14.0748978
8: -10.3340893, 4.3345008, -10.3340893, 4.3340945, -14.6681814, 14.6685896
9: -6.2449045, 6.2926207, -6.2449045, 6.2923393, -12.5372419, 12.5375242

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205717, upper bound: 12.3205614
time: 8.24 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205762, upper bound: 12.3205762
time: 3.27 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.5698814, 6.9411693, -6.7292461, 5.4973006, -14.0671825, 13.6704149
1: -7.6660314, 6.2734976, -5.9277387, 4.9580154, -12.6240463, 12.2012367
2: -9.7376099, 6.1946559, -7.5581732, 4.9471788, -14.6847887, 13.7528286
3: -10.4423542, 5.0843115, -8.1093235, 4.0547276, -14.4970818, 13.1936350
4: -10.2801332, 6.9778852, -8.0125427, 5.5043306, -15.7844629, 14.9904280
5: -8.7151575, 6.8729825, -6.8453302, 5.4444189, -14.1595764, 13.7183132
6: -7.7491550, 7.7519722, -6.1066017, 6.1685882, -13.9177427, 13.8585720
7: -8.7687416, 8.5399113, -6.8302069, 6.6626287, -15.4313698, 15.3701153
8: -12.6675615, 5.1925240, -9.9402676, 4.2549224, -16.9224834, 15.1327915
9: -7.6537256, 7.7102900, -6.0054545, 6.0592327, -13.7129574, 13.7157440

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=232, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203129, upper bound: 12.3204550
time: 2.76 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2

### Relational analysis result of IS_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206598, upper bound: 12.3206529
time: 4.29 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.5698814, 6.9411693, -6.9888253, 5.6913362, -14.2612171, 13.9299936
1: -7.6660314, 6.2734976, -6.1915340, 5.1479669, -12.8139982, 12.4650297
2: -9.7376099, 6.1946559, -7.8761301, 5.1190448, -14.8566551, 14.0707855
3: -10.4423542, 5.0843115, -8.4607792, 4.2022519, -14.6446056, 13.5450907
4: -10.2801332, 6.9778852, -8.3617630, 5.7133770, -15.9935083, 15.3396482
5: -8.7151575, 6.8729825, -7.1066108, 5.6384387, -14.3535957, 13.9795933
6: -7.7491550, 7.7519722, -6.3233042, 6.3838353, -14.1329899, 14.0752764
7: -8.7687416, 8.5399113, -7.1255670, 6.9491315, -15.7178726, 15.6654758
8: -12.6675615, 5.1925240, -10.3340893, 4.3340945, -17.0016518, 15.5266132
9: -7.6537256, 7.7102900, -6.2449045, 6.2923393, -13.9460649, 13.9551935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=232, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205548, upper bound: 12.3205441
time: 3.31 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205555, upper bound: 12.3205435
time: 3.45 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6.6311421, 5.4041176, -6.3711700, 5.1947541, -11.8258944, 11.7752876
1: -5.8615170, 4.8934021, -5.6187835, 4.7107477, -10.5722637, 10.5121851
2: -7.4551568, 4.8736391, -7.1474133, 4.6976337, -12.1527901, 12.0210514
3: -8.0158005, 4.0009851, -7.6897240, 3.8522098, -11.8680096, 11.6907091
4: -7.9324799, 5.4258709, -7.6171980, 5.2181773, -13.1506577, 13.0430689
5: -6.7383990, 5.3545151, -6.4703860, 5.1456547, -11.8840532, 11.8248997
6: -5.9947801, 6.0686231, -5.7556219, 5.8289323, -11.8237123, 11.8242426
7: -6.7572336, 6.5916080, -6.4867086, 6.3254375, -13.0826712, 13.0783157
8: -9.8059158, 4.1329851, -9.4268131, 3.9926686, -13.7985840, 13.5597982
9: -5.9251409, 5.9710159, -5.6891379, 5.7409520, -11.6660929, 11.6601524

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205376, upper bound: 12.3205456
time: 3.71 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205376, upper bound: 12.3205526
time: 3.49 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6.7018452, 5.4595895, -7.5252938, 6.1240201, -12.8258648, 12.9848824
1: -5.9277730, 4.9438858, -6.6774817, 5.5286865, -11.4564590, 11.6213665
2: -7.5375347, 4.9217110, -8.4952984, 5.4908624, -13.0283966, 13.4170084
3: -8.1041546, 4.0405626, -9.1167183, 4.4991651, -12.6033173, 13.1572809
4: -8.0195007, 5.4810748, -8.9916134, 6.1375375, -14.1570368, 14.4726887
5: -6.8122001, 5.4090919, -7.6577978, 6.0617657, -12.8739643, 13.0668888
6: -6.0566869, 6.1288586, -6.8124943, 6.8515630, -12.9082489, 12.9413528
7: -6.8307910, 6.6632018, -7.6638837, 7.4699054, -14.3006954, 14.3270855
8: -9.9102755, 4.1667218, -11.1308498, 4.6597052, -14.5699806, 15.2975712
9: -5.9880939, 6.0339165, -6.7170362, 6.7774935, -12.7655869, 12.7509518

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=58, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205406, upper bound: 12.3205455
time: 2.84 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205406, upper bound: 12.3205554
time: 3.52 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.2041111, 6.6488729, -6.3711700, 5.1947541, -13.3988647, 13.0200424
1: -7.3291855, 6.0148802, -5.6187835, 4.7107477, -12.0399332, 11.6336622
2: -9.3095264, 5.9450769, -7.1474133, 4.6976337, -14.0071583, 13.0924883
3: -9.9891529, 4.8798938, -7.6897240, 3.8522098, -13.8413620, 12.5696173
4: -9.8434048, 6.6857133, -7.6171980, 5.2181773, -15.0615826, 14.3029099
5: -8.3414049, 6.5839329, -6.4703860, 5.1456547, -13.4870586, 13.0543184
6: -7.4146471, 7.4319081, -5.7556219, 5.8289323, -13.2435770, 13.1875277
7: -8.3938370, 8.1761303, -6.4867086, 6.3254375, -14.7192745, 14.6628370
8: -12.1312695, 4.9800224, -9.4268131, 3.9926686, -16.1239376, 14.4068346
9: -7.3285289, 7.3833933, -5.6891379, 5.7409520, -13.0694780, 13.0725288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=231, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205356, upper bound: 12.3205360
time: 2.67 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205356, upper bound: 12.3205398
time: 4.34 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.2755203, 6.7052355, -7.5252938, 6.1240201, -14.3995399, 14.2305298
1: -7.3954949, 6.0653319, -6.6774817, 5.5286865, -12.9241810, 12.7428122
2: -9.3923550, 5.9930797, -8.4952984, 5.4908624, -14.8832169, 14.4883785
3: -10.0776501, 4.9197698, -9.1167183, 4.4991651, -14.5768147, 14.0364876
4: -9.9297180, 6.7410355, -8.9916134, 6.1375375, -16.0672550, 15.7326488
5: -8.4154749, 6.6391797, -7.6577978, 6.0617657, -14.4772396, 14.2969770
6: -7.4773717, 7.4933162, -6.8124943, 6.8515630, -14.3289337, 14.3058100
7: -8.4673424, 8.2479734, -7.6638837, 7.4699054, -15.9372454, 15.9118576
8: -12.2346125, 5.0170770, -11.1308498, 4.6597052, -16.8943176, 16.1479244
9: -7.3914361, 7.4462867, -6.7170362, 6.7774935, -14.1689301, 14.1633224

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=231, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205394, upper bound: 12.3205361
time: 3.32 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205394, upper bound: 12.3205417
time: 3.58 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 8.24 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 8.24
Output dim: 8, lower bound: -12.3205371, upper bound: 12.3204886
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.24
Output dim: 8, lower bound: -12.3206927, upper bound: 12.3207281
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.24
Output dim: 8, lower bound: -12.3205371, upper bound: 12.3204886
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.24
Output dim: 8, lower bound: -12.3206927, upper bound: 12.3207284
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 8.24
Output dim: 8, lower bound: -12.3205854, upper bound: 12.3206086
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.24
Output dim: 8, lower bound: -12.3205853, upper bound: 12.3206131
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.24
Output dim: 8, lower bound: -12.3205854, upper bound: 12.3206086
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.24
Output dim: 8, lower bound: -12.3205853, upper bound: 12.3206125
IS_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 8.24
Output dim: 8, lower bound: -12.3205720, upper bound: 12.3206164
IS_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 8.24
Output dim: 8, lower bound: -12.3205698, upper bound: 12.3206157
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.24
Output dim: 8, lower bound: -12.3205174, upper bound: 12.3204662
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.24
Output dim: 8, lower bound: -12.3206783, upper bound: 12.3207186
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 8.24
Output dim: 8, lower bound: -12.3205638, upper bound: 12.3206008
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.24
Output dim: 8, lower bound: -12.3205638, upper bound: 12.3206065
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.24
Output dim: 8, lower bound: -12.3205668, upper bound: 12.3206008
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.24
Output dim: 8, lower bound: -12.3205668, upper bound: 12.3206004
IS_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 8.24
Output dim: 8, lower bound: -12.3203360, upper bound: 12.3204809
IS_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 8.24
Output dim: 8, lower bound: -12.3206682, upper bound: 12.3206673
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.24
Output dim: 8, lower bound: -12.3205717, upper bound: 12.3205614
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.24
Output dim: 8, lower bound: -12.3205762, upper bound: 12.3205762
IS_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 8.24
Output dim: 8, lower bound: -12.3203129, upper bound: 12.3204550
IS_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 8.24
Output dim: 8, lower bound: -12.3206598, upper bound: 12.3206529
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.24
Output dim: 8, lower bound: -12.3205548, upper bound: 12.3205441
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.24
Output dim: 8, lower bound: -12.3205555, upper bound: 12.3205435
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 8.24
Output dim: 8, lower bound: -12.3205376, upper bound: 12.3205456
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.24
Output dim: 8, lower bound: -12.3205376, upper bound: 12.3205526
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.24
Output dim: 8, lower bound: -12.3205406, upper bound: 12.3205455
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.24
Output dim: 8, lower bound: -12.3205406, upper bound: 12.3205554
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 8.24
Output dim: 8, lower bound: -12.3205356, upper bound: 12.3205360
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.24
Output dim: 8, lower bound: -12.3205356, upper bound: 12.3205398
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.24
Output dim: 8, lower bound: -12.3205394, upper bound: 12.3205361
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.24
Output dim: 8, lower bound: -12.3205394, upper bound: 12.3205417

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.4719090, 3.6893673, -5.9847932, 4.9014654, -9.3733749, 9.6741600
1: -3.8317254, 3.3533330, -5.2398567, 4.4296875, -8.2614117, 8.5931892
2: -4.8953552, 3.4018815, -6.6828198, 4.4396553, -9.3350105, 10.0847015
3: -5.2999401, 2.7835197, -7.1864176, 3.6370516, -8.9369907, 9.9699364
4: -5.2807770, 3.7005892, -7.1177878, 4.9098225, -10.1905994, 10.8183765
5: -4.5096202, 3.7862389, -6.0801873, 4.8972936, -9.4069138, 9.8664265
6: -4.0742149, 4.2313728, -5.4367533, 5.5262775, -9.6004925, 9.6681261
7: -4.4900002, 4.4078498, -6.0630121, 5.9206610, -10.4106617, 10.4708595
8: -6.6041875, 3.0292170, -8.8436079, 3.8467312, -10.4509172, 11.8728247
9: -3.9896269, 4.0231175, -5.3422122, 5.3907056, -9.3803320, 9.3653297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=227, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 85

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205841, upper bound: 12.3205098
time: 3.51 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206054, upper bound: 12.3205216
time: 4.43 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6.2964301, 5.1500850, -6.7293825, 5.4974775, -11.7939072, 11.8794670
1: -5.5275245, 4.6504521, -5.9278474, 4.9581099, -10.4856319, 10.5782986
2: -7.0477347, 4.6515450, -7.5583863, 4.9472923, -11.9950275, 12.2099304
3: -7.5721588, 3.8114462, -8.1094646, 4.0548248, -11.6269836, 11.9209099
4: -7.4918504, 5.1576352, -8.0126896, 5.5044532, -12.9963026, 13.1703224
5: -6.4013290, 5.1249952, -6.8454456, 5.4445453, -11.8458719, 11.9704390
6: -5.7163124, 5.7939129, -6.1067233, 6.1687536, -11.8850660, 11.9006348
7: -6.3833666, 6.2306824, -6.8303452, 6.6627579, -13.0461245, 13.0610275
8: -9.3006182, 4.0157890, -9.9405537, 4.2551603, -13.5557747, 13.9563427
9: -5.6190548, 5.6695924, -6.0056019, 6.0593967, -11.6784496, 11.6751938

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206969, upper bound: 12.3207714
time: 3.92 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206969, upper bound: 12.3208946
time: 3.83 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.4719090, 3.6893673, -6.2469788, 5.0972643, -9.5691719, 9.9363461
1: -3.8317254, 3.3533330, -5.5060306, 4.6210604, -8.4527855, 8.8593616
2: -4.8953552, 3.4018815, -7.0034804, 4.6129932, -9.5083485, 10.4053593
3: -5.2999401, 2.7835197, -7.5404191, 3.7859464, -9.0858860, 10.3239365
4: -5.2807770, 3.7005892, -7.4696283, 5.1208153, -10.4015923, 11.1702175
5: -4.5096202, 3.7862389, -6.3437443, 5.0691795, -9.5788002, 10.1299820
6: -4.0742149, 4.2313728, -5.6550293, 5.7405562, -9.8147707, 9.8864021
7: -4.4900002, 4.4078498, -6.3609734, 6.2091994, -10.6991997, 10.7688236
8: -6.6041875, 3.0292170, -9.2407331, 3.9231200, -10.5273066, 12.2699490
9: -3.9896269, 4.0231175, -5.5835481, 5.6259208, -9.6155472, 9.6066656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=58, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=227, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 169

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3202834, upper bound: 12.3201809
time: 6.24 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203176, upper bound: 12.3201944
time: 4.23 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6.2964301, 5.1500850, -6.9895306, 5.6913362, -11.9877663, 12.1396160
1: -5.5275245, 4.6504521, -6.1919088, 5.1479669, -10.6754913, 10.8423595
2: -7.0477347, 4.6515450, -7.8762584, 5.1190448, -12.1667795, 12.5278034
3: -7.5721588, 3.8114462, -8.4608793, 4.2022519, -11.7744102, 12.2723236
4: -7.4918504, 5.1576352, -8.3618374, 5.7133770, -13.2052259, 13.5194721
5: -6.4013290, 5.1249952, -7.1066108, 5.6386104, -12.0399389, 12.2316036
6: -5.7163124, 5.7939129, -6.3234072, 6.3838353, -12.1001472, 12.1173191
7: -6.3833666, 6.2306824, -7.1255670, 6.9493318, -13.3326960, 13.3562489
8: -9.3006182, 4.0157890, -10.3340893, 4.3345008, -13.6351185, 14.3498783
9: -5.6190548, 5.6695924, -6.2449045, 6.2926207, -11.9116755, 11.9144964

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203613, upper bound: 12.3205384
time: 5.08 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203613, upper bound: 12.3207284
time: 4.20 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.4061422, 5.2237868, -6.3705287, 5.2092619, -11.6154041, 11.5943155
1: -5.6494083, 4.7356091, -5.5970054, 4.7029080, -10.3523159, 10.3326130
2: -7.1879139, 4.7221370, -7.1363387, 4.7011619, -11.8890762, 11.8584757
3: -7.7317543, 3.8716068, -7.6634912, 3.8529015, -11.5846558, 11.5350971
4: -7.6569457, 5.2461390, -7.5824776, 5.2162256, -12.8731709, 12.8286171
5: -6.5068545, 5.1737971, -6.4760828, 5.1746826, -11.6815376, 11.6498795
6: -5.7895226, 5.8601246, -5.7772946, 5.8527145, -11.6422367, 11.6374187
7: -6.5209870, 6.3589001, -6.4611411, 6.3045030, -12.8254900, 12.8200417
8: -9.4777937, 4.0166345, -9.4108839, 4.0556240, -13.5334177, 13.4275188
9: -5.7200379, 5.7727098, -5.6850262, 5.7371645, -11.4571991, 11.4577332

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207801, upper bound: 12.3207731
time: 4.82 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207801, upper bound: 12.3207737
time: 3.79 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.5656195, 6.1572371, -6.4462266, 5.2688723, -12.8344908, 12.6034641
1: -6.7126088, 5.5568876, -5.6672769, 4.7566080, -11.4692173, 11.2241650
2: -8.5414305, 5.5188890, -7.2238541, 4.7526150, -13.2940454, 12.7427425
3: -9.1644773, 4.5214167, -7.7571239, 3.8953388, -13.0598164, 12.2785406
4: -9.0367575, 6.1693931, -7.6744113, 5.2750931, -14.3118505, 13.8438015
5: -7.6994491, 6.0942087, -6.5552416, 5.2289586, -12.9284077, 12.6494484
6: -6.8512702, 6.8873243, -5.8439870, 5.9174738, -12.7687416, 12.7313118
7: -7.7029009, 7.5078802, -6.5389237, 6.3804150, -14.0833158, 14.0468044
8: -11.1889200, 4.6879587, -9.5219364, 4.0926042, -15.2815189, 14.2098932
9: -6.7522712, 6.8136611, -5.7521224, 5.8041139, -12.5563850, 12.5657835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 85

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207807, upper bound: 12.3207780
time: 3.42 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207807, upper bound: 12.3207798
time: 4.76 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.4061422, 5.2237868, -6.6311421, 5.4041176, -11.8102598, 11.8549290
1: -5.6494083, 4.7356091, -5.8615170, 4.8934021, -10.5428104, 10.5971251
2: -7.1879139, 4.7221370, -7.4551568, 4.8736391, -12.0615530, 12.1772938
3: -7.7317543, 3.8716068, -8.0158005, 4.0009851, -11.7327394, 11.8874054
4: -7.6569457, 5.2461390, -7.9324799, 5.4258709, -13.0828161, 13.1786194
5: -6.5068545, 5.1737971, -6.7383990, 5.3545151, -11.8613682, 11.9121962
6: -5.7895226, 5.8601246, -5.9947801, 6.0686231, -11.8581448, 11.8549042
7: -6.5209870, 6.3589001, -6.7572336, 6.5916080, -13.1125946, 13.1161337
8: -9.4777937, 4.0166345, -9.8059158, 4.1329851, -13.6107788, 13.8225498
9: -5.7200379, 5.7727098, -5.9251409, 5.9710159, -11.6910515, 11.6978493

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205759, upper bound: 12.3206049
time: 5.30 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205759, upper bound: 12.3206050
time: 5.29 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.5656195, 6.1572371, -6.7018452, 5.4595895, -13.0252094, 12.8590822
1: -6.7126088, 5.5568876, -5.9277730, 4.9438858, -11.6564922, 11.4846611
2: -8.5414305, 5.5188890, -7.5375347, 4.9217110, -13.4631405, 13.0564232
3: -9.1644773, 4.5214167, -8.1041546, 4.0405626, -13.2050400, 12.6255684
4: -9.0367575, 6.1693931, -8.0195007, 5.4810748, -14.5178318, 14.1888924
5: -7.6994491, 6.0942087, -6.8122001, 5.4090919, -13.1085377, 12.9064074
6: -6.8512702, 6.8873243, -6.0566869, 6.1288586, -12.9801292, 12.9440117
7: -7.7029009, 7.5078802, -6.8307910, 6.6632018, -14.3661022, 14.3386707
8: -11.1889200, 4.6879587, -9.9102755, 4.1667218, -15.3556423, 14.5982323
9: -6.7522712, 6.8136611, -5.9880939, 6.0339165, -12.7861853, 12.8017540

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=58, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205754, upper bound: 12.3206091
time: 7.74 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205754, upper bound: 12.3206132
time: 5.00 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -6.3705287, 5.2092619, -6.4362431, 5.2476768, -11.6182060, 11.6455050
1: -5.5970054, 4.7029080, -5.6775985, 4.7571487, -10.3541527, 10.3805065
2: -7.1363387, 4.7011619, -7.2236609, 4.7425995, -11.8789377, 11.9248219
3: -7.6634912, 3.8529015, -7.7697926, 3.8884609, -11.5519505, 11.6226940
4: -7.5824776, 5.2162256, -7.6937318, 5.2703624, -12.8528395, 12.9099579
5: -6.4760828, 5.1746826, -6.5375752, 5.1973267, -11.6734095, 11.7122564
6: -5.7772946, 5.8527145, -5.8166275, 5.8861771, -11.6634712, 11.6693420
7: -6.4611411, 6.3045030, -6.5525165, 6.3894095, -12.8505478, 12.8570194
8: -9.4108839, 4.0556240, -9.5224218, 4.0322313, -13.4431152, 13.5780458
9: -5.6850262, 5.7371645, -5.7470279, 5.7998395, -11.4848652, 11.4841919

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=29, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207737, upper bound: 12.3207806
time: 4.40 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207737, upper bound: 12.3207856
time: 3.20 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -6.4462266, 5.2688723, -7.6174564, 6.1980815, -12.6443081, 12.8863268
1: -5.6672769, 4.7566080, -6.7610564, 5.5937462, -11.2610207, 11.5176640
2: -7.2238541, 4.7526150, -8.6026030, 5.5539751, -12.7778292, 13.3552179
3: -7.7571239, 3.8953388, -9.2295790, 4.5502605, -12.3073845, 13.1249180
4: -7.6744113, 5.2750931, -9.0998135, 6.2108178, -13.8852291, 14.3749065
5: -6.5552416, 5.2289586, -7.7520099, 6.1345057, -12.6897449, 12.9809685
6: -5.8439870, 5.9174738, -6.8977504, 6.9318194, -12.7758064, 12.8152218
7: -6.5389237, 6.3804150, -7.7569189, 7.5601463, -14.0990696, 14.1373329
8: -9.5219364, 4.0926042, -11.2652454, 4.7154856, -14.2374220, 15.3578491
9: -5.7521224, 5.8041139, -6.7985163, 6.8601508, -12.6122732, 12.6026306

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=29, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 85

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207774, upper bound: 12.3207806
time: 4.34 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207774, upper bound: 12.3207876
time: 3.69 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.4719090, 3.6893673, -7.8376603, 6.3576269, -10.8295364, 11.5270271
1: -3.8317254, 3.3533330, -6.9912243, 5.7570386, -9.5887642, 10.3445568
2: -4.8953552, 3.4018815, -8.8818455, 5.6975007, -10.5928555, 12.2837248
3: -5.2999401, 2.7835197, -9.5386162, 4.6759253, -9.9758654, 12.3221359
4: -5.2807770, 3.7005892, -9.4049616, 6.3968949, -11.6776714, 13.1055508
5: -4.5096202, 3.7862389, -7.9671350, 6.3000007, -10.8096209, 11.7533741
6: -4.0742149, 4.2313728, -7.0917058, 7.1158094, -11.1900244, 11.3230782
7: -4.4900002, 4.4078498, -8.0182133, 7.8137722, -12.3037720, 12.4260635
8: -6.6041875, 3.0292170, -11.5959358, 4.7735186, -11.3777065, 14.6251526
9: -3.9896269, 4.0231175, -7.0047970, 7.0560646, -11.0456905, 11.0279140

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=58, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3202243, upper bound: 12.3200992
time: 4.01 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3202408, upper bound: 12.3200983
time: 5.68 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6.2964301, 5.1500850, -8.6048441, 6.9687753, -13.2652054, 13.7549286
1: -5.5275245, 4.6504521, -7.6986752, 6.2983317, -11.8258562, 12.3491249
2: -7.0477347, 4.6515450, -9.7788553, 6.2183223, -13.2660570, 14.4303999
3: -7.5721588, 3.8114462, -10.4863005, 5.1037650, -12.6759243, 14.2977448
4: -7.4918504, 5.1576352, -10.3225985, 7.0058084, -14.4976587, 15.4802341
5: -6.4013290, 5.1249952, -8.7506447, 6.9001713, -13.3014994, 13.8756390
6: -5.7163124, 5.7939129, -7.7805314, 7.7822499, -13.4985619, 13.5744438
7: -6.3833666, 6.2306824, -8.8051319, 8.5751514, -14.9585171, 15.0358133
8: -9.3006182, 4.0157890, -12.7189760, 5.2114010, -14.5120192, 16.7347641
9: -5.6190548, 5.6695924, -7.6848869, 7.7416420, -13.3606968, 13.3544788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=232, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3202799, upper bound: 12.3205152
time: 3.70 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3202799, upper bound: 12.3207193
time: 4.79 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.4061422, 5.2237868, -6.5307431, 5.3273740, -11.7335167, 11.7545300
1: -5.6494083, 4.7356091, -5.7598782, 4.8238630, -10.4732714, 10.4954872
2: -7.1879139, 4.7221370, -7.3339195, 4.8094854, -11.9973984, 12.0560570
3: -7.7317543, 3.8716068, -7.8829160, 3.9413977, -11.6731520, 11.7545204
4: -7.6569457, 5.2461390, -7.7990632, 5.3482261, -13.0051718, 13.0452023
5: -6.5068545, 5.1737971, -6.6346936, 5.2755542, -11.7824078, 11.8084908
6: -5.7895226, 5.8601246, -5.9102049, 5.9731164, -11.7626390, 11.7703295
7: -6.5209870, 6.3589001, -6.6444540, 6.4782176, -12.9992046, 13.0033541
8: -9.4777937, 4.0166345, -9.6615982, 4.1045771, -13.5823708, 13.6782322
9: -5.7200379, 5.7727098, -5.8306189, 5.8863440, -11.6063805, 11.6033268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=57, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205700, upper bound: 12.3206056
time: 4.77 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205700, upper bound: 12.3206061
time: 4.12 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.5629311, 6.1572371, -6.5307431, 5.3273740, -12.8903046, 12.6879807
1: -6.7111931, 5.5568876, -5.7598782, 4.8238630, -11.5350552, 11.3167658
2: -8.5409431, 5.5188890, -7.3339195, 4.8094854, -13.3504276, 12.8528070
3: -9.1640987, 4.5214167, -7.8829160, 3.9413977, -13.1054964, 12.4043331
4: -9.0364780, 6.1693931, -7.7990632, 5.3482261, -14.3847036, 13.9684544
5: -7.6994491, 6.0935555, -6.6346936, 5.2755542, -12.9750023, 12.7282486
6: -6.8508821, 6.8873243, -5.9102049, 5.9731164, -12.8239956, 12.7975292
7: -7.7029009, 7.5071216, -6.6444540, 6.4782176, -14.1811180, 14.1515751
8: -11.1889200, 4.6863728, -9.6615982, 4.1045771, -15.2934971, 14.3479710
9: -6.7522712, 6.8125873, -5.8306189, 5.8863440, -12.6386147, 12.6432037

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=57, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205700, upper bound: 12.3206080
time: 2.50 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205700, upper bound: 12.3206087
time: 2.70 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.4061422, 5.2237868, -7.7105088, 6.2758937, -12.6820354, 12.9342957
1: -5.6494083, 4.7356091, -6.8419647, 5.6589756, -11.3083830, 11.5775738
2: -7.1879139, 4.7221370, -8.7103539, 5.6195402, -12.8074541, 13.4324894
3: -7.7317543, 3.8716068, -9.3404226, 4.6019154, -12.3336687, 13.2120285
4: -7.6569457, 5.2461390, -9.2030525, 6.2869210, -13.9438667, 14.4491920
5: -6.5068545, 5.1737971, -7.8467627, 6.2111130, -12.7179680, 13.0205593
6: -5.7895226, 5.8601246, -6.9894147, 7.0163298, -12.8058519, 12.8495388
7: -6.5209870, 6.3589001, -7.8471293, 7.6470318, -14.1680183, 14.2060299
8: -9.4777937, 4.0166345, -11.4015112, 4.7855611, -14.2633553, 15.4181433
9: -5.7200379, 5.7727098, -6.8802962, 6.9447989, -12.6648350, 12.6530037

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205642, upper bound: 12.3206008
time: 5.19 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205642, upper bound: 12.3205999
time: 3.79 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.5629311, 6.1572371, -7.7105088, 6.2758937, -13.8388224, 13.8677464
1: -6.7111931, 5.5568876, -6.8419647, 5.6589756, -12.3701677, 12.3988523
2: -8.5409431, 5.5188890, -8.7103539, 5.6195402, -14.1604824, 14.2292423
3: -9.1640987, 4.5214167, -9.3404226, 4.6019154, -13.7660112, 13.8618383
4: -9.0364780, 6.1693931, -9.2030525, 6.2869210, -15.3233967, 15.3724432
5: -7.6994491, 6.0935555, -7.8467627, 6.2111130, -13.9105606, 13.9403181
6: -6.8508821, 6.8873243, -6.9894147, 7.0163298, -13.8672113, 13.8767376
7: -7.7029009, 7.5071216, -7.8471293, 7.6470318, -15.3499327, 15.3542500
8: -11.1889200, 4.6863728, -11.4015112, 4.7855611, -15.9744797, 16.0878830
9: -6.7522712, 6.8125873, -6.8802962, 6.9447989, -13.6970692, 13.6928825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205642, upper bound: 12.3206090
time: 14.48 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205642, upper bound: 12.3206090
time: 4.20 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -6.2469788, 5.0972643, -4.4719090, 3.6893673, -9.9363461, 9.5691729
1: -5.5060306, 4.6210604, -3.8317254, 3.3533330, -8.8593607, 8.4527855
2: -7.0034804, 4.6129932, -4.8953552, 3.4018815, -10.4053602, 9.5083485
3: -7.5404191, 3.7859464, -5.2999401, 2.7835197, -10.3239374, 9.0858860
4: -7.4696283, 5.1208153, -5.2807770, 3.7005892, -11.1702175, 10.4015923
5: -6.3437443, 5.0691795, -4.5096202, 3.7862389, -10.1299820, 9.5788002
6: -5.6550293, 5.7405562, -4.0742149, 4.2313728, -9.8864021, 9.8147707
7: -6.3609734, 6.2091994, -4.4900002, 4.4078498, -10.7688236, 10.6991997
8: -9.2407331, 3.9231200, -6.6041875, 3.0292170, -12.2699480, 10.5273066
9: -5.5835481, 5.6259208, -3.9896269, 4.0231175, -9.6066647, 9.6155462

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=29, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=58, inp2_unstable=54, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=227, inp2_unstable=223, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 169

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A1_B1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3201809, upper bound: 12.3202833
time: 3.78 seconds

## Relational analysis of IS_A2_B1_A1_B1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3201944, upper bound: 12.3203175
time: 4.11 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -6.9895306, 5.6913362, -6.2964301, 5.1500850, -12.1396151, 11.9877644
1: -6.1919088, 5.1479669, -5.5275245, 4.6504521, -10.8423595, 10.6754913
2: -7.8762584, 5.1190448, -7.0477347, 4.6515450, -12.5278034, 12.1667795
3: -8.4608793, 4.2022519, -7.5721588, 3.8114462, -12.2723236, 11.7744102
4: -8.3618374, 5.7133770, -7.4918504, 5.1576352, -13.5194721, 13.2052250
5: -7.1066108, 5.6386104, -6.4013290, 5.1249952, -12.2316046, 12.0399389
6: -6.3234072, 6.3838353, -5.7163124, 5.7939129, -12.1173162, 12.1001472
7: -7.1255670, 6.9493318, -6.3833666, 6.2306824, -13.3562489, 13.3326969
8: -10.3340893, 4.3345008, -9.3006182, 4.0157890, -14.3498783, 13.6351175
9: -6.2449045, 6.2926207, -5.6190548, 5.6695924, -11.9144945, 11.9116755

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=29, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1_B1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205385, upper bound: 12.3203612
time: 4.80 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205385, upper bound: 12.3206922
time: 3.07 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5.1597514, 4.2243500, -6.6311421, 5.4041176, -10.5638685, 10.8554907
1: -4.5061083, 3.8457608, -5.8615170, 4.8934021, -9.3995104, 9.7072773
2: -5.7244811, 3.8630471, -7.4551568, 4.8736391, -10.5981188, 11.3182030
3: -6.1857219, 3.1738603, -8.0158005, 4.0009851, -10.1867065, 11.1896601
4: -6.1667233, 4.2442379, -7.9324799, 5.4258709, -11.5925941, 12.1767178
5: -5.2219253, 4.2483940, -6.7383990, 5.3545151, -10.5764399, 10.9867935
6: -4.6424723, 4.7664061, -5.9947801, 6.0686231, -10.7110939, 10.7611837
7: -5.2428083, 5.1236596, -6.7572336, 6.5916080, -11.8344164, 11.8808928
8: -7.6338062, 3.3060670, -9.8059158, 4.1329851, -11.7667894, 13.1119823
9: -4.6105165, 4.6485400, -5.9251409, 5.9710159, -10.5815315, 10.5736809

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=226, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205605, upper bound: 12.3205603
time: 4.28 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205605, upper bound: 12.3205602
time: 2.40 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6.2761860, 5.1234202, -6.7018452, 5.4595895, -11.7357750, 11.8252649
1: -5.5314741, 4.6410146, -5.9277730, 4.9438858, -10.4753590, 10.5687866
2: -7.0332956, 4.6332264, -7.5375347, 4.9217110, -11.9550066, 12.1707611
3: -7.5735126, 3.8007462, -8.1041546, 4.0405626, -11.6140747, 11.9048996
4: -7.5034671, 5.1366544, -8.0195007, 5.4810748, -12.9845409, 13.1561546
5: -6.3759503, 5.0952325, -6.8122001, 5.4090919, -11.7850418, 11.9074326
6: -5.6668215, 5.7605472, -6.0566869, 6.1288586, -11.7956800, 11.8172340
7: -6.3867984, 6.2342095, -6.8307910, 6.6632018, -13.0499992, 13.0649996
8: -9.2895851, 3.9463463, -9.9102755, 4.1667218, -13.4563065, 13.8566208
9: -5.6075034, 5.6557236, -5.9880939, 6.0339165, -11.6414175, 11.6438160

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=58, inp2_unstable=58, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205613, upper bound: 12.3205711
time: 6.64 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205613, upper bound: 12.3205752
time: 4.54 seconds

## BFS IS instance: IS_A2_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -7.8292599, 6.3509970, -4.4719090, 3.6893673, -11.5186272, 10.8229055
1: -6.9833832, 5.7510753, -3.8317254, 3.3533330, -10.3367157, 9.5828009
2: -8.8719387, 5.6918168, -4.8953552, 3.4018815, -12.2738180, 10.5871716
3: -9.5280685, 4.6712527, -5.2999401, 2.7835197, -12.3115873, 9.9711924
4: -9.3947601, 6.3901892, -5.2807770, 3.7005892, -13.0953493, 11.6709652
5: -7.9586077, 6.2934709, -4.5096202, 3.7862389, -11.7448463, 10.8030910
6: -7.0841742, 7.1085596, -4.0742149, 4.2313728, -11.3155460, 11.1827745
7: -8.0094719, 7.8053064, -4.4900002, 4.4078498, -12.4173222, 12.2953062
8: -11.5835867, 4.7689857, -6.6041875, 3.0292170, -14.6128035, 11.3731709
9: -6.9973154, 7.0485334, -3.9896269, 4.0231175, -11.0204334, 11.0381603

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=29, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=58, inp2_unstable=54, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=223, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3200993, upper bound: 12.3202241
time: 4.81 seconds

## Relational analysis of IS_A2_B1_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3200983, upper bound: 12.3202408
time: 4.77 seconds

## BFS IS instance: IS_A2_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -8.5698814, 6.9411693, -6.2964301, 5.1500850, -13.7199669, 13.2375994
1: -7.6660314, 6.2734976, -5.5275245, 4.6504521, -12.3164835, 11.8010206
2: -9.7376099, 6.1946559, -7.0477347, 4.6515450, -14.3891544, 13.2423906
3: -10.4423542, 5.0843115, -7.5721588, 3.8114462, -14.2537994, 12.6564693
4: -10.2801332, 6.9778852, -7.4918504, 5.1576352, -15.4377689, 14.4697351
5: -8.7151575, 6.8729825, -6.4013290, 5.1249952, -13.8401518, 13.2743111
6: -7.7491550, 7.7519722, -5.7163124, 5.7939129, -13.5430670, 13.4682827
7: -8.7687416, 8.5399113, -6.3833666, 6.2306824, -14.9994240, 14.9232769
8: -12.6675615, 5.1925240, -9.3006182, 4.0157890, -16.6833496, 14.4931421
9: -7.6537256, 7.7102900, -5.6190548, 5.6695924, -13.3233185, 13.3293447

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=29, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=55, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=232, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205147, upper bound: 12.3202798
time: 4.32 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205147, upper bound: 12.3206788
time: 4.62 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.6990261, 5.4452829, -6.6311421, 5.4041176, -12.1031418, 12.0764236
1: -5.9436979, 4.9496870, -5.8615170, 4.8934021, -10.8371000, 10.8112040
2: -7.5469732, 4.9168863, -7.4551568, 4.8736391, -12.4206104, 12.3720417
3: -8.1254025, 4.0377951, -8.0158005, 4.0009851, -12.1263876, 12.0535955
4: -8.0460482, 5.4828219, -7.9324799, 5.4258709, -13.4719172, 13.4153023
5: -6.8005204, 5.3938394, -6.7383990, 5.3545151, -12.1550341, 12.1322365
6: -6.0375314, 6.1080351, -5.9947801, 6.0686231, -12.1061535, 12.1028156
7: -6.8515720, 6.6790986, -6.7572336, 6.5916080, -13.4431801, 13.4363317
8: -9.9226217, 4.1132174, -9.8059158, 4.1329851, -14.0556068, 13.9191332
9: -5.9894671, 6.0368752, -5.9251409, 5.9710159, -11.9604816, 11.9620132

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=57, inp2_unstable=59, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205456, upper bound: 12.3205381
time: 7.94 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205456, upper bound: 12.3205385
time: 3.52 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.8484807, 6.3685699, -6.7018452, 5.4595895, -13.3080692, 13.0704145
1: -6.9983768, 5.7631297, -5.9277730, 4.9438858, -11.9422626, 11.6909008
2: -8.8884344, 5.7051468, -7.5375347, 4.9217110, -13.8101444, 13.2426815
3: -9.5461559, 4.6807013, -8.1041546, 4.0405626, -13.5867186, 12.7848549
4: -9.4142504, 6.3972955, -8.0195007, 5.4810748, -14.8953247, 14.4167957
5: -7.9808164, 6.3042850, -6.8122001, 5.4090919, -13.3899069, 13.1164837
6: -7.0880589, 7.1261220, -6.0566869, 6.1288586, -13.2169161, 13.1828079
7: -8.0235529, 7.8185883, -6.8307910, 6.6632018, -14.6867523, 14.6493769
8: -11.6154938, 4.7814655, -9.9102755, 4.1667218, -15.7822151, 14.6917400
9: -7.0115933, 7.0682116, -5.9880939, 6.0339165, -13.0455074, 13.0563049

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=58, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205456, upper bound: 12.3205406
time: 3.59 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205456, upper bound: 12.3205441
time: 4.73 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5.1597514, 4.2243500, -6.3711700, 5.1947541, -10.3545055, 10.5955200
1: -4.5061083, 3.8457608, -5.6187835, 4.7107477, -9.2168560, 9.4645443
2: -5.7244811, 3.8630471, -7.1474133, 4.6976337, -10.4221134, 11.0104599
3: -6.1857219, 3.1738603, -7.6897240, 3.8522098, -10.0379314, 10.8635845
4: -6.1667233, 4.2442379, -7.6171980, 5.2181773, -11.3848991, 11.8614359
5: -5.2219253, 4.2483940, -6.4703860, 5.1456547, -10.3675795, 10.7187786
6: -4.6424723, 4.7664061, -5.7556219, 5.8289323, -10.4714022, 10.5220251
7: -5.2428083, 5.1236596, -6.4867086, 6.3254375, -11.5682459, 11.6103668
8: -7.6338062, 3.3060670, -9.4268131, 3.9926686, -11.6264744, 12.7328777
9: -4.6105165, 4.6485400, -5.6891379, 5.7409520, -10.3514671, 10.3376770

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=226, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203533, upper bound: 12.3203037
time: 5.53 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3201593, upper bound: 12.3202433
time: 4.22 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 11.14 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3205841, upper bound: 12.3205098
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3206054, upper bound: 12.3205216
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3206969, upper bound: 12.3207714
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3206969, upper bound: 12.3208946
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3202834, upper bound: 12.3201809
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3203176, upper bound: 12.3201944
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3203613, upper bound: 12.3205384
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3203613, upper bound: 12.3207284
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3207801, upper bound: 12.3207731
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3207801, upper bound: 12.3207737
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3207807, upper bound: 12.3207780
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3207807, upper bound: 12.3207798
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3205759, upper bound: 12.3206049
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3205759, upper bound: 12.3206050
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3205754, upper bound: 12.3206091
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3205754, upper bound: 12.3206132
IS_A1_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3207737, upper bound: 12.3207806
IS_A1_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3207737, upper bound: 12.3207856
IS_A1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3207774, upper bound: 12.3207806
IS_A1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3207774, upper bound: 12.3207876
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3202243, upper bound: 12.3200992
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3202408, upper bound: 12.3200983
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3202799, upper bound: 12.3205152
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3202799, upper bound: 12.3207193
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3205700, upper bound: 12.3206056
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3205700, upper bound: 12.3206061
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3205700, upper bound: 12.3206080
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3205700, upper bound: 12.3206087
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3205642, upper bound: 12.3206008
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3205642, upper bound: 12.3205999
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3205642, upper bound: 12.3206090
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3205642, upper bound: 12.3206090
IS_A2_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3201809, upper bound: 12.3202833
IS_A2_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3201944, upper bound: 12.3203175
IS_A2_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3205385, upper bound: 12.3203612
IS_A2_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3205385, upper bound: 12.3206922
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3205605, upper bound: 12.3205603
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3205605, upper bound: 12.3205602
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3205613, upper bound: 12.3205711
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3205613, upper bound: 12.3205752
IS_A2_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3200993, upper bound: 12.3202241
IS_A2_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3200983, upper bound: 12.3202408
IS_A2_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3205147, upper bound: 12.3202798
IS_A2_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3205147, upper bound: 12.3206788
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3205456, upper bound: 12.3205381
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3205456, upper bound: 12.3205385
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3205456, upper bound: 12.3205406
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3205456, upper bound: 12.3205441
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3203533, upper bound: 12.3203037
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 8, lower bound: -12.3201593, upper bound: 12.3202433
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.14
Output dim: 8, lower bound: -12.3205376, upper bound: 12.3205526
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.14
Output dim: 8, lower bound: -12.3205406, upper bound: 12.3205455
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.14
Output dim: 8, lower bound: -12.3205406, upper bound: 12.3205554
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.14
Output dim: 8, lower bound: -12.3205356, upper bound: 12.3205360
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.14
Output dim: 8, lower bound: -12.3205356, upper bound: 12.3205398
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.14
Output dim: 8, lower bound: -12.3205394, upper bound: 12.3205361
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.14
Output dim: 8, lower bound: -12.3205394, upper bound: 12.3205417

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 6.60 + 597.36 = 603.96 seconds
