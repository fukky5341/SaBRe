## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 106.6602947382


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797)
1: (-48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317)
2: (-65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747)
3: (-68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471)
4: (-64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390)
5: (-57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115)
6: (-54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235)
7: (-58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829)
8: (-71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617)
9: (-54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.50 + 11.33 = 12.83 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -106.7670618, upper bound: 106.7670618

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7554331, upper bound: 106.7556257
time: 9.01 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7563527, upper bound: 106.7563527
time: 7.55 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 16.73 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 16.73
Output dim: 0, lower bound: -106.7554331, upper bound: 106.7556257
IS_A2, status: Status.UNKNOWN, split count: 1, time: 16.73
Output dim: 0, lower bound: -106.7563527, upper bound: 106.7563527

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -40.4215355, 32.1656380, -52.4963608, 41.7913055, -82.2128220, 84.6620026
1: -32.5613441, 27.7784863, -42.6171036, 36.2078896, -68.7692337, 70.3955917
2: -43.4267426, 28.1031914, -56.7059441, 36.6510315, -80.0777435, 84.8091278
3: -45.9748688, 24.2641106, -59.9826431, 31.5825367, -77.5574036, 84.2467499
4: -42.8716888, 32.4032478, -55.8638458, 42.2842445, -85.1559296, 88.2670898
5: -38.8010483, 30.6035709, -50.5371666, 39.6374054, -78.4384537, 81.1407394
6: -36.7541809, 35.3119507, -47.8690758, 45.8901176, -82.6442871, 83.1810303
7: -39.3098488, 33.9188652, -51.2901459, 44.0228157, -83.3326645, 85.2089920
8: -47.4530792, 31.4979420, -62.1591644, 41.3959351, -88.8490143, 93.6570969
9: -36.3031998, 35.3297119, -47.2926941, 46.1051826, -82.4083710, 82.6224060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=63, inp2_unstable=64, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=148, inp2_unstable=150, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=217, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 168

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7550576, upper bound: 106.7550576
time: 8.33 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7550576, upper bound: 106.7556257
time: 8.54 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -43.0184174, 34.2086563, -50.9955406, 40.5937691, -83.6121826, 85.2041931
1: -34.7182274, 29.5691071, -41.3734207, 35.1553078, -69.8735275, 70.9425201
2: -46.2603874, 29.9053726, -55.0451431, 35.5759583, -81.8363342, 84.9505081
3: -48.9885712, 25.8325176, -58.2324982, 30.6709747, -79.6595459, 84.0649948
4: -45.6809273, 34.5013237, -54.2551613, 41.0516281, -86.7325516, 88.7564850
5: -41.2985191, 32.5255585, -49.0815697, 38.5158157, -79.8143311, 81.6071167
6: -39.1464157, 37.5944214, -46.4886284, 44.5845070, -83.7309113, 84.0830383
7: -41.8991547, 36.0853615, -49.7956772, 42.7667351, -84.6658936, 85.8810425
8: -50.5748444, 33.5495453, -60.3187370, 40.1566734, -90.7315140, 93.8682861
9: -38.6610641, 37.6220551, -45.9355354, 44.7589874, -83.4200516, 83.5575867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=63, inp2_unstable=64, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=154, inp2_unstable=150, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=194, inp2_unstable=213, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 170

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7556258, upper bound: 106.7551749
time: 8.19 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7556258, upper bound: 106.7563527
time: 9.96 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 19.78 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 19.78
Output dim: 0, lower bound: -106.7550576, upper bound: 106.7550576
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 19.78
Output dim: 0, lower bound: -106.7550576, upper bound: 106.7556257
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 19.78
Output dim: 0, lower bound: -106.7556258, upper bound: 106.7551749
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 19.78
Output dim: 0, lower bound: -106.7556258, upper bound: 106.7563527

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -40.4215355, 32.1656380, -40.4215355, 32.1656380, -72.5871735, 72.5871735
1: -32.5613441, 27.7784863, -32.5613441, 27.7784863, -60.3398285, 60.3398285
2: -43.4267426, 28.1031914, -43.4267426, 28.1031914, -71.5298920, 71.5298920
3: -45.9748688, 24.2641106, -45.9748688, 24.2641106, -70.2389832, 70.2389832
4: -42.8716888, 32.4032478, -42.8716888, 32.4032478, -75.2749329, 75.2749329
5: -38.8010483, 30.6035709, -38.8010483, 30.6035709, -69.4046173, 69.4046173
6: -36.7541809, 35.3119507, -36.7541809, 35.3119507, -72.0661316, 72.0661316
7: -39.3098488, 33.9188652, -39.3098488, 33.9188652, -73.2287140, 73.2287140
8: -47.4530792, 31.4979420, -47.4530792, 31.4979420, -78.9510193, 78.9510193
9: -36.3031998, 35.3297119, -36.3031998, 35.3297119, -71.6329117, 71.6329117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=63, inp2_unstable=63, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=148, inp2_unstable=148, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=191, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7373110, upper bound: 106.7343085
time: 9.27 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7345154, upper bound: 106.7336190
time: 7.89 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -40.4215355, 32.1656380, -43.0184174, 34.2086563, -74.6301880, 75.1840515
1: -32.5613441, 27.7784863, -34.7182274, 29.5691071, -62.1304512, 62.4967117
2: -43.4267426, 28.1031914, -46.2603874, 29.9053726, -73.3320923, 74.3635559
3: -45.9748688, 24.2641106, -48.9885712, 25.8325176, -71.8073883, 73.2526703
4: -42.8716888, 32.4032478, -45.6809273, 34.5013237, -77.3730164, 78.0841751
5: -38.8010483, 30.6035709, -41.2985191, 32.5255585, -71.3266068, 71.9020920
6: -36.7541809, 35.3119507, -39.1464157, 37.5944214, -74.3485794, 74.4583664
7: -39.3098488, 33.9188652, -41.8991547, 36.0853615, -75.3952103, 75.8180237
8: -47.4530792, 31.4979420, -50.5748444, 33.5495453, -81.0026245, 82.0727768
9: -36.3031998, 35.3297119, -38.6610641, 37.6220551, -73.9252548, 73.9907761

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=63, inp2_unstable=63, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=148, inp2_unstable=154, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=191, inp2_unstable=194, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7358105, upper bound: 106.7382072
time: 8.37 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7345154, upper bound: 106.7336190
time: 9.01 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -43.0184174, 34.2086563, -40.4215355, 32.1656380, -75.1840515, 74.6301880
1: -34.7182274, 29.5691071, -32.5613441, 27.7784863, -62.4967117, 62.1304512
2: -46.2603874, 29.9053726, -43.4267426, 28.1031914, -74.3635559, 73.3320923
3: -48.9885712, 25.8325176, -45.9748688, 24.2641106, -73.2526703, 71.8073883
4: -45.6809273, 34.5013237, -42.8716888, 32.4032478, -78.0841751, 77.3730164
5: -41.2985191, 32.5255585, -38.8010483, 30.6035709, -71.9020920, 71.3266068
6: -39.1464157, 37.5944214, -36.7541809, 35.3119507, -74.4583664, 74.3485794
7: -41.8991547, 36.0853615, -39.3098488, 33.9188652, -75.8180237, 75.3952103
8: -50.5748444, 33.5495453, -47.4530792, 31.4979420, -82.0727768, 81.0026245
9: -38.6610641, 37.6220551, -36.3031998, 35.3297119, -73.9907761, 73.9252548

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=63, inp2_unstable=63, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=154, inp2_unstable=148, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=194, inp2_unstable=191, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7377862, upper bound: 106.7343919
time: 9.92 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7333969, upper bound: 106.7334017
time: 6.65 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -43.0184174, 34.2086563, -43.0184174, 34.2086563, -77.2270737, 77.2270737
1: -34.7182274, 29.5691071, -34.7182274, 29.5691071, -64.2873230, 64.2873230
2: -46.2603874, 29.9053726, -46.2603874, 29.9053726, -76.1657486, 76.1657486
3: -48.9885712, 25.8325176, -48.9885712, 25.8325176, -74.8210754, 74.8210754
4: -45.6809273, 34.5013237, -45.6809273, 34.5013237, -80.1822510, 80.1822510
5: -41.2985191, 32.5255585, -41.2985191, 32.5255585, -73.8240662, 73.8240662
6: -39.1464157, 37.5944214, -39.1464157, 37.5944214, -76.7408218, 76.7408218
7: -41.8991547, 36.0853615, -41.8991547, 36.0853615, -77.9845123, 77.9845123
8: -50.5748444, 33.5495453, -50.5748444, 33.5495453, -84.1243820, 84.1243820
9: -38.6610641, 37.6220551, -38.6610641, 37.6220551, -76.2831192, 76.2831192

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=63, inp2_unstable=63, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=154, inp2_unstable=154, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=194, inp2_unstable=194, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7377862, upper bound: 106.7344097
time: 8.46 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7333969, upper bound: 106.7334073
time: 8.11 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 18.13 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 18.13
Output dim: 0, lower bound: -106.7373110, upper bound: 106.7343085
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.13
Output dim: 0, lower bound: -106.7345154, upper bound: 106.7336190
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 18.13
Output dim: 0, lower bound: -106.7358105, upper bound: 106.7382072
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 18.13
Output dim: 0, lower bound: -106.7345154, upper bound: 106.7336190
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 18.13
Output dim: 0, lower bound: -106.7377862, upper bound: 106.7343919
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.13
Output dim: 0, lower bound: -106.7333969, upper bound: 106.7334017
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 18.13
Output dim: 0, lower bound: -106.7377862, upper bound: 106.7344097
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 18.13
Output dim: 0, lower bound: -106.7333969, upper bound: 106.7334073

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -26.7422485, 21.2718410, -37.8471146, 30.1299877, -56.8722305, 59.1189537
1: -21.2280235, 18.2902088, -30.4349766, 25.9966240, -47.2246323, 48.7251816
2: -28.4548073, 18.6351051, -40.6121483, 26.3152657, -54.7700691, 59.2472534
3: -30.0922546, 15.9876270, -42.9920540, 22.7105942, -52.8028488, 58.9796829
4: -28.2785721, 21.3875751, -40.1298027, 30.3276749, -58.6062469, 61.5173721
5: -25.3948956, 20.3002529, -36.2888680, 28.6771679, -54.0720634, 56.5891190
6: -24.3348961, 23.3814526, -34.4116020, 33.0819054, -57.4168015, 57.7930489
7: -26.0041809, 22.4731750, -36.7924232, 31.7752075, -57.7793846, 59.2655907
8: -31.1208191, 20.5765305, -44.3611984, 29.4308472, -60.5516586, 64.9377289
9: -24.0275688, 23.3268414, -33.9962540, 33.0737152, -57.1012840, 57.3230934

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=63, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=147, inp2_unstable=147, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=148, inp2_unstable=184, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 168

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7364786, upper bound: 106.7364786
time: 8.50 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7364786, upper bound: 106.7364786
time: 8.21 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -22.5513592, 17.8861790, -30.0552025, 23.9169464, -46.4683075, 47.9413795
1: -17.6214981, 15.3477020, -23.9647007, 20.5707531, -38.1922531, 39.3124008
2: -23.7973385, 15.6399269, -32.0421410, 20.8775845, -44.6749191, 47.6820679
3: -25.1999550, 13.4318771, -33.9282684, 17.9947720, -43.1947250, 47.3601456
4: -23.6879063, 17.9321709, -31.7673378, 24.0477104, -47.7356186, 49.6995087
5: -21.2457676, 17.0805855, -28.6590309, 22.8299713, -44.0757256, 45.7396049
6: -20.4381771, 19.6731262, -27.3091831, 26.2767029, -46.7148819, 46.9823074
7: -21.8581924, 18.8887234, -29.1731853, 25.2266026, -47.0847855, 48.0619011
8: -26.0306473, 17.2440929, -34.9918060, 23.1664467, -49.1970863, 52.2358971
9: -20.1597729, 19.5271244, -26.9766388, 26.1724625, -46.3322296, 46.5037613

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=63, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=154, inp2_unstable=147, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=129, inp2_unstable=162, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7364786, upper bound: 106.7364786
time: 7.68 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7364786, upper bound: 106.7364786
time: 7.66 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -37.8471146, 30.1299877, -29.5187664, 23.4870911, -61.3342018, 59.6487465
1: -30.4349766, 25.9966240, -23.5490704, 20.2099152, -50.6448860, 49.5456848
2: -40.6121483, 26.3152657, -31.5047932, 20.5330448, -61.1451874, 57.8200531
3: -42.9920540, 22.7105942, -33.3346443, 17.6447906, -60.6368446, 56.0452385
4: -40.1298027, 30.3276749, -31.2796803, 23.6343670, -63.7641678, 61.6073532
5: -36.2888680, 28.6771679, -28.0859947, 22.3803577, -58.6692276, 56.7631607
6: -34.4116020, 33.0819054, -26.8713856, 25.8427086, -60.2543030, 59.9532890
7: -36.7924232, 31.7752075, -28.7505798, 24.8096333, -61.6020546, 60.5257797
8: -44.3611984, 29.4308472, -34.4302559, 22.7473793, -67.1085739, 63.8610878
9: -33.9962540, 33.0737152, -26.5568848, 25.7704887, -59.7667389, 59.6306000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=63, inp2_unstable=62, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=147, inp2_unstable=153, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=184, inp2_unstable=159, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 168

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7345154, upper bound: 106.7336190
time: 8.08 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7345154, upper bound: 106.7336190
time: 9.16 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -30.0552025, 23.9169464, -22.9962101, 18.2312794, -48.2864838, 46.9131546
1: -23.9647007, 20.5707531, -17.9922199, 15.6463699, -39.6110611, 38.5629730
2: -32.0421410, 20.8775845, -24.2844887, 15.9405718, -47.9827118, 45.1620674
3: -33.9282684, 17.9947720, -25.7163887, 13.6907778, -47.6190453, 43.7111588
4: -31.7673378, 24.0477104, -24.1710072, 18.3016148, -50.0689507, 48.2187157
5: -28.6590309, 22.8299713, -21.6525860, 17.3900032, -46.0490341, 44.4825554
6: -27.3091831, 26.2767029, -20.8454704, 20.0681534, -47.3773346, 47.1221733
7: -29.1731853, 25.2266026, -22.3190098, 19.2652969, -48.4384804, 47.5456123
8: -34.9918060, 23.1664467, -26.5726395, 17.5822601, -52.5740662, 49.7390785
9: -26.9766388, 26.1724625, -20.5671253, 19.9276485, -46.9042892, 46.7395859

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=63, inp2_unstable=62, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=147, inp2_unstable=158, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=162, inp2_unstable=128, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7345154, upper bound: 106.7336189
time: 7.75 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7345154, upper bound: 106.7336190
time: 7.90 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -29.5187664, 23.4870911, -37.8471146, 30.1299877, -59.6487465, 61.3342018
1: -23.5490704, 20.2099152, -30.4349766, 25.9966240, -49.5456848, 50.6448860
2: -31.5047932, 20.5330448, -40.6121483, 26.3152657, -57.8200531, 61.1451874
3: -33.3346443, 17.6447906, -42.9920540, 22.7105942, -56.0452385, 60.6368446
4: -31.2796803, 23.6343670, -40.1298027, 30.3276749, -61.6073532, 63.7641678
5: -28.0859947, 22.3803577, -36.2888680, 28.6771679, -56.7631607, 58.6692276
6: -26.8713856, 25.8427086, -34.4116020, 33.0819054, -59.9532890, 60.2543030
7: -28.7505798, 24.8096333, -36.7924232, 31.7752075, -60.5257797, 61.6020546
8: -34.4302559, 22.7473793, -44.3611984, 29.4308472, -63.8610916, 67.1085739
9: -26.5568848, 25.7704887, -33.9962540, 33.0737152, -59.6306000, 59.7667389

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=63, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=153, inp2_unstable=147, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=159, inp2_unstable=184, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 168

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7336190, upper bound: 106.7345154
time: 7.73 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7336190, upper bound: 106.7345154
time: 8.68 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -22.9962101, 18.2312794, -30.0552025, 23.9169464, -46.9131546, 48.2864838
1: -17.9922199, 15.6463699, -23.9647007, 20.5707531, -38.5629730, 39.6110611
2: -24.2844887, 15.9405718, -32.0421410, 20.8775845, -45.1620674, 47.9827118
3: -25.7163887, 13.6907778, -33.9282684, 17.9947720, -43.7111588, 47.6190453
4: -24.1710072, 18.3016148, -31.7673378, 24.0477104, -48.2187157, 50.0689507
5: -21.6525860, 17.3900032, -28.6590309, 22.8299713, -44.4825516, 46.0490341
6: -20.8454704, 20.0681534, -27.3091831, 26.2767029, -47.1221733, 47.3773346
7: -22.3190098, 19.2652969, -29.1731853, 25.2266026, -47.5456123, 48.4384804
8: -26.5726395, 17.5822601, -34.9918060, 23.1664467, -49.7390785, 52.5740662
9: -20.5671253, 19.9276485, -26.9766388, 26.1724625, -46.7395859, 46.9042892

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=63, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=158, inp2_unstable=147, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=128, inp2_unstable=162, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7336190, upper bound: 106.7345154
time: 8.31 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7336190, upper bound: 106.7345154
time: 8.35 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -29.5187664, 23.4870911, -40.5106430, 32.2322578, -61.7510185, 63.9977226
1: -23.5490704, 20.2099152, -32.6535645, 27.8397751, -51.3888435, 52.8634796
2: -31.5047932, 20.5330448, -43.5276222, 28.1626759, -59.6674576, 64.0606613
3: -33.3346443, 17.6447906, -46.0933571, 24.3177814, -57.6524277, 63.7381439
4: -31.2796803, 23.6343670, -43.0172424, 32.4772720, -63.7569504, 66.6516037
5: -28.0859947, 22.3803577, -38.8574066, 30.6514111, -58.7374039, 61.2377625
6: -26.8713856, 25.8427086, -36.8641205, 35.4313316, -62.3027153, 62.7068291
7: -28.7505798, 24.8096333, -39.4506950, 34.0039062, -62.7544861, 64.2603302
8: -34.4302559, 22.7473793, -47.5513649, 31.5370960, -65.9673538, 70.2987442
9: -26.5568848, 25.7704887, -36.4209099, 35.4289551, -61.9858398, 62.1913948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=63, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=153, inp2_unstable=154, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=159, inp2_unstable=187, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 168

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7334059, upper bound: 106.7334073
time: 7.33 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7334059, upper bound: 106.7334073
time: 7.93 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -22.9962101, 18.2312794, -31.5634136, 25.1066113, -48.1028175, 49.7946892
1: -17.9922199, 15.6463699, -25.2240715, 21.6079121, -39.6001244, 40.8704414
2: -24.2844887, 15.9405718, -33.6904984, 21.9086761, -46.1931610, 49.6310692
3: -25.7163887, 13.6907778, -35.6812286, 18.8970261, -44.6134071, 49.3720055
4: -24.1710072, 18.3016148, -33.3945770, 25.2608681, -49.4318657, 51.6961899
5: -21.6525860, 17.3900032, -30.1051502, 23.9419403, -45.5945282, 47.4951553
6: -20.8454704, 20.0681534, -28.6896553, 27.6141739, -48.4596443, 48.7578011
7: -22.3190098, 19.2652969, -30.6836796, 26.4995632, -48.8185730, 49.9489708
8: -26.5726395, 17.5822601, -36.7943916, 24.3335533, -50.9061928, 54.3766518
9: -20.5671253, 19.9276485, -28.3497181, 27.5052948, -48.0724182, 48.2773666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=63, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=158, inp2_unstable=154, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=128, inp2_unstable=167, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7334059, upper bound: 106.7334073
time: 7.17 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7334059, upper bound: 106.7334073
time: 7.00 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 15.71 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.71
Output dim: 0, lower bound: -106.7364786, upper bound: 106.7364786
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.71
Output dim: 0, lower bound: -106.7364786, upper bound: 106.7364786
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.71
Output dim: 0, lower bound: -106.7364786, upper bound: 106.7364786
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.71
Output dim: 0, lower bound: -106.7364786, upper bound: 106.7364786
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 15.71
Output dim: 0, lower bound: -106.7345154, upper bound: 106.7336190
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 15.71
Output dim: 0, lower bound: -106.7345154, upper bound: 106.7336190
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 15.71
Output dim: 0, lower bound: -106.7345154, upper bound: 106.7336189
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 15.71
Output dim: 0, lower bound: -106.7345154, upper bound: 106.7336190
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.71
Output dim: 0, lower bound: -106.7336190, upper bound: 106.7345154
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.71
Output dim: 0, lower bound: -106.7336190, upper bound: 106.7345154
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.71
Output dim: 0, lower bound: -106.7336190, upper bound: 106.7345154
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.71
Output dim: 0, lower bound: -106.7336190, upper bound: 106.7345154
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.71
Output dim: 0, lower bound: -106.7334059, upper bound: 106.7334073
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.71
Output dim: 0, lower bound: -106.7334059, upper bound: 106.7334073
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.71
Output dim: 0, lower bound: -106.7334059, upper bound: 106.7334073
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.71
Output dim: 0, lower bound: -106.7334059, upper bound: 106.7334073

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -26.7422485, 21.2718410, -26.7422485, 21.2718410, -48.0140839, 48.0140839
1: -21.2280235, 18.2902088, -21.2280235, 18.2902088, -39.5182304, 39.5182304
2: -28.4548073, 18.6351051, -28.4548073, 18.6351051, -47.0899124, 47.0899124
3: -30.0922546, 15.9876270, -30.0922546, 15.9876270, -46.0798798, 46.0798798
4: -28.2785721, 21.3875751, -28.2785721, 21.3875751, -49.6661415, 49.6661415
5: -25.3948956, 20.3002529, -25.3948956, 20.3002529, -45.6951485, 45.6951485
6: -24.3348961, 23.3814526, -24.3348961, 23.3814526, -47.7163467, 47.7163467
7: -26.0041809, 22.4731750, -26.0041809, 22.4731750, -48.4773560, 48.4773560
8: -31.1208191, 20.5765305, -31.1208191, 20.5765305, -51.6973495, 51.6973495
9: -24.0275688, 23.3268414, -24.0275688, 23.3268414, -47.3544083, 47.3544083

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=147, inp2_unstable=147, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=148, inp2_unstable=148, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7279963, upper bound: 106.7258219
time: 9.80 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7277806, upper bound: 106.7249537
time: 8.57 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -26.7422485, 21.2718410, -22.5513592, 17.8861790, -44.6284142, 43.8232002
1: -21.2280235, 18.2902088, -17.6214981, 15.3477020, -36.5757217, 35.9117050
2: -28.4548073, 18.6351051, -23.7973385, 15.6399269, -44.0947342, 42.4324417
3: -30.0922546, 15.9876270, -25.1999550, 13.4318771, -43.5241318, 41.1875839
4: -28.2785721, 21.3875751, -23.6879063, 17.9321709, -46.2107430, 45.0754814
5: -25.3948956, 20.3002529, -21.2457676, 17.0805855, -42.4754753, 41.5460129
6: -24.3348961, 23.3814526, -20.4381771, 19.6731262, -44.0080223, 43.8196297
7: -26.0041809, 22.4731750, -21.8581924, 18.8887234, -44.8928986, 44.3313599
8: -31.1208191, 20.5765305, -26.0306473, 17.2440929, -48.3649101, 46.6071777
9: -24.0275688, 23.3268414, -20.1597729, 19.5271244, -43.5546951, 43.4866066

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=147, inp2_unstable=154, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=148, inp2_unstable=129, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7285212, upper bound: 106.7251571
time: 8.24 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7277806, upper bound: 106.7249537
time: 8.53 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -22.5513592, 17.8861790, -26.7422485, 21.2718410, -43.8232002, 44.6284142
1: -17.6214981, 15.3477020, -21.2280235, 18.2902088, -35.9117050, 36.5757217
2: -23.7973385, 15.6399269, -28.4548073, 18.6351051, -42.4324417, 44.0947342
3: -25.1999550, 13.4318771, -30.0922546, 15.9876270, -41.1875839, 43.5241318
4: -23.6879063, 17.9321709, -28.2785721, 21.3875751, -45.0754814, 46.2107430
5: -21.2457676, 17.0805855, -25.3948956, 20.3002529, -41.5460129, 42.4754753
6: -20.4381771, 19.6731262, -24.3348961, 23.3814526, -43.8196297, 44.0080223
7: -21.8581924, 18.8887234, -26.0041809, 22.4731750, -44.3313599, 44.8928986
8: -26.0306473, 17.2440929, -31.1208191, 20.5765305, -46.6071777, 48.3649101
9: -20.1597729, 19.5271244, -24.0275688, 23.3268414, -43.4866066, 43.5546951

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=154, inp2_unstable=147, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=129, inp2_unstable=148, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7244161, upper bound: 106.7250799
time: 9.75 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7242420, upper bound: 106.7242420
time: 7.27 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -22.5513592, 17.8861790, -22.5513592, 17.8861790, -40.4375305, 40.4375305
1: -17.6214981, 15.3477020, -17.6214981, 15.3477020, -32.9692001, 32.9692001
2: -23.7973385, 15.6399269, -23.7973385, 15.6399269, -39.4372635, 39.4372635
3: -25.1999550, 13.4318771, -25.1999550, 13.4318771, -38.6318321, 38.6318321
4: -23.6879063, 17.9321709, -23.6879063, 17.9321709, -41.6200790, 41.6200790
5: -21.2457676, 17.0805855, -21.2457676, 17.0805855, -38.3263397, 38.3263397
6: -20.4381771, 19.6731262, -20.4381771, 19.6731262, -40.1113052, 40.1113052
7: -21.8581924, 18.8887234, -21.8581924, 18.8887234, -40.7469025, 40.7469025
8: -26.0306473, 17.2440929, -26.0306473, 17.2440929, -43.2747383, 43.2747383
9: -20.1597729, 19.5271244, -20.1597729, 19.5271244, -39.6868896, 39.6868896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=154, inp2_unstable=154, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=129, inp2_unstable=129, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7250799, upper bound: 106.7244161
time: 8.52 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7242420, upper bound: 106.7242420
time: 9.19 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -26.7422485, 21.2718410, -29.5187664, 23.4870911, -50.2293320, 50.7905998
1: -21.2280235, 18.2902088, -23.5490704, 20.2099152, -41.4379349, 41.8392792
2: -28.4548073, 18.6351051, -31.5047932, 20.5330448, -48.9878464, 50.1398964
3: -30.0922546, 15.9876270, -33.3346443, 17.6447906, -47.7370453, 49.3222733
4: -28.2785721, 21.3875751, -31.2796803, 23.6343670, -51.9129410, 52.6672478
5: -25.3948956, 20.3002529, -28.0859947, 22.3803577, -47.7752533, 48.3862457
6: -24.3348961, 23.3814526, -26.8713856, 25.8427086, -50.1776047, 50.2528343
7: -26.0041809, 22.4731750, -28.7505798, 24.8096333, -50.8138123, 51.2237511
8: -31.1208191, 20.5765305, -34.4302559, 22.7473793, -53.8681984, 55.0067825
9: -24.0275688, 23.3268414, -26.5568848, 25.7704887, -49.7980576, 49.8837280

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=147, inp2_unstable=153, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=148, inp2_unstable=159, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7236513, upper bound: 106.7268637
time: 9.03 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7233881, upper bound: 106.7261975
time: 8.98 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -22.5513592, 17.8861790, -29.5187664, 23.4870911, -46.0384483, 47.4049301
1: -17.6214981, 15.3477020, -23.5490704, 20.2099152, -37.8314133, 38.8967743
2: -23.7973385, 15.6399269, -31.5047932, 20.5330448, -44.3303795, 47.1447182
3: -25.1999550, 13.4318771, -33.3346443, 17.6447906, -42.8447456, 46.7665215
4: -23.6879063, 17.9321709, -31.2796803, 23.6343670, -47.3222733, 49.2118530
5: -21.2457676, 17.0805855, -28.0859947, 22.3803577, -43.6261253, 45.1665726
6: -20.4381771, 19.6731262, -26.8713856, 25.8427086, -46.2808838, 46.5445061
7: -21.8581924, 18.8887234, -28.7505798, 24.8096333, -46.6678238, 47.6392937
8: -26.0306473, 17.2440929, -34.4302559, 22.7473793, -48.7780266, 51.6743431
9: -20.1597729, 19.5271244, -26.5568848, 25.7704887, -45.9302521, 46.0840073

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=154, inp2_unstable=153, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=129, inp2_unstable=159, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7236513, upper bound: 106.7268638
time: 9.43 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7233881, upper bound: 106.7261975
time: 9.38 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -26.7422485, 21.2718410, -22.9962101, 18.2312794, -44.9735222, 44.2680473
1: -21.2280235, 18.2902088, -17.9922199, 15.6463699, -36.8743820, 36.2824287
2: -28.4548073, 18.6351051, -24.2844887, 15.9405718, -44.3953781, 42.9195938
3: -30.0922546, 15.9876270, -25.7163887, 13.6907778, -43.7830315, 41.7040100
4: -28.2785721, 21.3875751, -24.1710072, 18.3016148, -46.5801849, 45.5585709
5: -25.3948956, 20.3002529, -21.6525860, 17.3900032, -42.7848969, 41.9528389
6: -24.3348961, 23.3814526, -20.8454704, 20.0681534, -44.4030495, 44.2269211
7: -26.0041809, 22.4731750, -22.3190098, 19.2652969, -45.2694778, 44.7921829
8: -31.1208191, 20.5765305, -26.5726395, 17.5822601, -48.7030754, 47.1491699
9: -24.0275688, 23.3268414, -20.5671253, 19.9276485, -43.9552155, 43.8939667

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=147, inp2_unstable=158, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=148, inp2_unstable=128, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7227044, upper bound: 106.7212512
time: 9.37 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7219503, upper bound: 106.7210984
time: 9.27 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -22.5513592, 17.8861790, -22.9962101, 18.2312794, -40.7826385, 40.8823776
1: -17.6214981, 15.3477020, -17.9922199, 15.6463699, -33.2678680, 33.3399200
2: -23.7973385, 15.6399269, -24.2844887, 15.9405718, -39.7379112, 39.9244156
3: -25.1999550, 13.4318771, -25.7163887, 13.6907778, -38.8907318, 39.1482658
4: -23.6879063, 17.9321709, -24.1710072, 18.3016148, -41.9895210, 42.1031799
5: -21.2457676, 17.0805855, -21.6525860, 17.3900032, -38.6357689, 38.7331696
6: -20.4381771, 19.6731262, -20.8454704, 20.0681534, -40.5063324, 40.5185966
7: -21.8581924, 18.8887234, -22.3190098, 19.2652969, -41.1234856, 41.2077293
8: -26.0306473, 17.2440929, -26.5726395, 17.5822601, -43.6129036, 43.8167305
9: -20.1597729, 19.5271244, -20.5671253, 19.9276485, -40.0874138, 40.0942497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=154, inp2_unstable=158, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=129, inp2_unstable=128, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7222155, upper bound: 106.7221259
time: 7.58 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7219503, upper bound: 106.7210983
time: 9.12 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -29.5187664, 23.4870911, -26.7422485, 21.2718410, -50.7905998, 50.2293320
1: -23.5490704, 20.2099152, -21.2280235, 18.2902088, -41.8392792, 41.4379349
2: -31.5047932, 20.5330448, -28.4548073, 18.6351051, -50.1398964, 48.9878464
3: -33.3346443, 17.6447906, -30.0922546, 15.9876270, -49.3222733, 47.7370453
4: -31.2796803, 23.6343670, -28.2785721, 21.3875751, -52.6672478, 51.9129410
5: -28.0859947, 22.3803577, -25.3948956, 20.3002529, -48.3862457, 47.7752533
6: -26.8713856, 25.8427086, -24.3348961, 23.3814526, -50.2528343, 50.1776047
7: -28.7505798, 24.8096333, -26.0041809, 22.4731750, -51.2237511, 50.8138123
8: -34.4302559, 22.7473793, -31.1208191, 20.5765305, -55.0067825, 53.8681984
9: -26.5568848, 25.7704887, -24.0275688, 23.3268414, -49.8837280, 49.7980576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=153, inp2_unstable=147, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=159, inp2_unstable=148, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7268637, upper bound: 106.7236513
time: 8.87 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7261975, upper bound: 106.7233881
time: 9.14 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -29.5187664, 23.4870911, -22.5513592, 17.8861790, -47.4049301, 46.0384483
1: -23.5490704, 20.2099152, -17.6214981, 15.3477020, -38.8967743, 37.8314133
2: -31.5047932, 20.5330448, -23.7973385, 15.6399269, -47.1447182, 44.3303795
3: -33.3346443, 17.6447906, -25.1999550, 13.4318771, -46.7665215, 42.8447456
4: -31.2796803, 23.6343670, -23.6879063, 17.9321709, -49.2118530, 47.3222733
5: -28.0859947, 22.3803577, -21.2457676, 17.0805855, -45.1665726, 43.6261253
6: -26.8713856, 25.8427086, -20.4381771, 19.6731262, -46.5445061, 46.2808838
7: -28.7505798, 24.8096333, -21.8581924, 18.8887234, -47.6392937, 46.6678238
8: -34.4302559, 22.7473793, -26.0306473, 17.2440929, -51.6743431, 48.7780266
9: -26.5568848, 25.7704887, -20.1597729, 19.5271244, -46.0840073, 45.9302521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=153, inp2_unstable=154, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=159, inp2_unstable=129, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7268637, upper bound: 106.7236513
time: 8.40 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7261975, upper bound: 106.7233881
time: 9.83 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -22.9962101, 18.2312794, -26.7422485, 21.2718410, -44.2680473, 44.9735222
1: -17.9922199, 15.6463699, -21.2280235, 18.2902088, -36.2824287, 36.8743820
2: -24.2844887, 15.9405718, -28.4548073, 18.6351051, -42.9195938, 44.3953781
3: -25.7163887, 13.6907778, -30.0922546, 15.9876270, -41.7040100, 43.7830315
4: -24.1710072, 18.3016148, -28.2785721, 21.3875751, -45.5585709, 46.5801849
5: -21.6525860, 17.3900032, -25.3948956, 20.3002529, -41.9528389, 42.7848969
6: -20.8454704, 20.0681534, -24.3348961, 23.3814526, -44.2269211, 44.4030495
7: -22.3190098, 19.2652969, -26.0041809, 22.4731750, -44.7921829, 45.2694778
8: -26.5726395, 17.5822601, -31.1208191, 20.5765305, -47.1491699, 48.7030754
9: -20.5671253, 19.9276485, -24.0275688, 23.3268414, -43.8939667, 43.9552155

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=158, inp2_unstable=147, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=128, inp2_unstable=148, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7212512, upper bound: 106.7227044
time: 8.30 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2

### Relational analysis result of IS_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7210983, upper bound: 106.7219504
time: 8.81 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -22.9962101, 18.2312794, -22.5513592, 17.8861790, -40.8823776, 40.7826385
1: -17.9922199, 15.6463699, -17.6214981, 15.3477020, -33.3399200, 33.2678680
2: -24.2844887, 15.9405718, -23.7973385, 15.6399269, -39.9244156, 39.7379112
3: -25.7163887, 13.6907778, -25.1999550, 13.4318771, -39.1482658, 38.8907318
4: -24.1710072, 18.3016148, -23.6879063, 17.9321709, -42.1031799, 41.9895210
5: -21.6525860, 17.3900032, -21.2457676, 17.0805855, -38.7331696, 38.6357689
6: -20.8454704, 20.0681534, -20.4381771, 19.6731262, -40.5185966, 40.5063324
7: -22.3190098, 19.2652969, -21.8581924, 18.8887234, -41.2077293, 41.1234856
8: -26.5726395, 17.5822601, -26.0306473, 17.2440929, -43.8167305, 43.6129036
9: -20.5671253, 19.9276485, -20.1597729, 19.5271244, -40.0942497, 40.0874138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=158, inp2_unstable=154, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=128, inp2_unstable=129, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7221260, upper bound: 106.7222156
time: 9.38 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7210984, upper bound: 106.7219504
time: 8.71 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -29.5187664, 23.4870911, -29.5187664, 23.4870911, -53.0058479, 53.0058479
1: -23.5490704, 20.2099152, -23.5490704, 20.2099152, -43.7589874, 43.7589874
2: -31.5047932, 20.5330448, -31.5047932, 20.5330448, -52.0378304, 52.0378304
3: -33.3346443, 17.6447906, -33.3346443, 17.6447906, -50.9794350, 50.9794350
4: -31.2796803, 23.6343670, -31.2796803, 23.6343670, -54.9140472, 54.9140472
5: -28.0859947, 22.3803577, -28.0859947, 22.3803577, -50.4663544, 50.4663544
6: -26.8713856, 25.8427086, -26.8713856, 25.8427086, -52.7140884, 52.7140884
7: -28.7505798, 24.8096333, -28.7505798, 24.8096333, -53.5602112, 53.5602112
8: -34.4302559, 22.7473793, -34.4302559, 22.7473793, -57.1776314, 57.1776314
9: -26.5568848, 25.7704887, -26.5568848, 25.7704887, -52.3273735, 52.3273735

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=153, inp2_unstable=153, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=159, inp2_unstable=159, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7265818, upper bound: 106.7221501
time: 8.74 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7257399, upper bound: 106.7219436
time: 9.83 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -29.5187664, 23.4870911, -22.9962101, 18.2312794, -47.7500381, 46.4832954
1: -23.5490704, 20.2099152, -17.9922199, 15.6463699, -39.1954346, 38.2021332
2: -31.5047932, 20.5330448, -24.2844887, 15.9405718, -47.4453621, 44.8175278
3: -33.3346443, 17.6447906, -25.7163887, 13.6907778, -47.0254211, 43.3611755
4: -31.2796803, 23.6343670, -24.1710072, 18.3016148, -49.5812874, 47.8053741
5: -28.0859947, 22.3803577, -21.6525860, 17.3900032, -45.4759979, 44.0329437
6: -26.8713856, 25.8427086, -20.8454704, 20.0681534, -46.9395332, 46.6881790
7: -28.7505798, 24.8096333, -22.3190098, 19.2652969, -48.0158730, 47.1286430
8: -34.4302559, 22.7473793, -26.5726395, 17.5822601, -52.0125122, 49.3200188
9: -26.5568848, 25.7704887, -20.5671253, 19.9276485, -46.4845352, 46.3376122

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=153, inp2_unstable=158, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=159, inp2_unstable=128, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7265818, upper bound: 106.7221500
time: 9.06 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7257399, upper bound: 106.7219436
time: 9.04 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -22.9962101, 18.2312794, -29.5184345, 23.4868336, -46.4830437, 47.7497063
1: -17.9922199, 15.6463699, -23.5488224, 20.2096786, -38.2018967, 39.1951866
2: -24.2844887, 15.9405718, -31.5044842, 20.5327835, -44.8172722, 47.4450531
3: -25.7163887, 13.6907778, -33.3342934, 17.6444931, -43.3608780, 47.0250702
4: -24.1710072, 18.3016148, -31.2792587, 23.6340942, -47.8050995, 49.5808716
5: -21.6525860, 17.3900032, -28.0856876, 22.3801250, -44.0327110, 45.4756927
6: -20.8454704, 20.0681534, -26.8709679, 25.8424377, -46.6879082, 46.9391136
7: -22.3190098, 19.2652969, -28.7502499, 24.8094006, -47.1284103, 48.0155487
8: -26.5726395, 17.5822601, -34.4298973, 22.7471027, -49.3197403, 52.0121536
9: -20.5671253, 19.9276485, -26.5565567, 25.7702217, -46.3373489, 46.4842072

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=158, inp2_unstable=153, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=128, inp2_unstable=159, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7210479, upper bound: 106.7217888
time: 8.97 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7208772, upper bound: 106.7208795
time: 8.42 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -22.9962101, 18.2312794, -22.9962101, 18.2312794, -41.2274857, 41.2274857
1: -17.9922199, 15.6463699, -17.9922199, 15.6463699, -33.6385803, 33.6385803
2: -24.2844887, 15.9405718, -24.2844887, 15.9405718, -40.2250595, 40.2250595
3: -25.7163887, 13.6907778, -25.7163887, 13.6907778, -39.4071617, 39.4071617
4: -24.1710072, 18.3016148, -24.1710072, 18.3016148, -42.4726105, 42.4726105
5: -21.6525860, 17.3900032, -21.6525860, 17.3900032, -39.0425873, 39.0425873
6: -20.8454704, 20.0681534, -20.8454704, 20.0681534, -40.9136238, 40.9136238
7: -22.3190098, 19.2652969, -22.3190098, 19.2652969, -41.5843048, 41.5843048
8: -26.5726395, 17.5822601, -26.5726395, 17.5822601, -44.1548958, 44.1548958
9: -20.5671253, 19.9276485, -20.5671253, 19.9276485, -40.4947739, 40.4947739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=62, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=158, inp2_unstable=158, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=128, inp2_unstable=128, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7218084, upper bound: 106.7210532
time: 8.83 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7208772, upper bound: 106.7208795
time: 7.58 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 17.94 seconds
IS_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 17.94
Output dim: 0, lower bound: -106.7279963, upper bound: 106.7258219
IS_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 17.94
Output dim: 0, lower bound: -106.7277806, upper bound: 106.7249537
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.94
Output dim: 0, lower bound: -106.7285212, upper bound: 106.7251571
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.94
Output dim: 0, lower bound: -106.7277806, upper bound: 106.7249537
IS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 17.94
Output dim: 0, lower bound: -106.7244161, upper bound: 106.7250799
IS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 17.94
Output dim: 0, lower bound: -106.7242420, upper bound: 106.7242420
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.94
Output dim: 0, lower bound: -106.7250799, upper bound: 106.7244161
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.94
Output dim: 0, lower bound: -106.7242420, upper bound: 106.7242420
IS_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 17.94
Output dim: 0, lower bound: -106.7236513, upper bound: 106.7268637
IS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 17.94
Output dim: 0, lower bound: -106.7233881, upper bound: 106.7261975
IS_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 17.94
Output dim: 0, lower bound: -106.7236513, upper bound: 106.7268638
IS_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 17.94
Output dim: 0, lower bound: -106.7233881, upper bound: 106.7261975
IS_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 17.94
Output dim: 0, lower bound: -106.7227044, upper bound: 106.7212512
IS_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 17.94
Output dim: 0, lower bound: -106.7219503, upper bound: 106.7210984
IS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 17.94
Output dim: 0, lower bound: -106.7222155, upper bound: 106.7221259
IS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 17.94
Output dim: 0, lower bound: -106.7219503, upper bound: 106.7210983
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.94
Output dim: 0, lower bound: -106.7268637, upper bound: 106.7236513
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.94
Output dim: 0, lower bound: -106.7261975, upper bound: 106.7233881
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.94
Output dim: 0, lower bound: -106.7268637, upper bound: 106.7236513
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.94
Output dim: 0, lower bound: -106.7261975, upper bound: 106.7233881
IS_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 17.94
Output dim: 0, lower bound: -106.7212512, upper bound: 106.7227044
IS_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 17.94
Output dim: 0, lower bound: -106.7210983, upper bound: 106.7219504
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.94
Output dim: 0, lower bound: -106.7221260, upper bound: 106.7222156
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.94
Output dim: 0, lower bound: -106.7210984, upper bound: 106.7219504
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.94
Output dim: 0, lower bound: -106.7265818, upper bound: 106.7221501
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.94
Output dim: 0, lower bound: -106.7257399, upper bound: 106.7219436
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.94
Output dim: 0, lower bound: -106.7265818, upper bound: 106.7221500
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.94
Output dim: 0, lower bound: -106.7257399, upper bound: 106.7219436
IS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 17.94
Output dim: 0, lower bound: -106.7210479, upper bound: 106.7217888
IS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 17.94
Output dim: 0, lower bound: -106.7208772, upper bound: 106.7208795
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.94
Output dim: 0, lower bound: -106.7218084, upper bound: 106.7210532
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.94
Output dim: 0, lower bound: -106.7208772, upper bound: 106.7208795

## BFS IS instance: IS_A1_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -24.0195961, 19.0861931, -17.8702641, 14.2024946, -38.2220917, 36.9564476
1: -18.9567909, 16.4109135, -13.7971811, 12.1453905, -31.1021805, 30.2080936
2: -25.4804630, 16.7213516, -18.7773094, 12.4067907, -37.8872452, 35.4986610
3: -26.9289074, 14.3488464, -19.8063393, 10.6257715, -37.5546799, 34.1551819
4: -25.3517647, 19.1783981, -18.7144279, 14.2162466, -39.5680122, 37.8928223
5: -22.7362747, 18.2413273, -16.7799950, 13.5830059, -36.3192711, 35.0213127
6: -21.8413792, 20.9841995, -16.2330551, 15.5810909, -37.4224625, 37.2172546
7: -23.3485241, 20.1918030, -17.2967930, 15.0185223, -38.3670425, 37.4885941
8: -27.8650684, 18.4097118, -20.5747604, 13.6283646, -41.4934311, 38.9844742
9: -21.5682793, 20.9100895, -15.9917793, 15.4268293, -36.9951096, 36.9018707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=62, inp2_unstable=61, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=147, inp2_unstable=146, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=135, inp2_unstable=116, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7337897, upper bound: 106.7337897
time: 7.96 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7337897, upper bound: 106.7337897
time: 8.29 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 17.79 seconds
IS_A1_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 17.79
Output dim: 0, lower bound: -106.7337897, upper bound: 106.7337897
IS_A1_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 17.79
Output dim: 0, lower bound: -106.7337897, upper bound: 106.7337897
IS_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 0, lower bound: -106.7277806, upper bound: 106.7249537
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 0, lower bound: -106.7285212, upper bound: 106.7251571
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 0, lower bound: -106.7277806, upper bound: 106.7249537
IS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 0, lower bound: -106.7244161, upper bound: 106.7250799
IS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 0, lower bound: -106.7242420, upper bound: 106.7242420
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 0, lower bound: -106.7250799, upper bound: 106.7244161
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 0, lower bound: -106.7242420, upper bound: 106.7242420
IS_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 0, lower bound: -106.7236513, upper bound: 106.7268637
IS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 0, lower bound: -106.7233881, upper bound: 106.7261975
IS_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 0, lower bound: -106.7236513, upper bound: 106.7268638
IS_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 0, lower bound: -106.7233881, upper bound: 106.7261975
IS_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 0, lower bound: -106.7227044, upper bound: 106.7212512
IS_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 0, lower bound: -106.7219503, upper bound: 106.7210984
IS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 0, lower bound: -106.7222155, upper bound: 106.7221259
IS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 0, lower bound: -106.7219503, upper bound: 106.7210983
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 0, lower bound: -106.7268637, upper bound: 106.7236513
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 0, lower bound: -106.7261975, upper bound: 106.7233881
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 0, lower bound: -106.7268637, upper bound: 106.7236513
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 0, lower bound: -106.7261975, upper bound: 106.7233881
IS_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 0, lower bound: -106.7212512, upper bound: 106.7227044
IS_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 0, lower bound: -106.7210983, upper bound: 106.7219504
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 0, lower bound: -106.7221260, upper bound: 106.7222156
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 0, lower bound: -106.7210984, upper bound: 106.7219504
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 0, lower bound: -106.7265818, upper bound: 106.7221501
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 0, lower bound: -106.7257399, upper bound: 106.7219436
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 0, lower bound: -106.7265818, upper bound: 106.7221500
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 0, lower bound: -106.7257399, upper bound: 106.7219436
IS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 0, lower bound: -106.7210479, upper bound: 106.7217888
IS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 0, lower bound: -106.7208772, upper bound: 106.7208795
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 0, lower bound: -106.7218084, upper bound: 106.7210532
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 0, lower bound: -106.7208772, upper bound: 106.7208795

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 12.83 + 595.33 = 608.16 seconds
