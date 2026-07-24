## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 143.61867486269998


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766)
1: (-62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901)
2: (-83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913)
3: (-88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352)
4: (-81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725)
5: (-73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507)
6: (-70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321)
7: (-76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043)
8: (-92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513)
9: (-69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.53 + 10.19 = 11.73 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -143.7624373, upper bound: 143.7624373

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6649315, upper bound: 143.6590435
time: 7.26 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6417354, upper bound: 143.6417354
time: 4.89 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 12.30 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 12.30
Output dim: 4, lower bound: -143.6649315, upper bound: 143.6590435
IS_A2, status: Status.UNKNOWN, split count: 1, time: 12.30
Output dim: 4, lower bound: -143.6417354, upper bound: 143.6417354

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -75.7754288, 60.4595566, -76.0228043, 60.6544914, -136.4299011, 136.4823456
1: -62.7865219, 53.6056404, -62.9943390, 53.7792549, -116.5657806, 116.5999756
2: -83.1312408, 54.2274513, -83.4031906, 54.4029045, -137.5341492, 137.6306458
3: -88.4644012, 46.7679405, -88.7539597, 46.9208794, -135.3852844, 135.5218964
4: -81.7224503, 62.8103371, -81.9910431, 63.0132256, -144.7356415, 144.8013763
5: -72.9203491, 56.7705574, -73.1580811, 56.9551659, -129.8755188, 129.9286194
6: -70.0489120, 66.8218460, -70.2768860, 67.0410385, -137.0899506, 137.0987244
7: -75.7611084, 63.7964745, -76.0090866, 64.0053253, -139.7664185, 139.8055573
8: -91.7253265, 63.5594482, -92.0233536, 63.7623940, -155.4877167, 155.5827942
9: -69.1330185, 67.9258499, -69.3594666, 68.1468353, -137.2798462, 137.2852783

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=58, inp2_unstable=59, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 155

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5611779, upper bound: 143.5582645
time: 6.95 seconds

## Relational analysis of IS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6042710, upper bound: 143.6012072
time: 7.12 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6591245, upper bound: 143.6533346
time: 7.13 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -97.5423889, 77.7365036, -75.3719025, 60.1434021, -157.6857758, 153.1083984
1: -81.1014328, 68.8907394, -62.4460297, 53.3217430, -134.4231720, 131.3367615
2: -106.9554520, 69.4951782, -82.6866074, 53.9403191, -160.8957672, 152.1817932
3: -113.8497009, 60.1707802, -87.9906235, 46.5180473, -160.3677368, 148.1614075
4: -105.0295410, 80.6163635, -81.2843552, 62.4789162, -167.5084534, 161.9007263
5: -93.9048538, 72.9319458, -72.5331955, 56.4702110, -150.3750610, 145.4651489
6: -89.9386978, 86.1439285, -69.6767349, 66.4631271, -156.4018097, 155.8206635
7: -97.3581924, 82.0536880, -75.3552322, 63.4553146, -160.8135071, 157.4089050
8: -117.8584747, 81.4248352, -91.2375107, 63.2285652, -181.0870361, 172.6623077
9: -88.8761597, 87.3950424, -68.7620773, 67.5658798, -156.4420166, 156.1571198

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=58, inp2_unstable=59, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=143, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6043070, upper bound: 143.6060369
time: 5.39 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6360029, upper bound: 143.6360029
time: 4.80 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.82 seconds
IS_A1_A1, status: Status.VERIFIED, split count: 2, time: 24.82
Output dim: 4, lower bound: -143.6042710, upper bound: 143.6012072
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 24.82
Output dim: 4, lower bound: -143.6591245, upper bound: 143.6533346
IS_A2_A1, status: Status.VERIFIED, split count: 2, time: 24.82
Output dim: 4, lower bound: -143.6043070, upper bound: 143.6060369
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 24.82
Output dim: 4, lower bound: -143.6360029, upper bound: 143.6360029

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -69.6014862, 55.5721283, -75.0562973, 59.8893547, -129.4908447, 130.6284180
1: -57.5442352, 49.2468033, -62.1738701, 53.0968056, -110.6410370, 111.4206696
2: -76.2999649, 49.8429146, -82.3339157, 53.7164612, -130.0164185, 132.1768341
3: -81.2436676, 42.9487839, -87.6237564, 46.3229675, -127.5666351, 130.5725403
4: -75.0962753, 57.7355690, -80.9539642, 62.2187538, -137.3150330, 138.6895294
5: -66.9880066, 52.1712303, -72.2294388, 56.2351303, -123.2231369, 124.4006653
6: -64.3609085, 61.3479958, -69.3863754, 66.1842422, -130.5451508, 130.7343750
7: -69.5795670, 58.5862923, -75.0414658, 63.1896706, -132.7692108, 133.6277466
8: -84.2888260, 58.4792328, -90.8593063, 62.9672394, -147.2560577, 149.3385315
9: -63.4707680, 62.3343391, -68.4729309, 67.2716141, -130.7423706, 130.8072662

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=57, inp2_unstable=59, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=143, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=254, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5532604, upper bound: 143.5505580
time: 7.56 seconds

## Relational analysis of IS_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6218093, upper bound: 143.6167587
time: 7.05 seconds

## Relational analysis of IS_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6215558, upper bound: 143.6150368
time: 7.87 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6536557, upper bound: 143.6478474
time: 5.82 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -92.7847443, 73.9378281, -74.4557419, 59.4117165, -152.1964417, 148.3935699
1: -77.0551910, 65.5191956, -61.6666145, 52.6722412, -129.7274323, 127.1858063
2: -101.6757355, 66.1023636, -81.6698456, 53.2871246, -154.9628601, 147.7722168
3: -108.2916336, 57.2319260, -86.9202042, 45.9520912, -154.2437134, 144.1521301
4: -99.8842316, 76.6764984, -80.2934647, 61.7204514, -161.6046753, 156.9699707
5: -89.3289337, 69.3778458, -71.6516953, 55.7859383, -145.1148682, 141.0295105
6: -85.5430298, 81.9075851, -68.8303680, 65.6469193, -151.1899414, 150.7379456
7: -92.5772018, 78.0383911, -74.4347992, 62.6819992, -155.2592010, 152.4731903
8: -112.0712662, 77.4210510, -90.1228485, 62.4574814, -174.5287170, 167.5438843
9: -84.4948578, 83.0564651, -67.9184189, 66.7302933, -151.2251587, 150.9748840

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=57, inp2_unstable=59, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=150, inp2_unstable=143, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_A2_A1

### Relational analysis result of IS_A2_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6080573, upper bound: 143.6071179
time: 6.31 seconds

## Relational analysis of IS_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6046223, upper bound: 143.6046223
time: 5.73 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 28.82 seconds
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 28.82
Output dim: 4, lower bound: -143.6215558, upper bound: 143.6150368
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 28.82
Output dim: 4, lower bound: -143.6536557, upper bound: 143.6478474
IS_A2_A2_A1, status: Status.VERIFIED, split count: 3, time: 28.82
Output dim: 4, lower bound: -143.6080573, upper bound: 143.6071179
IS_A2_A2_A2, status: Status.VERIFIED, split count: 3, time: 28.82
Output dim: 4, lower bound: -143.6046223, upper bound: 143.6046223

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -60.1921425, 48.0266457, -54.1093216, 43.0580673, -103.2502136, 102.1359711
1: -49.5463829, 42.5732384, -44.3531303, 38.2283821, -87.7747650, 86.9263687
2: -65.8350601, 43.1281242, -59.0175743, 38.7437057, -104.5787659, 102.1456985
3: -70.2417374, 37.1330986, -63.1709328, 33.3657455, -103.6074753, 100.3040314
4: -64.8974838, 49.9110641, -58.2614861, 44.7559204, -109.6534042, 108.1725464
5: -57.9308662, 45.1230888, -52.0762177, 40.5297394, -98.4606018, 97.1993027
6: -55.6722679, 52.9649506, -50.0372009, 47.5170021, -103.1892700, 103.0021515
7: -60.1176567, 50.6422882, -53.9902649, 45.4923630, -105.6100159, 104.6325531
8: -72.7970810, 50.4812851, -65.2306747, 45.0668983, -117.8639755, 115.7119598
9: -54.8002014, 53.7192802, -49.1485901, 48.0444603, -102.8446579, 102.8678589

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=57, inp2_unstable=58, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=143, inp2_unstable=146, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=247, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_A2_B1_A1

### Relational analysis result of IS_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6142211, upper bound: 143.6088700
time: 7.47 seconds

## Relational analysis of IS_A1_A2_B1_A2

### Relational analysis result of IS_A1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6142211, upper bound: 143.6150368
time: 7.53 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -68.6913910, 54.8447533, -70.2233658, 56.0304489, -124.7218323, 125.0681152
1: -56.7701378, 48.6021652, -58.0620766, 49.6714935, -106.4416351, 106.6642227
2: -75.2887421, 49.1937599, -76.9707794, 50.2709312, -125.5596771, 126.1645279
3: -80.1793442, 42.3870239, -81.9777756, 43.3372650, -123.5166092, 124.3647842
4: -74.1113510, 56.9816933, -75.7273102, 58.2185173, -132.3298645, 132.7089996
5: -66.1118317, 51.4911728, -67.5798874, 52.6260071, -118.7378387, 119.0710602
6: -63.5200577, 60.5366554, -64.9224091, 61.8794899, -125.3995514, 125.4590607
7: -68.6644058, 57.8179817, -70.1859283, 59.1108818, -127.7752838, 128.0039062
8: -83.1810226, 57.7130318, -84.9802246, 58.9008827, -142.0819092, 142.6932526
9: -62.6321259, 61.5033379, -64.0223846, 62.8635902, -125.4957123, 125.5257263

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=57, inp2_unstable=58, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=143, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=254, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6226807, upper bound: 143.6194041
time: 8.25 seconds

## Relational analysis of IS_A1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6226807, upper bound: 143.6478474
time: 7.92 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 19.89 seconds
IS_A1_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 19.89
Output dim: 4, lower bound: -143.6142211, upper bound: 143.6088700
IS_A1_A2_B1_A2, status: Status.VERIFIED, split count: 4, time: 19.89
Output dim: 4, lower bound: -143.6142211, upper bound: 143.6150368
IS_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 19.89
Output dim: 4, lower bound: -143.6226807, upper bound: 143.6194041
IS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 19.89
Output dim: 4, lower bound: -143.6226807, upper bound: 143.6478474

## BFS IS instance: IS_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -48.9253693, 38.9495392, -70.2233658, 56.0304489, -104.9558182, 109.1728973
1: -39.9948921, 34.5751038, -58.0620766, 49.6714935, -89.6663818, 92.6371765
2: -53.2756615, 35.0823135, -76.9707794, 50.2709312, -103.5465927, 112.0530930
3: -57.0935478, 30.1711178, -81.9777756, 43.3372650, -100.4308167, 112.1488953
4: -52.7179947, 40.4957924, -75.7273102, 58.2185173, -110.9365082, 116.2230988
5: -47.1085320, 36.6577263, -67.5798874, 52.6260071, -99.7345428, 104.2376099
6: -45.2901878, 42.9477539, -64.9224091, 61.8794899, -107.1696777, 107.8701630
7: -48.7971802, 41.1083984, -70.1859283, 59.1108818, -107.9080505, 111.2943268
8: -58.9783325, 40.8071785, -84.9802246, 58.9008827, -117.8792114, 125.7873993
9: -44.4083176, 43.3711090, -64.0223846, 62.8635902, -107.2719116, 107.3934937

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=58, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=146, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=241, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 49

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5830405, upper bound: 143.5892827
time: 7.78 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6086069, upper bound: 143.6134763
time: 7.08 seconds

## BFS IS instance: IS_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -64.7955780, 51.7317848, -70.2233658, 56.0304489, -120.8260040, 121.9551544
1: -53.4576569, 45.8433304, -58.0620766, 49.6714935, -103.1291504, 103.9054108
2: -70.9617386, 46.4162064, -76.9707794, 50.2709312, -121.2326660, 123.3869858
3: -75.6238785, 39.9820251, -81.9777756, 43.3372650, -118.9611435, 121.9597931
4: -69.8961029, 53.7560272, -75.7273102, 58.2185173, -128.1146240, 129.4833374
5: -62.3621979, 48.5795937, -67.5798874, 52.6260071, -114.9882050, 116.1594849
6: -59.9222946, 57.0658646, -64.9224091, 61.8794899, -121.8017807, 121.9882660
7: -64.7473068, 54.5286636, -70.1859283, 59.1108818, -123.8581848, 124.7145920
8: -78.4421463, 54.4347458, -84.9802246, 58.9008827, -137.3430328, 139.4149780
9: -59.0426178, 57.9482040, -64.0223846, 62.8635902, -121.9062042, 121.9705887

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=58, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=143, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=253, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.5897358, upper bound: 143.6260278
time: 8.44 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5825059, upper bound: 143.6181681
time: 8.71 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 40.19 seconds
IS_A1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 40.19
Output dim: 4, lower bound: -143.5830405, upper bound: 143.5892827
IS_A1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 40.19
Output dim: 4, lower bound: -143.6086069, upper bound: 143.6134763
IS_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 40.19
Output dim: 4, lower bound: -143.5897358, upper bound: 143.6260278
IS_A1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 40.19
Output dim: 4, lower bound: -143.5825059, upper bound: 143.6181681

## BFS IS instance: IS_A1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -62.2472801, 49.7036285, -57.6382790, 46.0151749, -108.2624512, 107.3418961
1: -51.3141098, 44.0522003, -47.4716949, 40.8250618, -92.1391754, 91.5238953
2: -68.1471100, 44.6006966, -63.0678406, 41.3031883, -109.4503021, 107.6685181
3: -72.6302414, 38.4240532, -67.1931992, 35.6435051, -108.2737350, 105.6172409
4: -67.1568222, 51.6689262, -62.1924133, 47.9130211, -115.0698242, 113.8613434
5: -59.9010963, 46.6725502, -55.4231682, 43.2120056, -103.1130753, 102.0957184
6: -57.5709076, 54.8162079, -53.3056717, 50.7634811, -108.3343887, 108.1218796
7: -62.1968994, 52.3752441, -57.5917969, 48.4770432, -110.6739426, 109.9670410
8: -75.3762741, 52.3408470, -69.8353806, 48.5643578, -123.9406281, 122.1762238
9: -56.7151070, 55.6637077, -52.5275879, 51.5761986, -108.2913055, 108.1912994

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=57, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=143, inp2_unstable=142, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=250, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 107

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6246378, upper bound: 143.6181681
time: 7.85 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6246378, upper bound: 143.6181681
time: 8.20 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 19.82 seconds
IS_A1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 19.82
Output dim: 4, lower bound: -143.6246378, upper bound: 143.6181681
IS_A1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 19.82
Output dim: 4, lower bound: -143.6246378, upper bound: 143.6181681

## BFS IS instance: IS_A1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -52.4100266, 41.8666000, -57.6382790, 46.0151749, -98.4252014, 99.5048752
1: -43.0761375, 37.1460266, -47.4716949, 40.8250618, -83.9011993, 84.6177216
2: -57.2795563, 37.5978012, -63.0678406, 41.3031883, -98.5827484, 100.6656418
3: -61.0660591, 32.4149246, -67.1931992, 35.6435051, -96.7095490, 99.6081085
4: -56.5757446, 43.6117973, -62.1924133, 47.9130211, -104.4887619, 105.8042068
5: -50.4002914, 39.3099976, -55.4231682, 43.2120056, -93.6122818, 94.7331696
6: -48.4932442, 46.1533203, -53.3056717, 50.7634811, -99.2567291, 99.4589920
7: -52.3478088, 44.0692749, -57.5917969, 48.4770432, -100.8248520, 101.6610718
8: -63.5343361, 44.2504311, -69.8353806, 48.5643578, -112.0986938, 114.0858154
9: -47.7454872, 46.8593788, -52.5275879, 51.5761986, -99.3216858, 99.3869629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=57, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=142, inp2_unstable=142, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=244, inp2_unstable=250, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5865119, upper bound: 143.5819898
time: 7.80 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5900305, upper bound: 143.5844942
time: 8.18 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6263022, upper bound: 143.6207308
time: 6.85 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -49.3865967, 39.4502754, -57.6382790, 46.0151749, -95.4017715, 97.0885468
1: -40.5630302, 35.0370331, -47.4716949, 40.8250618, -81.3880844, 82.5087280
2: -53.9154205, 35.4234886, -63.0678406, 41.3031883, -95.2186127, 98.4913254
3: -57.5452614, 30.6224766, -67.1931992, 35.6435051, -93.1887665, 97.8156738
4: -53.3789139, 41.1305809, -62.1924133, 47.9130211, -101.2919235, 103.3229904
5: -47.5030251, 37.0650330, -55.4231682, 43.2120056, -90.7150192, 92.4881973
6: -45.7314301, 43.5351028, -53.3056717, 50.7634811, -96.4949112, 96.8407669
7: -49.3551788, 41.5209198, -57.5917969, 48.4770432, -97.8322220, 99.1127167
8: -59.9189529, 41.7493744, -69.8353806, 48.5643578, -108.4833069, 111.5847549
9: -45.0488167, 44.1828194, -52.5275879, 51.5761986, -96.6250153, 96.7104034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=57, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=146, inp2_unstable=142, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=241, inp2_unstable=250, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5865119, upper bound: 143.5819898
time: 8.23 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5900305, upper bound: 143.5844942
time: 8.91 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6263022, upper bound: 143.6207308
time: 6.51 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 46.22 seconds
IS_A1_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 7, time: 46.22
Output dim: 4, lower bound: -143.5900305, upper bound: 143.5844942
IS_A1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 46.22
Output dim: 4, lower bound: -143.6263022, upper bound: 143.6207308
IS_A1_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 46.22
Output dim: 4, lower bound: -143.5900305, upper bound: 143.5844942
IS_A1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 46.22
Output dim: 4, lower bound: -143.6263022, upper bound: 143.6207308

## BFS IS instance: IS_A1_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -51.5796356, 41.2058372, -53.7174149, 42.8914986, -94.4711304, 94.9232330
1: -42.3723373, 36.5594673, -44.1440277, 38.0519714, -80.4243088, 80.7034912
2: -56.3609505, 37.0142937, -58.7353630, 38.5423241, -94.9032745, 95.7496338
3: -60.1019325, 31.9029083, -62.6337204, 33.2080994, -93.3100281, 94.5366287
4: -55.6826210, 42.9229774, -57.9612160, 44.6611633, -100.3437729, 100.8841858
5: -49.6057892, 38.6919327, -51.6681786, 40.2961655, -89.9019547, 90.3600845
6: -47.7285881, 45.4170570, -49.6778374, 47.2833099, -95.0118866, 95.0948944
7: -51.5163727, 43.3731918, -53.6615829, 45.1887932, -96.7051620, 97.0347519
8: -62.5217285, 43.5426941, -65.0578842, 45.2283325, -107.7500610, 108.6005783
9: -46.9862099, 46.1074448, -48.9375000, 48.0245438, -95.0107574, 95.0449448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=56, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=142, inp2_unstable=141, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=243, inp2_unstable=248, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 107

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6154372, upper bound: 143.6112682
time: 7.30 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6154372, upper bound: 143.6318373
time: 7.25 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -48.5937805, 38.8181725, -53.7174149, 42.8914986, -91.4852753, 92.5355835
1: -39.8905258, 34.4759903, -44.1440277, 38.0519714, -77.9424973, 78.6200180
2: -53.0376625, 34.8658371, -58.7353630, 38.5423241, -91.5799866, 93.6011810
3: -56.6258011, 30.1339054, -62.6337204, 33.2080994, -89.8339005, 92.7676163
4: -52.5275803, 40.4718018, -57.9612160, 44.6611633, -97.1887360, 98.4330063
5: -46.7441483, 36.4738045, -51.6681786, 40.2961655, -87.0403061, 88.1419754
6: -45.0008698, 42.8316689, -49.6778374, 47.2833099, -92.2841797, 92.5095062
7: -48.5622406, 40.8546677, -53.6615829, 45.1887932, -93.7510376, 94.5162430
8: -58.9503937, 41.0730629, -65.0578842, 45.2283325, -104.1787262, 106.1309357
9: -44.3229332, 43.4647255, -48.9375000, 48.0245438, -92.3474731, 92.4022217

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=56, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=145, inp2_unstable=141, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=241, inp2_unstable=248, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5859774, upper bound: 143.5830211
time: 7.81 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.5859774, upper bound: 143.6207308
time: 8.34 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 29.80 seconds
IS_A1_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 8, time: 29.80
Output dim: 4, lower bound: -143.6154372, upper bound: 143.6112682
IS_A1_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 29.80
Output dim: 4, lower bound: -143.6154372, upper bound: 143.6318373
IS_A1_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 29.80
Output dim: 4, lower bound: -143.5859774, upper bound: 143.5830211
IS_A1_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 29.80
Output dim: 4, lower bound: -143.5859774, upper bound: 143.6207308

## BFS IS instance: IS_A1_A2_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -48.6060562, 38.8379021, -53.7174149, 42.8914986, -91.4975510, 92.5553055
1: -39.8501663, 34.4563637, -44.1440277, 38.0519714, -77.9021378, 78.6003799
2: -53.0693130, 34.9242706, -58.7353630, 38.5423241, -91.6116333, 93.6596298
3: -56.6526566, 30.0714684, -62.6337204, 33.2080994, -89.8607559, 92.7051773
4: -52.4883156, 40.4533119, -57.9612160, 44.6611633, -97.1494675, 98.4145203
5: -46.7606239, 36.4757004, -51.6681786, 40.2961655, -87.0567780, 88.1438675
6: -44.9910164, 42.7791290, -49.6778374, 47.2833099, -92.2743225, 92.4569702
7: -48.5422668, 40.8767815, -53.6615829, 45.1887932, -93.7310638, 94.5383606
8: -58.8920021, 41.0069656, -65.0578842, 45.2283325, -104.1203308, 106.0648499
9: -44.2650642, 43.4144592, -48.9375000, 48.0245438, -92.2896042, 92.3519592

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=56, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=140, inp2_unstable=141, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=242, inp2_unstable=248, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6094533, upper bound: 143.6299096
time: 7.26 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6073386, upper bound: 143.6291295
time: 8.04 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -45.7414055, 36.5441017, -53.7174149, 42.8914986, -88.6329041, 90.2615128
1: -37.4723473, 32.4578171, -44.1440277, 38.0519714, -75.5243225, 76.6018295
2: -49.8767242, 32.8612862, -58.7353630, 38.5423241, -88.4190521, 91.5966339
3: -53.3171234, 28.3745461, -62.6337204, 33.2080994, -86.5252228, 91.0082474
4: -49.4631157, 38.0994072, -57.9612160, 44.6611633, -94.1242828, 96.0606155
5: -44.0129509, 34.3447800, -51.6681786, 40.2961655, -84.3091049, 86.0129471
6: -42.3765144, 40.2961197, -49.6778374, 47.2833099, -89.6598053, 89.9739532
7: -45.7069893, 38.4557228, -53.6615829, 45.1887932, -90.8957825, 92.1173096
8: -55.4664497, 38.6428375, -65.0578842, 45.2283325, -100.6947708, 103.7007141
9: -41.7156143, 40.8792725, -48.9375000, 48.0245438, -89.7401581, 89.8167725

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=56, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=141, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=238, inp2_unstable=248, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.5792537, upper bound: 143.6192116
time: 7.06 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.5784785, upper bound: 143.6190814
time: 7.53 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 39.97 seconds
IS_A1_A2_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 39.97
Output dim: 4, lower bound: -143.6094533, upper bound: 143.6299096
IS_A1_A2_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 39.97
Output dim: 4, lower bound: -143.6073386, upper bound: 143.6291295
IS_A1_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 39.97
Output dim: 4, lower bound: -143.5792537, upper bound: 143.6192116
IS_A1_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 39.97
Output dim: 4, lower bound: -143.5784785, upper bound: 143.6190814

## BFS IS instance: IS_A1_A2_B2_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -46.1245193, 36.8651047, -47.2441216, 37.7457848, -83.8702927, 84.1092148
1: -37.7774048, 32.7082253, -38.7313232, 33.4896126, -71.2670135, 71.4395447
2: -50.3197594, 33.1623993, -51.5665398, 33.9424400, -84.2621994, 84.7289429
3: -53.7177238, 28.5606709, -54.9725685, 29.2602215, -82.9779434, 83.5332260
4: -49.8040504, 38.4064827, -50.9525070, 39.3241577, -89.1281891, 89.3589706
5: -44.3674049, 34.6048622, -45.4236870, 35.4227524, -79.7901611, 80.0285339
6: -42.7047882, 40.5870209, -43.7025108, 41.5654259, -84.2702179, 84.2895355
7: -46.0361786, 38.7668686, -47.1220551, 39.6931534, -85.7293320, 85.8889160
8: -55.9005623, 38.9793892, -57.2579765, 39.9359207, -95.8364868, 96.2373581
9: -41.9949646, 41.1811829, -43.0165558, 42.2014503, -84.1964111, 84.1977386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=55, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=137, inp2_unstable=136, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=240, inp2_unstable=242, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 97

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_A2_B2_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6351532, upper bound: 143.6289552
time: 7.15 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_A2_B2_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6351532, upper bound: 143.6291295
time: 7.81 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -46.8188171, 37.4156837, -51.6063576, 41.1996994, -88.0185165, 89.0220261
1: -38.3574066, 33.1970329, -42.3659592, 36.5583687, -74.9157715, 75.5629883
2: -51.0885391, 33.6527176, -56.3907814, 37.0173569, -88.1058960, 90.0434952
3: -54.5414047, 28.9828091, -60.1477509, 31.9251728, -86.4665756, 89.1305618
4: -50.5574760, 38.9793320, -55.6821709, 42.9123993, -93.4698792, 94.6614990
5: -45.0378036, 35.1253281, -49.6265526, 38.6712799, -83.7090836, 84.7518768
6: -43.3463631, 41.2006493, -47.7560081, 45.4170227, -88.7633820, 88.9566422
7: -46.7391472, 39.3568077, -51.5199890, 43.3656387, -90.1047821, 90.8767929
8: -56.7392311, 39.5461731, -62.5396614, 43.5086784, -100.2479019, 102.0858307
9: -42.6311188, 41.8057861, -46.9965286, 46.1142769, -88.7453918, 88.8023148

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=55, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=137, inp2_unstable=142, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=240, inp2_unstable=245, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6351532, upper bound: 143.6289552
time: 7.78 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6351532, upper bound: 143.6291295
time: 7.07 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -43.2399445, 34.5522118, -47.2441216, 37.7457848, -80.9857330, 81.7963333
1: -35.3819504, 30.6967182, -38.7313232, 33.4896126, -68.8715591, 69.4280396
2: -47.0997391, 31.0885563, -51.5665398, 33.9424400, -81.0421753, 82.6550980
3: -50.3588676, 26.8540096, -54.9725685, 29.2602215, -79.6190872, 81.8265610
4: -46.7536278, 36.0379524, -50.9525070, 39.3241577, -86.0777817, 86.9904556
5: -41.5977974, 32.4550209, -45.4236870, 35.4227524, -77.0205536, 77.8787079
6: -40.0709572, 38.0890045, -43.7025108, 41.5654259, -81.6363831, 81.7915192
7: -43.1736069, 36.3281631, -47.1220551, 39.6931534, -82.8667450, 83.4502182
8: -52.4460106, 36.5999641, -57.2579765, 39.9359207, -92.3819275, 93.8579407
9: -39.4283257, 38.6300430, -43.0165558, 42.2014503, -81.6297760, 81.6465988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=55, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=136, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=233, inp2_unstable=242, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 155

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_A2_B2_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6234197, upper bound: 143.6179899
time: 7.56 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_A2_B2_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6234197, upper bound: 143.6190814
time: 7.48 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 30.91 seconds
IS_A1_A2_B2_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 10, time: 30.91
Output dim: 4, lower bound: -143.6351532, upper bound: 143.6289552
IS_A1_A2_B2_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 30.91
Output dim: 4, lower bound: -143.6351532, upper bound: 143.6291295
IS_A1_A2_B2_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 10, time: 30.91
Output dim: 4, lower bound: -143.6351532, upper bound: 143.6289552
IS_A1_A2_B2_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 30.91
Output dim: 4, lower bound: -143.6351532, upper bound: 143.6291295
IS_A1_A2_B2_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 10, time: 30.91
Output dim: 4, lower bound: -143.6234197, upper bound: 143.6179899
IS_A1_A2_B2_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 30.91
Output dim: 4, lower bound: -143.6234197, upper bound: 143.6190814
IS_A1_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 30.91
Output dim: 4, lower bound: -143.5784785, upper bound: 143.6190814

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 11.73 + 595.44 = 607.16 seconds
