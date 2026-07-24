## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 143.61867486269998


## IAR start

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
execution time: IAR + RelationalAnalysis = 0.87 + 10.10 = 10.97 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -143.7624373, upper bound: 143.7624373

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6649315, upper bound: 143.6590435
time: 7.10 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6417354, upper bound: 143.6417354
time: 4.72 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 11.90 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 11.90
Output dim: 4, lower bound: -143.6649315, upper bound: 143.6590435
NS_A2, status: Status.UNKNOWN, split count: 1, time: 11.90
Output dim: 4, lower bound: -143.6417354, upper bound: 143.6417354

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5611779, upper bound: 143.5582645
time: 6.88 seconds

## Relational analysis of NS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6281070, upper bound: 143.6230854
time: 8.47 seconds

## Relational analysis of NS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6433969, upper bound: 143.6382785
time: 8.03 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6371482, upper bound: 143.6307675
time: 7.11 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6043070, upper bound: 143.6060369
time: 5.51 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6360029, upper bound: 143.6360029
time: 4.61 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 21.37 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 21.37
Output dim: 4, lower bound: -143.6433969, upper bound: 143.6382785
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 21.37
Output dim: 4, lower bound: -143.6371482, upper bound: 143.6307675
NS_A2_A1, status: Status.VERIFIED, split count: 2, time: 21.37
Output dim: 4, lower bound: -143.6043070, upper bound: 143.6060369
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 21.37
Output dim: 4, lower bound: -143.6360029, upper bound: 143.6360029

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -73.1887054, 58.4024124, -63.3361092, 50.5659752, -123.7546768, 121.7385254
1: -60.6066475, 51.7852783, -52.3021126, 44.8536224, -105.4602661, 104.0873795
2: -80.2745285, 52.3839226, -69.3896103, 45.3595428, -125.6340714, 121.7735214
3: -85.4281769, 45.1847534, -73.8587341, 39.1610374, -124.5892105, 119.0434875
4: -78.9420166, 60.6921005, -68.3463745, 52.6280136, -131.5700378, 129.0384521
5: -70.4220963, 54.8367004, -60.9037361, 47.4708176, -117.8929138, 115.7404327
6: -67.6599960, 64.5350800, -58.5599480, 55.8253784, -123.4853745, 123.0950241
7: -73.1732635, 61.6124496, -63.3172226, 53.2898560, -126.4631195, 124.9296570
8: -88.6111374, 61.4341278, -76.7509613, 53.3445740, -141.9556885, 138.1850586
9: -66.7699127, 65.6064072, -57.7709007, 56.7680130, -123.5379181, 123.3772964

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 107

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6371482, upper bound: 143.6307675
time: 8.14 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6371482, upper bound: 143.6307675
time: 7.57 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -67.2843857, 53.7105026, -60.3147316, 48.1566544, -115.4410400, 114.0252380
1: -55.6351700, 47.6327477, -49.7530441, 42.7395401, -98.3747101, 97.3857803
2: -73.7522812, 48.1802025, -66.0325241, 43.1788940, -116.9311752, 114.2127151
3: -78.5066605, 41.5620346, -70.3337479, 37.3511238, -115.8577805, 111.8957672
4: -72.5993500, 55.8614197, -65.1410522, 50.1459618, -122.7453079, 121.0024643
5: -64.7299194, 50.4312134, -58.0015259, 45.2322235, -109.9621429, 108.4327393
6: -62.2188683, 59.3184280, -55.7842064, 53.1830788, -115.4019470, 115.1026306
7: -67.2688980, 56.6357193, -60.3232727, 50.7358780, -118.0047760, 116.9589920
8: -81.5095673, 56.5814590, -73.1486816, 50.8548470, -132.3644104, 129.7301331
9: -61.3850937, 60.3099022, -55.0624542, 54.0677643, -115.4528503, 115.3723602

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 97

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6371482, upper bound: 143.6307675
time: 7.03 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6371482, upper bound: 143.6307675
time: 7.39 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 114

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_A2_A1

### Relational analysis result of NS_A2_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6080573, upper bound: 143.6071179
time: 6.22 seconds

## Relational analysis of NS_A2_A2_A2

### Relational analysis result of NS_A2_A2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6046223, upper bound: 143.6046223
time: 5.57 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 12.66 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 12.66
Output dim: 4, lower bound: -143.6371482, upper bound: 143.6307675
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 12.66
Output dim: 4, lower bound: -143.6371482, upper bound: 143.6307675
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 12.66
Output dim: 4, lower bound: -143.6371482, upper bound: 143.6307675
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 12.66
Output dim: 4, lower bound: -143.6371482, upper bound: 143.6307675
NS_A2_A2_A1, status: Status.VERIFIED, split count: 3, time: 12.66
Output dim: 4, lower bound: -143.6080573, upper bound: 143.6071179
NS_A2_A2_A2, status: Status.VERIFIED, split count: 3, time: 12.66
Output dim: 4, lower bound: -143.6046223, upper bound: 143.6046223

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -63.0904427, 50.3723373, -63.3361092, 50.5659752, -113.6564178, 113.7084503
1: -52.0955849, 44.6813126, -52.3021126, 44.8536224, -96.9492035, 96.9834137
2: -69.1193008, 45.1851921, -69.3896103, 45.3595428, -114.4788437, 114.5747986
3: -73.5709991, 39.0092659, -73.8587341, 39.1610374, -112.7320251, 112.8679962
4: -68.0793457, 52.4264336, -68.3463745, 52.6280136, -120.7073593, 120.7728043
5: -60.6674805, 47.2874222, -60.9037361, 47.4708176, -108.1382980, 108.1911545
6: -58.3334427, 55.6075516, -58.5599480, 55.8253784, -114.1588135, 114.1674957
7: -63.0707817, 53.0822906, -63.3172226, 53.2898560, -116.3606262, 116.3994980
8: -76.4548416, 53.1430626, -76.7509613, 53.3445740, -129.7993774, 129.8940277
9: -57.5458794, 56.5483932, -57.7709007, 56.7680130, -114.3138885, 114.3192902

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 114

### Candidate
type: A, layer: 1, pos: 114

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6040887, upper bound: 143.5975187
time: 7.53 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6374784, upper bound: 143.6323003
time: 8.41 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -60.0731010, 47.9659500, -63.3361092, 50.5659752, -110.6390610, 111.3020630
1: -49.5509109, 42.5702400, -52.3021126, 44.8536224, -94.4045181, 94.8723526
2: -65.7671661, 43.0079918, -69.3896103, 45.3595428, -111.1267090, 112.3975983
3: -70.0504608, 37.2018280, -73.8587341, 39.1610374, -109.2114868, 111.0605621
4: -64.8787537, 49.9474792, -68.3463745, 52.6280136, -117.5067673, 118.2938385
5: -57.7691994, 45.0515289, -60.9037361, 47.4708176, -105.2400055, 105.9552612
6: -55.5617676, 52.9695053, -58.5599480, 55.8253784, -111.3871307, 111.5294495
7: -60.0811615, 50.5318680, -63.3172226, 53.2898560, -113.3710175, 113.8490753
8: -72.8579025, 50.6566162, -76.7509613, 53.3445740, -126.2024765, 127.4075775
9: -54.8414917, 53.8521233, -57.7709007, 56.7680130, -111.6094971, 111.6230240

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 97

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 114

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6040887, upper bound: 143.5975187
time: 6.80 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6374784, upper bound: 143.6323003
time: 7.83 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -63.0904427, 50.3723373, -60.3147316, 48.1566544, -111.2471008, 110.6870728
1: -52.0955849, 44.6813126, -49.7530441, 42.7395401, -94.8351288, 94.4343414
2: -69.1193008, 45.1851921, -66.0325241, 43.1788940, -112.2981949, 111.2177048
3: -73.5709991, 39.0092659, -70.3337479, 37.3511238, -110.9221115, 109.3430099
4: -68.0793457, 52.4264336, -65.1410522, 50.1459618, -118.2253036, 117.5674820
5: -60.6674805, 47.2874222, -58.0015259, 45.2322235, -105.8997040, 105.2889404
6: -58.3334427, 55.6075516, -55.7842064, 53.1830788, -111.5165253, 111.3917542
7: -63.0707817, 53.0822906, -60.3232727, 50.7358780, -113.8066483, 113.4055557
8: -76.4548416, 53.1430626, -73.1486816, 50.8548470, -127.3096848, 126.2917480
9: -57.5458794, 56.5483932, -55.0624542, 54.0677643, -111.6136475, 111.6108475

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 114

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of NS_A1_B2_A1_A1

### Relational analysis result of NS_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5992520, upper bound: 143.5956080
time: 7.36 seconds

## Relational analysis of NS_A1_B2_A1_A2

### Relational analysis result of NS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6310065, upper bound: 143.6246490
time: 6.95 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -60.0731010, 47.9659500, -60.3147316, 48.1566544, -108.2297516, 108.2806854
1: -49.5509109, 42.5702400, -49.7530441, 42.7395401, -92.2904434, 92.3232803
2: -65.7671661, 43.0079918, -66.0325241, 43.1788940, -108.9460602, 109.0405045
3: -70.0504608, 37.2018280, -70.3337479, 37.3511238, -107.4015732, 107.5355759
4: -64.8787537, 49.9474792, -65.1410522, 50.1459618, -115.0247116, 115.0885162
5: -57.7691994, 45.0515289, -58.0015259, 45.2322235, -103.0014191, 103.0530548
6: -55.5617676, 52.9695053, -55.7842064, 53.1830788, -108.7448425, 108.7537079
7: -60.0811615, 50.5318680, -60.3232727, 50.7358780, -110.8170395, 110.8551331
8: -72.8579025, 50.6566162, -73.1486816, 50.8548470, -123.7127533, 123.8052979
9: -54.8414917, 53.8521233, -55.0624542, 54.0677643, -108.9092484, 108.9145813

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 114

### Candidate
type: A, layer: 1, pos: 114

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5976753, upper bound: 143.5897620
time: 7.01 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6310065, upper bound: 143.6246490
time: 6.43 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 16.23 seconds
NS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 16.23
Output dim: 4, lower bound: -143.6040887, upper bound: 143.5975187
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.23
Output dim: 4, lower bound: -143.6374784, upper bound: 143.6323003
NS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 16.23
Output dim: 4, lower bound: -143.6040887, upper bound: 143.5975187
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.23
Output dim: 4, lower bound: -143.6374784, upper bound: 143.6323003
NS_A1_B2_A1_A1, status: Status.VERIFIED, split count: 4, time: 16.23
Output dim: 4, lower bound: -143.5992520, upper bound: 143.5956080
NS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 16.23
Output dim: 4, lower bound: -143.6310065, upper bound: 143.6246490
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 16.23
Output dim: 4, lower bound: -143.5976753, upper bound: 143.5897620
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.23
Output dim: 4, lower bound: -143.6310065, upper bound: 143.6246490

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -62.1838493, 49.6478195, -58.5778503, 46.7602081, -108.9440536, 108.2256470
1: -51.3243675, 44.0390854, -48.2647514, 41.4869423, -92.8112869, 92.3038330
2: -68.1119308, 44.5383720, -64.1071014, 41.9696465, -110.0815735, 108.6454697
3: -72.5105743, 38.4497566, -68.2942123, 36.2245102, -108.7350769, 106.7439651
4: -67.0977783, 51.6756210, -63.2004890, 48.6863976, -115.7841721, 114.8761139
5: -59.7947845, 46.6101913, -56.3263588, 43.9130287, -103.7077866, 102.9365463
6: -57.4957504, 54.7992897, -54.1702309, 51.5933990, -109.0891418, 108.9695206
7: -62.1589966, 52.3164902, -58.5329666, 49.2708473, -111.4298401, 110.8494568
8: -75.3511353, 52.3798752, -70.9659348, 49.3386116, -124.6897430, 123.3458099
9: -56.7103348, 55.7201729, -53.3881683, 52.4262581, -109.1365738, 109.1083374

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6176144, upper bound: 143.6140612
time: 8.27 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6176144, upper bound: 143.6431936
time: 8.01 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -59.1632080, 47.2370644, -58.5778503, 46.7602081, -105.9234085, 105.8148956
1: -48.7828674, 41.9275093, -48.2647514, 41.4869423, -90.2697906, 90.1922607
2: -64.7570724, 42.3609428, -64.1071014, 41.9696465, -106.7267151, 106.4680481
3: -68.9864655, 36.6403008, -68.2942123, 36.2245102, -105.2109756, 104.9345093
4: -63.8929634, 49.1941833, -63.2004890, 48.6863976, -112.5793610, 112.3946686
5: -56.8947144, 44.3714828, -56.3263588, 43.9130287, -100.8077393, 100.6978455
6: -54.7221909, 52.1626091, -54.1702309, 51.5933990, -106.3155823, 106.3328400
7: -59.1663818, 49.7649841, -58.5329666, 49.2708473, -108.4372253, 108.2979507
8: -71.7527618, 49.8888931, -70.9659348, 49.3386116, -121.0913696, 120.8548279
9: -54.0047264, 53.0260124, -53.3881683, 52.4262581, -106.4309769, 106.4141846

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6046808, upper bound: 143.6012106
time: 5.85 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6046808, upper bound: 143.6323003
time: 8.04 seconds

## BFS NS instance: NS_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -58.3348007, 46.5684128, -59.4037666, 47.4269676, -105.7617493, 105.9721832
1: -48.0612221, 41.3167114, -48.9837418, 42.0960045, -90.1572113, 90.3004532
2: -63.8399925, 41.7974396, -65.0213394, 42.5309639, -106.3709488, 106.8187714
3: -68.0093613, 36.0743752, -69.2685089, 36.7890358, -104.7983856, 105.3428802
4: -62.9367409, 48.4866333, -64.1543655, 49.3916740, -112.3284073, 112.6409988
5: -56.0926514, 43.7313461, -57.1259422, 44.5512695, -100.6439209, 100.8572845
6: -53.9464836, 51.3785782, -54.9437714, 52.3751373, -106.3216248, 106.3223495
7: -58.2891808, 49.0654221, -59.4074593, 49.9679146, -108.2570877, 108.4728851
8: -70.6732559, 49.1390533, -72.0422440, 50.0863342, -120.7595901, 121.1812897
9: -53.1658020, 52.2091446, -54.2245941, 53.2402382, -106.4060364, 106.4337387

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of NS_A1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5986425, upper bound: 143.5905501
time: 7.80 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.5986425, upper bound: 143.6253502
time: 7.68 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -59.1632080, 47.2370644, -55.5232658, 44.3190117, -103.4822006, 102.7603073
1: -48.7828674, 41.9275093, -45.7112160, 39.3542480, -88.1370926, 87.6387253
2: -64.7570724, 42.3609428, -60.7119102, 39.7697449, -104.5268097, 103.0728531
3: -68.9864655, 36.6403008, -64.7278824, 34.3945503, -103.3810043, 101.3681793
4: -63.8929634, 49.1941833, -59.9514809, 46.1770935, -110.0700455, 109.1456604
5: -56.8947144, 44.3714828, -53.3938026, 41.6512680, -98.5459824, 97.7652893
6: -54.7221909, 52.1626091, -51.3613815, 48.9347954, -103.6569748, 103.5239868
7: -59.1663818, 49.7649841, -55.5041428, 46.6962357, -105.8626022, 105.2691116
8: -71.7527618, 49.8888931, -67.3236008, 46.8094559, -118.5622177, 117.2124939
9: -54.0047264, 53.0260124, -50.6578598, 49.7145691, -103.7192841, 103.6838684

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 97

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5992520, upper bound: 143.5956080
time: 5.89 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.5992520, upper bound: 143.6246490
time: 7.90 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 14.67 seconds
NS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 14.67
Output dim: 4, lower bound: -143.6176144, upper bound: 143.6140612
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 4, lower bound: -143.6176144, upper bound: 143.6431936
NS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 14.67
Output dim: 4, lower bound: -143.6046808, upper bound: 143.6012106
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 4, lower bound: -143.6046808, upper bound: 143.6323003
NS_A1_B2_A1_A2_B1, status: Status.VERIFIED, split count: 5, time: 14.67
Output dim: 4, lower bound: -143.5986425, upper bound: 143.5905501
NS_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 4, lower bound: -143.5986425, upper bound: 143.6253502
NS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 14.67
Output dim: 4, lower bound: -143.5992520, upper bound: 143.5956080
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 4, lower bound: -143.5992520, upper bound: 143.6246490

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -58.3348007, 46.5684128, -58.5778503, 46.7602081, -105.0950089, 105.1462479
1: -48.0612221, 41.3167114, -48.2647514, 41.4869423, -89.5481567, 89.5814667
2: -63.8399925, 41.7974396, -64.1071014, 41.9696465, -105.8096390, 105.9045334
3: -68.0093613, 36.0743752, -68.2942123, 36.2245102, -104.2338638, 104.3685913
4: -62.9367409, 48.4866333, -63.2004890, 48.6863976, -111.6231232, 111.6871185
5: -56.0926514, 43.7313461, -56.3263588, 43.9130287, -100.0056763, 100.0577011
6: -53.9464836, 51.3785782, -54.1702309, 51.5933990, -105.5398865, 105.5488052
7: -58.2891808, 49.0654221, -58.5329666, 49.2708473, -107.5600281, 107.5983887
8: -70.6732559, 49.1390533, -70.9659348, 49.3386116, -120.0118713, 120.1049881
9: -53.1658020, 52.2091446, -53.3881683, 52.4262581, -105.5920563, 105.5973129

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 114

### Candidate
type: A, layer: 1, pos: 114

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5788108, upper bound: 143.6160673
time: 7.51 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6037809, upper bound: 143.6378023
time: 8.17 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -55.2852898, 44.1310692, -58.5778503, 46.7602081, -102.0454941, 102.7089005
1: -45.5127487, 39.1875381, -48.2647514, 41.4869423, -86.9996719, 87.4522858
2: -60.4503479, 39.6013908, -64.1071014, 41.9696465, -102.4199829, 103.7084885
3: -64.4486008, 34.2476234, -68.2942123, 36.2245102, -100.6731110, 102.5418243
4: -59.6930046, 45.9816208, -63.2004890, 48.6863976, -108.3793945, 109.1821136
5: -53.1648216, 41.4733047, -56.3263588, 43.9130287, -97.0778427, 97.7996674
6: -51.1421280, 48.7246971, -54.1702309, 51.5933990, -102.7355270, 102.8949280
7: -55.2654152, 46.4955292, -58.5329666, 49.2708473, -104.5362473, 105.0284958
8: -67.0368423, 46.6139221, -70.9659348, 49.3386116, -116.3754578, 117.5798569
9: -50.4405251, 49.5024757, -53.3881683, 52.4262581, -102.8667831, 102.8906403

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 97

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 114

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5526807, upper bound: 143.5914390
time: 6.59 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.5917679, upper bound: 143.6270754
time: 6.54 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -58.3348007, 46.5684128, -55.5232658, 44.3190117, -102.6538010, 102.0916595
1: -48.0612221, 41.3167114, -45.7112160, 39.3542480, -87.4154587, 87.0279236
2: -63.8399925, 41.7974396, -60.7119102, 39.7697449, -103.6097412, 102.5093536
3: -68.0093613, 36.0743752, -64.7278824, 34.3945503, -102.4038925, 100.8022614
4: -62.9367409, 48.4866333, -59.9514809, 46.1770935, -109.1138077, 108.4381104
5: -56.0926514, 43.7313461, -53.3938026, 41.6512680, -97.7439194, 97.1251450
6: -53.9464836, 51.3785782, -51.3613815, 48.9347954, -102.8812790, 102.7399445
7: -58.2891808, 49.0654221, -55.5041428, 46.6962357, -104.9854050, 104.5695572
8: -70.6732559, 49.1390533, -67.3236008, 46.8094559, -117.4827118, 116.4626541
9: -53.1658020, 52.2091446, -50.6578598, 49.7145691, -102.8803558, 102.8670044

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 97

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 114

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_A1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5563139, upper bound: 143.5862295
time: 7.66 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.5932657, upper bound: 143.6200050
time: 8.63 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -55.2852898, 44.1310692, -55.5232658, 44.3190117, -99.6042862, 99.6543121
1: -45.5127487, 39.1875381, -45.7112160, 39.3542480, -84.8669815, 84.8987579
2: -60.4503479, 39.6013908, -60.7119102, 39.7697449, -100.2200775, 100.3133011
3: -64.4486008, 34.2476234, -64.7278824, 34.3945503, -98.8431320, 98.9755020
4: -59.6930046, 45.9816208, -59.9514809, 46.1770935, -105.8700790, 105.9331055
5: -53.1648216, 41.4733047, -53.3938026, 41.6512680, -94.8160858, 94.8671112
6: -51.1421280, 48.7246971, -51.3613815, 48.9347954, -100.0769196, 100.0860748
7: -55.2654152, 46.4955292, -55.5041428, 46.6962357, -101.9616165, 101.9996567
8: -67.0368423, 46.6139221, -67.3236008, 46.8094559, -113.8462982, 113.9375229
9: -50.4405251, 49.5024757, -50.6578598, 49.7145691, -100.1550903, 100.1603394

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 97

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 114

### Candidate
type: A, layer: 1, pos: 114

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 120

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.5876240, upper bound: 143.6226009
time: 7.03 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5885757, upper bound: 143.5821168
time: 7.78 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 28.58 seconds
NS_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 28.58
Output dim: 4, lower bound: -143.5788108, upper bound: 143.6160673
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 28.58
Output dim: 4, lower bound: -143.6037809, upper bound: 143.6378023
NS_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 28.58
Output dim: 4, lower bound: -143.5526807, upper bound: 143.5914390
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 28.58
Output dim: 4, lower bound: -143.5917679, upper bound: 143.6270754
NS_A1_B2_A1_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 28.58
Output dim: 4, lower bound: -143.5563139, upper bound: 143.5862295
NS_A1_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 28.58
Output dim: 4, lower bound: -143.5932657, upper bound: 143.6200050
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 28.58
Output dim: 4, lower bound: -143.5876240, upper bound: 143.6226009
NS_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 28.58
Output dim: 4, lower bound: -143.5885757, upper bound: 143.5821168

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -57.4789352, 45.8866234, -54.6469078, 43.6286697, -101.1076050, 100.5335312
1: -47.3330612, 40.7110252, -44.9252586, 38.7061081, -86.0391693, 85.6362839
2: -62.8946495, 41.1940002, -59.7642975, 39.2005615, -102.0952072, 100.9582977
3: -67.0137405, 35.5430031, -63.7217712, 33.7825584, -100.7962952, 99.2647705
4: -62.0139656, 47.7761459, -58.9592896, 45.4252815, -107.4392319, 106.7354355
5: -55.2725220, 43.0943718, -52.5612221, 40.9888039, -96.2613144, 95.6555939
6: -53.1548958, 50.6182709, -50.5335503, 48.1030273, -101.2579193, 101.1518250
7: -57.4314461, 48.3466187, -54.5935402, 45.9723778, -103.4038162, 102.9401550
8: -69.6302490, 48.4111748, -66.1762772, 45.9945908, -115.6248245, 114.5874481
9: -52.3815727, 51.4322357, -49.7878761, 48.8629150, -101.2444839, 101.2201080

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 107

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6218011, upper bound: 143.6176785
time: 7.55 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6218011, upper bound: 143.6378023
time: 8.84 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -54.4708977, 43.4813881, -54.6469078, 43.6286697, -98.0995636, 98.1282959
1: -44.8225670, 38.6117783, -44.9252586, 38.7061081, -83.5286713, 83.5370331
2: -59.5500526, 39.0274429, -59.7642975, 39.2005615, -98.7506104, 98.7917404
3: -63.5016518, 33.7427864, -63.7217712, 33.7825584, -97.2842102, 97.4645538
4: -58.8150673, 45.3063965, -58.9592896, 45.4252815, -104.2403412, 104.2656860
5: -52.3845062, 40.8672981, -52.5612221, 40.9888039, -93.3733063, 93.4285202
6: -50.3885193, 48.0024071, -50.5335503, 48.1030273, -98.4915314, 98.5359573
7: -54.4486427, 45.8130493, -54.5935402, 45.9723778, -100.4210205, 100.4065857
8: -66.0436096, 45.9199066, -66.1762772, 45.9945908, -112.0381851, 112.0961838
9: -49.6950188, 48.7643890, -49.7878761, 48.8629150, -98.5579300, 98.5522614

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 97

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5929084, upper bound: 143.5898967
time: 6.01 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.5929084, upper bound: 143.6270754
time: 8.18 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -54.4046097, 43.4374390, -54.7088776, 43.6693382, -98.0739441, 98.1463165
1: -44.7226830, 38.5364532, -45.0210495, 38.7784767, -83.5011520, 83.5575027
2: -59.4978752, 39.0290260, -59.8116188, 39.1958046, -98.6936798, 98.8406448
3: -63.4377556, 33.6327553, -63.7809067, 33.8897285, -97.3274689, 97.4136658
4: -58.6960258, 45.2262154, -59.0735550, 45.5018730, -104.1978989, 104.2997742
5: -52.3282928, 40.8077965, -52.6134911, 41.0452499, -93.3735428, 93.4212875
6: -50.3102913, 47.8889923, -50.6077728, 48.2125015, -98.5227890, 98.4967575
7: -54.3504372, 45.7677307, -54.6873817, 46.0137558, -100.3641891, 100.4551086
8: -65.8845139, 45.7955360, -66.3303986, 46.1154556, -111.9999695, 112.1259308
9: -49.5662689, 48.6469727, -49.9123535, 48.9764938, -98.5427628, 98.5593185

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 114

### Candidate
type: A, layer: 1, pos: 114

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of NS_A1_B2_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B2_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of NS_A1_B2_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A1_B2_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A1_B2_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_A1_A2_B2_A2_A1

### Relational analysis result of NS_A1_B2_A1_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6119541, upper bound: 143.6063971
time: 7.69 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_A2_A2

### Relational analysis result of NS_A1_B2_A1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6267486, upper bound: 143.6200050
time: 7.39 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -52.6878891, 42.0630798, -48.7977333, 38.9651527, -91.6530457, 90.8608093
1: -43.3444023, 37.3578644, -40.0980949, 34.6131287, -77.9575119, 77.4559402
2: -57.5765991, 37.7529602, -53.2681122, 34.9848824, -92.5614777, 91.0210724
3: -61.3705177, 32.6577263, -56.7620049, 30.2809162, -91.6514359, 89.4197311
4: -56.8780556, 43.8438377, -52.6681786, 40.6396751, -97.5177307, 96.5120163
5: -50.6568794, 39.5186157, -46.9038200, 36.5832672, -87.2401352, 86.4224396
6: -48.7402649, 46.4324837, -45.1485939, 43.0002289, -91.7404938, 91.5810699
7: -52.6380348, 44.2931938, -48.7039795, 40.9870796, -93.6251144, 92.9971771
8: -63.9092941, 44.4926910, -59.2207451, 41.3155823, -105.2248688, 103.7134247
9: -48.0650063, 47.1644135, -44.5032959, 43.6604843, -91.7254868, 91.6676865

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6282401, upper bound: 143.6220214
time: 6.69 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6282401, upper bound: 143.6226009
time: 7.91 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 15.49 seconds
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 15.49
Output dim: 4, lower bound: -143.6218011, upper bound: 143.6176785
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 15.49
Output dim: 4, lower bound: -143.6218011, upper bound: 143.6378023
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 15.49
Output dim: 4, lower bound: -143.5929084, upper bound: 143.5898967
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 15.49
Output dim: 4, lower bound: -143.5929084, upper bound: 143.6270754
NS_A1_B2_A1_A2_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 15.49
Output dim: 4, lower bound: -143.6119541, upper bound: 143.6063971
NS_A1_B2_A1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 15.49
Output dim: 4, lower bound: -143.6267486, upper bound: 143.6200050
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 15.49
Output dim: 4, lower bound: -143.6282401, upper bound: 143.6220214
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 15.49
Output dim: 4, lower bound: -143.6282401, upper bound: 143.6226009

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -45.6197433, 36.4023895, -54.6469078, 43.6286697, -89.2484131, 91.0492859
1: -37.2289429, 32.2918549, -44.9252586, 38.7061081, -75.9350510, 77.2171173
2: -49.7389755, 32.8265572, -59.7642975, 39.2005615, -88.9395370, 92.5908508
3: -53.2857819, 28.1831169, -63.7217712, 33.7825584, -87.0683441, 91.9048843
4: -49.2329559, 37.8283310, -58.9592896, 45.4252815, -94.6582184, 96.7876205
5: -43.9249458, 34.2472343, -52.5612221, 40.9888039, -84.9137421, 86.8084564
6: -42.1888084, 40.0387802, -50.5335503, 48.1030273, -90.2918320, 90.5723267
7: -45.5401154, 38.3637886, -54.5935402, 45.9723778, -91.5124893, 92.9573212
8: -55.0829735, 38.1415367, -66.1762772, 45.9945908, -101.0775528, 104.3178101
9: -41.5034752, 40.6114159, -49.7878761, 48.8629150, -90.3663864, 90.3992767

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 97

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 114

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6154315, upper bound: 143.6152325
time: 7.53 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6133117, upper bound: 143.6140790
time: 6.89 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -54.4046097, 43.4374390, -54.6469078, 43.6286697, -98.0332794, 98.0843353
1: -44.7226830, 38.5364532, -44.9252586, 38.7061081, -83.4287796, 83.4617157
2: -59.4978752, 39.0290260, -59.7642975, 39.2005615, -98.6984406, 98.7933197
3: -63.4377556, 33.6327553, -63.7217712, 33.7825584, -97.2203140, 97.3545227
4: -58.6960258, 45.2262154, -58.9592896, 45.4252815, -104.1212845, 104.1855011
5: -52.3282928, 40.8077965, -52.5612221, 40.9888039, -93.3170929, 93.3690186
6: -50.3102913, 47.8889923, -50.5335503, 48.1030273, -98.4133072, 98.4225388
7: -54.3504372, 45.7677307, -54.5935402, 45.9723778, -100.3228149, 100.3612671
8: -65.8845139, 45.7955360, -66.1762772, 45.9945908, -111.8790894, 111.9718094
9: -49.5662689, 48.6469727, -49.7878761, 48.8629150, -98.4291763, 98.4348373

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 114

### Candidate
type: A, layer: 1, pos: 114

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 240

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 120

### Candidate
type: A, layer: 1, pos: 120

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6154315, upper bound: 143.6356385
time: 6.83 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6133117, upper bound: 143.6348930
time: 7.36 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -51.5427666, 41.1451073, -54.6469078, 43.6286697, -95.1714325, 95.7920074
1: -42.3408318, 36.5412178, -44.9252586, 38.7061081, -81.0469284, 81.4664688
2: -56.3123550, 36.9633713, -59.7642975, 39.2005615, -95.5129166, 96.7276688
3: -60.0986366, 31.9271641, -63.7217712, 33.7825584, -93.8811951, 95.6489334
4: -55.6614075, 42.8771057, -58.9592896, 45.4252815, -101.0866699, 101.8363953
5: -49.5795784, 38.6869087, -52.5612221, 40.9888039, -90.5683670, 91.2481308
6: -47.6804390, 45.4054184, -50.5335503, 48.1030273, -95.7834625, 95.9389648
7: -51.5121765, 43.3579636, -54.5935402, 45.9723778, -97.4845581, 97.9514923
8: -62.4694176, 43.4277306, -66.1762772, 45.9945908, -108.4639893, 109.6040039
9: -47.0145187, 46.1109047, -49.7878761, 48.8629150, -95.8774261, 95.8987427

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 114

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

### Candidate
type: A, layer: 1, pos: 240

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.5858440, upper bound: 143.6253843
time: 6.16 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.5850633, upper bound: 143.6252647
time: 6.54 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -52.7850761, 42.1611824, -54.2515717, 43.3084793, -96.0935516, 96.4127502
1: -43.3928719, 37.4136543, -44.6455002, 38.4610596, -81.8539276, 82.0591583
2: -57.7237778, 37.8825531, -59.3103104, 38.8716545, -96.5954285, 97.1928482
3: -61.5342789, 32.6359177, -63.2424431, 33.6085625, -95.1428299, 95.8783569
4: -56.9531288, 43.9057693, -58.5813942, 45.1286697, -102.0818024, 102.4871674
5: -50.7808800, 39.6288719, -52.1763496, 40.7121696, -91.4930496, 91.8052216
6: -48.8248177, 46.4726143, -50.1881332, 47.8126717, -96.6374817, 96.6607513
7: -52.7309265, 44.4204254, -54.2297783, 45.6333237, -98.3642502, 98.6502075
8: -63.9352951, 44.4813042, -65.7795410, 45.7441063, -109.6793976, 110.2608490
9: -48.1022606, 47.2067375, -49.4989243, 48.5696678, -96.6719055, 96.7056580

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_A1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 114

### Candidate
type: A, layer: 1, pos: 114

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 240

### Candidate
type: A, layer: 1, pos: 240

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 120

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 120

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A1_B2_A1_A2_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6247659, upper bound: 143.6180822
time: 7.98 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6251183, upper bound: 143.6184580
time: 6.23 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -48.5616112, 38.7787895, -48.7977333, 38.9651527, -87.5267639, 87.5764999
1: -39.9010239, 34.4477921, -40.0980949, 34.6131287, -74.5141449, 74.5458832
2: -53.0084457, 34.8181000, -53.2681122, 34.9848824, -87.9933319, 88.0862122
3: -56.4851723, 30.1354961, -56.7620049, 30.2809162, -86.7660904, 86.8974915
4: -52.4120941, 40.4455833, -52.6681786, 40.6396751, -93.0517731, 93.1137619
5: -46.6767883, 36.4063988, -46.9038200, 36.5832672, -83.2600479, 83.3102112
6: -44.9314308, 42.7919235, -45.1485939, 43.0002289, -87.9316559, 87.9405212
7: -48.4673424, 40.7876549, -48.7039795, 40.9870796, -89.4544220, 89.4916382
8: -58.9357300, 41.1216660, -59.2207451, 41.3155823, -100.2513046, 100.3424072
9: -44.2875481, 43.4502602, -44.5032959, 43.6604843, -87.9480286, 87.9535446

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 114

### Candidate
type: A, layer: 1, pos: 114

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 240

### Candidate
type: A, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 120

### Candidate
type: A, layer: 1, pos: 120

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 62

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6190002, upper bound: 143.6127906
time: 6.71 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6173544, upper bound: 143.6109978
time: 6.65 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -53.0750237, 42.3555450, -48.7977333, 38.9651527, -92.0401764, 91.1532669
1: -43.6540909, 37.6219177, -40.0980949, 34.6131287, -78.2672195, 77.7200089
2: -57.9953156, 38.0062180, -53.2681122, 34.9848824, -92.9801941, 91.2743225
3: -61.8384171, 32.8951836, -56.7620049, 30.2809162, -92.1193237, 89.6571884
4: -57.2989464, 44.1532784, -52.6681786, 40.6396751, -97.9386215, 96.8214569
5: -51.0248909, 39.7743111, -46.9038200, 36.5832672, -87.6081543, 86.6781235
6: -49.1241455, 46.7695007, -45.1485939, 43.0002289, -92.1243591, 91.9180832
7: -53.0176239, 44.5872612, -48.7039795, 40.9870796, -94.0046997, 93.2912445
8: -64.3926926, 44.8100700, -59.2207451, 41.3155823, -105.7082748, 104.0308075
9: -48.3996391, 47.4964828, -44.5032959, 43.6604843, -92.0601196, 91.9997482

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 114

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 114

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 120

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6137843, upper bound: 143.6094161
time: 7.05 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6282557, upper bound: 143.6226009
time: 7.34 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 15.26 seconds
NS_A1_B1_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 15.26
Output dim: 4, lower bound: -143.6154315, upper bound: 143.6152325
NS_A1_B1_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 15.26
Output dim: 4, lower bound: -143.6133117, upper bound: 143.6140790
NS_A1_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 15.26
Output dim: 4, lower bound: -143.6154315, upper bound: 143.6356385
NS_A1_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 15.26
Output dim: 4, lower bound: -143.6133117, upper bound: 143.6348930
NS_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 15.26
Output dim: 4, lower bound: -143.5858440, upper bound: 143.6253843
NS_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 15.26
Output dim: 4, lower bound: -143.5850633, upper bound: 143.6252647
NS_A1_B2_A1_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 15.26
Output dim: 4, lower bound: -143.6247659, upper bound: 143.6180822
NS_A1_B2_A1_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 15.26
Output dim: 4, lower bound: -143.6251183, upper bound: 143.6184580
NS_A1_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 15.26
Output dim: 4, lower bound: -143.6190002, upper bound: 143.6127906
NS_A1_B2_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 15.26
Output dim: 4, lower bound: -143.6173544, upper bound: 143.6109978
NS_A1_B2_A2_B2_A2_B1_A2_A1, status: Status.VERIFIED, split count: 8, time: 15.26
Output dim: 4, lower bound: -143.6137843, upper bound: 143.6094161
NS_A1_B2_A2_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 15.26
Output dim: 4, lower bound: -143.6282557, upper bound: 143.6226009

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -51.8528786, 41.4081841, -48.1373558, 38.4546509, -90.3075180, 89.5455399
1: -42.5905266, 36.7394753, -39.4825134, 34.1189232, -76.7094498, 76.2219772
2: -56.6754951, 37.2152481, -52.5577660, 34.5747185, -91.2502060, 89.7730103
3: -60.4164581, 32.0694771, -56.0171318, 29.8073788, -90.2238388, 88.0866013
4: -55.9292259, 43.1254730, -51.9070129, 40.0602188, -95.9894409, 95.0324783
5: -49.8649292, 38.8911781, -46.2811813, 36.0918541, -85.9567871, 85.1723633
6: -47.9500542, 45.6348419, -44.5202103, 42.3524513, -90.3025055, 90.1550522
7: -51.7698479, 43.6040421, -48.0158195, 40.4483871, -92.2182312, 91.6198578
8: -62.8144951, 43.7121773, -58.3368073, 40.6738167, -103.4883041, 102.0489807
9: -47.2313766, 46.3512230, -43.8334618, 43.0077362, -90.2390976, 90.1846771

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 155

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6409530, upper bound: 143.6347086
time: 6.91 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6409530, upper bound: 143.6348930
time: 7.21 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -52.5587082, 41.9683609, -52.5142403, 41.9204178, -94.4791107, 94.4826050
1: -43.1807594, 37.2366676, -43.1290054, 37.1973343, -80.3780823, 80.3656769
2: -57.4557076, 37.7145042, -57.3977509, 37.6604462, -95.1161499, 95.1122589
3: -61.2557220, 32.5007668, -61.2106133, 32.4822083, -93.7379303, 93.7113724
4: -56.6964455, 43.7073593, -56.6551514, 43.6602516, -100.3566895, 100.3625107
5: -50.5470657, 39.4183617, -50.4991493, 39.3497009, -89.8967667, 89.9175110
6: -48.6052208, 46.2583008, -48.5903320, 46.2176476, -94.8228683, 94.8486328
7: -52.4856453, 44.2016640, -52.4290924, 44.1311302, -96.6167755, 96.6307526
8: -63.6665268, 44.2886467, -63.6345978, 44.2601242, -107.9266357, 107.9232483
9: -47.8784714, 46.9862938, -47.8269768, 46.9336433, -94.8121185, 94.8132553

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6409530, upper bound: 143.6347086
time: 6.16 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6409530, upper bound: 143.6348930
time: 7.97 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -48.9821243, 39.1083183, -48.1373558, 38.4546509, -87.4367676, 87.2456589
1: -40.2022476, 34.7365036, -39.4825134, 34.1189232, -74.3211670, 74.2190018
2: -53.4767227, 35.1441841, -52.5577660, 34.5747185, -88.0514374, 87.7019424
3: -57.0696259, 30.3671036, -56.0171318, 29.8073788, -86.8770065, 86.3842316
4: -52.8933144, 40.7669601, -51.9070129, 40.0602188, -92.9535294, 92.6739731
5: -47.1102257, 36.7558365, -46.2811813, 36.0918541, -83.2020721, 83.0370102
6: -45.3209572, 43.1457672, -44.5202103, 42.3524513, -87.6734085, 87.6659775
7: -48.9265823, 41.1819763, -48.0158195, 40.4483871, -89.3749695, 89.1977997
8: -59.3834419, 41.3352509, -58.3368073, 40.6738167, -100.0572586, 99.6720581
9: -44.6715431, 43.8071823, -43.8334618, 43.0077362, -87.6792755, 87.6406326

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 97

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6295538, upper bound: 143.6241908
time: 7.55 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6295538, upper bound: 143.6252647
time: 8.34 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -49.7704124, 39.7334747, -52.5142403, 41.9204178, -91.6908264, 92.2477112
1: -40.8601074, 35.2923584, -43.1290054, 37.1973343, -78.0574417, 78.4213638
2: -54.3481979, 35.7021027, -57.3977509, 37.6604462, -92.0086365, 93.0998535
3: -58.0036125, 30.8450394, -61.2106133, 32.4822083, -90.4858093, 92.0556488
4: -53.7450371, 41.4157295, -56.6551514, 43.6602516, -97.4052811, 98.0708771
5: -47.8707809, 37.3477516, -50.4991493, 39.3497009, -87.2204819, 87.8469009
6: -46.0484123, 43.8402824, -48.5903320, 46.2176476, -92.2660599, 92.4306183
7: -49.7231712, 41.8511543, -52.4290924, 44.1311302, -93.8543015, 94.2802353
8: -60.3334351, 41.9770737, -63.6345978, 44.2601242, -104.5935516, 105.6116714
9: -45.3918953, 44.5151329, -47.8269768, 46.9336433, -92.3255310, 92.3421097

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 97

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6295538, upper bound: 143.6241908
time: 7.42 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6295538, upper bound: 143.6252647
time: 7.48 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 10.97 + 596.55 = 607.52 seconds
