## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 71.4340763181


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-44.4740562, 35.5138130, -44.4740562, 35.5138130, -79.9878693, 79.9878693)
1: (-36.5142250, 31.1372204, -36.5142250, 31.1372204, -67.6514435, 67.6514435)
2: (-47.2705040, 29.3448944, -47.2705040, 29.3448944, -76.6154022, 76.6154022)
3: (-53.3106308, 26.5060616, -53.3106308, 26.5060616, -79.8166733, 79.8166733)
4: (-47.9498138, 36.7448006, -47.9498138, 36.7448006, -84.6946030, 84.6946030)
5: (-42.0963745, 32.5022964, -42.0963745, 32.5022964, -74.5986557, 74.5986557)
6: (-39.7821922, 40.5336456, -39.7821922, 40.5336456, -80.3158417, 80.3158417)
7: (-45.6046028, 33.5817642, -45.6046028, 33.5817642, -79.1863556, 79.1863556)
8: (-52.0067253, 35.6603088, -52.0067253, 35.6603088, -87.6670380, 87.6670380)
9: (-39.7509995, 39.6626358, -39.7509995, 39.6626358, -79.4136353, 79.4136353)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.79 + 12.96 = 13.74 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -71.5055819, upper bound: 71.5055819

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5016530, upper bound: 71.5012952
time: 12.48 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5055819, upper bound: 71.5055819
time: 10.17 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 22.74 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 22.74
Output dim: 7, lower bound: -71.5016530, upper bound: 71.5012952
NS_A2, status: Status.UNKNOWN, split count: 1, time: 22.74
Output dim: 7, lower bound: -71.5055819, upper bound: 71.5055819

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -46.5624542, 37.1875648, -44.0417290, 35.1715164, -81.7339630, 81.2292862
1: -38.2388077, 32.5995865, -36.1552773, 30.8384533, -69.0772629, 68.7548676
2: -49.5592117, 30.7943001, -46.8070374, 29.0556030, -78.6148148, 77.6013336
3: -55.7948990, 27.7589378, -52.7884941, 26.2491531, -82.0440445, 80.5474319
4: -50.1887779, 38.4514961, -47.4766045, 36.3903618, -86.5791397, 85.9280930
5: -44.0803566, 34.0508308, -41.6850204, 32.1862297, -76.2665863, 75.7358246
6: -41.6876373, 42.4006195, -39.3908920, 40.1441383, -81.8317719, 81.7915039
7: -47.7220955, 35.2712860, -45.1623611, 33.2461433, -80.9682388, 80.4336395
8: -54.5168991, 37.3696175, -51.4986191, 35.3147926, -89.8316956, 88.8682404
9: -41.6416779, 41.5152817, -39.3610001, 39.2764053, -80.9180832, 80.8762817

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of NS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

### Candidate
type: B, layer: 1, pos: 132

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5010179, upper bound: 71.5007204
time: 10.64 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4994814, upper bound: 71.4992349
time: 12.24 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -44.3130836, 35.3869553, -44.4740562, 35.5138130, -79.8268890, 79.8610077
1: -36.3800430, 31.0262375, -36.5142250, 31.1372204, -67.5172653, 67.5404663
2: -47.0975685, 29.2355003, -47.2705040, 29.3448944, -76.4424591, 76.5060043
3: -53.1175156, 26.4096642, -53.3106308, 26.5060616, -79.6235657, 79.7202835
4: -47.7734413, 36.6128082, -47.9498138, 36.7448006, -84.5182343, 84.5626221
5: -41.9432259, 32.3841743, -42.0963745, 32.5022964, -74.4455109, 74.4805450
6: -39.6357498, 40.3893509, -39.7821922, 40.5336456, -80.1693954, 80.1715317
7: -45.4414406, 33.4542694, -45.6046028, 33.5817642, -79.0231934, 79.0588684
8: -51.8169899, 35.5310555, -52.0067253, 35.6603088, -87.4772797, 87.5377808
9: -39.6056976, 39.5192642, -39.7509995, 39.6626358, -79.2683334, 79.2702637

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 163

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4807071, upper bound: 71.4803869
time: 11.03 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5053807, upper bound: 71.5053807
time: 10.73 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 22.56 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 22.56
Output dim: 7, lower bound: -71.5010179, upper bound: 71.5007204
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 22.56
Output dim: 7, lower bound: -71.4994814, upper bound: 71.4992349
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 22.56
Output dim: 7, lower bound: -71.4807071, upper bound: 71.4803869
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 22.56
Output dim: 7, lower bound: -71.5053807, upper bound: 71.5053807

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -44.1188774, 35.2566643, -43.3555450, 34.6296120, -78.7484894, 78.6122055
1: -36.2138443, 30.9124832, -35.5867538, 30.3635445, -66.5773849, 66.4992218
2: -46.9251442, 29.1846046, -46.0651932, 28.5968952, -75.5220413, 75.2498016
3: -52.8569489, 26.3303375, -51.9655876, 25.8437595, -78.7006989, 78.2959290
4: -47.5185890, 36.4423790, -46.7271423, 35.8258820, -83.3444672, 83.1695099
5: -41.7714233, 32.2660904, -41.0363350, 31.6851768, -73.4566040, 73.3024292
6: -39.4895325, 40.1858978, -38.7696228, 39.5246696, -79.0142059, 78.9555206
7: -45.1962242, 33.4036674, -44.4554825, 32.7113190, -77.9075470, 77.8591309
8: -51.6367455, 35.4288788, -50.6889801, 34.7652130, -86.4019623, 86.1178589
9: -39.4550171, 39.3330765, -38.7443466, 38.6621094, -78.1171188, 78.0774231

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 105

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4994756, upper bound: 71.4992325
time: 11.92 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4994756, upper bound: 71.4992349
time: 10.63 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -45.6339073, 36.4558029, -43.9157143, 35.0724831, -80.7063904, 80.3715210
1: -37.4667244, 31.9517670, -36.0507355, 30.7506676, -68.2173920, 68.0025024
2: -48.5414467, 30.1585960, -46.6690331, 28.9694405, -77.5108795, 76.8276291
3: -54.6942558, 27.1992722, -52.6385651, 26.1736889, -80.8679428, 79.8378372
4: -49.1727066, 37.6851540, -47.3385201, 36.2863770, -85.4590836, 85.0236664
5: -43.2002373, 33.3591194, -41.5662842, 32.0931625, -75.2933960, 74.9253998
6: -40.8394165, 41.5607109, -39.2756577, 40.0302353, -80.8696442, 80.8363647
7: -46.7641945, 34.5194855, -45.0319939, 33.1444664, -79.9086609, 79.5514603
8: -53.4107170, 36.6173325, -51.3484573, 35.2126465, -88.6233521, 87.9657898
9: -40.8029099, 40.6765137, -39.2470207, 39.1625938, -79.9654999, 79.9235229

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4994756, upper bound: 71.4992326
time: 13.34 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4994756, upper bound: 71.4992349
time: 13.02 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -42.6834679, 34.0802078, -36.3525925, 29.0057049, -71.6891708, 70.4328003
1: -35.0245132, 29.8779507, -29.7729130, 25.3706875, -60.3951950, 59.6508560
2: -45.2927055, 28.0610466, -38.2220306, 23.3407364, -68.6334305, 66.2830734
3: -51.1915131, 25.3853664, -43.7800522, 21.2520161, -72.4435196, 69.1654205
4: -46.0157356, 35.2666206, -39.2321930, 30.0492954, -76.0650177, 74.4988098
5: -40.3759995, 31.1554832, -34.2959175, 26.2793121, -66.6553116, 65.4514008
6: -38.1303596, 38.9300308, -32.2500305, 33.2993622, -71.4297180, 71.1800461
7: -43.8101196, 32.0579033, -37.5569725, 26.3333130, -70.1434326, 69.6148758
8: -49.8225327, 34.1707726, -41.9428444, 28.8525047, -78.6750336, 76.1136017
9: -38.1068878, 38.0348625, -32.2077904, 32.2199135, -70.3267975, 70.2426300

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4798983, upper bound: 71.4794833
time: 12.84 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4807071, upper bound: 71.4803871
time: 12.84 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -44.3130836, 35.3869553, -43.9017410, 35.0558815, -79.3689575, 79.2886810
1: -36.3800430, 31.0262375, -36.0386200, 30.7354298, -67.1154709, 67.0648575
2: -47.0975685, 29.2355003, -46.6373482, 28.9329338, -76.0305023, 75.8728485
3: -53.1175156, 26.4096642, -52.6357841, 26.1471291, -79.2646484, 79.0454483
4: -47.7734413, 36.6128082, -47.3311234, 36.2726784, -84.0461121, 83.9439316
5: -41.9432259, 32.3841743, -41.5455132, 32.0710106, -74.0142212, 73.9296875
6: -39.6357498, 40.3893509, -39.2524910, 40.0221863, -79.6579361, 79.6418304
7: -45.4414406, 33.4542694, -45.0320129, 33.0921097, -78.5335541, 78.4862747
8: -51.8169899, 35.5310555, -51.3076019, 35.1835213, -87.0004959, 86.8386536
9: -39.6056976, 39.5192642, -39.2256432, 39.1423416, -78.7480392, 78.7449036

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4960525, upper bound: 71.4962281
time: 11.56 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5049634, upper bound: 71.5049634
time: 9.14 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 21.57 seconds
NS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 21.57
Output dim: 7, lower bound: -71.4994756, upper bound: 71.4992325
NS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 21.57
Output dim: 7, lower bound: -71.4994756, upper bound: 71.4992349
NS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 21.57
Output dim: 7, lower bound: -71.4994756, upper bound: 71.4992326
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 21.57
Output dim: 7, lower bound: -71.4994756, upper bound: 71.4992349
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.57
Output dim: 7, lower bound: -71.4798983, upper bound: 71.4794833
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.57
Output dim: 7, lower bound: -71.4807071, upper bound: 71.4803871
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.57
Output dim: 7, lower bound: -71.4960525, upper bound: 71.4962281
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.57
Output dim: 7, lower bound: -71.5049634, upper bound: 71.5049634

## BFS NS instance: NS_A1_A1_B1

### Backsubstitution after applying NS history:
0: -44.1188774, 35.2566643, -41.1070404, 32.8524170, -76.9712982, 76.3636856
1: -36.2138443, 30.9124832, -33.7263031, 28.8178673, -65.0317078, 64.6387711
2: -46.9251442, 29.1846046, -43.6586266, 27.1397572, -74.0649033, 72.8432312
3: -52.8569489, 26.3303375, -49.2449570, 24.5498104, -77.4067459, 75.5752945
4: -47.5185890, 36.4423790, -44.2713051, 33.9811630, -81.4997559, 80.7136841
5: -41.7714233, 32.2660904, -38.9163895, 30.0532646, -71.8246918, 71.1824799
6: -39.4895325, 40.1858978, -36.7579651, 37.4887352, -76.9782715, 76.9438477
7: -45.1962242, 33.4036674, -42.1336899, 31.0278740, -76.2240982, 75.5373535
8: -51.6367455, 35.4288788, -48.0512657, 32.9905396, -84.6272888, 83.4801331
9: -39.4550171, 39.3330765, -36.7385178, 36.6633377, -76.1183472, 76.0715942

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 163

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 105

### Candidate
type: A, layer: 1, pos: 132

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 105

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 132

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_A1_B1_A1

### Relational analysis result of NS_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4980650, upper bound: 71.4979185
time: 11.88 seconds

## Relational analysis of NS_A1_A1_B1_A2

### Relational analysis result of NS_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4977893, upper bound: 71.4977775
time: 13.18 seconds

## BFS NS instance: NS_A1_A1_B2

### Backsubstitution after applying NS history:
0: -44.1188774, 35.2566643, -43.1249428, 34.4502602, -78.5691376, 78.3815994
1: -36.2138443, 30.9124832, -35.3945122, 30.1989040, -66.4127502, 66.3069916
2: -46.9251442, 29.1846046, -45.8025780, 28.4285965, -75.3537445, 74.9871826
3: -52.8569489, 26.3303375, -51.6970253, 25.7011547, -78.5580902, 78.0273590
4: -47.5185890, 36.4423790, -46.4727478, 35.6340599, -83.1526489, 82.9151230
5: -41.7714233, 32.2660904, -40.8216972, 31.5087090, -73.2801361, 73.0877838
6: -39.4895325, 40.1858978, -38.5526886, 39.3152084, -78.8047409, 78.7385864
7: -45.1962242, 33.4036674, -44.2138176, 32.5058212, -77.7020416, 77.6174774
8: -51.6367455, 35.4288788, -50.4050369, 34.5721779, -86.2089233, 85.8339157
9: -39.4550171, 39.3330765, -38.5307121, 38.4486389, -77.9036407, 77.8637848

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 163

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 105

### Candidate
type: A, layer: 1, pos: 132

### Candidate
type: B, layer: 1, pos: 105

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 132

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_A1_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5004742, upper bound: 71.5001833
time: 13.00 seconds

## Relational analysis of NS_A1_A1_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5004742, upper bound: 71.5006914
time: 12.35 seconds

## BFS NS instance: NS_A1_A2_B1

### Backsubstitution after applying NS history:
0: -45.6339073, 36.4558029, -41.1070404, 32.8524170, -78.4863281, 77.5628357
1: -37.4667244, 31.9517670, -33.7263031, 28.8178673, -66.2845917, 65.6780548
2: -48.5414467, 30.1585960, -43.6586266, 27.1397572, -75.6811981, 73.8172150
3: -54.6942558, 27.1992722, -49.2449570, 24.5498104, -79.2440643, 76.4442291
4: -49.1727066, 37.6851540, -44.2713051, 33.9811630, -83.1538696, 81.9564590
5: -43.2002373, 33.3591194, -38.9163895, 30.0532646, -73.2535019, 72.2755127
6: -40.8394165, 41.5607109, -36.7579651, 37.4887352, -78.3281250, 78.3186722
7: -46.7641945, 34.5194855, -42.1336899, 31.0278740, -77.7920685, 76.6531677
8: -53.4107170, 36.6173325, -48.0512657, 32.9905396, -86.4012527, 84.6685944
9: -40.8029099, 40.6765137, -36.7385178, 36.6633377, -77.4662476, 77.4150314

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 132

### Candidate
type: A, layer: 1, pos: 105

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 105

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 132

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_A2_B1_A1

### Relational analysis result of NS_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4964344, upper bound: 71.4963616
time: 12.37 seconds

## Relational analysis of NS_A1_A2_B1_A2

### Relational analysis result of NS_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4964729, upper bound: 71.4963465
time: 11.25 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: -45.6339073, 36.4558029, -43.1249428, 34.4502602, -80.0841675, 79.5807419
1: -37.4667244, 31.9517670, -35.3945122, 30.1989040, -67.6656265, 67.3462753
2: -48.5414467, 30.1585960, -45.8025780, 28.4285965, -76.9700394, 75.9611664
3: -54.6942558, 27.1992722, -51.6970253, 25.7011547, -80.3954086, 78.8963013
4: -49.1727066, 37.6851540, -46.4727478, 35.6340599, -84.8067627, 84.1578979
5: -43.2002373, 33.3591194, -40.8216972, 31.5087090, -74.7089462, 74.1808167
6: -40.8394165, 41.5607109, -38.5526886, 39.3152084, -80.1546173, 80.1134033
7: -46.7641945, 34.5194855, -44.2138176, 32.5058212, -79.2700119, 78.7332916
8: -53.4107170, 36.6173325, -50.4050369, 34.5721779, -87.9828796, 87.0223694
9: -40.8029099, 40.6765137, -38.5307121, 38.4486389, -79.2515488, 79.2072296

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 105

### Candidate
type: A, layer: 1, pos: 132

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 105

### Candidate
type: B, layer: 1, pos: 132

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_A2_B2_A1

### Relational analysis result of NS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4784732, upper bound: 71.4777598
time: 14.87 seconds

## Relational analysis of NS_A1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of NS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_A2_B2_A1

### Relational analysis result of NS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4964344, upper bound: 71.4963616
time: 12.65 seconds

## Relational analysis of NS_A1_A2_B2_A2

### Relational analysis result of NS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4964729, upper bound: 71.4963465
time: 11.69 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -39.7795525, 31.7873955, -35.7383766, 28.5200081, -68.2995529, 67.5257721
1: -32.6221962, 27.8801556, -29.2652512, 24.9460869, -57.5682793, 57.1454086
2: -42.1819191, 26.1701946, -37.5601845, 22.9353733, -65.1172943, 63.7303772
3: -47.6884499, 23.7079315, -43.0435905, 20.8928318, -68.5812836, 66.7515182
4: -42.8472404, 32.8851166, -38.5644417, 29.5442886, -72.3915253, 71.4495468
5: -37.6415291, 29.0439835, -33.7187004, 25.8305111, -63.4720268, 62.7626762
6: -35.5350266, 36.3039322, -31.7010117, 32.7452011, -68.2802277, 68.0049438
7: -40.8163414, 29.8665562, -36.9241600, 25.8598366, -66.6761703, 66.7907028
8: -46.4175224, 31.8745613, -41.2215233, 28.3661461, -74.7836533, 73.0960693
9: -35.5148048, 35.4530869, -31.6587734, 31.6720428, -67.1868439, 67.1118546

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 132

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4761070, upper bound: 71.4758186
time: 12.27 seconds

## Relational analysis of NS_A2_B1_A1_A2

### Relational analysis result of NS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4760793, upper bound: 71.4757942
time: 11.26 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -41.7854576, 33.3735085, -36.2410278, 28.9175873, -70.7030334, 69.6145325
1: -34.2794647, 29.2506714, -29.6806774, 25.2930222, -59.5724678, 58.9313431
2: -44.3082390, 27.4467525, -38.1001015, 23.2654915, -67.5737305, 65.5468521
3: -50.1232338, 24.8487015, -43.6469460, 21.1861382, -71.3093719, 68.4956360
4: -45.0337830, 34.5265198, -39.1106567, 29.9570866, -74.9908524, 73.6371765
5: -39.5318909, 30.4909897, -34.1914253, 26.1967220, -65.7286072, 64.6824112
6: -37.3099556, 38.1184311, -32.1492882, 33.1987114, -70.5086670, 70.2677078
7: -42.8817787, 31.3314476, -37.4414749, 26.2441883, -69.1259613, 68.7729187
8: -48.7507706, 33.4445419, -41.8103256, 28.7633018, -77.5140686, 75.2548676
9: -37.2933502, 37.2238998, -32.1072617, 32.1195641, -69.4129028, 69.3311615

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 132

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4686659, upper bound: 71.4681905
time: 13.04 seconds

## Relational analysis of NS_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4784920, upper bound: 71.4780513
time: 10.20 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -40.7159424, 32.4908829, -42.3164368, 33.7905006, -74.5064392, 74.8073196
1: -33.4328423, 28.4339218, -34.7309647, 29.6144123, -63.0472527, 63.1648865
2: -43.0767593, 26.5804100, -44.8963242, 27.8150787, -70.8918381, 71.4767303
3: -48.8159027, 24.1471233, -50.7340889, 25.1813660, -73.9972687, 74.8812103
4: -43.8428116, 33.6400452, -45.6059570, 34.9702301, -78.8130417, 79.2459869
5: -38.4821167, 29.6048279, -40.0305672, 30.8833961, -69.3655090, 69.6353912
6: -36.2408295, 37.2229004, -37.7880287, 38.6109810, -74.8518066, 75.0109253
7: -41.7971382, 30.2853279, -43.4219856, 31.7748756, -73.5720139, 73.7072983
8: -47.4248848, 32.5396271, -49.4015007, 33.8890381, -81.3139191, 81.9411240
9: -36.2638016, 36.2526245, -37.7777977, 37.7158508, -73.9796524, 74.0304260

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 253

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4957230, upper bound: 71.4958626
time: 11.77 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4957205, upper bound: 71.4958543
time: 13.45 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -43.3417473, 34.6110458, -43.9017410, 35.0558815, -78.3976212, 78.5127640
1: -35.5796432, 30.3374596, -36.0386200, 30.7354298, -66.3150711, 66.3760757
2: -46.0305824, 28.5475540, -46.6373482, 28.9329338, -74.9635162, 75.1848907
3: -51.9520226, 25.8140297, -52.6357841, 26.1471291, -78.0991516, 78.4498138
4: -46.7155304, 35.8135757, -47.3311234, 36.2726784, -82.9881897, 83.1446915
5: -41.0121765, 31.6538296, -41.5455132, 32.0710106, -73.0831909, 73.1993332
6: -38.7354012, 39.5256691, -39.2524910, 40.0221863, -78.7575836, 78.7781525
7: -44.4572411, 32.6425552, -45.0320129, 33.0921097, -77.5493393, 77.6745377
8: -50.6475258, 34.7355537, -51.3076019, 35.1835213, -85.8310471, 86.0431519
9: -38.7166595, 38.6443520, -39.2256432, 39.1423416, -77.8589935, 77.8699951

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 163

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4918167, upper bound: 71.4926561
time: 12.84 seconds

## Relational analysis of NS_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4648927, upper bound: 71.4657529
time: 11.91 seconds

## Relational analysis of NS_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.5049121, upper bound: 71.5049121
time: 9.18 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 39.77 seconds
NS_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 39.77
Output dim: 7, lower bound: -71.4980650, upper bound: 71.4979185
NS_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 39.77
Output dim: 7, lower bound: -71.4977893, upper bound: 71.4977775
NS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 39.77
Output dim: 7, lower bound: -71.5004742, upper bound: 71.5001833
NS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 39.77
Output dim: 7, lower bound: -71.5004742, upper bound: 71.5006914
NS_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 39.77
Output dim: 7, lower bound: -71.4964344, upper bound: 71.4963616
NS_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 39.77
Output dim: 7, lower bound: -71.4964729, upper bound: 71.4963465
NS_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 39.77
Output dim: 7, lower bound: -71.4964344, upper bound: 71.4963616
NS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 39.77
Output dim: 7, lower bound: -71.4964729, upper bound: 71.4963465
NS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 39.77
Output dim: 7, lower bound: -71.4761070, upper bound: 71.4758186
NS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 39.77
Output dim: 7, lower bound: -71.4760793, upper bound: 71.4757942
NS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 39.77
Output dim: 7, lower bound: -71.4686659, upper bound: 71.4681905
NS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 39.77
Output dim: 7, lower bound: -71.4784920, upper bound: 71.4780513
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 39.77
Output dim: 7, lower bound: -71.4957230, upper bound: 71.4958626
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 39.77
Output dim: 7, lower bound: -71.4957205, upper bound: 71.4958543
NS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 39.77
Output dim: 7, lower bound: -71.4648927, upper bound: 71.4657529
NS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 39.77
Output dim: 7, lower bound: -71.5049121, upper bound: 71.5049121

## BFS NS instance: NS_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -37.6619873, 30.1328239, -39.4746056, 31.5576038, -69.2195892, 69.6074295
1: -30.8358021, 26.3334637, -32.3650475, 27.6602707, -58.4960709, 58.6985092
2: -39.8192368, 24.6105652, -41.8630714, 25.9863510, -65.8055878, 66.4736328
3: -45.1197357, 22.2627602, -47.2889442, 23.5262852, -68.6460114, 69.5517044
4: -40.5304642, 31.1197929, -42.5036354, 32.6367149, -73.1671600, 73.6234283
5: -35.6345139, 27.4114170, -37.3677101, 28.8271618, -64.4616776, 64.7791138
6: -33.5723991, 34.3922424, -35.2665138, 36.0236282, -69.5960159, 69.6587448
7: -38.6841812, 27.9706764, -40.4871025, 29.6577415, -68.3419189, 68.4577637
8: -43.8625412, 30.1283035, -46.0920601, 31.6464500, -75.5089874, 76.2203674
9: -33.5311623, 33.4791603, -35.2398605, 35.1824188, -68.7135773, 68.7190094

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A1_A1_B1_A1_B1

### Relational analysis result of NS_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4979689, upper bound: 71.4978674
time: 11.21 seconds

## Relational analysis of NS_A1_A1_B1_A1_B2

### Relational analysis result of NS_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4979689, upper bound: 71.4979470
time: 9.45 seconds

## BFS NS instance: NS_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -39.5015717, 31.5959988, -39.2108917, 31.3493176, -70.8508911, 70.8068924
1: -32.3700829, 27.6100368, -32.1479721, 27.4737129, -59.8437958, 59.7580032
2: -41.8214378, 25.8584633, -41.5765266, 25.8035889, -67.6250305, 67.4349899
3: -47.3304100, 23.3616982, -46.9720459, 23.3616772, -70.6920853, 70.3337402
4: -42.5509453, 32.6222000, -42.2229347, 32.4186249, -74.9695587, 74.8451233
5: -37.3733025, 28.7707195, -37.1160088, 28.6301384, -66.0034409, 65.8867264
6: -35.2424545, 36.0549545, -35.0280495, 35.7874908, -71.0299454, 71.0829926
7: -40.5546951, 29.4538536, -40.2214966, 29.4433270, -69.9980240, 69.6753540
8: -46.0674477, 31.6098156, -45.7762985, 31.4327469, -77.5001984, 77.3861160
9: -35.2024231, 35.1320686, -35.0012093, 34.9480209, -70.1504364, 70.1332703

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 132

### Candidate
type: B, layer: 1, pos: 105

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A1_A1_B1_A2_B1

### Relational analysis result of NS_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4979689, upper bound: 71.4978674
time: 11.20 seconds

## Relational analysis of NS_A1_A1_B1_A2_B2

### Relational analysis result of NS_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4979689, upper bound: 71.4979470
time: 12.02 seconds

## BFS NS instance: NS_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -43.3251610, 34.6258392, -42.6887360, 34.1148567, -77.4400101, 77.3145599
1: -35.5540962, 30.3602257, -35.0220413, 29.8942413, -65.4483337, 65.3822556
2: -46.0657654, 28.6509628, -45.3381577, 28.1713428, -74.2371063, 73.9891205
3: -51.9011917, 25.8560734, -51.1250458, 25.4407101, -77.3418961, 76.9810944
4: -46.6529427, 35.7890205, -45.9634209, 35.2573280, -81.9102631, 81.7524414
5: -41.0134392, 31.6859818, -40.4145737, 31.2029572, -72.2163696, 72.1005554
6: -38.7667084, 39.4689674, -38.1486435, 38.8977814, -77.6644897, 77.6176147
7: -44.3863182, 32.7806396, -43.7278061, 32.2104340, -76.5967407, 76.5084457
8: -50.6943779, 34.7900734, -49.9064484, 34.2275391, -84.9219131, 84.6965179
9: -38.7373543, 38.6242714, -38.1151772, 38.0491943, -76.7865219, 76.7394485

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 105

### Candidate
type: B, layer: 1, pos: 105

### Candidate
type: A, layer: 1, pos: 132

### Candidate
type: B, layer: 1, pos: 132

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of NS_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 85

### Candidate
type: A, layer: 1, pos: 85

### Candidate
type: B, layer: 1, pos: 119

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_A1_B2_B1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4976791, upper bound: 71.4974973
time: 11.22 seconds

## Relational analysis of NS_A1_A1_B2_B1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4974455, upper bound: 71.4973985
time: 62.61 seconds

## BFS NS instance: NS_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -44.1188774, 35.2566643, -42.6199493, 34.0496521, -78.1685257, 77.8765945
1: -36.2138443, 30.9124832, -34.9752922, 29.8486748, -66.0625153, 65.8877716
2: -46.9251442, 29.1846046, -45.2571220, 28.0880680, -75.0132141, 74.4417267
3: -52.8569489, 26.3303375, -51.0883102, 25.3986626, -78.2555847, 77.4186478
4: -47.5185890, 36.4423790, -45.9179840, 35.2177505, -82.7363434, 82.3603668
5: -41.7714233, 32.2660904, -40.3422394, 31.1391335, -72.9105530, 72.6083298
6: -39.4895325, 40.1858978, -38.0898285, 38.8597717, -78.3493042, 78.2757263
7: -45.1962242, 33.4036674, -43.6988716, 32.1098518, -77.3060684, 77.1025238
8: -51.6367455, 35.4288788, -49.8033905, 34.1624680, -85.7992096, 85.2322617
9: -39.4550171, 39.3330765, -38.0717392, 37.9961052, -77.4511261, 77.4048157

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 163

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 105

### Candidate
type: A, layer: 1, pos: 132

### Candidate
type: B, layer: 1, pos: 105

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 132

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 85

### Candidate
type: A, layer: 1, pos: 85

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_A1_B2_B2_A1

### Relational analysis result of NS_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -71.4829436, upper bound: 71.4823757
time: 15.08 seconds

## Relational analysis of NS_A1_A1_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 13.74 + 588.96 = 602.70 seconds
