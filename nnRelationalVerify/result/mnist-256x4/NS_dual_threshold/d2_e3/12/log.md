## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 12)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.314959394


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219)
1: (-0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226)
2: (-0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511)
3: (-0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240)
4: (-0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287)
5: (-0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725)
6: (-0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234)
7: (-0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564)
8: (-0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595)
9: (-0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.86 + 2.63 = 4.50 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3252106, upper bound: 1.3309396
time: 1.22 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3309396, upper bound: 1.3309396
time: 1.21 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.61 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.61
Output dim: 1, lower bound: -1.3252106, upper bound: 1.3309396
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.61
Output dim: 1, lower bound: -1.3309396, upper bound: 1.3309396

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.5600910, 0.6107074, -0.6792153, 0.7483127, -1.3084037, 1.2899227
1: -0.0118597, 1.1067646, -0.2160076, 1.1223013, -1.1341610, 1.3227723
2: -0.4062488, 0.5647241, -0.4806015, 0.7046829, -1.1109316, 1.0453258
3: -0.2885821, 0.4228401, -0.3698099, 0.5288432, -0.8174253, 0.7926499
4: -0.4552769, 0.4922526, -0.5661561, 0.5929308, -1.0482078, 1.0584087
5: -0.4845304, 0.5231618, -0.5966969, 0.6358859, -1.1204163, 1.1198586
6: -0.4522866, 0.5976913, -0.5455428, 0.7156302, -1.1679168, 1.1432341
7: -0.4034567, 0.6169886, -0.4952183, 0.7569211, -1.1603777, 1.1122069
8: -0.4719390, 0.6249494, -0.6211940, 0.7365156, -1.2084546, 1.2461433
9: -0.4793802, 0.5691741, -0.5763472, 0.7067938, -1.1861739, 1.1455212

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 230

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3179355
time: 1.25 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3038354
time: 1.88 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.6199672, 0.6830270, -0.7201101, 0.7847198, -1.4046869, 1.4031370
1: -0.1155903, 1.1148309, -0.2734933, 1.1269344, -1.2425246, 1.3883243
2: -0.4441343, 0.6385815, -0.5034105, 0.7500329, -1.1941671, 1.1419921
3: -0.3261516, 0.4786346, -0.4058629, 0.5595815, -0.8857331, 0.8844975
4: -0.5125127, 0.5421219, -0.6087081, 0.6288819, -1.1413946, 1.1508300
5: -0.5421376, 0.5796869, -0.6259659, 0.6826422, -1.2247798, 1.2056528
6: -0.4994634, 0.6603864, -0.5740133, 0.7451505, -1.2446139, 1.2343998
7: -0.4469948, 0.6892314, -0.5338485, 0.8026321, -1.2496269, 1.2230799
8: -0.5468315, 0.6815135, -0.6814606, 0.7661344, -1.3129659, 1.3629742
9: -0.5293852, 0.6405196, -0.6107609, 0.7424142, -1.2717994, 1.2512804

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3309396, upper bound: 1.3252106
time: 1.44 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3309396, upper bound: 1.3309396
time: 1.44 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.78 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.78
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3179355
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 4.78
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3038354
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.78
Output dim: 1, lower bound: -1.3309396, upper bound: 1.3252106
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.78
Output dim: 1, lower bound: -1.3309396, upper bound: 1.3309396

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.5566272, 0.6064003, -0.5836190, 0.6407256, -1.1973526, 1.1900194
1: -0.0060864, 1.1062855, -0.0673438, 1.1101651, -1.1162515, 1.1736293
2: -0.4040061, 0.5605116, -0.4221041, 0.5945376, -0.9985437, 0.9826157
3: -0.2863833, 0.4194815, -0.3036317, 0.4450496, -0.7314329, 0.7231132
4: -0.4518747, 0.4892785, -0.4779670, 0.5132749, -0.9651496, 0.9672456
5: -0.4811544, 0.5198162, -0.5078886, 0.5459127, -1.0270671, 1.0277047
6: -0.4496066, 0.5939437, -0.4703194, 0.6237649, -1.0733714, 1.0642631
7: -0.4008349, 0.6132606, -0.4202437, 0.6585019, -1.0593367, 1.0335042
8: -0.4674564, 0.6216476, -0.5015241, 0.6477755, -1.1152320, 1.1231717
9: -0.4763896, 0.5650529, -0.5006652, 0.5992055, -1.0755951, 1.0657181

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2934728, upper bound: 1.3163970
time: 1.38 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3179355
time: 1.46 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.6199672, 0.6830270, -0.5600910, 0.6107074, -1.2306746, 1.2431180
1: -0.1155903, 1.1148309, -0.0118597, 1.1067646, -1.2223549, 1.1266905
2: -0.4441343, 0.6385815, -0.4062488, 0.5647241, -1.0088584, 1.0448303
3: -0.3261516, 0.4786346, -0.2885821, 0.4228401, -0.7489917, 0.7672168
4: -0.5125127, 0.5421219, -0.4552769, 0.4922526, -1.0047653, 0.9973989
5: -0.5421376, 0.5796869, -0.4845304, 0.5231618, -1.0652993, 1.0642173
6: -0.4994634, 0.6603864, -0.4522866, 0.5976913, -1.0971547, 1.1126730
7: -0.4469948, 0.6892314, -0.4034567, 0.6169886, -1.0639834, 1.0926881
8: -0.5468315, 0.6815135, -0.4719390, 0.6249494, -1.1717808, 1.1534525
9: -0.5293852, 0.6405196, -0.4793802, 0.5691741, -1.0985593, 1.1198997

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 230

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3179355, upper bound: 1.2940208
time: 1.86 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3038354, upper bound: 1.2940208
time: 1.77 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.6199672, 0.6830270, -0.6199672, 0.6830270, -1.3029943, 1.3029943
1: -0.1155903, 1.1148309, -0.1155903, 1.1148309, -1.2304211, 1.2304211
2: -0.4441343, 0.6385815, -0.4441343, 0.6385815, -1.0827157, 1.0827157
3: -0.3261516, 0.4786346, -0.3261516, 0.4786346, -0.8047862, 0.8047862
4: -0.5125127, 0.5421219, -0.5125127, 0.5421219, -1.0546346, 1.0546346
5: -0.5421376, 0.5796869, -0.5421376, 0.5796869, -1.1218245, 1.1218245
6: -0.4994634, 0.6603864, -0.4994634, 0.6603864, -1.1598499, 1.1598499
7: -0.4469948, 0.6892314, -0.4469948, 0.6892314, -1.1362262, 1.1362262
8: -0.5468315, 0.6815135, -0.5468315, 0.6815135, -1.2283450, 1.2283450
9: -0.5293852, 0.6405196, -0.5293852, 0.6405196, -1.1699047, 1.1699047

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 230

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3179355, upper bound: 1.3038354
time: 2.29 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3038354, upper bound: 1.3038354
time: 1.99 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 6.18 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.18
Output dim: 1, lower bound: -1.2934728, upper bound: 1.3163970
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.18
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3179355
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.18
Output dim: 1, lower bound: -1.3179355, upper bound: 1.2940208
NS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 6.18
Output dim: 1, lower bound: -1.3038354, upper bound: 1.2940208
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.18
Output dim: 1, lower bound: -1.3179355, upper bound: 1.3038354
NS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 6.18
Output dim: 1, lower bound: -1.3038354, upper bound: 1.3038354

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.5359708, 0.5824189, -0.5624803, 0.6147116, -1.1506824, 1.1448991
1: 0.0269089, 1.1033458, -0.0304336, 1.1072524, -1.0803435, 1.1337793
2: -0.3906489, 0.5367982, -0.4085525, 0.5685153, -0.9591642, 0.9453506
3: -0.2738248, 0.3994227, -0.2901121, 0.4249665, -0.6987913, 0.6895348
4: -0.4315488, 0.4716068, -0.4575237, 0.4953170, -0.9268659, 0.9291304
5: -0.4624305, 0.4998228, -0.4875703, 0.5256924, -0.9881229, 0.9873931
6: -0.4341901, 0.5713786, -0.4539331, 0.6011360, -1.0353260, 1.0253117
7: -0.3851972, 0.5923215, -0.4046595, 0.6333213, -1.0185184, 0.9969810
8: -0.4425591, 0.6018406, -0.4746424, 0.6278753, -1.0704343, 1.0764830
9: -0.4586116, 0.5420443, -0.4825547, 0.5739532, -1.0325648, 1.0245990

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2934728, upper bound: 1.3124087
time: 1.32 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2934728, upper bound: 1.3163970
time: 1.30 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.5337406, 0.5801033, -0.5829151, 0.6398898, -1.1736305, 1.1630185
1: 0.0317519, 1.1030662, -0.0661684, 1.1100713, -1.0783194, 1.1692345
2: -0.3890726, 0.5345331, -0.4216688, 0.5936667, -0.9827393, 0.9562020
3: -0.2730072, 0.3973796, -0.3031871, 0.4444034, -0.7174106, 0.7005667
4: -0.4294085, 0.4695599, -0.4773042, 0.5126981, -0.9421066, 0.9468641
5: -0.4604208, 0.4977116, -0.5072356, 0.5452541, -1.0056748, 1.0049472
6: -0.4326363, 0.5689713, -0.4697697, 0.6230379, -1.0556742, 1.0387410
7: -0.3836042, 0.5880114, -0.4197416, 0.6576673, -1.0412714, 1.0077530
8: -0.4403208, 0.5996875, -0.5006523, 0.6471361, -1.0874569, 1.1003399
9: -0.4565496, 0.5397090, -0.5000865, 0.5983859, -1.0549355, 1.0397956

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3141063
time: 1.59 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3179355
time: 1.83 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.5319269, 0.5792231, -0.5566272, 0.6064003, -1.1383272, 1.1358502
1: 0.0232654, 1.1029636, -0.0060864, 1.1062855, -1.0830201, 1.1090500
2: -0.3884865, 0.5334979, -0.4040061, 0.5605116, -0.9489981, 0.9375040
3: -0.2724631, 0.3955417, -0.2863833, 0.4194815, -0.6919447, 0.6819249
4: -0.4275658, 0.4688424, -0.4518747, 0.4892785, -0.9168444, 0.9207171
5: -0.4595576, 0.4961770, -0.4811544, 0.5198162, -0.9793738, 0.9773314
6: -0.4312571, 0.5676770, -0.4496066, 0.5939437, -1.0252008, 1.0172836
7: -0.3817583, 0.5965335, -0.4008349, 0.6132606, -0.9950190, 0.9973683
8: -0.4383688, 0.5984533, -0.4674564, 0.6216476, -1.0600164, 1.0659096
9: -0.4560432, 0.5395763, -0.4763896, 0.5650529, -1.0210960, 1.0159659

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3179355, upper bound: 1.2940208
time: 1.59 seconds

## Relational analysis of NS_A2_B1_A1_A2

### Relational analysis result of NS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3179355, upper bound: 1.2940208
time: 1.74 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.5319269, 0.5792231, -0.6162727, 0.6786109, -1.2105379, 1.1954958
1: 0.0232654, 1.1029636, -0.1094486, 1.1143456, -1.0910802, 1.2124121
2: -0.3884865, 0.5334979, -0.4417984, 0.6340393, -1.0225258, 0.9752963
3: -0.2724631, 0.3955417, -0.3238474, 0.4752129, -0.7476761, 0.7193891
4: -0.4275658, 0.4688424, -0.5089602, 0.5391225, -0.9666884, 0.9778026
5: -0.4595576, 0.4961770, -0.5384184, 0.5762362, -1.0357938, 1.0345955
6: -0.4312571, 0.5676770, -0.4964301, 0.6565983, -1.0878555, 1.0641072
7: -0.3817583, 0.5965335, -0.4442691, 0.6852144, -1.0669727, 1.0408025
8: -0.4383688, 0.5984533, -0.5422481, 0.6778173, -1.1161861, 1.1407014
9: -0.4560432, 0.5395763, -0.5263677, 0.6360234, -1.0920665, 1.0659440

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3210300, upper bound: 1.3038354
time: 1.73 seconds

## Relational analysis of NS_A2_B2_A1_A2

### Relational analysis result of NS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3210359, upper bound: 1.3038354
time: 1.77 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 5.34 seconds
NS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 5.34
Output dim: 1, lower bound: -1.2934728, upper bound: 1.3124087
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.34
Output dim: 1, lower bound: -1.2934728, upper bound: 1.3163970
NS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 5.34
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3141063
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.34
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3179355
NS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 5.34
Output dim: 1, lower bound: -1.3179355, upper bound: 1.2940208
NS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 5.34
Output dim: 1, lower bound: -1.3179355, upper bound: 1.2940208
NS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 5.34
Output dim: 1, lower bound: -1.3210300, upper bound: 1.3038354
NS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 5.34
Output dim: 1, lower bound: -1.3210359, upper bound: 1.3038354

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.5359708, 0.5824189, -0.5111784, 0.5577636, -1.0937344, 1.0935974
1: 0.0269089, 1.1033458, 0.0544443, 1.1000473, -1.0731385, 1.0489014
2: -0.3906489, 0.5367982, -0.3749264, 0.5118566, -0.9025055, 0.9117246
3: -0.2738248, 0.3994227, -0.2613841, 0.3760073, -0.6498321, 0.6608068
4: -0.4315488, 0.4716068, -0.4072045, 0.4514893, -0.8830382, 0.8788113
5: -0.4624305, 0.4998228, -0.4421038, 0.4766369, -0.9390674, 0.9419266
6: -0.4341901, 0.5713786, -0.4166759, 0.5450221, -0.9792122, 0.9880545
7: -0.3851972, 0.5923215, -0.3661570, 0.5756422, -0.9608394, 0.9584786
8: -0.4425591, 0.6018406, -0.4170480, 0.5785314, -1.0210905, 1.0188886
9: -0.4586116, 0.5420443, -0.4380647, 0.5184230, -0.9770346, 0.9801090

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 217

## Relational analysis of NS_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2934728, upper bound: 1.3163970
time: 1.33 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2934728, upper bound: 1.3163970
time: 1.42 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.5337406, 0.5801033, -0.5312601, 0.5785346, -1.1122752, 1.1113634
1: 0.0317519, 1.1030662, 0.0242562, 1.1028700, -1.0711181, 1.0788100
2: -0.3890726, 0.5345331, -0.3880513, 0.5328032, -0.9218758, 0.9225844
3: -0.2730072, 0.3973796, -0.2721088, 0.3949141, -0.6679213, 0.6694884
4: -0.4294085, 0.4695599, -0.4269114, 0.4682858, -0.8976942, 0.8964713
5: -0.4604208, 0.4977116, -0.4589974, 0.4955454, -0.9559661, 0.9567090
6: -0.4326363, 0.5689713, -0.4307886, 0.5669497, -0.9995860, 0.9997599
7: -0.3836042, 0.5880114, -0.3812566, 0.5958712, -0.9794753, 0.9692680
8: -0.4403208, 0.5996875, -0.4376837, 0.5978137, -1.0381346, 1.0373712
9: -0.4565496, 0.5397090, -0.4554692, 0.5388991, -0.9954487, 0.9951782

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 217

## Relational analysis of NS_A1_B1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3179355
time: 1.39 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3179355
time: 1.52 seconds

## BFS NS instance: NS_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.5089711, 0.5556483, -0.5359248, 0.5823253, -1.0912964, 1.0915730
1: 0.0544052, 1.0997175, 0.0288265, 1.1033713, -1.0489661, 1.0708910
2: -0.3736412, 0.5096530, -0.3904791, 0.5367792, -0.9104204, 0.9001321
3: -0.2599269, 0.3738821, -0.2741256, 0.3994395, -0.6593664, 0.6480078
4: -0.4049840, 0.4498860, -0.4315558, 0.4713601, -0.8763441, 0.8814418
5: -0.4404315, 0.4744489, -0.4622352, 0.4997769, -0.9402084, 0.9366841
6: -0.4151016, 0.5427112, -0.4341749, 0.5713304, -0.9864321, 0.9768862
7: -0.3643802, 0.5765744, -0.3852647, 0.5899265, -0.9543066, 0.9618391
8: -0.4147213, 0.5765094, -0.4425697, 0.6017658, -1.0164871, 1.0190791
9: -0.4362389, 0.5164666, -0.4584090, 0.5418674, -0.9781063, 0.9748756

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 217

## Relational analysis of NS_A2_B1_A1_A1_B1

### Relational analysis result of NS_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3167880, upper bound: 1.2940208
time: 1.59 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2

### Relational analysis result of NS_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3179355, upper bound: 1.2940208
time: 1.58 seconds

## BFS NS instance: NS_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.5080959, 0.5546181, -0.5559875, 0.6055978, -1.1136937, 1.1106055
1: 0.0586885, 1.0996149, -0.0049440, 1.1061953, -1.0475068, 1.1045588
2: -0.3729362, 0.5086761, -0.4035886, 0.5597267, -0.9326628, 0.9122647
3: -0.2597930, 0.3731009, -0.2859719, 0.4188614, -0.6786544, 0.6590728
4: -0.4041750, 0.4489416, -0.4512466, 0.4887250, -0.8929000, 0.9001882
5: -0.4395361, 0.4738726, -0.4805275, 0.5191966, -0.9587327, 0.9544001
6: -0.4145058, 0.5416850, -0.4491126, 0.5932455, -1.0077513, 0.9907976
7: -0.3638188, 0.5728714, -0.4003532, 0.6125097, -0.9763286, 0.9732246
8: -0.4138752, 0.5755936, -0.4666297, 0.6210339, -1.0349090, 1.0422232
9: -0.4355216, 0.5153548, -0.4758342, 0.5642802, -0.9998018, 0.9911889

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 217

## Relational analysis of NS_A2_B1_A1_A2_B1

### Relational analysis result of NS_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3167880, upper bound: 1.2940208
time: 1.55 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2

### Relational analysis result of NS_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3179355, upper bound: 1.2940208
time: 1.70 seconds

## BFS NS instance: NS_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.5089711, 0.5556483, -0.5942690, 0.6522570, -1.1612281, 1.1499174
1: 0.0544052, 1.0997175, -0.0711237, 1.1114092, -1.0570040, 1.1708412
2: -0.3736412, 0.5096530, -0.4279551, 0.6067816, -0.9804228, 0.9376081
3: -0.2599269, 0.3738821, -0.3099001, 0.4549386, -0.7148655, 0.6837822
4: -0.4049840, 0.4498860, -0.4880311, 0.5210609, -0.9260449, 0.9379171
5: -0.4404315, 0.4744489, -0.5170826, 0.5556321, -0.9960636, 0.9915315
6: -0.4151016, 0.5427112, -0.4788408, 0.6338319, -1.0489335, 1.0215520
7: -0.3643802, 0.5765744, -0.4283801, 0.6588243, -1.0232044, 1.0049546
8: -0.4147213, 0.5765094, -0.5150120, 0.6567606, -1.0714819, 1.0915214
9: -0.4362389, 0.5164666, -0.5082582, 0.6095885, -1.0458274, 1.0247248

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 217

## Relational analysis of NS_A2_B2_A1_A1_B1

### Relational analysis result of NS_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3204445, upper bound: 1.3038354
time: 2.01 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2

### Relational analysis result of NS_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3210300, upper bound: 1.3038354
time: 1.71 seconds

## BFS NS instance: NS_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.5080959, 0.5546181, -0.6155738, 0.6777685, -1.1858644, 1.1701918
1: 0.0586885, 1.0996149, -0.1082081, 1.1142524, -1.0555639, 1.2078229
2: -0.3729362, 0.5086761, -0.4413519, 0.6331742, -1.0061103, 0.9500279
3: -0.2597930, 0.3731009, -0.3234054, 0.4745660, -0.7343590, 0.6965063
4: -0.4041750, 0.4489416, -0.5082879, 0.5385493, -0.9427243, 0.9572295
5: -0.4395361, 0.4738726, -0.5377108, 0.5755817, -1.0151178, 1.0115833
6: -0.4145058, 0.5416850, -0.4958572, 0.6558757, -1.0703816, 1.0375422
7: -0.3638188, 0.5728714, -0.4437564, 0.6843848, -1.0482036, 1.0166278
8: -0.4138752, 0.5755936, -0.5413824, 0.6771141, -1.0909894, 1.1169760
9: -0.4355216, 0.5153548, -0.5257930, 0.6351595, -1.0706811, 1.0411477

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 217

## Relational analysis of NS_A2_B2_A1_A2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3204445, upper bound: 1.3038354
time: 1.74 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2

### Relational analysis result of NS_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3210359, upper bound: 1.3038354
time: 1.68 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.24 seconds
NS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 1, lower bound: -1.2934728, upper bound: 1.3163970
NS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 1, lower bound: -1.2934728, upper bound: 1.3163970
NS_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3179355
NS_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3179355
NS_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 1, lower bound: -1.3167880, upper bound: 1.2940208
NS_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 1, lower bound: -1.3179355, upper bound: 1.2940208
NS_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 1, lower bound: -1.3167880, upper bound: 1.2940208
NS_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 1, lower bound: -1.3179355, upper bound: 1.2940208
NS_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 1, lower bound: -1.3204445, upper bound: 1.3038354
NS_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 1, lower bound: -1.3210300, upper bound: 1.3038354
NS_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 1, lower bound: -1.3204445, upper bound: 1.3038354
NS_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 1, lower bound: -1.3210359, upper bound: 1.3038354

## BFS NS instance: NS_A1_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.5236636, 0.5695848, -0.4636731, 0.5064614, -1.0301250, 1.0332578
1: 0.0464727, 1.1016070, 0.1290958, 1.0941297, -1.0476570, 0.9725112
2: -0.3825613, 0.5238751, -0.3425300, 0.4614716, -0.8440329, 0.8664051
3: -0.2672290, 0.3878395, -0.2349656, 0.3304005, -0.5976295, 0.6228051
4: -0.4194828, 0.4612495, -0.3607326, 0.4100166, -0.8294994, 0.8219821
5: -0.4520263, 0.4881359, -0.4003922, 0.4307064, -0.8827327, 0.8885282
6: -0.4255446, 0.5578691, -0.3818131, 0.4925461, -0.9180907, 0.9396822
7: -0.3759742, 0.5790245, -0.3288448, 0.5261226, -0.9020969, 0.9078693
8: -0.4299211, 0.5899684, -0.3660956, 0.5326337, -0.9625547, 0.9560640
9: -0.4479873, 0.5293438, -0.3957516, 0.4678778, -0.9158651, 0.9250954

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_A1_B2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2602195, upper bound: 1.2944729
time: 1.44 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2934728, upper bound: 1.3163970
time: 1.40 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.5315273, 0.5777867, -0.4866615, 0.5314576, -1.0629849, 1.0644481
1: 0.0340035, 1.1027186, 0.0945388, 1.0968542, -1.0628507, 1.0081798
2: -0.3877271, 0.5321351, -0.3583340, 0.4858378, -0.8735649, 0.8904691
3: -0.2714550, 0.3952416, -0.2478983, 0.3526277, -0.6240827, 0.6431400
4: -0.4271926, 0.4678645, -0.3832129, 0.4302394, -0.8574320, 0.8510773
5: -0.4586722, 0.4956032, -0.4207595, 0.4531127, -0.9117849, 0.9163626
6: -0.4310688, 0.5665027, -0.3989345, 0.5179152, -0.9489841, 0.9654372
7: -0.3818682, 0.5874839, -0.3472258, 0.5485812, -0.9304495, 0.9347097
8: -0.4379968, 0.5975547, -0.3911136, 0.5547950, -0.9927918, 0.9886684
9: -0.4547732, 0.5374604, -0.4161556, 0.4924040, -0.9471772, 0.9536160

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_A1_B2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2602195, upper bound: 1.2944729
time: 1.27 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2934728, upper bound: 1.3163970
time: 1.29 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.5208316, 0.5666386, -0.4819598, 0.5264870, -1.0473186, 1.0485984
1: 0.0523192, 1.1012396, 0.0999865, 1.0963242, -1.0440049, 1.0012530
2: -0.3805902, 0.5209765, -0.3551681, 0.4809905, -0.8615807, 0.8761446
3: -0.2660812, 0.3852300, -0.2453127, 0.3480676, -0.6141487, 0.6305427
4: -0.4167515, 0.4586978, -0.3786127, 0.4261956, -0.8429471, 0.8373105
5: -0.4495094, 0.4854519, -0.4166677, 0.4486606, -0.8981701, 0.9021196
6: -0.4235670, 0.5548007, -0.3954148, 0.5128335, -0.9364005, 0.9502156
7: -0.3739284, 0.5740952, -0.3434046, 0.5453074, -0.9192358, 0.9174998
8: -0.4270632, 0.5872338, -0.3859842, 0.5503387, -0.9774019, 0.9732180
9: -0.4454057, 0.5263902, -0.4121363, 0.4876176, -0.9330232, 0.9385265

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_A2_B2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2873330, upper bound: 1.2962926
time: 1.72 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_B2

### Relational analysis result of NS_A1_B1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3179355
time: 1.53 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.5291389, 0.5753057, -0.5057616, 0.5519804, -1.0811193, 1.0810673
1: 0.0390732, 1.1024154, 0.0647011, 1.0992770, -1.0602038, 1.0377142
2: -0.3860480, 0.5297031, -0.3713025, 0.5060596, -0.8921077, 0.9010055
3: -0.2705483, 0.3930494, -0.2585052, 0.3709176, -0.6414658, 0.6515546
4: -0.4248972, 0.4656858, -0.4019116, 0.4468358, -0.8717331, 0.8675973
5: -0.4565298, 0.4933414, -0.4374521, 0.4715293, -0.9280591, 0.9307935
6: -0.4294035, 0.5639216, -0.4128788, 0.5389856, -0.9683892, 0.9768004
7: -0.3801556, 0.5830317, -0.3621447, 0.5683970, -0.9485525, 0.9451764
8: -0.4355954, 0.5952489, -0.4115027, 0.5732332, -1.0088286, 1.0067517
9: -0.4525753, 0.5349636, -0.4330632, 0.5126354, -0.9652107, 0.9680268

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 230

## Relational analysis of NS_A1_B1_A2_B2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3179355
time: 1.55 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3179355
time: 1.52 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.4969252, 0.5429235, -0.4900588, 0.5342579, -1.0311831, 1.0329823
1: 0.0738707, 1.0980746, 0.0995532, 1.0970882, -1.0232174, 0.9985214
2: -0.3656223, 0.4969363, -0.3601754, 0.4886651, -0.8542874, 0.8571117
3: -0.2533855, 0.3624807, -0.2493618, 0.3560323, -0.6094178, 0.6118425
4: -0.3931844, 0.4396157, -0.3865961, 0.4325664, -0.8257508, 0.8262118
5: -0.4301167, 0.4628758, -0.4232252, 0.4558412, -0.8859578, 0.8861011
6: -0.4065333, 0.5294395, -0.4015682, 0.5210432, -0.9275765, 0.9310077
7: -0.3552409, 0.5633604, -0.3503738, 0.5429649, -0.8982059, 0.9137342
8: -0.4021941, 0.5648668, -0.3949189, 0.5576044, -0.9597985, 0.9597858
9: -0.4255238, 0.5038677, -0.4186925, 0.4945267, -0.9200505, 0.9225602

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_A1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2946007, upper bound: 1.2846301
time: 1.70 seconds

## Relational analysis of NS_A2_B1_A1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3167880, upper bound: 1.2940208
time: 1.75 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.5044761, 0.5509656, -0.5108692, 0.5562087, -1.0606847, 1.0618348
1: 0.0615809, 1.0990847, 0.0686061, 1.0998279, -1.0382470, 1.0304787
2: -0.3706865, 0.5049374, -0.3740159, 0.5104873, -0.8811738, 0.8789532
3: -0.2575313, 0.3696528, -0.2607377, 0.3758604, -0.6333917, 0.6303905
4: -0.4005781, 0.4461009, -0.4069910, 0.4502766, -0.8508547, 0.8530920
5: -0.4366308, 0.4701809, -0.4410574, 0.4759845, -0.9126153, 0.9112384
6: -0.4119453, 0.5377803, -0.4165713, 0.5438412, -0.9557865, 0.9543515
7: -0.3610137, 0.5716773, -0.3664819, 0.5629153, -0.9239289, 0.9381592
8: -0.4101073, 0.5721756, -0.4168391, 0.5776028, -0.9877101, 0.9890147
9: -0.4322841, 0.5118319, -0.4367770, 0.5160432, -0.9483273, 0.9486089

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_A1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2958528, upper bound: 1.2846301
time: 1.78 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3179355, upper bound: 1.2940208
time: 1.82 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.4954782, 0.5412195, -0.5089271, 0.5543306, -1.0498089, 1.0501466
1: 0.0791359, 1.0979170, 0.0700128, 1.0995702, -1.0204343, 1.0279042
2: -0.3644927, 0.4953310, -0.3728465, 0.5085770, -0.8730697, 0.8681775
3: -0.2529013, 0.3611288, -0.2597435, 0.3740061, -0.6269073, 0.6208723
4: -0.3918183, 0.4381295, -0.4050795, 0.4487818, -0.8406001, 0.8432090
5: -0.4286753, 0.4618331, -0.4395320, 0.4741649, -0.9028402, 0.9013652
6: -0.4054807, 0.5277659, -0.4151838, 0.5418135, -0.9472942, 0.9429497
7: -0.3541903, 0.5590453, -0.3649352, 0.5625196, -0.9167099, 0.9239805
8: -0.4006810, 0.5633901, -0.4148262, 0.5758059, -0.9764869, 0.9782163
9: -0.4242274, 0.5020942, -0.4351947, 0.5143205, -0.9385479, 0.9372889

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_A1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2951756, upper bound: 1.2876766
time: 1.97 seconds

## Relational analysis of NS_A2_B1_A1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3167880, upper bound: 1.2940208
time: 1.77 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.5034521, 0.5497794, -0.5307298, 0.5767708, -1.0802230, 1.0805092
1: 0.0660850, 1.0989603, 0.0383545, 1.1026266, -1.0365416, 1.0606058
2: -0.3698841, 0.5038040, -0.3869973, 0.5312194, -0.9011034, 0.8908012
3: -0.2573126, 0.3687309, -0.2713650, 0.3945620, -0.6518745, 0.6400959
4: -0.3996223, 0.4450329, -0.4264784, 0.4668923, -0.8665146, 0.8715113
5: -0.4356104, 0.4695351, -0.4577700, 0.4948130, -0.9304234, 0.9273051
6: -0.4112444, 0.5365903, -0.4305307, 0.5655408, -0.9767852, 0.9671210
7: -0.3603396, 0.5678397, -0.3814179, 0.5829607, -0.9433002, 0.9492577
8: -0.4091071, 0.5711157, -0.4372500, 0.5966811, -1.0057882, 1.0083658
9: -0.4314324, 0.5105666, -0.4538525, 0.5363149, -0.9677473, 0.9644191

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 230

## Relational analysis of NS_A2_B1_A1_A2_B2_B1

### Relational analysis result of NS_A2_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3179355, upper bound: 1.2940208
time: 1.63 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2_B2

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3179355, upper bound: 1.2940208
time: 1.71 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.4969252, 0.5429235, -0.5447036, 0.5915207, -1.0884459, 1.0876272
1: 0.0738707, 1.0980746, 0.0151584, 1.1045932, -1.0307224, 1.0829161
2: -0.3656223, 0.4969363, -0.3962368, 0.5459374, -0.9115597, 0.8931730
3: -0.2533855, 0.3624807, -0.2783684, 0.4078941, -0.6612796, 0.6408490
4: -0.3931844, 0.4396157, -0.4401668, 0.4789951, -0.8721795, 0.8797825
5: -0.4301167, 0.4628758, -0.4696465, 0.5082506, -0.9383672, 0.9325224
6: -0.4065333, 0.5294395, -0.4404496, 0.5808281, -0.9873614, 0.9698890
7: -0.3552409, 0.5633604, -0.3918501, 0.5998964, -0.9551373, 0.9552105
8: -0.4021941, 0.5648668, -0.4522401, 0.6101419, -1.0123360, 1.0171070
9: -0.4255238, 0.5038677, -0.4660689, 0.5506709, -0.9761947, 0.9699366

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_A1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2970830, upper bound: 1.2912687
time: 2.08 seconds

## Relational analysis of NS_A2_B2_A1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3204445, upper bound: 1.3038354
time: 1.84 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.5044761, 0.5509656, -0.5680363, 0.6204430, -1.1249191, 1.1190019
1: 0.0615809, 1.0990847, -0.0241520, 1.1078504, -1.0462695, 1.1232367
2: -0.3706865, 0.5049374, -0.4113927, 0.5742399, -0.9449263, 0.9163301
3: -0.2575313, 0.3696528, -0.2932460, 0.4305193, -0.6880506, 0.6628988
4: -0.4005781, 0.4461009, -0.4630776, 0.4990926, -0.8996707, 0.9091785
5: -0.4366308, 0.4701809, -0.4922821, 0.5308097, -0.9674405, 0.9624630
6: -0.4119453, 0.5377803, -0.4584314, 0.6061757, -1.0181210, 0.9962117
7: -0.3610137, 0.5716773, -0.4094744, 0.6257339, -0.9867475, 0.9811517
8: -0.4101073, 0.5721756, -0.4822144, 0.6324508, -1.0425581, 1.0543900
9: -0.4322841, 0.5118319, -0.4862618, 0.5784352, -1.0107193, 0.9980937

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 230

## Relational analysis of NS_A2_B2_A1_A1_B2_B1

### Relational analysis result of NS_A2_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3210300, upper bound: 1.3038354
time: 1.65 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2_B2

### Relational analysis result of NS_A2_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3210300, upper bound: 1.3038354
time: 1.81 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.4954782, 0.5412195, -0.5642322, 0.6158370, -1.1113153, 1.1054517
1: 0.0791359, 1.0979170, -0.0195584, 1.1073411, -1.0282052, 1.1174753
2: -0.3644927, 0.4953310, -0.4090044, 0.5697302, -0.9342229, 0.9043354
3: -0.2529013, 0.3611288, -0.2908283, 0.4268133, -0.6797146, 0.6519572
4: -0.3918183, 0.4381295, -0.4593379, 0.4959229, -0.8877412, 0.8974674
5: -0.4286753, 0.4618331, -0.4886380, 0.5271684, -0.9558437, 0.9504711
6: -0.4054807, 0.5277659, -0.4554650, 0.6021655, -1.0076462, 0.9832309
7: -0.3541903, 0.5590453, -0.4065316, 0.6230137, -0.9772040, 0.9655769
8: -0.4006810, 0.5633901, -0.4772540, 0.6289029, -1.0295839, 1.0406442
9: -0.4242274, 0.5020942, -0.4830405, 0.5741217, -0.9983491, 0.9851347

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_A1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2977580, upper bound: 1.2953678
time: 2.18 seconds

## Relational analysis of NS_A2_B2_A1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3204445, upper bound: 1.3038354
time: 1.75 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.5034521, 0.5497794, -0.5887982, 0.6455985, -1.1490506, 1.1385777
1: 0.0660850, 1.0989603, -0.0601727, 1.1106710, -1.0445861, 1.1591330
2: -0.3698841, 0.5038040, -0.4244859, 0.5998881, -0.9697722, 0.9282899
3: -0.2573126, 0.3687309, -0.3064822, 0.4499249, -0.7072375, 0.6752131
4: -0.3996223, 0.4450329, -0.4828985, 0.5164502, -0.9160725, 0.9279314
5: -0.4356104, 0.4695351, -0.5119177, 0.5504732, -0.9860836, 0.9814528
6: -0.4112444, 0.5365903, -0.4745851, 0.6280543, -1.0392988, 1.0111754
7: -0.3603396, 0.5678397, -0.4245357, 0.6505935, -1.0109330, 0.9923754
8: -0.4091071, 0.5711157, -0.5082787, 0.6516898, -1.0607969, 1.0793945
9: -0.4314324, 0.5105666, -0.5036650, 0.6029748, -1.0344071, 1.0142317

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 230

## Relational analysis of NS_A2_B2_A1_A2_B2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3210359, upper bound: 1.3038354
time: 5.10 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_B2

### Relational analysis result of NS_A2_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3210359, upper bound: 1.3038354
time: 1.84 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 9.06 seconds
NS_A1_B1_A1_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 9.06
Output dim: 1, lower bound: -1.2602195, upper bound: 1.2944729
NS_A1_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 9.06
Output dim: 1, lower bound: -1.2934728, upper bound: 1.3163970
NS_A1_B1_A1_B2_B2_A1, status: Status.VERIFIED, split count: 6, time: 9.06
Output dim: 1, lower bound: -1.2602195, upper bound: 1.2944729
NS_A1_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 9.06
Output dim: 1, lower bound: -1.2934728, upper bound: 1.3163970
NS_A1_B1_A2_B2_B1_B1, status: Status.VERIFIED, split count: 6, time: 9.06
Output dim: 1, lower bound: -1.2873330, upper bound: 1.2962926
NS_A1_B1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 9.06
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3179355
NS_A1_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 9.06
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3179355
NS_A1_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 9.06
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3179355
NS_A2_B1_A1_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 9.06
Output dim: 1, lower bound: -1.2946007, upper bound: 1.2846301
NS_A2_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 9.06
Output dim: 1, lower bound: -1.3167880, upper bound: 1.2940208
NS_A2_B1_A1_A1_B2_A1, status: Status.VERIFIED, split count: 6, time: 9.06
Output dim: 1, lower bound: -1.2958528, upper bound: 1.2846301
NS_A2_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 9.06
Output dim: 1, lower bound: -1.3179355, upper bound: 1.2940208
NS_A2_B1_A1_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 9.06
Output dim: 1, lower bound: -1.2951756, upper bound: 1.2876766
NS_A2_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 9.06
Output dim: 1, lower bound: -1.3167880, upper bound: 1.2940208
NS_A2_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 9.06
Output dim: 1, lower bound: -1.3179355, upper bound: 1.2940208
NS_A2_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 9.06
Output dim: 1, lower bound: -1.3179355, upper bound: 1.2940208
NS_A2_B2_A1_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 9.06
Output dim: 1, lower bound: -1.2970830, upper bound: 1.2912687
NS_A2_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 9.06
Output dim: 1, lower bound: -1.3204445, upper bound: 1.3038354
NS_A2_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 9.06
Output dim: 1, lower bound: -1.3210300, upper bound: 1.3038354
NS_A2_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 9.06
Output dim: 1, lower bound: -1.3210300, upper bound: 1.3038354
NS_A2_B2_A1_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 9.06
Output dim: 1, lower bound: -1.2977580, upper bound: 1.2953678
NS_A2_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 9.06
Output dim: 1, lower bound: -1.3204445, upper bound: 1.3038354
NS_A2_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 9.06
Output dim: 1, lower bound: -1.3210359, upper bound: 1.3038354
NS_A2_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 9.06
Output dim: 1, lower bound: -1.3210359, upper bound: 1.3038354

## BFS NS instance: NS_A1_B1_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.4976032, 0.5412019, -0.4636731, 0.5064614, -1.0040646, 1.0048749
1: 0.1010497, 1.0975657, 0.1290958, 1.0941297, -0.9930800, 0.9684699
2: -0.3647572, 0.4954609, -0.3425300, 0.4614716, -0.8262288, 0.8379909
3: -0.2529820, 0.3634610, -0.2349656, 0.3304005, -0.5833825, 0.5984266
4: -0.3939459, 0.4384053, -0.3607326, 0.4100166, -0.8039626, 0.7991379
5: -0.4292126, 0.4631453, -0.4003922, 0.4307064, -0.8599190, 0.8635375
6: -0.4073245, 0.5284358, -0.3818131, 0.4925461, -0.8998706, 0.9102489
7: -0.3569316, 0.5388317, -0.3288448, 0.5261226, -0.8830542, 0.8676764
8: -0.4031583, 0.5642452, -0.3660956, 0.5326337, -0.9357920, 0.9303408
9: -0.4248061, 0.5004928, -0.3957516, 0.4678778, -0.8926839, 0.8962443

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2861321, upper bound: 1.2923757
time: 1.60 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2861321, upper bound: 1.3163970
time: 1.94 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.5053723, 0.5493285, -0.4866615, 0.5314576, -1.0368299, 1.0359900
1: 0.0885868, 1.0986685, 0.0945388, 1.0968542, -1.0082674, 1.0041296
2: -0.3698707, 0.5036298, -0.3583340, 0.4858378, -0.8557085, 0.8619638
3: -0.2571736, 0.3707772, -0.2478983, 0.3526277, -0.6098013, 0.6186755
4: -0.4015606, 0.4449536, -0.3832129, 0.4302394, -0.8318000, 0.8281664
5: -0.4357916, 0.4705329, -0.4207595, 0.4531127, -0.8889043, 0.8912925
6: -0.4127927, 0.5369732, -0.3989345, 0.5179152, -0.9307079, 0.9359078
7: -0.3627637, 0.5471736, -0.3472258, 0.5485812, -0.9113449, 0.8943994
8: -0.4111483, 0.5717471, -0.3911136, 0.5547950, -0.9659433, 0.9628607
9: -0.4315248, 0.5085297, -0.4161556, 0.4924040, -0.9239289, 0.9246854

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2863988, upper bound: 1.2923757
time: 1.90 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2863988, upper bound: 1.3163970
time: 1.91 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.5208316, 0.5666386, -0.4564242, 0.4970009, -1.0178325, 1.0230628
1: 0.0523192, 1.1012396, 0.1577360, 1.0929112, -1.0405920, 0.9435035
2: -0.3805902, 0.5209765, -0.3366998, 0.4525022, -0.8330923, 0.8576763
3: -0.2660812, 0.3852300, -0.2303572, 0.3236118, -0.5896930, 0.6155872
4: -0.4167515, 0.4586978, -0.3538352, 0.4024940, -0.8192455, 0.8125330
5: -0.4495094, 0.4854519, -0.3930120, 0.4222236, -0.8717331, 0.8784640
6: -0.4235670, 0.5548007, -0.3765635, 0.4835171, -0.9070841, 0.9313643
7: -0.3739284, 0.5740952, -0.3237260, 0.5035393, -0.8774678, 0.8978212
8: -0.4270632, 0.5872338, -0.3583039, 0.5248247, -0.9518880, 0.9455377
9: -0.4454057, 0.5263902, -0.3885528, 0.4575875, -0.9029931, 0.9149430

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_A2_B2_B1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2642203, upper bound: 1.3003487
time: 1.66 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2642203, upper bound: 1.3179355
time: 1.49 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.4507602, 0.4922982, -0.5057616, 0.5519804, -1.0027406, 0.9980599
1: 0.1512642, 1.0927367, 0.0647011, 1.0992770, -0.9480128, 1.0280356
2: -0.3334853, 0.4479538, -0.3713025, 0.5060596, -0.8395449, 0.8192562
3: -0.2279799, 0.3180028, -0.2585052, 0.3709176, -0.5988975, 0.5765080
4: -0.3484447, 0.3984188, -0.4019116, 0.4468358, -0.7952806, 0.8003304
5: -0.3887716, 0.4187043, -0.4374521, 0.4715293, -0.8603009, 0.8561563
6: -0.3721896, 0.4784056, -0.4128788, 0.5389856, -0.9111753, 0.8912843
7: -0.3185889, 0.5113881, -0.3621447, 0.5683970, -0.8869858, 0.8735329
8: -0.3520579, 0.5200887, -0.4115027, 0.5732332, -0.9252911, 0.9315914
9: -0.3846697, 0.4538932, -0.4330632, 0.5126354, -0.8973050, 0.8869565

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2876766, upper bound: 1.2962926
time: 1.60 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3179355
time: 1.23 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.4118216, 0.4507490, -0.5057616, 0.5519804, -0.9638020, 0.9565106
1: 0.1988588, 1.0893105, 0.0647011, 1.0992770, -0.9004183, 1.0246093
2: -0.3070550, 0.4107959, -0.3713025, 0.5060596, -0.8131146, 0.7820983
3: -0.2069236, 0.2802933, -0.2585052, 0.3709176, -0.5778412, 0.5387985
4: -0.3116455, 0.3645775, -0.4019116, 0.4468358, -0.7584813, 0.7664890
5: -0.3547037, 0.3825886, -0.4374521, 0.4715293, -0.8262330, 0.8200407
6: -0.3428573, 0.4367436, -0.4128788, 0.5389856, -0.8818429, 0.8496223
7: -0.2868362, 0.4842940, -0.3621447, 0.5683970, -0.8552332, 0.8464388
8: -0.3097364, 0.4828212, -0.4115027, 0.5732332, -0.8829696, 0.8943239
9: -0.3537707, 0.4140335, -0.4330632, 0.5126354, -0.8664061, 0.8470967

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_A2_B2_B2_A2_A1

### Relational analysis result of NS_A1_B1_A2_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2642203, upper bound: 1.3003466
time: 3.57 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_A2_A2

### Relational analysis result of NS_A1_B1_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3179355
time: 1.45 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.4717157, 0.5140141, -0.4900588, 0.5342579, -1.0059736, 1.0040729
1: 0.1303866, 1.0946854, 0.0995532, 1.0970882, -0.9667016, 0.9951323
2: -0.3474411, 0.4688564, -0.3601754, 0.4886651, -0.8361062, 0.8290318
3: -0.2387603, 0.3383325, -0.2493618, 0.3560323, -0.5947926, 0.5876943
4: -0.3685658, 0.4162898, -0.3865961, 0.4325664, -0.8011322, 0.8028859
5: -0.4068288, 0.4375032, -0.4232252, 0.4558412, -0.8626699, 0.8607284
6: -0.3879540, 0.5005443, -0.4015682, 0.5210432, -0.9089972, 0.9021124
7: -0.3358265, 0.5224375, -0.3503738, 0.5429649, -0.8787915, 0.8728112
8: -0.3748931, 0.5397864, -0.3949189, 0.5576044, -0.9324975, 0.9347053
9: -0.4023169, 0.4744403, -0.4186925, 0.4945267, -0.8968436, 0.8931328

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_A1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2988142, upper bound: 1.2642203
time: 2.21 seconds

## Relational analysis of NS_A2_B1_A1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2988142, upper bound: 1.2940208
time: 1.86 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.4790125, 0.5220827, -0.5108692, 0.5562087, -1.0352211, 1.0329520
1: 0.1180222, 1.0955698, 0.0686061, 1.0998279, -0.9818057, 1.0269637
2: -0.3525190, 0.4767044, -0.3740159, 0.5104873, -0.8630063, 0.8507203
3: -0.2429261, 0.3453759, -0.2607377, 0.3758604, -0.6187865, 0.6061136
4: -0.3756931, 0.4227908, -0.4069910, 0.4502766, -0.8259696, 0.8297818
5: -0.4133590, 0.4447988, -0.4410574, 0.4759845, -0.8893435, 0.8858563
6: -0.3933792, 0.5086889, -0.4165713, 0.5438412, -0.9372203, 0.9252602
7: -0.3416108, 0.5307057, -0.3664819, 0.5629153, -0.9045261, 0.8971876
8: -0.3828212, 0.5468925, -0.4168391, 0.5776028, -0.9604240, 0.9637316
9: -0.4088534, 0.4824251, -0.4367770, 0.5160432, -0.9248965, 0.9192021

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_A1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2998979, upper bound: 1.2642203
time: 1.58 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2998979, upper bound: 1.2940208
time: 2.40 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.4696727, 0.5115258, -0.5089271, 0.5543306, -1.0240033, 1.0204529
1: 0.1367386, 1.0944096, 0.0700128, 1.0995702, -0.9628315, 1.0243968
2: -0.3458682, 0.4665022, -0.3728465, 0.5085770, -0.8544452, 0.8393487
3: -0.2378523, 0.3364035, -0.2597435, 0.3740061, -0.6118584, 0.5961471
4: -0.3666200, 0.4142280, -0.4050795, 0.4487818, -0.8154018, 0.8193076
5: -0.4048102, 0.4354247, -0.4395320, 0.4741649, -0.8789751, 0.8749567
6: -0.3864535, 0.4981180, -0.4151838, 0.5418135, -0.9282670, 0.9133019
7: -0.3343161, 0.5170068, -0.3649352, 0.5625196, -0.8968357, 0.8819420
8: -0.3727204, 0.5376713, -0.4148262, 0.5758059, -0.9485263, 0.9524975
9: -0.4002903, 0.4718558, -0.4351947, 0.5143205, -0.9146107, 0.9070505

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_A1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2993237, upper bound: 1.2642203
time: 1.67 seconds

## Relational analysis of NS_A2_B1_A1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2993237, upper bound: 1.2940208
time: 2.16 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.5034521, 0.5497794, -0.4523562, 0.4938715, -0.9973236, 1.0021355
1: 0.0660850, 1.0989603, 0.1507265, 1.0928903, -1.0268053, 0.9482338
2: -0.3698841, 0.5038040, -0.3345054, 0.4494804, -0.8193645, 0.8383094
3: -0.2573126, 0.3687309, -0.2288367, 0.3195572, -0.5768698, 0.5975676
4: -0.3996223, 0.4450329, -0.3499819, 0.3997155, -0.7993377, 0.7950149
5: -0.4356104, 0.4695351, -0.3901005, 0.4201046, -0.8557150, 0.8596357
6: -0.4112444, 0.5365903, -0.3733964, 0.4800335, -0.8912779, 0.9099867
7: -0.3603396, 0.5678397, -0.3199374, 0.5113049, -0.8716444, 0.8877772
8: -0.4091071, 0.5711157, -0.3538229, 0.5215430, -0.9306501, 0.9249387
9: -0.4314324, 0.5105666, -0.3857024, 0.4553435, -0.8867759, 0.8962690

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_A1_A2_B2_B1_A1

### Relational analysis result of NS_A2_B1_A1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2962926, upper bound: 1.2876766
time: 2.03 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2_B1_A2

### Relational analysis result of NS_A2_B1_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3179355, upper bound: 1.2940208
time: 2.06 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.5034521, 0.5497794, -0.4136339, 0.4525276, -0.9559797, 0.9634133
1: 0.0660850, 1.0989603, 0.1983559, 1.0894860, -1.0234010, 0.9006044
2: -0.3698841, 0.5038040, -0.3082157, 0.4122961, -0.7821801, 0.8120197
3: -0.2573126, 0.3687309, -0.2078697, 0.2820550, -0.5393676, 0.5766006
4: -0.3996223, 0.4450329, -0.3133922, 0.3660452, -0.7656676, 0.7584251
5: -0.4356104, 0.4695351, -0.3562112, 0.3841321, -0.8197425, 0.8257463
6: -0.4112444, 0.5365903, -0.3442317, 0.4385809, -0.8498254, 0.8808221
7: -0.3603396, 0.5678397, -0.2883692, 0.4841652, -0.8445047, 0.8562089
8: -0.4091071, 0.5711157, -0.3117437, 0.4844669, -0.8935740, 0.8828595
9: -0.4314324, 0.5105666, -0.3549318, 0.4156649, -0.8470972, 0.8654984

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_B1

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3003466, upper bound: 1.2642203
time: 1.95 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_B2

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3179355, upper bound: 1.2940208
time: 1.62 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.4717157, 0.5140141, -0.5447036, 0.5915207, -1.0632365, 1.0587177
1: 0.1303866, 1.0946854, 0.0151584, 1.1045932, -0.9742066, 1.0795270
2: -0.3474411, 0.4688564, -0.3962368, 0.5459374, -0.8933786, 0.8650932
3: -0.2387603, 0.3383325, -0.2783684, 0.4078941, -0.6466544, 0.6167008
4: -0.3685658, 0.4162898, -0.4401668, 0.4789951, -0.8475609, 0.8564566
5: -0.4068288, 0.4375032, -0.4696465, 0.5082506, -0.9150794, 0.9071498
6: -0.3879540, 0.5005443, -0.4404496, 0.5808281, -0.9687821, 0.9409938
7: -0.3358265, 0.5224375, -0.3918501, 0.5998964, -0.9357229, 0.9142876
8: -0.3748931, 0.5397864, -0.4522401, 0.6101419, -0.9850349, 0.9920266
9: -0.4023169, 0.4744403, -0.4660689, 0.5506709, -0.9529877, 0.9405092

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_A1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3037741, upper bound: 1.2722414
time: 1.65 seconds

## Relational analysis of NS_A2_B2_A1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3037741, upper bound: 1.3038354
time: 1.90 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.5044761, 0.5509656, -0.4866615, 0.5314576, -1.0359337, 1.0376271
1: 0.0615809, 1.0990847, 0.0945388, 1.0968542, -1.0352733, 1.0045459
2: -0.3706865, 0.5049374, -0.3583340, 0.4858378, -0.8565242, 0.8632714
3: -0.2575313, 0.3696528, -0.2478983, 0.3526277, -0.6101590, 0.6175511
4: -0.4005781, 0.4461009, -0.3832129, 0.4302394, -0.8308175, 0.8293138
5: -0.4366308, 0.4701809, -0.4207595, 0.4531127, -0.8897435, 0.8909404
6: -0.4119453, 0.5377803, -0.3989345, 0.5179152, -0.9298606, 0.9367148
7: -0.3610137, 0.5716773, -0.3472258, 0.5485812, -0.9095949, 0.9189031
8: -0.4101073, 0.5721756, -0.3911136, 0.5547950, -0.9649023, 0.9632893
9: -0.4322841, 0.5118319, -0.4161556, 0.4924040, -0.9246881, 0.9279875

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_A1_A1_B2_B1_A1

### Relational analysis result of NS_A2_B2_A1_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2980847, upper bound: 1.2912688
time: 1.73 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2_B1_A2

### Relational analysis result of NS_A2_B2_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3210300, upper bound: 1.3038354
time: 1.70 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.5044761, 0.5509656, -0.4498065, 0.4922269, -0.9967030, 1.0007721
1: 0.0615809, 1.0990847, 0.1398905, 1.0933566, -1.0317757, 0.9591942
2: -0.3706865, 0.5049374, -0.3333498, 0.4491442, -0.8198307, 0.8382871
3: -0.2575313, 0.3696528, -0.2280732, 0.3169137, -0.5744450, 0.5977260
4: -0.4005781, 0.4461009, -0.3478066, 0.3982305, -0.7988086, 0.7939075
5: -0.4366308, 0.4701809, -0.3885369, 0.4191990, -0.8558298, 0.8587179
6: -0.4119453, 0.5377803, -0.3712578, 0.4782004, -0.8901457, 0.9090381
7: -0.3610137, 0.5716773, -0.3172958, 0.5215938, -0.8826075, 0.8889730
8: -0.4101073, 0.5721756, -0.3511316, 0.5196081, -0.9297154, 0.9233072
9: -0.4322841, 0.5118319, -0.3866844, 0.4547167, -0.8870008, 0.8985163

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_A1_A1_B2_B2_B1

### Relational analysis result of NS_A2_B2_A1_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3053147, upper bound: 1.2722414
time: 1.74 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2_B2_B2

### Relational analysis result of NS_A2_B2_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3210300, upper bound: 1.3038354
time: 3.78 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.4696727, 0.5115258, -0.5642322, 0.6158370, -1.0855098, 1.0757580
1: 0.1367386, 1.0944096, -0.0195584, 1.1073411, -0.9706024, 1.1139679
2: -0.3458682, 0.4665022, -0.4090044, 0.5697302, -0.9155983, 0.8755066
3: -0.2378523, 0.3364035, -0.2908283, 0.4268133, -0.6646656, 0.6272319
4: -0.3666200, 0.4142280, -0.4593379, 0.4959229, -0.8625429, 0.8735659
5: -0.4048102, 0.4354247, -0.4886380, 0.5271684, -0.9319786, 0.9240627
6: -0.3864535, 0.4981180, -0.4554650, 0.6021655, -0.9886190, 0.9535831
7: -0.3343161, 0.5170068, -0.4065316, 0.6230137, -0.9573299, 0.9235384
8: -0.3727204, 0.5376713, -0.4772540, 0.6289029, -1.0016233, 1.0149254
9: -0.4002903, 0.4718558, -0.4830405, 0.5741217, -0.9744120, 0.9548963

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_A1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3043612, upper bound: 1.2722414
time: 2.08 seconds

## Relational analysis of NS_A2_B2_A1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3043612, upper bound: 1.3038354
time: 2.10 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.5034521, 0.5497794, -0.5057616, 0.5519804, -1.0554326, 1.0555410
1: 0.0660850, 1.0989603, 0.0647011, 1.0992770, -1.0331920, 1.0342591
2: -0.3698841, 0.5038040, -0.3713025, 0.5060596, -0.8759437, 0.8751065
3: -0.2573126, 0.3687309, -0.2585052, 0.3709176, -0.6282302, 0.6272361
4: -0.3996223, 0.4450329, -0.4019116, 0.4468358, -0.8464582, 0.8469445
5: -0.4356104, 0.4695351, -0.4374521, 0.4715293, -0.9071398, 0.9069872
6: -0.4112444, 0.5365903, -0.4128788, 0.5389856, -0.9502300, 0.9494691
7: -0.3603396, 0.5678397, -0.3621447, 0.5683970, -0.9287366, 0.9299845
8: -0.4091071, 0.5711157, -0.4115027, 0.5732332, -0.9823403, 0.9826185
9: -0.4314324, 0.5105666, -0.4330632, 0.5126354, -0.9440677, 0.9436298

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_A1_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_A1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2985506, upper bound: 1.2953678
time: 1.91 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3210359, upper bound: 1.3038354
time: 1.87 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.5034521, 0.5497794, -0.4689400, 0.5132192, -1.0166714, 1.0187194
1: 0.0660850, 1.0989603, 0.1095104, 1.0954794, -1.0293944, 0.9894499
2: -0.3698841, 0.5038040, -0.3466184, 0.4690445, -0.8389286, 0.8504223
3: -0.2573126, 0.3687309, -0.2389302, 0.3353671, -0.5926797, 0.6076611
4: -0.3996223, 0.4450329, -0.3661454, 0.4152110, -0.8148333, 0.8111783
5: -0.4356104, 0.4695351, -0.4056133, 0.4380153, -0.8736258, 0.8751484
6: -0.4112444, 0.5365903, -0.3855340, 0.4992392, -0.9104836, 0.9221244
7: -0.3603396, 0.5678397, -0.3325789, 0.5411037, -0.9014432, 0.9004186
8: -0.4091071, 0.5711157, -0.3719761, 0.5381780, -0.9472851, 0.9430918
9: -0.4314324, 0.5105666, -0.4037425, 0.4754081, -0.9068404, 0.9143091

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_A1_A2_B2_B2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3057681, upper bound: 1.2722414
time: 1.87 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_B2_B2

### Relational analysis result of NS_A2_B2_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3210359, upper bound: 1.3038354
time: 1.63 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 5.69 seconds
NS_A1_B1_A1_B2_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 1, lower bound: -1.2861321, upper bound: 1.2923757
NS_A1_B1_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.69
Output dim: 1, lower bound: -1.2861321, upper bound: 1.3163970
NS_A1_B1_A1_B2_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 1, lower bound: -1.2863988, upper bound: 1.2923757
NS_A1_B1_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.69
Output dim: 1, lower bound: -1.2863988, upper bound: 1.3163970
NS_A1_B1_A2_B2_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 1, lower bound: -1.2642203, upper bound: 1.3003487
NS_A1_B1_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.69
Output dim: 1, lower bound: -1.2642203, upper bound: 1.3179355
NS_A1_B1_A2_B2_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 1, lower bound: -1.2876766, upper bound: 1.2962926
NS_A1_B1_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.69
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3179355
NS_A1_B1_A2_B2_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 1, lower bound: -1.2642203, upper bound: 1.3003466
NS_A1_B1_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.69
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3179355
NS_A2_B1_A1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 1, lower bound: -1.2988142, upper bound: 1.2642203
NS_A2_B1_A1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 1, lower bound: -1.2988142, upper bound: 1.2940208
NS_A2_B1_A1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 1, lower bound: -1.2998979, upper bound: 1.2642203
NS_A2_B1_A1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 1, lower bound: -1.2998979, upper bound: 1.2940208
NS_A2_B1_A1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 1, lower bound: -1.2993237, upper bound: 1.2642203
NS_A2_B1_A1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 1, lower bound: -1.2993237, upper bound: 1.2940208
NS_A2_B1_A1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 1, lower bound: -1.2962926, upper bound: 1.2876766
NS_A2_B1_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.69
Output dim: 1, lower bound: -1.3179355, upper bound: 1.2940208
NS_A2_B1_A1_A2_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 1, lower bound: -1.3003466, upper bound: 1.2642203
NS_A2_B1_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.69
Output dim: 1, lower bound: -1.3179355, upper bound: 1.2940208
NS_A2_B2_A1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 1, lower bound: -1.3037741, upper bound: 1.2722414
NS_A2_B2_A1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 1, lower bound: -1.3037741, upper bound: 1.3038354
NS_A2_B2_A1_A1_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 1, lower bound: -1.2980847, upper bound: 1.2912688
NS_A2_B2_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.69
Output dim: 1, lower bound: -1.3210300, upper bound: 1.3038354
NS_A2_B2_A1_A1_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 1, lower bound: -1.3053147, upper bound: 1.2722414
NS_A2_B2_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.69
Output dim: 1, lower bound: -1.3210300, upper bound: 1.3038354
NS_A2_B2_A1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 1, lower bound: -1.3043612, upper bound: 1.2722414
NS_A2_B2_A1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 1, lower bound: -1.3043612, upper bound: 1.3038354
NS_A2_B2_A1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 1, lower bound: -1.2985506, upper bound: 1.2953678
NS_A2_B2_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.69
Output dim: 1, lower bound: -1.3210359, upper bound: 1.3038354
NS_A2_B2_A1_A2_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 1, lower bound: -1.3057681, upper bound: 1.2722414
NS_A2_B2_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.69
Output dim: 1, lower bound: -1.3210359, upper bound: 1.3038354

## BFS NS instance: NS_A1_B1_A1_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.4976032, 0.5412019, -0.4383603, 0.4771355, -0.9747387, 0.9795622
1: 0.1010497, 1.0975657, 0.1867004, 1.0909578, -0.9899081, 0.9108652
2: -0.3647572, 0.4954609, -0.3241676, 0.4335284, -0.7982856, 0.8196286
3: -0.2529820, 0.3634610, -0.2200601, 0.3061879, -0.5591699, 0.5835211
4: -0.3939459, 0.4384053, -0.3366429, 0.3864498, -0.7803957, 0.7750481
5: -0.4292126, 0.4631453, -0.3768767, 0.4043428, -0.8335554, 0.8400219
6: -0.4073245, 0.5284358, -0.3630565, 0.4636914, -0.8710159, 0.8914923
7: -0.3569316, 0.5388317, -0.3092609, 0.4854263, -0.8423579, 0.8480926
8: -0.4031583, 0.5642452, -0.3386118, 0.5072618, -0.9104201, 0.9028570
9: -0.4248061, 0.5004928, -0.3726639, 0.4380332, -0.8628393, 0.8731566

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 230

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2861321, upper bound: 1.3163970
time: 2.54 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2861321, upper bound: 1.3163970
time: 2.12 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.5053723, 0.5493285, -0.4609815, 0.5018930, -1.0072653, 1.0103101
1: 0.0885868, 1.0986685, 0.1517130, 1.0933654, -1.0047786, 0.9469554
2: -0.3698707, 0.5036298, -0.3398001, 0.4571700, -0.8270407, 0.8434299
3: -0.2571736, 0.3707772, -0.2329021, 0.3280224, -0.5851960, 0.6036793
4: -0.4015606, 0.4449536, -0.3581706, 0.4064540, -0.8080146, 0.8031242
5: -0.4357916, 0.4705329, -0.3970063, 0.4267280, -0.8625195, 0.8675393
6: -0.4127927, 0.5369732, -0.3799877, 0.4884244, -0.9012170, 0.9169609
7: -0.3627637, 0.5471736, -0.3274313, 0.5069483, -0.8697120, 0.8746048
8: -0.4111483, 0.5717471, -0.3632787, 0.5291944, -0.9403427, 0.9350258
9: -0.4315248, 0.5085297, -0.3924963, 0.4623132, -0.8938380, 0.9010260

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 230

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2863988, upper bound: 1.3163970
time: 1.90 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2863988, upper bound: 1.3163970
time: 1.72 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.4947543, 0.5378940, -0.4564242, 0.4970009, -0.9917552, 0.9943181
1: 0.1077393, 1.0971915, 0.1577360, 1.0929112, -0.9851720, 0.9394554
2: -0.3626146, 0.4923242, -0.3366998, 0.4525022, -0.8151168, 0.8290240
3: -0.2516344, 0.3607724, -0.2303572, 0.3236118, -0.5752462, 0.5911296
4: -0.3912132, 0.4356283, -0.3538352, 0.4024940, -0.7937073, 0.7894635
5: -0.4264724, 0.4602815, -0.3930120, 0.4222236, -0.8486960, 0.8532935
6: -0.4052232, 0.5251771, -0.3765635, 0.4835171, -0.8887403, 0.9017407
7: -0.3547880, 0.5328362, -0.3237260, 0.5035393, -0.8583273, 0.8565622
8: -0.4001114, 0.5613880, -0.3583039, 0.5248247, -0.9249362, 0.9196919
9: -0.4220546, 0.4971047, -0.3885528, 0.4575875, -0.8796421, 0.8856575

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 230

## Relational analysis of NS_A1_B1_A2_B2_B1_B2_A2_A1

### Relational analysis result of NS_A1_B1_A2_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2642203, upper bound: 1.3179355
time: 2.13 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_B2_A2_A2

### Relational analysis result of NS_A1_B1_A2_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2642203, upper bound: 1.3179355
time: 2.22 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.4507602, 0.4922982, -0.4795894, 0.5222655, -0.9730257, 0.9718876
1: 0.1512642, 1.0927367, 0.1219975, 1.0955881, -0.9443239, 0.9707392
2: -0.3334853, 0.4479538, -0.3526646, 0.4769843, -0.8104696, 0.8006184
3: -0.2279799, 0.3180028, -0.2434595, 0.3459948, -0.5739747, 0.5614623
4: -0.3484447, 0.3984188, -0.3763299, 0.4229207, -0.7713654, 0.7747487
5: -0.3887716, 0.4187043, -0.4135728, 0.4452986, -0.8340702, 0.8322771
6: -0.3721896, 0.4784056, -0.3938413, 0.5090361, -0.8812257, 0.8722469
7: -0.3185889, 0.5113881, -0.3422615, 0.5262406, -0.8448294, 0.8536496
8: -0.3520579, 0.5200887, -0.3835237, 0.5472079, -0.8992658, 0.9036124
9: -0.3846697, 0.4538932, -0.4090680, 0.4823717, -0.8670415, 0.8629612

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3011778, upper bound: 1.3183554
time: 1.69 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3011778, upper bound: 1.3219162
time: 1.75 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.3870632, 0.4219994, -0.5057616, 0.5519804, -0.9390435, 0.9277610
1: 0.2558922, 1.0861112, 0.0647011, 1.0992770, -0.8433849, 1.0214100
2: -0.2890756, 0.3819105, -0.3713025, 0.5060596, -0.7951353, 0.7532130
3: -0.1920240, 0.2566100, -0.2585052, 0.3709176, -0.5629416, 0.5151151
4: -0.2880441, 0.3415043, -0.4019116, 0.4468358, -0.7348800, 0.7434158
5: -0.3316787, 0.3561159, -0.4374521, 0.4715293, -0.8032080, 0.7935680
6: -0.3245516, 0.4084516, -0.4128788, 0.5389856, -0.8635373, 0.8213304
7: -0.2677214, 0.4439036, -0.3621447, 0.5683970, -0.8361183, 0.8060483
8: -0.2828393, 0.4579817, -0.4115027, 0.5732332, -0.8560725, 0.8694844
9: -0.3280032, 0.3846691, -0.4330632, 0.5126354, -0.8406386, 0.8177323

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_A2_B2_B2_A2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2876766, upper bound: 1.2962926
time: 1.71 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_A2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2876766, upper bound: 1.3179355
time: 1.90 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.4774697, 0.5201433, -0.4523562, 0.4938715, -0.9713411, 0.9724994
1: 0.1235183, 1.0953584, 0.1507265, 1.0928903, -0.9693720, 0.9446319
2: -0.3512909, 0.4748844, -0.3345054, 0.4494804, -0.8007713, 0.8093898
3: -0.2423010, 0.3439299, -0.2288367, 0.3195572, -0.5618582, 0.5727666
4: -0.3742348, 0.4211727, -0.3499819, 0.3997155, -0.7739503, 0.7711546
5: -0.4117870, 0.4432196, -0.3901005, 0.4201046, -0.8318915, 0.8333201
6: -0.3922521, 0.5068179, -0.3733964, 0.4800335, -0.8722856, 0.8802143
7: -0.3405009, 0.5257887, -0.3199374, 0.5113049, -0.8518058, 0.8457261
8: -0.3811941, 0.5452612, -0.3538229, 0.5215430, -0.9027371, 0.8990842
9: -0.4072751, 0.4803803, -0.3857024, 0.4553435, -0.8626186, 0.8660827

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_A1_A2_B2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3183554, upper bound: 1.3011778
time: 2.15 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3183554, upper bound: 1.3164195
time: 3.54 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.5034521, 0.5497794, -0.3888060, 0.4236670, -0.9271191, 0.9385854
1: 0.0660850, 1.0989603, 0.2555143, 1.0862572, -1.0201722, 0.8434460
2: -0.3698841, 0.5038040, -0.2901651, 0.3833129, -0.7531970, 0.7939690
3: -0.2573126, 0.3687309, -0.1929125, 0.2583126, -0.5156252, 0.5616434
4: -0.3996223, 0.4450329, -0.2897259, 0.3428826, -0.7425050, 0.7347588
5: -0.4356104, 0.4695351, -0.3330986, 0.3575686, -0.7931790, 0.8026337
6: -0.4112444, 0.5365903, -0.3258739, 0.4101963, -0.8214407, 0.8624642
7: -0.3603396, 0.5678397, -0.2692118, 0.4434388, -0.8037784, 0.8370515
8: -0.4091071, 0.5711157, -0.2847706, 0.4595459, -0.8686529, 0.8558863
9: -0.4314324, 0.5105666, -0.3294047, 0.3861737, -0.8176061, 0.8399713

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_B2_A1

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2962926, upper bound: 1.2876766
time: 1.85 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_B2_A2

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2962926, upper bound: 1.2940208
time: 2.01 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.4790125, 0.5220827, -0.4866615, 0.5314576, -1.0104702, 1.0087442
1: 0.1180222, 1.0955698, 0.0945388, 1.0968542, -0.9788320, 1.0010310
2: -0.3525190, 0.4767044, -0.3583340, 0.4858378, -0.8383567, 0.8350384
3: -0.2429261, 0.3453759, -0.2478983, 0.3526277, -0.5955538, 0.5932742
4: -0.3756931, 0.4227908, -0.3832129, 0.4302394, -0.8059325, 0.8060036
5: -0.4133590, 0.4447988, -0.4207595, 0.4531127, -0.8664717, 0.8655583
6: -0.3933792, 0.5086889, -0.3989345, 0.5179152, -0.9112945, 0.9076235
7: -0.3416108, 0.5307057, -0.3472258, 0.5485812, -0.8901920, 0.8779315
8: -0.3828212, 0.5468925, -0.3911136, 0.5547950, -0.9376162, 0.9380061
9: -0.4088534, 0.4824251, -0.4161556, 0.4924040, -0.9012573, 0.8985807

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_A1_A1_B2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3194258, upper bound: 1.3037133
time: 1.84 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3194258, upper bound: 1.3219162
time: 1.75 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.5044761, 0.5509656, -0.4243139, 0.4627862, -0.9672623, 0.9752795
1: 0.0615809, 1.0990847, 0.1976863, 1.0901189, -1.0285380, 0.9013984
2: -0.3706865, 0.5049374, -0.3148894, 0.4200093, -0.7906957, 0.8198268
3: -0.2575313, 0.3696528, -0.2129396, 0.2925202, -0.5500515, 0.5825924
4: -0.4005781, 0.4461009, -0.3235008, 0.3745518, -0.7751299, 0.7696017
5: -0.4366308, 0.4701809, -0.3648991, 0.3923786, -0.8290094, 0.8350800
6: -0.4119453, 0.5377803, -0.3524112, 0.4491811, -0.8611264, 0.8901915
7: -0.3610137, 0.5716773, -0.2975834, 0.4807875, -0.8418011, 0.8692607
8: -0.4101073, 0.5721756, -0.3234432, 0.4941097, -0.9042170, 0.8956189
9: -0.4322841, 0.5118319, -0.3607434, 0.4246919, -0.8569760, 0.8725753

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_A1_A1_B2_B2_B2_A1

### Relational analysis result of NS_A2_B2_A1_A1_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2980847, upper bound: 1.2912688
time: 3.01 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2_B2_B2_A2

### Relational analysis result of NS_A2_B2_A1_A1_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2980847, upper bound: 1.3038354
time: 2.86 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.4774697, 0.5201433, -0.5057616, 0.5519804, -1.0294501, 1.0259049
1: 0.1235183, 1.0953584, 0.0647011, 1.0992770, -0.9757588, 1.0306573
2: -0.3512909, 0.4748844, -0.3713025, 0.5060596, -0.8573505, 0.8461868
3: -0.2423010, 0.3439299, -0.2585052, 0.3709176, -0.6132185, 0.6024351
4: -0.3742348, 0.4211727, -0.4019116, 0.4468358, -0.8210707, 0.8230842
5: -0.4117870, 0.4432196, -0.4374521, 0.4715293, -0.8833163, 0.8806716
6: -0.3922521, 0.5068179, -0.4128788, 0.5389856, -0.9312377, 0.9196967
7: -0.3405009, 0.5257887, -0.3621447, 0.5683970, -0.9088979, 0.8879334
8: -0.3811941, 0.5452612, -0.4115027, 0.5732332, -0.9544272, 0.9567640
9: -0.4072751, 0.4803803, -0.4330632, 0.5126354, -0.9199104, 0.9134435

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_A1_A2_B2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3194258, upper bound: 1.3037133
time: 1.72 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3194258, upper bound: 1.3219162
time: 1.96 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.5034521, 0.5497794, -0.4433549, 0.4836905, -0.9871426, 0.9931343
1: 0.0660850, 1.0989603, 0.1673493, 1.0921822, -1.0260972, 0.9316109
2: -0.3698841, 0.5038040, -0.3280977, 0.4399850, -0.8098691, 0.8319017
3: -0.2573126, 0.3687309, -0.2237861, 0.3108808, -0.5681934, 0.5925170
4: -0.3996223, 0.4450329, -0.3416336, 0.3914540, -0.7910763, 0.7866665
5: -0.4356104, 0.4695351, -0.3818991, 0.4111936, -0.8468040, 0.8514342
6: -0.4112444, 0.5365903, -0.3666428, 0.4700547, -0.8812991, 0.9032332
7: -0.3603396, 0.5678397, -0.3128262, 0.4997778, -0.8601173, 0.8806659
8: -0.4091071, 0.5711157, -0.3442032, 0.5126029, -0.9217100, 0.9153190
9: -0.4314324, 0.5105666, -0.3774804, 0.4452704, -0.8767028, 0.8880470

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_A1_A2_B2_B2_B2_A1

### Relational analysis result of NS_A2_B2_A1_A2_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2985506, upper bound: 1.2953678
time: 1.78 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_B2_B2_A2

### Relational analysis result of NS_A2_B2_A1_A2_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2985506, upper bound: 1.3038354
time: 2.25 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 6.34 seconds
NS_A1_B1_A1_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 6.34
Output dim: 1, lower bound: -1.2861321, upper bound: 1.3163970
NS_A1_B1_A1_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 6.34
Output dim: 1, lower bound: -1.2861321, upper bound: 1.3163970
NS_A1_B1_A1_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 6.34
Output dim: 1, lower bound: -1.2863988, upper bound: 1.3163970
NS_A1_B1_A1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 6.34
Output dim: 1, lower bound: -1.2863988, upper bound: 1.3163970
NS_A1_B1_A2_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 6.34
Output dim: 1, lower bound: -1.2642203, upper bound: 1.3179355
NS_A1_B1_A2_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 6.34
Output dim: 1, lower bound: -1.2642203, upper bound: 1.3179355
NS_A1_B1_A2_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 6.34
Output dim: 1, lower bound: -1.3011778, upper bound: 1.3183554
NS_A1_B1_A2_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 6.34
Output dim: 1, lower bound: -1.3011778, upper bound: 1.3219162
NS_A1_B1_A2_B2_B2_A2_A2_B1, status: Status.VERIFIED, split count: 8, time: 6.34
Output dim: 1, lower bound: -1.2876766, upper bound: 1.2962926
NS_A1_B1_A2_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.34
Output dim: 1, lower bound: -1.2876766, upper bound: 1.3179355
NS_A2_B1_A1_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.34
Output dim: 1, lower bound: -1.3183554, upper bound: 1.3011778
NS_A2_B1_A1_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.34
Output dim: 1, lower bound: -1.3183554, upper bound: 1.3164195
NS_A2_B1_A1_A2_B2_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 6.34
Output dim: 1, lower bound: -1.2962926, upper bound: 1.2876766
NS_A2_B1_A1_A2_B2_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 6.34
Output dim: 1, lower bound: -1.2962926, upper bound: 1.2940208
NS_A2_B2_A1_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.34
Output dim: 1, lower bound: -1.3194258, upper bound: 1.3037133
NS_A2_B2_A1_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.34
Output dim: 1, lower bound: -1.3194258, upper bound: 1.3219162
NS_A2_B2_A1_A1_B2_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 6.34
Output dim: 1, lower bound: -1.2980847, upper bound: 1.2912688
NS_A2_B2_A1_A1_B2_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 6.34
Output dim: 1, lower bound: -1.2980847, upper bound: 1.3038354
NS_A2_B2_A1_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.34
Output dim: 1, lower bound: -1.3194258, upper bound: 1.3037133
NS_A2_B2_A1_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.34
Output dim: 1, lower bound: -1.3194258, upper bound: 1.3219162
NS_A2_B2_A1_A2_B2_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 6.34
Output dim: 1, lower bound: -1.2985506, upper bound: 1.2953678
NS_A2_B2_A1_A2_B2_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 6.34
Output dim: 1, lower bound: -1.2985506, upper bound: 1.3038354

## BFS NS instance: NS_A1_B1_A1_B2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.4195511, 0.4566368, -0.4383603, 0.4771355, -0.8966867, 0.8949971
1: 0.2152798, 1.0889233, 0.1867004, 1.0909578, -0.8756779, 0.9022229
2: -0.3111796, 0.4139076, -0.3241676, 0.4335284, -0.7447081, 0.7380753
3: -0.2094079, 0.2880500, -0.2200601, 0.3061879, -0.5155958, 0.5081102
4: -0.3187003, 0.3698373, -0.3366429, 0.3864498, -0.7051501, 0.7064802
5: -0.3601501, 0.3859894, -0.3768767, 0.4043428, -0.7644929, 0.7628661
6: -0.3489935, 0.4431649, -0.3630565, 0.4636914, -0.8126850, 0.8062215
7: -0.2941607, 0.4678082, -0.3092609, 0.4854263, -0.7795870, 0.7770691
8: -0.3180781, 0.4890719, -0.3386118, 0.5072618, -0.8253399, 0.8276837
9: -0.3561982, 0.4179266, -0.3726639, 0.4380332, -0.7942314, 0.7905905

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B2_A1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2934728, upper bound: 1.3159074
time: 1.42 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B2_A1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2923241, upper bound: 1.3158269
time: 1.65 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.3826607, 0.4172364, -0.4383603, 0.4771355, -0.8597962, 0.8555967
1: 0.2614568, 1.0856235, 0.1867004, 1.0909578, -0.8295010, 0.8989230
2: -0.2861410, 0.3769957, -0.3241676, 0.4335284, -0.7196695, 0.7011633
3: -0.1893389, 0.2523209, -0.2200601, 0.3061879, -0.4955267, 0.4723810
4: -0.2837990, 0.3377727, -0.3366429, 0.3864498, -0.6702488, 0.6744156
5: -0.3278795, 0.3514731, -0.3768767, 0.4043428, -0.7322224, 0.7283497
6: -0.3212488, 0.4036347, -0.3630565, 0.4636914, -0.7849402, 0.7666912
7: -0.2641141, 0.4414544, -0.3092609, 0.4854263, -0.7495404, 0.7507153
8: -0.2779790, 0.4537472, -0.3386118, 0.5072618, -0.7852408, 0.7923589
9: -0.3242825, 0.3800508, -0.3726639, 0.4380332, -0.7623158, 0.7527147

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B2_A2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2884928, upper bound: 1.2949342
time: 1.93 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B2_A2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2660204, upper bound: 1.2929015
time: 1.43 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.4268915, 0.4647911, -0.4609815, 0.5018930, -0.9287845, 0.9257726
1: 0.2027959, 1.0897297, 0.1517130, 1.0933654, -0.8905695, 0.9380167
2: -0.3163114, 0.4216826, -0.3398001, 0.4571700, -0.7734815, 0.7614827
3: -0.2136319, 0.2951229, -0.2329021, 0.3280224, -0.5416543, 0.5280250
4: -0.3256792, 0.3764105, -0.3581706, 0.4064540, -0.7321332, 0.7345811
5: -0.3667505, 0.3933562, -0.3970063, 0.4267280, -0.7934785, 0.7903626
6: -0.3544782, 0.4512754, -0.3799877, 0.4884244, -0.8429025, 0.8312631
7: -0.3000087, 0.4758574, -0.3274313, 0.5069483, -0.8069570, 0.8032887
8: -0.3260714, 0.4962513, -0.3632787, 0.5291944, -0.8552658, 0.8595301
9: -0.3626973, 0.4259938, -0.3924963, 0.4623132, -0.8250105, 0.8184901

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2_A1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2934728, upper bound: 1.3159074
time: 1.52 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2_A1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2923241, upper bound: 1.3158269
time: 1.53 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.3900038, 0.4253857, -0.4609815, 0.5018930, -0.8918967, 0.8863672
1: 0.2489527, 1.0864346, 0.1517130, 1.0933654, -0.8444127, 0.9347216
2: -0.2912732, 0.3847993, -0.3398001, 0.4571700, -0.7484432, 0.7245994
3: -0.1935736, 0.2593949, -0.2329021, 0.3280224, -0.5215960, 0.4922971
4: -0.2907842, 0.3443463, -0.3581706, 0.4064540, -0.6972381, 0.7025169
5: -0.3344799, 0.3588617, -0.3970063, 0.4267280, -0.7612078, 0.7558680
6: -0.3267310, 0.4117451, -0.3799877, 0.4884244, -0.8151554, 0.7917328
7: -0.2699609, 0.4495219, -0.3274313, 0.5069483, -0.7769091, 0.7769532
8: -0.2859757, 0.4609244, -0.3632787, 0.5291944, -0.8151700, 0.8242031
9: -0.3307793, 0.3881198, -0.3924963, 0.4623132, -0.7930925, 0.7806162

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2_A2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2903861, upper bound: 1.2958187
time: 1.87 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2_A2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2660204, upper bound: 1.2937811
time: 1.46 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.4184496, 0.4551106, -0.4564242, 0.4970009, -0.9154506, 0.9115348
1: 0.2202593, 1.0887475, 0.1577360, 1.0929112, -0.8726519, 0.9310114
2: -0.3102138, 0.4125317, -0.3366998, 0.4525022, -0.7627159, 0.7492315
3: -0.2089160, 0.2870457, -0.2303572, 0.3236118, -0.5325278, 0.5174029
4: -0.3177019, 0.3685614, -0.3538352, 0.4024940, -0.7201959, 0.7223965
5: -0.3589278, 0.3850805, -0.3930120, 0.4222236, -0.7811514, 0.7780926
6: -0.3481911, 0.4417605, -0.3765635, 0.4835171, -0.8317082, 0.8183240
7: -0.2934254, 0.4629646, -0.3237260, 0.5035393, -0.7969648, 0.7866906
8: -0.3169301, 0.4878358, -0.3583039, 0.5248247, -0.8417549, 0.8461397
9: -0.3549829, 0.4162546, -0.3885528, 0.4575875, -0.8125704, 0.8048073

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A2_B2_B1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2930208, upper bound: 1.3176706
time: 1.62 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2930208, upper bound: 1.3174784
time: 1.55 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.3793514, 0.4134324, -0.4564242, 0.4970009, -0.8763524, 0.8698566
1: 0.2690878, 1.0852551, 0.1577360, 1.0929112, -0.8238235, 0.9275191
2: -0.2836862, 0.3737034, -0.3366998, 0.4525022, -0.7361884, 0.7104032
3: -0.1875729, 0.2491793, -0.2303572, 0.3236118, -0.5111847, 0.4795365
4: -0.2807067, 0.3346017, -0.3538352, 0.4024940, -0.6832007, 0.6884369
5: -0.3247475, 0.3483454, -0.3930120, 0.4222236, -0.7469711, 0.7413574
6: -0.3187922, 0.3999296, -0.3765635, 0.4835171, -0.8023093, 0.7764932
7: -0.2615775, 0.4354421, -0.3237260, 0.5035393, -0.7651168, 0.7591681
8: -0.2744384, 0.4504401, -0.3583039, 0.5248247, -0.7992632, 0.8087441
9: -0.3211792, 0.3761925, -0.3885528, 0.4575875, -0.7787667, 0.7647452

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A2_B2_B1_B2_A2_A2_A1

### Relational analysis result of NS_A1_B1_A2_B2_B1_B2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2887963, upper bound: 1.2965754
time: 1.97 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_B2_A2_A2_A2

### Relational analysis result of NS_A1_B1_A2_B2_B1_B2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2660204, upper bound: 1.2943401
time: 1.58 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.4131809, 0.4569310, -0.4795894, 0.5222655, -0.9354464, 0.9365204
1: 0.3335374, 1.0902195, 0.1219975, 1.0955881, -0.7620507, 0.9682220
2: -0.3071634, 0.4202842, -0.3526646, 0.4769843, -0.7841477, 0.7729488
3: -0.2058630, 0.3016580, -0.2434595, 0.3459948, -0.5518578, 0.5451175
4: -0.3310969, 0.3644254, -0.3763299, 0.4229207, -0.7540176, 0.7407553
5: -0.3594356, 0.3879419, -0.4135728, 0.4452986, -0.8047342, 0.8015146
6: -0.3599293, 0.4365174, -0.3938413, 0.5090361, -0.8689654, 0.8303586
7: -0.3087140, 0.3825606, -0.3422615, 0.5262406, -0.8349546, 0.7248222
8: -0.3339059, 0.4714856, -0.3835237, 0.5472079, -0.8811138, 0.8550092
9: -0.3638552, 0.3994102, -0.4090680, 0.4823717, -0.8462269, 0.8084782

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3011778, upper bound: 1.3183554
time: 1.35 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3011778, upper bound: 1.3183554
time: 1.44 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.4262245, 0.4637488, -0.4795894, 0.5222655, -0.9484900, 0.9433382
1: 0.2070149, 1.0896072, 0.1219975, 1.0955881, -0.8885732, 0.9676097
2: -0.3156461, 0.4207659, -0.3526646, 0.4769843, -0.7926304, 0.7734305
3: -0.2133930, 0.2945376, -0.2434595, 0.3459948, -0.5593878, 0.5379971
4: -0.3250946, 0.3755192, -0.3763299, 0.4229207, -0.7480153, 0.7518491
5: -0.3659154, 0.3928891, -0.4135728, 0.4452986, -0.8112141, 0.8064618
6: -0.3540035, 0.4503496, -0.3938413, 0.5090361, -0.8630396, 0.8441908
7: -0.2996252, 0.4714411, -0.3422615, 0.5262406, -0.8258657, 0.8137026
8: -0.3254003, 0.4954395, -0.3835237, 0.5472079, -0.8726082, 0.8789632
9: -0.3618652, 0.4247929, -0.4090680, 0.4823717, -0.8442370, 0.8338609

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3011766, upper bound: 1.3219162
time: 1.57 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3011766, upper bound: 1.3216608
time: 1.50 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.3870632, 0.4219994, -0.4795894, 0.5222655, -0.9093287, 0.9015888
1: 0.2558922, 1.0861112, 0.1219975, 1.0955881, -0.8396959, 0.9641137
2: -0.2890756, 0.3819105, -0.3526646, 0.4769843, -0.7660599, 0.7345752
3: -0.1920240, 0.2566100, -0.2434595, 0.3459948, -0.5380188, 0.5000694
4: -0.2880441, 0.3415043, -0.3763299, 0.4229207, -0.7109648, 0.7178342
5: -0.3316787, 0.3561159, -0.4135728, 0.4452986, -0.7769773, 0.7696887
6: -0.3245516, 0.4084516, -0.3938413, 0.5090361, -0.8335877, 0.8022929
7: -0.2677214, 0.4439036, -0.3422615, 0.5262406, -0.7939619, 0.7861651
8: -0.2828393, 0.4579817, -0.3835237, 0.5472079, -0.8300472, 0.8415054
9: -0.3280032, 0.3846691, -0.4090680, 0.4823717, -0.8103750, 0.7937371

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A2_B2_B2_A2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_B2_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2638366, upper bound: 1.2973939
time: 1.83 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_A2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_B2_A2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2511941, upper bound: 1.2952187
time: 2.01 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.4774697, 0.5201433, -0.4199749, 0.4642798, -0.9417495, 0.9401182
1: 0.1235183, 1.0953584, 0.3237756, 1.0909228, -0.9674046, 0.7715828
2: -0.3512909, 0.4748844, -0.3118159, 0.4273157, -0.7786065, 0.7867002
3: -0.2423010, 0.3439299, -0.2097563, 0.3082188, -0.5505198, 0.5536863
4: -0.3742348, 0.4211727, -0.3375647, 0.3703794, -0.7446142, 0.7587374
5: -0.4117870, 0.4432196, -0.3654354, 0.3947331, -0.8065201, 0.8086550
6: -0.3922521, 0.5068179, -0.3650163, 0.4438908, -0.8361430, 0.8718343
7: -0.3405009, 0.5257887, -0.3141954, 0.3883569, -0.7288578, 0.8399841
8: -0.3811941, 0.5452612, -0.3413161, 0.4780260, -0.8592200, 0.8865773
9: -0.4072751, 0.4803803, -0.3697757, 0.4065810, -0.8138560, 0.8501561

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 218

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A2_B1_A1_A2_B2_B1_A2_B1_B1

### Relational analysis result of NS_A2_B1_A1_A2_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3177390, upper bound: 1.3011616
time: 2.02 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2_B1_A2_B1_B2

### Relational analysis result of NS_A2_B1_A1_A2_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3177390, upper bound: 1.3011778
time: 1.70 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.4774697, 0.5201433, -0.4277210, 0.4651801, -0.9426497, 0.9478643
1: 0.1235183, 1.0953584, 0.2065891, 1.0897280, -0.9662098, 0.8887693
2: -0.3512909, 0.4748844, -0.3165881, 0.4221664, -0.7734573, 0.7914724
3: -0.2423010, 0.3439299, -0.2141852, 0.2959982, -0.5382992, 0.5581151
4: -0.3742348, 0.4211727, -0.3265392, 0.3767133, -0.7509482, 0.7477118
5: -0.4117870, 0.4432196, -0.3671418, 0.3941824, -0.8059695, 0.8103613
6: -0.3922521, 0.5068179, -0.3551311, 0.4518512, -0.8441033, 0.8619490
7: -0.3405009, 0.5257887, -0.3008942, 0.4711357, -0.8116366, 0.8266829
8: -0.3811941, 0.5452612, -0.3270510, 0.4967850, -0.8779790, 0.8723122
9: -0.4072751, 0.4803803, -0.3630718, 0.4261006, -0.8333756, 0.8434521

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_B1_A1_A2_B2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_A2_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3182548, upper bound: 1.3156830
time: 1.77 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_A2_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3182396, upper bound: 1.3156830
time: 1.65 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.4790125, 0.5220827, -0.4479859, 0.4945248, -0.9735373, 0.9700686
1: 0.1180222, 1.0955698, 0.2836002, 1.0939550, -0.9759328, 0.8119696
2: -0.3525190, 0.4767044, -0.3309644, 0.4564796, -0.8089986, 0.8076688
3: -0.2429261, 0.3453759, -0.2257794, 0.3352846, -0.5782107, 0.5711553
4: -0.3756931, 0.4227908, -0.3645084, 0.3948840, -0.7705771, 0.7872992
5: -0.4133590, 0.4447988, -0.3901332, 0.4226848, -0.8360438, 0.8349320
6: -0.3933792, 0.5086889, -0.3859535, 0.4744122, -0.8677914, 0.8946425
7: -0.3416108, 0.5307057, -0.3367546, 0.4126185, -0.7542293, 0.8674603
8: -0.3828212, 0.5468925, -0.3718436, 0.5049437, -0.8877649, 0.9187361
9: -0.4088534, 0.4824251, -0.3943095, 0.4360902, -0.8449435, 0.8767346

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 218

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A2_B2_A1_A1_B2_B1_A2_B1_B1

### Relational analysis result of NS_A2_B2_A1_A1_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3185296, upper bound: 1.3035761
time: 1.82 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2_B1_A2_B1_B2

### Relational analysis result of NS_A2_B2_A1_A1_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3185296, upper bound: 1.3037133
time: 1.68 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.4790125, 0.5220827, -0.4609815, 0.5018930, -0.9809055, 0.9830643
1: 0.1180222, 1.0955698, 0.1517130, 1.0933654, -0.9753432, 0.9438568
2: -0.3525190, 0.4767044, -0.3398001, 0.4571700, -0.8096890, 0.8165046
3: -0.2429261, 0.3453759, -0.2329021, 0.3280224, -0.5709485, 0.5782779
4: -0.3756931, 0.4227908, -0.3581706, 0.4064540, -0.7821470, 0.7809614
5: -0.4133590, 0.4447988, -0.3970063, 0.4267280, -0.8400869, 0.8418051
6: -0.3933792, 0.5086889, -0.3799877, 0.4884244, -0.8818035, 0.8886766
7: -0.3416108, 0.5307057, -0.3274313, 0.5069483, -0.8485591, 0.8581370
8: -0.3828212, 0.5468925, -0.3632787, 0.5291944, -0.9120156, 0.9101712
9: -0.4088534, 0.4824251, -0.3924963, 0.4623132, -0.8711666, 0.8749214

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_A1_A1_B2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_A1_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3194241, upper bound: 1.3216608
time: 1.87 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_A1_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3193967, upper bound: 1.3216608
time: 2.08 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.4774697, 0.5201433, -0.4664812, 0.5144531, -0.9919228, 0.9866245
1: 0.1235183, 1.0953584, 0.2568278, 1.0960970, -0.9725788, 0.8385306
2: -0.3512909, 0.4748844, -0.3435791, 0.4759363, -0.8272271, 0.8184634
3: -0.2423010, 0.3439299, -0.2363504, 0.3531684, -0.5954694, 0.5802803
4: -0.3742348, 0.4211727, -0.3825910, 0.4110281, -0.7852629, 0.8037637
5: -0.4117870, 0.4432196, -0.4064068, 0.4410938, -0.8528808, 0.8496264
6: -0.3922521, 0.5068179, -0.3997318, 0.4947057, -0.8869578, 0.9065497
7: -0.3405009, 0.5257887, -0.3515932, 0.4292631, -0.7697640, 0.8773819
8: -0.3811941, 0.5452612, -0.3919654, 0.5226738, -0.9038679, 0.9372265
9: -0.4072751, 0.4803803, -0.4106461, 0.4555485, -0.8628235, 0.8910264

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_A1_A2_B2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_A2_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3194241, upper bound: 1.3037102
time: 1.67 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_A2_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3193968, upper bound: 1.3037102
time: 1.71 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.4774697, 0.5201433, -0.4795894, 0.5222655, -0.9997352, 0.9997327
1: 0.1235183, 1.0953584, 0.1219975, 1.0955881, -0.9720699, 0.9733608
2: -0.3512909, 0.4748844, -0.3526646, 0.4769843, -0.8282752, 0.8275490
3: -0.2423010, 0.3439299, -0.2434595, 0.3459948, -0.5882958, 0.5873894
4: -0.3742348, 0.4211727, -0.3763299, 0.4229207, -0.7971555, 0.7975026
5: -0.4117870, 0.4432196, -0.4135728, 0.4452986, -0.8570856, 0.8567923
6: -0.3922521, 0.5068179, -0.3938413, 0.5090361, -0.9012882, 0.9006592
7: -0.3405009, 0.5257887, -0.3422615, 0.5262406, -0.8667415, 0.8680502
8: -0.3811941, 0.5452612, -0.3835237, 0.5472079, -0.9284019, 0.9287848
9: -0.4072751, 0.4803803, -0.4090680, 0.4823717, -0.8896468, 0.8894483

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_A1_A2_B2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_A2_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3194241, upper bound: 1.3216608
time: 1.76 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_A2_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3193968, upper bound: 1.3216608
time: 1.64 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 5.39 seconds
NS_A1_B1_A1_B2_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 9, time: 5.39
Output dim: 1, lower bound: -1.2934728, upper bound: 1.3159074
NS_A1_B1_A1_B2_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 9, time: 5.39
Output dim: 1, lower bound: -1.2923241, upper bound: 1.3158269
NS_A1_B1_A1_B2_B1_A2_B2_A2_A1, status: Status.VERIFIED, split count: 9, time: 5.39
Output dim: 1, lower bound: -1.2884928, upper bound: 1.2949342
NS_A1_B1_A1_B2_B1_A2_B2_A2_A2, status: Status.VERIFIED, split count: 9, time: 5.39
Output dim: 1, lower bound: -1.2660204, upper bound: 1.2929015
NS_A1_B1_A1_B2_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 9, time: 5.39
Output dim: 1, lower bound: -1.2934728, upper bound: 1.3159074
NS_A1_B1_A1_B2_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 9, time: 5.39
Output dim: 1, lower bound: -1.2923241, upper bound: 1.3158269
NS_A1_B1_A1_B2_B2_A2_B2_A2_A1, status: Status.VERIFIED, split count: 9, time: 5.39
Output dim: 1, lower bound: -1.2903861, upper bound: 1.2958187
NS_A1_B1_A1_B2_B2_A2_B2_A2_A2, status: Status.VERIFIED, split count: 9, time: 5.39
Output dim: 1, lower bound: -1.2660204, upper bound: 1.2937811
NS_A1_B1_A2_B2_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 9, time: 5.39
Output dim: 1, lower bound: -1.2930208, upper bound: 1.3176706
NS_A1_B1_A2_B2_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 5.39
Output dim: 1, lower bound: -1.2930208, upper bound: 1.3174784
NS_A1_B1_A2_B2_B1_B2_A2_A2_A1, status: Status.VERIFIED, split count: 9, time: 5.39
Output dim: 1, lower bound: -1.2887963, upper bound: 1.2965754
NS_A1_B1_A2_B2_B1_B2_A2_A2_A2, status: Status.VERIFIED, split count: 9, time: 5.39
Output dim: 1, lower bound: -1.2660204, upper bound: 1.2943401
NS_A1_B1_A2_B2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 9, time: 5.39
Output dim: 1, lower bound: -1.3011778, upper bound: 1.3183554
NS_A1_B1_A2_B2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 5.39
Output dim: 1, lower bound: -1.3011778, upper bound: 1.3183554
NS_A1_B1_A2_B2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 5.39
Output dim: 1, lower bound: -1.3011766, upper bound: 1.3219162
NS_A1_B1_A2_B2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 5.39
Output dim: 1, lower bound: -1.3011766, upper bound: 1.3216608
NS_A1_B1_A2_B2_B2_A2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 5.39
Output dim: 1, lower bound: -1.2638366, upper bound: 1.2973939
NS_A1_B1_A2_B2_B2_A2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 5.39
Output dim: 1, lower bound: -1.2511941, upper bound: 1.2952187
NS_A2_B1_A1_A2_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 5.39
Output dim: 1, lower bound: -1.3177390, upper bound: 1.3011616
NS_A2_B1_A1_A2_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 5.39
Output dim: 1, lower bound: -1.3177390, upper bound: 1.3011778
NS_A2_B1_A1_A2_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.39
Output dim: 1, lower bound: -1.3182548, upper bound: 1.3156830
NS_A2_B1_A1_A2_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.39
Output dim: 1, lower bound: -1.3182396, upper bound: 1.3156830
NS_A2_B2_A1_A1_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 5.39
Output dim: 1, lower bound: -1.3185296, upper bound: 1.3035761
NS_A2_B2_A1_A1_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 5.39
Output dim: 1, lower bound: -1.3185296, upper bound: 1.3037133
NS_A2_B2_A1_A1_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.39
Output dim: 1, lower bound: -1.3194241, upper bound: 1.3216608
NS_A2_B2_A1_A1_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.39
Output dim: 1, lower bound: -1.3193967, upper bound: 1.3216608
NS_A2_B2_A1_A2_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.39
Output dim: 1, lower bound: -1.3194241, upper bound: 1.3037102
NS_A2_B2_A1_A2_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.39
Output dim: 1, lower bound: -1.3193968, upper bound: 1.3037102
NS_A2_B2_A1_A2_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.39
Output dim: 1, lower bound: -1.3194241, upper bound: 1.3216608
NS_A2_B2_A1_A2_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.39
Output dim: 1, lower bound: -1.3193968, upper bound: 1.3216608

## BFS NS instance: NS_A1_B1_A1_B2_B1_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.3082405, 0.3442434, -0.4009246, 0.4506611, -0.7589016, 0.7451680
1: 0.4611007, 1.0789244, 0.2474016, 1.0880945, -0.6269938, 0.8315228
2: -0.2336403, 0.3125380, -0.3075938, 0.4077835, -0.6414237, 0.6201318
3: -0.1492895, 0.2105336, -0.2064972, 0.2841538, -0.4334433, 0.4170308
4: -0.2368676, 0.2701490, -0.3140051, 0.3651786, -0.6020463, 0.5841540
5: -0.2712739, 0.2807794, -0.3556081, 0.3780738, -0.6493477, 0.6363875
6: -0.2784777, 0.3261680, -0.3459184, 0.4233593, -0.7018371, 0.6720864
7: -0.2253797, 0.3062526, -0.2912915, 0.4471793, -0.6725590, 0.5975441
8: -0.2247561, 0.3676424, -0.3135851, 0.4671625, -0.6919186, 0.6812275
9: -0.2720845, 0.3037117, -0.3517307, 0.3924817, -0.6645662, 0.6554425

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 92

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B2_A1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156830, upper bound: 1.3216608
time: 1.46 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B2_A1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156830, upper bound: 1.3216608
time: 1.45 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.3558824, 0.3955187, -0.4170001, 0.4525359, -0.8084183, 0.8125188
1: 0.3768287, 1.0839835, 0.2316272, 1.0882753, -0.7114466, 0.8523563
2: -0.2681018, 0.3614399, -0.3087735, 0.4101875, -0.6782893, 0.6702135
3: -0.1755451, 0.2447728, -0.2075423, 0.2857314, -0.4612764, 0.4523150
4: -0.2747468, 0.3145800, -0.3163486, 0.3666931, -0.6414399, 0.6309286
5: -0.3088662, 0.3314286, -0.3571191, 0.3824915, -0.6913577, 0.6885477
6: -0.3153585, 0.3744123, -0.3471414, 0.4395072, -0.7548658, 0.7215537
7: -0.2597804, 0.3618024, -0.2925755, 0.4539089, -0.7136893, 0.6543779
8: -0.2690353, 0.4160919, -0.3153661, 0.4859500, -0.7549853, 0.7314579
9: -0.3136851, 0.3413160, -0.3532165, 0.4132264, -0.7269114, 0.6945325

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 92

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B2_A1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156830, upper bound: 1.3216608
time: 1.39 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B2_A1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156830, upper bound: 1.3216608
time: 1.47 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.3138752, 0.3501884, -0.4381081, 0.4755638, -0.7894390, 0.7882965
1: 0.4511299, 1.0795171, 0.1997561, 1.0905111, -0.6393812, 0.8797610
2: -0.2376855, 0.3182051, -0.3233204, 0.4321806, -0.6698662, 0.6415255
3: -0.1523104, 0.2145496, -0.2195429, 0.3061141, -0.4584246, 0.4340925
4: -0.2411290, 0.2753645, -0.3364365, 0.3853101, -0.6264392, 0.6118010
5: -0.2756355, 0.2867728, -0.3758623, 0.4036191, -0.6792547, 0.6626351
6: -0.2828000, 0.3317590, -0.3629501, 0.4625330, -0.7453330, 0.6947091
7: -0.2292982, 0.3128504, -0.3095681, 0.4732316, -0.7025298, 0.6224185
8: -0.2297464, 0.3733312, -0.3383914, 0.5063767, -0.7361231, 0.7117226
9: -0.2769655, 0.3077619, -0.3716885, 0.4357564, -0.7127219, 0.6794504

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 92

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2_A1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156830, upper bound: 1.3216608
time: 1.81 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2_A1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156830, upper bound: 1.3216608
time: 1.70 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.3625568, 0.4028900, -0.4398343, 0.4775372, -0.8400939, 0.8427243
1: 0.3643215, 1.0846618, 0.1962954, 1.0907189, -0.7263974, 0.8883664
2: -0.2727486, 0.3684868, -0.3245596, 0.4340459, -0.7067945, 0.6930463
3: -0.1796073, 0.2508092, -0.2205450, 0.3077683, -0.4873756, 0.4713542
4: -0.2810243, 0.3205429, -0.3380723, 0.3868994, -0.6679237, 0.6586151
5: -0.3147154, 0.3382251, -0.3774514, 0.4053601, -0.7200755, 0.7156765
6: -0.3203051, 0.3816808, -0.3642397, 0.4644716, -0.7847767, 0.7459205
7: -0.2650413, 0.3696998, -0.3109230, 0.4756416, -0.7406828, 0.6806228
8: -0.2762261, 0.4225958, -0.3402674, 0.5080917, -0.7843177, 0.7628632
9: -0.3195050, 0.3486479, -0.3732557, 0.4377336, -0.7572386, 0.7219036

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2_A1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156830, upper bound: 1.3216608
time: 1.65 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2_A1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156830, upper bound: 1.3216608
time: 1.64 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.3817640, 0.4294367, -0.3476897, 0.3859191, -0.7676830, 0.7771264
1: 0.2828701, 1.0861295, 0.4036264, 1.0832027, -0.8003325, 0.6825031
2: -0.2941188, 0.3875186, -0.2621003, 0.3522919, -0.6464107, 0.6496189
3: -0.1956249, 0.2656523, -0.1696054, 0.2391730, -0.4347979, 0.4352577
4: -0.2957382, 0.3478971, -0.2673730, 0.3068024, -0.6025406, 0.6152701
5: -0.3382992, 0.3587180, -0.3019186, 0.3225188, -0.6608180, 0.6606367
6: -0.3315937, 0.4021745, -0.3094994, 0.3654703, -0.6970640, 0.7116739
7: -0.2760433, 0.4245241, -0.2538279, 0.3436828, -0.6197261, 0.6783520
8: -0.2927136, 0.4483834, -0.2605860, 0.4078263, -0.7005399, 0.7089694
9: -0.3347033, 0.3714372, -0.3065471, 0.3312361, -0.6659394, 0.6779843

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A2_B2_B1_B2_A2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_B1_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156830, upper bound: 1.3216608
time: 1.70 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_B2_A2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_B1_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156830, upper bound: 1.3216608
time: 1.56 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.3831711, 0.4309734, -0.3948162, 0.4381524, -0.8213235, 0.8257896
1: 0.2804095, 1.0862722, 0.3145420, 1.0880083, -0.8075988, 0.7717302
2: -0.2950900, 0.3889869, -0.2950334, 0.4022259, -0.6973159, 0.6840203
3: -0.1964213, 0.2669947, -0.1983229, 0.2820933, -0.4785147, 0.4653176
4: -0.2970622, 0.3491417, -0.3118619, 0.3490675, -0.6461297, 0.6610036
5: -0.3395472, 0.3601372, -0.3434271, 0.3704920, -0.7100392, 0.7035643
6: -0.3326325, 0.4037088, -0.3445442, 0.4170130, -0.7496455, 0.7482531
7: -0.2771521, 0.4260702, -0.2910946, 0.3998163, -0.6769685, 0.7171648
8: -0.2942274, 0.4497426, -0.3115369, 0.4539140, -0.7481414, 0.7612796
9: -0.3359312, 0.3729569, -0.3478212, 0.3831898, -0.7191210, 0.7207781

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A2_B2_B1_B2_A2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_B1_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156830, upper bound: 1.3216608
time: 1.49 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_B2_A2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_B1_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156830, upper bound: 1.3216608
time: 1.85 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.4131809, 0.4569310, -0.4600815, 0.5011582, -0.9143391, 0.9170125
1: 0.3335374, 1.0902195, 0.1502132, 1.0933038, -0.7597665, 0.9400063
2: -0.3071634, 0.4202842, -0.3393328, 0.4564064, -0.7635698, 0.7596170
3: -0.2058630, 0.3016580, -0.2321909, 0.3271129, -0.5329759, 0.5338489
4: -0.3310969, 0.3644254, -0.3572670, 0.4059009, -0.7369977, 0.7216923
5: -0.3594356, 0.3879419, -0.3963980, 0.4258647, -0.7853003, 0.7843398
6: -0.3599293, 0.4365174, -0.3792961, 0.4876015, -0.8475307, 0.8158134
7: -0.3087140, 0.3825606, -0.3265963, 0.5092017, -0.8179157, 0.7091569
8: -0.3339059, 0.4714856, -0.3622497, 0.5284564, -0.8623623, 0.8337352
9: -0.3638552, 0.3994102, -0.3919076, 0.4617257, -0.8255808, 0.7913178

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 218

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_B2_A1_B1_B1

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2797563, upper bound: 1.3054145
time: 1.54 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_B2_A1_B1_B2

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1_B2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2797563, upper bound: 1.2944430
time: 1.54 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.4131809, 0.4569310, -0.4578659, 0.4984745, -0.9116554, 0.9147969
1: 0.3335374, 1.0902195, 0.1566248, 1.0930262, -0.7594888, 0.9335947
2: -0.3071634, 0.4202842, -0.3376423, 0.4539044, -0.7610679, 0.7579265
3: -0.2058630, 0.3016580, -0.2311576, 0.3250178, -0.5308807, 0.5328157
4: -0.3310969, 0.3644254, -0.3552038, 0.4036905, -0.7347873, 0.7196292
5: -0.3594356, 0.3879419, -0.3942271, 0.4236055, -0.7830411, 0.7821690
6: -0.3599293, 0.4365174, -0.3776568, 0.4850110, -0.8449403, 0.8141742
7: -0.3087140, 0.3825606, -0.3249323, 0.5038370, -0.8125510, 0.7074929
8: -0.3339059, 0.4714856, -0.3598798, 0.5261694, -0.8600752, 0.8313653
9: -0.3638552, 0.3994102, -0.3897578, 0.4589576, -0.8228128, 0.7891680

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 218

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2935390, upper bound: 1.2947329
time: 2.02 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2797563, upper bound: 1.2944711
time: 1.86 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.3897326, 0.4380474, -0.3675665, 0.4080493, -0.7977819, 0.8056140
1: 0.2688017, 1.0869250, 0.3724678, 1.0852914, -0.8164897, 0.7144573
2: -0.2995570, 0.3957464, -0.2760567, 0.3734620, -0.6730190, 0.6718031
3: -0.2001263, 0.2731453, -0.1814736, 0.2563307, -0.4564571, 0.4546189
4: -0.3031284, 0.3548637, -0.2864382, 0.3247175, -0.6278459, 0.6413020
5: -0.3452817, 0.3667591, -0.3191042, 0.3429012, -0.6881828, 0.6858633
6: -0.3373888, 0.4107629, -0.3246495, 0.3870727, -0.7244616, 0.7354124
7: -0.2822170, 0.4334618, -0.2700784, 0.3628655, -0.6450826, 0.7035401
8: -0.3011583, 0.4559909, -0.2825416, 0.4274605, -0.7286188, 0.7385325
9: -0.3415689, 0.3799776, -0.3239120, 0.3529807, -0.6945496, 0.7038896

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156830, upper bound: 1.3216608
time: 1.68 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156830, upper bound: 1.3216608
time: 1.68 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.3911990, 0.4396518, -0.4149448, 0.4601533, -0.8513523, 0.8545966
1: 0.2662597, 1.0870744, 0.2819335, 1.0900816, -0.8238218, 0.8051409
2: -0.3005711, 0.3972789, -0.3089370, 0.4232727, -0.7238439, 0.7062160
3: -0.2009571, 0.2745485, -0.2101040, 0.3015153, -0.5024724, 0.4846525
4: -0.3045120, 0.3561630, -0.3310097, 0.3668704, -0.6713824, 0.6871727
5: -0.3465850, 0.3682372, -0.3613285, 0.3906530, -0.7372380, 0.7295657
6: -0.3384750, 0.4123650, -0.3595850, 0.4390353, -0.7775103, 0.7719499
7: -0.2833767, 0.4350526, -0.3072207, 0.4197822, -0.7031589, 0.7422733
8: -0.3027403, 0.4574108, -0.3334495, 0.4734252, -0.7761655, 0.7908603
9: -0.3428513, 0.3815620, -0.3654664, 0.4048248, -0.7476761, 0.7470284

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156830, upper bound: 1.3216608
time: 16.68 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156830, upper bound: 1.3216608
time: 1.93 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B2_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.4774697, 0.5201433, -0.3957809, 0.4381112, -0.9155809, 0.9159242
1: 0.1235183, 1.0953584, 0.3585365, 1.0884182, -0.9649000, 0.7368219
2: -0.3512909, 0.4748844, -0.2952483, 0.4022765, -0.7535673, 0.7701327
3: -0.2423010, 0.3439299, -0.1958928, 0.2848564, -0.5271574, 0.5398228
4: -0.3742348, 0.4211727, -0.3145334, 0.3491776, -0.7234124, 0.7357061
5: -0.4117870, 0.4432196, -0.3440704, 0.3705487, -0.7823358, 0.7872900
6: -0.3922521, 0.5068179, -0.3469011, 0.4176344, -0.8098865, 0.8537190
7: -0.3405009, 0.5257887, -0.2946765, 0.3677167, -0.7082176, 0.8204652
8: -0.3811941, 0.5452612, -0.3149286, 0.4547357, -0.8359298, 0.8601898
9: -0.4072751, 0.4803803, -0.3486924, 0.3810481, -0.7883232, 0.8290727

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 218

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A2_B1_A1_A2_B2_B1_A2_B1_B1_B1

### Relational analysis result of NS_A2_B1_A1_A2_B2_B1_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2940094, upper bound: 1.2938550
time: 1.92 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2_B1_A2_B1_B1_B2

### Relational analysis result of NS_A2_B1_A1_A2_B2_B1_A2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2935439, upper bound: 1.2797563
time: 2.12 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B2_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.4774697, 0.5201433, -0.3992541, 0.4418675, -0.9193372, 0.9193974
1: 0.1235183, 1.0953584, 0.3535465, 1.0887779, -0.9652597, 0.7418119
2: -0.3512909, 0.4748844, -0.2976267, 0.4058709, -0.7571617, 0.7725110
3: -0.2423010, 0.3439299, -0.1978829, 0.2882102, -0.5305111, 0.5418128
4: -0.3742348, 0.4211727, -0.3178397, 0.3522211, -0.7264559, 0.7390124
5: -0.4117870, 0.4432196, -0.3471373, 0.3740204, -0.7858074, 0.7903569
6: -0.3922521, 0.5068179, -0.3495015, 0.4214035, -0.8136556, 0.8563194
7: -0.3405009, 0.5257887, -0.2974784, 0.3706797, -0.7111806, 0.8232671
8: -0.3811941, 0.5452612, -0.3187165, 0.4580791, -0.8392732, 0.8639777
9: -0.4072751, 0.4803803, -0.3517189, 0.3847135, -0.7919886, 0.8320992

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 218

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A2_B1_A1_A2_B2_B1_A2_B1_B2_B1

### Relational analysis result of NS_A2_B1_A1_A2_B2_B1_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2940094, upper bound: 1.2938550
time: 2.14 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2_B1_A2_B1_B2_B2

### Relational analysis result of NS_A2_B1_A1_A2_B2_B1_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2935439, upper bound: 1.2797563
time: 1.67 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.3605784, 0.4004564, -0.3908759, 0.4394192, -0.7999976, 0.7913323
1: 0.3811918, 1.0845523, 0.2694148, 1.0870802, -0.7058885, 0.8151375
2: -0.2712478, 0.3661932, -0.3004375, 0.3970646, -0.6683124, 0.6666307
3: -0.1775515, 0.2494680, -0.2008592, 0.2745309, -0.4520824, 0.4503272
4: -0.2796757, 0.3185699, -0.3044890, 0.3559811, -0.6356567, 0.6230590
5: -0.3128918, 0.3359492, -0.3464392, 0.3678030, -0.6806948, 0.6823884
6: -0.3193224, 0.3794353, -0.3384739, 0.4121773, -0.7314997, 0.7179092
7: -0.2643050, 0.3578502, -0.2834473, 0.4326379, -0.6969429, 0.6412975
8: -0.2747832, 0.4206780, -0.3027366, 0.4572612, -0.7320445, 0.7234147
9: -0.3177744, 0.3456388, -0.3427221, 0.3811976, -0.6989719, 0.6883609

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B1_A1_A2_B2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_A2_B2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3216608, upper bound: 1.3156830
time: 2.55 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_A2_B2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3216608, upper bound: 1.3156830
time: 2.74 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.4089793, 0.4536694, -0.3923126, 0.4409908, -0.8499701, 0.8459820
1: 0.2888865, 1.0894464, 0.2669398, 1.0872266, -0.7983401, 0.8225065
2: -0.3048329, 0.4170644, -0.3014309, 0.3985663, -0.7033992, 0.7184952
3: -0.2067885, 0.2956299, -0.2016733, 0.2759064, -0.4826950, 0.4973032
4: -0.3252114, 0.3616270, -0.3058454, 0.3572539, -0.6824653, 0.6674724
5: -0.3560255, 0.3847345, -0.3477163, 0.3692501, -0.7252757, 0.7324508
6: -0.3550143, 0.4325083, -0.3395383, 0.4137466, -0.7687609, 0.7720466
7: -0.3022560, 0.4158733, -0.2845846, 0.4341895, -0.7364454, 0.7004579
8: -0.3267933, 0.4676265, -0.3042876, 0.4586521, -0.7854455, 0.7719141
9: -0.3602240, 0.3985804, -0.3439786, 0.3827493, -0.7429733, 0.7425590

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B1_A1_A2_B2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_A2_B2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3216608, upper bound: 1.3156830
time: 1.56 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_A2_B2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3216608, upper bound: 1.3156830
time: 1.56 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B2_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.4790125, 0.5220827, -0.4412251, 0.4872455, -0.9662580, 0.9633078
1: 0.1180222, 1.0955698, 0.2932698, 1.0931714, -0.9751492, 0.8023001
2: -0.3525190, 0.4767044, -0.3263556, 0.4493724, -0.8018913, 0.8030601
3: -0.2429261, 0.3453759, -0.2219231, 0.3287449, -0.5716710, 0.5672989
4: -0.3756931, 0.4227908, -0.3578961, 0.3889861, -0.7646792, 0.7806869
5: -0.4133590, 0.4447988, -0.3841870, 0.4159575, -0.8293165, 0.8289858
6: -0.3933792, 0.5086889, -0.3809144, 0.4669976, -0.8603768, 0.8896034
7: -0.3416108, 0.5307057, -0.3313251, 0.4066195, -0.7482303, 0.8620308
8: -0.3828212, 0.5468925, -0.3644846, 0.4984654, -0.8812866, 0.9113771
9: -0.4088534, 0.4824251, -0.3883395, 0.4289881, -0.8378415, 0.8707646

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_A1_A1_B2_B1_A2_B1_B1_A1

### Relational analysis result of NS_A2_B2_A1_A1_B2_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3185030, upper bound: 1.3035565
time: 1.66 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2_B1_A2_B1_B1_A2

### Relational analysis result of NS_A2_B2_A1_A1_B2_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3184422, upper bound: 1.3035565
time: 1.80 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B2_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.4790125, 0.5220827, -0.4448265, 0.4911231, -0.9701356, 0.9669092
1: 0.1180222, 1.0955698, 0.2881188, 1.0935887, -0.9755665, 0.8074511
2: -0.3525190, 0.4767044, -0.3288107, 0.4531585, -0.8056774, 0.8055151
3: -0.2429261, 0.3453759, -0.2239772, 0.3322287, -0.5751548, 0.5693531
4: -0.3756931, 0.4227908, -0.3614183, 0.3921278, -0.7678208, 0.7842091
5: -0.4133590, 0.4447988, -0.3873544, 0.4195412, -0.8329002, 0.8321532
6: -0.3933792, 0.5086889, -0.3835986, 0.4709473, -0.8643265, 0.8922876
7: -0.3416108, 0.5307057, -0.3342174, 0.4098151, -0.7514259, 0.8649231
8: -0.3828212, 0.5468925, -0.3684047, 0.5019163, -0.8847375, 0.9152972
9: -0.4088534, 0.4824251, -0.3915196, 0.4327710, -0.8416243, 0.8739446

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 218

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A2_B2_A1_A1_B2_B1_A2_B1_B2_B1

### Relational analysis result of NS_A2_B2_A1_A1_B2_B1_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2942738, upper bound: 1.2954961
time: 1.71 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2_B1_A2_B1_B2_B2

### Relational analysis result of NS_A2_B2_A1_A1_B2_B1_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2937217, upper bound: 1.2788995
time: 1.90 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.3612094, 0.4013626, -0.4381081, 0.4755638, -0.8367732, 0.8394707
1: 0.3736446, 1.0845826, 0.1997561, 1.0905111, -0.7168665, 0.8848265
2: -0.2718044, 0.3670424, -0.3233204, 0.4321806, -0.7039850, 0.6903628
3: -0.1783482, 0.2498956, -0.2195429, 0.3061141, -0.4844623, 0.4694384
4: -0.2801110, 0.3193086, -0.3364365, 0.3853101, -0.6654211, 0.6557451
5: -0.3135588, 0.3367206, -0.3758623, 0.4036191, -0.7171779, 0.7125829
6: -0.3196295, 0.3802543, -0.3629501, 0.4625330, -0.7821625, 0.7432044
7: -0.2644885, 0.3632020, -0.3095681, 0.4732316, -0.7377201, 0.6727701
8: -0.2752351, 0.4213722, -0.3383914, 0.5063767, -0.7816117, 0.7597636
9: -0.3184032, 0.3468053, -0.3716885, 0.4357564, -0.7541597, 0.7184938

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_A1_A1_B2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_A1_B2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3216608, upper bound: 1.3216608
time: 1.53 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_A1_B2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3216608, upper bound: 1.3216608
time: 1.49 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.4126673, 0.4582149, -0.4398343, 0.4775372, -0.8902045, 0.8980492
1: 0.2713261, 1.0897832, 0.1962954, 1.0907189, -0.8193927, 0.8934878
2: -0.3076867, 0.4213812, -0.3245596, 0.4340459, -0.7417326, 0.7459408
3: -0.2097002, 0.2990242, -0.2205450, 0.3077683, -0.5174685, 0.5195692
4: -0.3285842, 0.3653242, -0.3380723, 0.3868994, -0.7154835, 0.7033964
5: -0.3596137, 0.3886566, -0.3774514, 0.4053601, -0.7649738, 0.7661080
6: -0.3576057, 0.4369122, -0.3642397, 0.4644716, -0.8220773, 0.8011520
7: -0.3047889, 0.4278767, -0.3109230, 0.4756416, -0.7804306, 0.7387998
8: -0.3305759, 0.4714776, -0.3402674, 0.5080917, -0.8386676, 0.8117450
9: -0.3637143, 0.4035081, -0.3732557, 0.4377336, -0.8014479, 0.7767638

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_A1_A1_B2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_A1_B2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3216608, upper bound: 1.3216608
time: 1.40 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_A1_B2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3216608, upper bound: 1.3216608
time: 1.39 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.3605784, 0.4004564, -0.4483897, 0.4949597, -0.8555381, 0.8488461
1: 0.3811918, 1.0845523, 0.2830227, 1.0940017, -0.7128099, 0.8015296
2: -0.2712478, 0.3661932, -0.3312395, 0.4569044, -0.7281522, 0.6974327
3: -0.1775515, 0.2494680, -0.2260097, 0.3356754, -0.5132269, 0.4754777
4: -0.2796757, 0.3185699, -0.3649036, 0.3952360, -0.6749117, 0.6834735
5: -0.3128918, 0.3359492, -0.3904884, 0.4230865, -0.7359784, 0.7264376
6: -0.3193224, 0.3794353, -0.3862546, 0.4748552, -0.7941776, 0.7656899
7: -0.2643050, 0.3578502, -0.3370789, 0.4129768, -0.6772818, 0.6949291
8: -0.2747832, 0.4206780, -0.3722832, 0.5053307, -0.7801139, 0.7929612
9: -0.3177744, 0.3456388, -0.3946660, 0.4365145, -0.7542889, 0.7403048

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A2_B2_A1_A2_B2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_A2_B2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3185030, upper bound: 1.3035565
time: 1.58 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_A2_B2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3185030, upper bound: 1.3037102
time: 1.84 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.4089793, 0.4536694, -0.4493577, 0.4960021, -0.9049814, 0.9030272
1: 0.2888865, 1.0894464, 0.2816383, 1.0941141, -0.8052275, 0.8078081
2: -0.3048329, 0.4170644, -0.3318995, 0.4579219, -0.7627547, 0.7489638
3: -0.2067885, 0.2956299, -0.2265619, 0.3366119, -0.5434005, 0.5221918
4: -0.3252114, 0.3616270, -0.3658502, 0.3960806, -0.7212920, 0.7274772
5: -0.3560255, 0.3847345, -0.3913397, 0.4240499, -0.7800754, 0.7760742
6: -0.3550143, 0.4325083, -0.3869761, 0.4759166, -0.8309309, 0.8194844
7: -0.3022560, 0.4158733, -0.3378565, 0.4138355, -0.7160915, 0.7537298
8: -0.3267933, 0.4676265, -0.3733368, 0.5062584, -0.8330517, 0.8409632
9: -0.3602240, 0.3985804, -0.3955208, 0.4375314, -0.7977554, 0.7941012

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A2_B2_A1_A2_B2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3184422, upper bound: 1.3035565
time: 1.63 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_A2_B2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3184422, upper bound: 1.3037102
time: 1.76 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.3605784, 0.4004564, -0.4566187, 0.4959320, -0.8565105, 0.8570751
1: 0.3811918, 1.0845523, 0.1700718, 1.0925230, -0.7113312, 0.9144805
2: -0.2712478, 0.3661932, -0.3361755, 0.4516086, -0.7228564, 0.7023687
3: -0.1775515, 0.2494680, -0.2301233, 0.3239626, -0.5015141, 0.4795913
4: -0.2796757, 0.3185699, -0.3540469, 0.4017666, -0.6814423, 0.6726168
5: -0.3128918, 0.3359492, -0.3924111, 0.4221721, -0.7350640, 0.7283602
6: -0.3193224, 0.3794353, -0.3768010, 0.4828413, -0.8021637, 0.7562364
7: -0.2643050, 0.3578502, -0.3243939, 0.4918253, -0.7561303, 0.6822441
8: -0.2747832, 0.4206780, -0.3585722, 0.5243761, -0.7991593, 0.7792503
9: -0.3177744, 0.3456388, -0.3879938, 0.4557919, -0.7735662, 0.7336326

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_A1_A2_B2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_A2_B2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3216608, upper bound: 1.3216608
time: 1.70 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_A2_B2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3216608, upper bound: 1.3216608
time: 1.68 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.4089793, 0.4536694, -0.4583475, 0.4979068, -0.9068861, 0.9120169
1: 0.2888865, 1.0894464, 0.1666330, 1.0927155, -0.8038290, 0.9228135
2: -0.3048329, 0.4170644, -0.3374157, 0.4534702, -0.7583031, 0.7544801
3: -0.2067885, 0.2956299, -0.2311222, 0.3256215, -0.5324101, 0.5267521
4: -0.3252114, 0.3616270, -0.3556767, 0.4033581, -0.7285695, 0.7173038
5: -0.3560255, 0.3847345, -0.3939996, 0.4239157, -0.7799413, 0.7787341
6: -0.3550143, 0.4325083, -0.3780932, 0.4847789, -0.8397932, 0.8106015
7: -0.3022560, 0.4158733, -0.3257511, 0.4942158, -0.7964718, 0.7416244
8: -0.3267933, 0.4676265, -0.3604428, 0.5260953, -0.8528886, 0.8280693
9: -0.3602240, 0.3985804, -0.3895618, 0.4577678, -0.8179918, 0.7881421

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_A1_A2_B2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3216608, upper bound: 1.3216608
time: 1.54 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_A2_B2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3216608, upper bound: 1.3216608
time: 1.53 seconds

## Summary of splitting at layer (split count: 9)
- Time for NS candidates: 5.22 seconds
NS_A1_B1_A1_B2_B1_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.3156830, upper bound: 1.3216608
NS_A1_B1_A1_B2_B1_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.3156830, upper bound: 1.3216608
NS_A1_B1_A1_B2_B1_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.3156830, upper bound: 1.3216608
NS_A1_B1_A1_B2_B1_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.3156830, upper bound: 1.3216608
NS_A1_B1_A1_B2_B2_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.3156830, upper bound: 1.3216608
NS_A1_B1_A1_B2_B2_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.3156830, upper bound: 1.3216608
NS_A1_B1_A1_B2_B2_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.3156830, upper bound: 1.3216608
NS_A1_B1_A1_B2_B2_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.3156830, upper bound: 1.3216608
NS_A1_B1_A2_B2_B1_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.3156830, upper bound: 1.3216608
NS_A1_B1_A2_B2_B1_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.3156830, upper bound: 1.3216608
NS_A1_B1_A2_B2_B1_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.3156830, upper bound: 1.3216608
NS_A1_B1_A2_B2_B1_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.3156830, upper bound: 1.3216608
NS_A1_B1_A2_B2_B2_A1_B2_A1_B1_B1, status: Status.VERIFIED, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.2797563, upper bound: 1.3054145
NS_A1_B1_A2_B2_B2_A1_B2_A1_B1_B2, status: Status.VERIFIED, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.2797563, upper bound: 1.2944430
NS_A1_B1_A2_B2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.2935390, upper bound: 1.2947329
NS_A1_B1_A2_B2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.2797563, upper bound: 1.2944711
NS_A1_B1_A2_B2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.3156830, upper bound: 1.3216608
NS_A1_B1_A2_B2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.3156830, upper bound: 1.3216608
NS_A1_B1_A2_B2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.3156830, upper bound: 1.3216608
NS_A1_B1_A2_B2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.3156830, upper bound: 1.3216608
NS_A2_B1_A1_A2_B2_B1_A2_B1_B1_B1, status: Status.VERIFIED, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.2940094, upper bound: 1.2938550
NS_A2_B1_A1_A2_B2_B1_A2_B1_B1_B2, status: Status.VERIFIED, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.2935439, upper bound: 1.2797563
NS_A2_B1_A1_A2_B2_B1_A2_B1_B2_B1, status: Status.VERIFIED, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.2940094, upper bound: 1.2938550
NS_A2_B1_A1_A2_B2_B1_A2_B1_B2_B2, status: Status.VERIFIED, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.2935439, upper bound: 1.2797563
NS_A2_B1_A1_A2_B2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.3216608, upper bound: 1.3156830
NS_A2_B1_A1_A2_B2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.3216608, upper bound: 1.3156830
NS_A2_B1_A1_A2_B2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.3216608, upper bound: 1.3156830
NS_A2_B1_A1_A2_B2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.3216608, upper bound: 1.3156830
NS_A2_B2_A1_A1_B2_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.3185030, upper bound: 1.3035565
NS_A2_B2_A1_A1_B2_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.3184422, upper bound: 1.3035565
NS_A2_B2_A1_A1_B2_B1_A2_B1_B2_B1, status: Status.VERIFIED, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.2942738, upper bound: 1.2954961
NS_A2_B2_A1_A1_B2_B1_A2_B1_B2_B2, status: Status.VERIFIED, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.2937217, upper bound: 1.2788995
NS_A2_B2_A1_A1_B2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.3216608, upper bound: 1.3216608
NS_A2_B2_A1_A1_B2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.3216608, upper bound: 1.3216608
NS_A2_B2_A1_A1_B2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.3216608, upper bound: 1.3216608
NS_A2_B2_A1_A1_B2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.3216608, upper bound: 1.3216608
NS_A2_B2_A1_A2_B2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.3185030, upper bound: 1.3035565
NS_A2_B2_A1_A2_B2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.3185030, upper bound: 1.3037102
NS_A2_B2_A1_A2_B2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.3184422, upper bound: 1.3035565
NS_A2_B2_A1_A2_B2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.3184422, upper bound: 1.3037102
NS_A2_B2_A1_A2_B2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.3216608, upper bound: 1.3216608
NS_A2_B2_A1_A2_B2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.3216608, upper bound: 1.3216608
NS_A2_B2_A1_A2_B2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.3216608, upper bound: 1.3216608
NS_A2_B2_A1_A2_B2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 1, lower bound: -1.3216608, upper bound: 1.3216608

## BFS NS instance: NS_A1_B1_A1_B2_B1_A2_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.3082405, 0.3442434, -0.3305022, 0.3677142, -0.6759547, 0.6747457
1: 0.4611007, 1.0789244, 0.4307171, 1.0813572, -0.6202565, 0.6482073
2: -0.2336403, 0.3125380, -0.2497076, 0.3349289, -0.5685692, 0.5622456
3: -0.1492895, 0.2105336, -0.1605810, 0.2267638, -0.3760533, 0.3711146
4: -0.2368676, 0.2701490, -0.2541013, 0.2908361, -0.5277038, 0.5242503
5: -0.2712739, 0.2807794, -0.2885478, 0.3043025, -0.5755764, 0.5693272
6: -0.2784777, 0.3261680, -0.2960895, 0.3483195, -0.6267973, 0.6222575
7: -0.2253797, 0.3062526, -0.2415373, 0.3259238, -0.5513035, 0.5477899
8: -0.2247561, 0.3676424, -0.2450191, 0.3903544, -0.6151105, 0.6126615
9: -0.2720845, 0.3037117, -0.2915607, 0.3189402, -0.5910246, 0.5952724

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B2_A1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B2_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3093136, upper bound: 1.3038548
time: 1.70 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B2_A1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B2_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2831133, upper bound: 1.3019566
time: 3.42 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A2_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.3082405, 0.3442434, -0.3767651, 0.4184147, -0.7266552, 0.7210085
1: 0.4611007, 1.0789244, 0.3451349, 1.0861604, -0.6250597, 0.7337896
2: -0.2336403, 0.3125380, -0.2825639, 0.3833486, -0.6169888, 0.5951019
3: -0.1492895, 0.2105336, -0.1876713, 0.2647523, -0.4140418, 0.3982050
4: -0.2368676, 0.2701490, -0.2947618, 0.3330985, -0.5699661, 0.5649108
5: -0.2712739, 0.2807794, -0.3273866, 0.3523792, -0.6236531, 0.6081660
6: -0.2784777, 0.3261680, -0.3311198, 0.3972769, -0.6757546, 0.6572878
7: -0.2253797, 0.3062526, -0.2767344, 0.3809578, -0.6063374, 0.5829870
8: -0.2247561, 0.3676424, -0.2919782, 0.4364348, -0.6611909, 0.6596206
9: -0.2720845, 0.3037117, -0.3320170, 0.3637194, -0.6358039, 0.6357287

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 92

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B2_A1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B2_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3093136, upper bound: 1.3044313
time: 2.19 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B2_A1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B2_A1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2831133, upper bound: 1.3025210
time: 1.51 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A2_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.3558824, 0.3955187, -0.3305022, 0.3677142, -0.7235966, 0.7260209
1: 0.3768287, 1.0839835, 0.4307171, 1.0813572, -0.7045286, 0.6532664
2: -0.2681018, 0.3614399, -0.2497076, 0.3349289, -0.6030307, 0.6111475
3: -0.1755451, 0.2447728, -0.1605810, 0.2267638, -0.4023089, 0.4053538
4: -0.2747468, 0.3145800, -0.2541013, 0.2908361, -0.5655829, 0.5686813
5: -0.3088662, 0.3314286, -0.2885478, 0.3043025, -0.6131687, 0.6199764
6: -0.3153585, 0.3744123, -0.2960895, 0.3483195, -0.6636781, 0.6705018
7: -0.2597804, 0.3618024, -0.2415373, 0.3259238, -0.5857042, 0.6033397
8: -0.2690353, 0.4160919, -0.2450191, 0.3903544, -0.6593897, 0.6611110
9: -0.3136851, 0.3413160, -0.2915607, 0.3189402, -0.6326252, 0.6328768

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B2_A1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B2_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2835605, upper bound: 1.3125051
time: 1.51 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B2_A1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B2_A1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2835605, upper bound: 1.3019518
time: 1.70 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A2_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.3558824, 0.3955187, -0.3767651, 0.4184147, -0.7742971, 0.7722838
1: 0.3768287, 1.0839835, 0.3451349, 1.0861604, -0.7093318, 0.7388487
2: -0.2681018, 0.3614399, -0.2825639, 0.3833486, -0.6514503, 0.6440038
3: -0.1755451, 0.2447728, -0.1876713, 0.2647523, -0.4402974, 0.4324441
4: -0.2747468, 0.3145800, -0.2947618, 0.3330985, -0.6078453, 0.6093419
5: -0.3088662, 0.3314286, -0.3273866, 0.3523792, -0.6612454, 0.6588151
6: -0.3153585, 0.3744123, -0.3311198, 0.3972769, -0.7126354, 0.7055321
7: -0.2597804, 0.3618024, -0.2767344, 0.3809578, -0.6407381, 0.6385368
8: -0.2690353, 0.4160919, -0.2919782, 0.4364348, -0.7054701, 0.7080702
9: -0.3136851, 0.3413160, -0.3320170, 0.3637194, -0.6774045, 0.6733330

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 82

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B2_A1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B2_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3088085, upper bound: 1.3041676
time: 1.96 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B2_A1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B2_A1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2835605, upper bound: 1.3025403
time: 1.53 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A2_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.3138752, 0.3501884, -0.3493451, 0.3878033, -0.7016785, 0.6995335
1: 0.4511299, 1.0795171, 0.4030430, 1.0833983, -0.6322684, 0.6764742
2: -0.2376855, 0.3182051, -0.2632957, 0.3540984, -0.5917839, 0.5815009
3: -0.1523104, 0.2145496, -0.1705022, 0.2404320, -0.3927424, 0.3850518
4: -0.2411290, 0.2753645, -0.2690840, 0.3083367, -0.5494657, 0.5444486
5: -0.2756355, 0.2867728, -0.3033022, 0.3242373, -0.5998728, 0.5900749
6: -0.2828000, 0.3317590, -0.3108925, 0.3672643, -0.6500643, 0.6426515
7: -0.2292982, 0.3128504, -0.2553672, 0.3438728, -0.5731710, 0.5682176
8: -0.2297464, 0.3733312, -0.2625882, 0.4095356, -0.6392820, 0.6359194
9: -0.2769655, 0.3077619, -0.3080138, 0.3329991, -0.6099645, 0.6157756

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 82

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2_A1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B2_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3108949, upper bound: 1.3043913
time: 1.77 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2_A1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B2_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2831133, upper bound: 1.3025734
time: 1.42 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A2_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.3138752, 0.3501884, -0.3964202, 0.4398684, -0.7537436, 0.7466086
1: 0.4511299, 1.0795171, 0.3134469, 1.0881832, -0.6370533, 0.7660702
2: -0.2376855, 0.3182051, -0.2961259, 0.4038712, -0.6415567, 0.6143310
3: -0.1523104, 0.2145496, -0.1991642, 0.2836977, -0.4360082, 0.4137138
4: -0.2411290, 0.2753645, -0.3134409, 0.3504633, -0.5915923, 0.5888054
5: -0.2756355, 0.2867728, -0.3448481, 0.3720673, -0.6477028, 0.6316209
6: -0.2828000, 0.3317590, -0.3457928, 0.4187528, -0.7015529, 0.6775518
7: -0.2292982, 0.3128504, -0.2924671, 0.4003470, -0.6296452, 0.6053175
8: -0.2297464, 0.3733312, -0.3133549, 0.4554634, -0.6852098, 0.6866862
9: -0.2769655, 0.3077619, -0.3492286, 0.3848137, -0.6617792, 0.6569905

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 92

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2_A1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B2_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3108949, upper bound: 1.3050034
time: 1.89 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2_A1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B2_A1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2831133, upper bound: 1.3030804
time: 1.29 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A2_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.3625568, 0.4028900, -0.3493451, 0.3878033, -0.7503601, 0.7522351
1: 0.3643215, 1.0846618, 0.4030430, 1.0833983, -0.7190769, 0.6816189
2: -0.2727486, 0.3684868, -0.2632957, 0.3540984, -0.6268470, 0.6317825
3: -0.1796073, 0.2508092, -0.1705022, 0.2404320, -0.4200393, 0.4213114
4: -0.2810243, 0.3205429, -0.2690840, 0.3083367, -0.5893610, 0.5896269
5: -0.3147154, 0.3382251, -0.3033022, 0.3242373, -0.6389528, 0.6415273
6: -0.3203051, 0.3816808, -0.3108925, 0.3672643, -0.6875694, 0.6925733
7: -0.2650413, 0.3696998, -0.2553672, 0.3438728, -0.6089140, 0.6250670
8: -0.2762261, 0.4225958, -0.2625882, 0.4095356, -0.6857617, 0.6851840
9: -0.3195050, 0.3486479, -0.3080138, 0.3329991, -0.6525040, 0.6566617

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2_A1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B2_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3101413, upper bound: 1.3042383
time: 2.11 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2_A1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B2_A1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2835605, upper bound: 1.3025640
time: 1.43 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A2_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.3625568, 0.4028900, -0.3964202, 0.4398684, -0.8024251, 0.7993101
1: 0.3643215, 1.0846618, 0.3134469, 1.0881832, -0.7238617, 0.7712150
2: -0.2727486, 0.3684868, -0.2961259, 0.4038712, -0.6766198, 0.6646127
3: -0.1796073, 0.2508092, -0.1991642, 0.2836977, -0.4633050, 0.4499733
4: -0.2810243, 0.3205429, -0.3134409, 0.3504633, -0.6314876, 0.6339838
5: -0.3147154, 0.3382251, -0.3448481, 0.3720673, -0.6867828, 0.6830732
6: -0.3203051, 0.3816808, -0.3457928, 0.4187528, -0.7390580, 0.7274736
7: -0.2650413, 0.3696998, -0.2924671, 0.4003470, -0.6653882, 0.6621668
8: -0.2762261, 0.4225958, -0.3133549, 0.4554634, -0.7316895, 0.7359507
9: -0.3195050, 0.3486479, -0.3492286, 0.3848137, -0.7043186, 0.6978765

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 82
type: B, layer: 1, pos: 82

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2_A1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B2_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3101413, upper bound: 1.3047259
time: 1.90 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2_A1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B2_A1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2835605, upper bound: 1.3030603
time: 1.38 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1_B2_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.3092689, 0.3452412, -0.3476897, 0.3859191, -0.6951879, 0.6929310
1: 0.4644021, 1.0790741, 0.4036264, 1.0832027, -0.6188006, 0.6754477
2: -0.2343680, 0.3134990, -0.2621003, 0.3522919, -0.5866600, 0.5755993
3: -0.1494814, 0.2114100, -0.1696054, 0.2391730, -0.3886544, 0.3810154
4: -0.2378029, 0.2710679, -0.2673730, 0.3068024, -0.5446053, 0.5384409
5: -0.2720293, 0.2817774, -0.3019186, 0.3225188, -0.5945481, 0.5836960
6: -0.2794973, 0.3271467, -0.3094994, 0.3654703, -0.6449676, 0.6366462
7: -0.2264068, 0.3038366, -0.2538279, 0.3436828, -0.5700896, 0.5576645
8: -0.2258945, 0.3687320, -0.2605860, 0.4078263, -0.6337209, 0.6293180
9: -0.2730074, 0.3039846, -0.3065471, 0.3312361, -0.6042434, 0.6105317

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A2_B2_B1_B2_A2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_B1_B2_A2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2831133, upper bound: 1.3126844
time: 1.48 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_B2_A2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_B1_B2_A2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2831133, upper bound: 1.3020468
time: 1.71 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1_B2_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.3565360, 0.3960547, -0.3476897, 0.3859191, -0.7424551, 0.7437445
1: 0.3854798, 1.0841163, 0.4036264, 1.0832027, -0.6977229, 0.6804899
2: -0.2684604, 0.3619757, -0.2621003, 0.3522919, -0.6207523, 0.6240760
3: -0.1753359, 0.2454401, -0.1696054, 0.2391730, -0.4145089, 0.4150455
4: -0.2757101, 0.3150066, -0.2673730, 0.3068024, -0.5825126, 0.5823796
5: -0.3092817, 0.3319448, -0.3019186, 0.3225188, -0.6318005, 0.6338634
6: -0.3161935, 0.3749942, -0.3094994, 0.3654703, -0.6816638, 0.6844937
7: -0.2608929, 0.3554863, -0.2538279, 0.3436828, -0.6045757, 0.6093142
8: -0.2702264, 0.4167316, -0.2605860, 0.4078263, -0.6780527, 0.6773176
9: -0.3142033, 0.3414215, -0.3065471, 0.3312361, -0.6454394, 0.6479685

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 82
type: A, layer: 1, pos: 82

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A2_B2_B1_B2_A2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_B1_B2_A2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2831133, upper bound: 1.3127560
time: 1.45 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_B2_A2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_B1_B2_A2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2831133, upper bound: 1.3020468
time: 1.52 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.50 + 596.39 = 600.89 seconds
