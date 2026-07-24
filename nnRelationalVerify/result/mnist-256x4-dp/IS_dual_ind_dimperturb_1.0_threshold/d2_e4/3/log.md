## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 82.0484663031


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=251, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-56.7985649, 46.7652855, -56.7985649, 46.7652855, -103.5638504, 103.5638428)
1: (-49.9042206, 38.4526215, -49.9042206, 38.4526215, -88.3568420, 88.3568420)
2: (-63.7191658, 41.6070213, -63.7191658, 41.6070213, -105.3261871, 105.3261871)
3: (-70.5155945, 36.0443954, -70.5155945, 36.0443954, -106.5599899, 106.5599899)
4: (-64.2876892, 46.5763397, -64.2876892, 46.5763397, -110.8640137, 110.8640137)
5: (-54.7288971, 44.6650429, -54.7288971, 44.6650429, -99.3939285, 99.3939285)
6: (-54.0188065, 51.2609940, -54.0188065, 51.2609940, -105.2798004, 105.2798004)
7: (-61.0627823, 49.1374741, -61.0627823, 49.1374741, -110.2002487, 110.2002411)
8: (-73.9399872, 45.3020477, -73.9399872, 45.3020477, -119.2420273, 119.2420273)
9: (-51.7169037, 53.1263313, -51.7169037, 53.1263313, -104.8432312, 104.8432312)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.35 + 13.98 = 15.33 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -82.1305969, upper bound: 82.1305969

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1286541, upper bound: 82.1285629
time: 10.50 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1285001, upper bound: 82.1285001
time: 9.10 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 19.72 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 19.72
Output dim: 1, lower bound: -82.1286541, upper bound: 82.1285629
IS_A2, status: Status.UNKNOWN, split count: 1, time: 19.72
Output dim: 1, lower bound: -82.1285001, upper bound: 82.1285001

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -50.5296860, 41.8674126, -54.7670555, 45.1827927, -95.7124710, 96.6344681
1: -44.6550293, 33.9995918, -48.2069817, 37.0133934, -81.6684265, 82.2065735
2: -56.8586960, 37.1124496, -61.5004959, 40.1534958, -97.0121689, 98.6129227
3: -63.2734413, 32.2411385, -68.1748810, 34.8132973, -98.0867386, 100.4160156
4: -57.6169624, 41.4024391, -62.1269951, 44.9001274, -102.5170746, 103.5294266
5: -48.7414551, 40.0365715, -52.7921524, 43.1647415, -91.9061890, 92.8287201
6: -48.3009415, 45.7399406, -52.1684837, 49.4750862, -97.7760239, 97.9084244
7: -54.8291016, 43.8889122, -59.0457764, 47.4379730, -102.2670670, 102.9346695
8: -66.4406815, 40.1198196, -71.5143814, 43.6251373, -110.0658112, 111.6341858
9: -46.0375252, 47.5133362, -49.8808250, 51.3133392, -97.3508453, 97.3941650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=49, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=239, inp2_unstable=248, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1198188, upper bound: 82.1209419
time: 14.54 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1188913, upper bound: 82.1189248
time: 9.81 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -51.3580055, 42.5393066, -53.8314629, 44.4437447, -95.8017502, 96.3707733
1: -45.3729782, 34.5430069, -47.4201202, 36.3591118, -81.7320862, 81.9631195
2: -57.7843285, 37.7049484, -60.4729462, 39.4876633, -97.2719879, 98.1778946
3: -64.3268433, 32.7433357, -67.0763779, 34.2431030, -98.5699463, 99.8197174
4: -58.5576897, 42.0762978, -61.1133270, 44.1330452, -102.6907349, 103.1896133
5: -49.5278320, 40.6730690, -51.8958855, 42.4634933, -91.9913254, 92.5689545
6: -49.0917816, 46.4812279, -51.3098755, 48.6486473, -97.7404327, 97.7911072
7: -55.7193451, 44.5904121, -58.0998268, 46.6545944, -102.3739395, 102.6902390
8: -67.5297318, 40.7562675, -70.3814240, 42.8632317, -110.3929596, 111.1376953
9: -46.7693329, 48.2775269, -49.0294647, 50.4680367, -97.2373581, 97.3069916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=49, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=112, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=237, inp2_unstable=247, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1200299, upper bound: 82.1210409
time: 15.26 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1190754, upper bound: 82.1190754
time: 9.43 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 26.04 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 26.04
Output dim: 1, lower bound: -82.1198188, upper bound: 82.1209419
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 26.04
Output dim: 1, lower bound: -82.1188913, upper bound: 82.1189248
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 26.04
Output dim: 1, lower bound: -82.1200299, upper bound: 82.1210409
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 26.04
Output dim: 1, lower bound: -82.1190754, upper bound: 82.1190754

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -50.5296860, 41.8674126, -52.4990578, 43.4019547, -93.9316330, 94.3664627
1: -44.6550293, 33.9995918, -46.2926979, 35.3889313, -80.0439606, 80.2922897
2: -56.8586960, 37.1124496, -58.9961548, 38.5110931, -95.3697815, 96.1085968
3: -63.2734413, 32.2411385, -65.4950638, 33.4269028, -96.7003479, 97.7362061
4: -57.6169624, 41.4024391, -59.6864510, 43.0196152, -100.6365662, 101.0888901
5: -48.7414551, 40.0365715, -50.6108208, 41.4600525, -90.2015076, 90.6473846
6: -48.3009415, 45.7399406, -50.0798721, 47.4561615, -95.7570953, 95.8198090
7: -54.8291016, 43.8889122, -56.7665024, 45.5133705, -100.3424683, 100.6554108
8: -66.4406815, 40.1198196, -68.7949066, 41.7410278, -108.1817093, 108.9147186
9: -46.0375252, 47.5133362, -47.7997360, 49.2470856, -95.2845917, 95.3130722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=48, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=239, inp2_unstable=244, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 94

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1188913, upper bound: 82.1189247
time: 11.91 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1188913, upper bound: 82.1189248
time: 9.78 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -47.9873047, 39.8544350, -53.0593224, 44.0446625, -92.0319595, 92.9137573
1: -42.4847183, 32.1558380, -46.8833237, 35.4417915, -77.9264908, 79.0391617
2: -54.0379753, 35.2572250, -59.7007790, 38.8806267, -92.9186020, 94.9580078
3: -60.2545280, 30.6833725, -66.6474152, 33.8159561, -94.0704803, 97.3307877
4: -54.8615723, 39.2788582, -60.6867828, 43.3451881, -98.2067566, 99.9656372
5: -46.2971115, 38.1117287, -51.1359024, 42.0442657, -88.3413696, 89.2476196
6: -45.9475403, 43.4582520, -50.8068275, 47.9626617, -93.9102020, 94.2650681
7: -52.2608261, 41.7094536, -57.7951889, 45.9993515, -98.2601624, 99.5046387
8: -63.3606720, 37.9874458, -70.0667801, 41.8604164, -105.2210846, 108.0542297
9: -43.6672211, 45.1763382, -48.1614532, 49.8936081, -93.5608292, 93.3377914

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=48, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=117, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=234, inp2_unstable=241, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 94

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1173823, upper bound: 82.1173359
time: 10.86 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1172840, upper bound: 82.1173010
time: 12.54 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -51.3580055, 42.5393066, -51.5490723, 42.6506729, -94.0086823, 94.0883789
1: -45.3729782, 34.5430069, -45.4942970, 34.7229309, -80.0959015, 80.0373077
2: -57.7843285, 37.7049484, -57.9519196, 37.8343658, -95.6186981, 95.6568680
3: -64.3268433, 32.7433357, -64.3789825, 32.8489075, -97.1757507, 97.1223145
4: -58.5576897, 42.0762978, -58.6591721, 42.2419128, -100.7996063, 100.7354660
5: -49.5278320, 40.6730690, -49.7008247, 40.7515106, -90.2793274, 90.3738861
6: -49.0917816, 46.4812279, -49.2098465, 46.6170731, -95.7088547, 95.6910629
7: -55.7193451, 44.5904121, -55.8067207, 44.7181320, -100.4374771, 100.3971329
8: -67.5297318, 40.7562675, -67.6469498, 40.9658546, -108.4955902, 108.4032135
9: -46.7693329, 48.2775269, -46.9337845, 48.3873749, -95.1566925, 95.2113113

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=48, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=112, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=237, inp2_unstable=242, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 94

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1190754, upper bound: 82.1190754
time: 11.30 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1190754, upper bound: 82.1190754
time: 9.25 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -48.9038582, 40.5923347, -52.2670135, 43.4084244, -92.3122864, 92.8593445
1: -43.2754898, 32.7636070, -46.2098808, 34.8834839, -78.1589737, 78.9734879
2: -55.0608597, 35.9113960, -58.8250580, 38.3122635, -93.3731155, 94.7364502
3: -61.4069138, 31.2384987, -65.7088089, 33.3285065, -94.7354126, 96.9473114
4: -55.8967590, 40.0262222, -59.8249931, 42.6944771, -98.5912323, 99.8512115
5: -47.1664467, 38.8145218, -50.3733215, 41.4487610, -88.6152039, 89.1878433
6: -46.8179436, 44.2774162, -50.0751343, 47.2570801, -94.0750275, 94.3525238
7: -53.2387085, 42.4848480, -56.9916496, 45.3289413, -98.5676270, 99.4764938
8: -64.5527496, 38.6982422, -69.0951157, 41.2118340, -105.7645721, 107.7933578
9: -44.4814835, 46.0185547, -47.4350166, 49.1702385, -93.6517181, 93.4535675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=48, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=112, inp2_unstable=117, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=234, inp2_unstable=240, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 94

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1174793, upper bound: 82.1174054
time: 10.21 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1173651, upper bound: 82.1173651
time: 9.88 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 21.49 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.49
Output dim: 1, lower bound: -82.1188913, upper bound: 82.1189247
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.49
Output dim: 1, lower bound: -82.1188913, upper bound: 82.1189248
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.49
Output dim: 1, lower bound: -82.1173823, upper bound: 82.1173359
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.49
Output dim: 1, lower bound: -82.1172840, upper bound: 82.1173010
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.49
Output dim: 1, lower bound: -82.1190754, upper bound: 82.1190754
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.49
Output dim: 1, lower bound: -82.1190754, upper bound: 82.1190754
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.49
Output dim: 1, lower bound: -82.1174793, upper bound: 82.1174054
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.49
Output dim: 1, lower bound: -82.1173651, upper bound: 82.1173651

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -48.4329872, 40.2072258, -52.4990578, 43.4019547, -91.8349457, 92.7062607
1: -42.8717613, 32.4849548, -46.2926979, 35.3889313, -78.2606888, 78.7776413
2: -54.5342445, 35.5874519, -58.9961548, 38.5110931, -93.0453339, 94.5836029
3: -60.7848053, 30.9539413, -65.4950638, 33.4269028, -94.2117081, 96.4490051
4: -55.3499374, 39.6578026, -59.6864510, 43.0196152, -98.3695450, 99.3442535
5: -46.7239456, 38.4505692, -50.6108208, 41.4600525, -88.1839981, 89.0613708
6: -46.3644981, 43.8635902, -50.0798721, 47.4561615, -93.8206482, 93.9434662
7: -52.7160759, 42.0955429, -56.7665024, 45.5133705, -98.2294464, 98.8620453
8: -63.9041672, 38.3690300, -68.7949066, 41.7410278, -105.6451950, 107.1639328
9: -44.0978394, 45.5886230, -47.7997360, 49.2470856, -93.3449173, 93.3883514

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=48, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=235, inp2_unstable=244, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1198188, upper bound: 82.1209419
time: 14.77 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1198188, upper bound: 82.1209419
time: 12.66 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -49.0971451, 40.9261818, -52.4990578, 43.4019547, -92.4990997, 93.4252319
1: -43.5386963, 32.5983849, -46.2926979, 35.3889313, -78.9276276, 78.8910828
2: -55.3518600, 36.0223236, -58.9961548, 38.5110931, -93.8629456, 95.0184708
3: -62.0587654, 31.4003963, -65.4950638, 33.4269028, -95.4856644, 96.8954620
4: -56.4579201, 40.0593567, -59.6864510, 43.0196152, -99.4775162, 99.7458038
5: -47.3515167, 39.1028900, -50.6108208, 41.4600525, -88.8115616, 89.7137070
6: -47.1814308, 44.4539680, -50.0798721, 47.4561615, -94.6375885, 94.5338364
7: -53.8425102, 42.6591682, -56.7665024, 45.5133705, -99.3558807, 99.4256668
8: -65.2805557, 38.5690727, -68.7949066, 41.7410278, -107.0215836, 107.3639679
9: -44.5461807, 46.3238754, -47.7997360, 49.2470856, -93.7932587, 94.1235962

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=48, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=117, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=231, inp2_unstable=244, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1198188, upper bound: 82.1209419
time: 11.83 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1198188, upper bound: 82.1209419
time: 15.76 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -44.2396851, 36.8277473, -51.6477928, 42.9070892, -87.1467743, 88.4755325
1: -39.2223816, 29.4948196, -45.6569138, 34.4310074, -73.6533813, 75.1517181
2: -49.8582115, 32.5338402, -58.1272888, 37.8556290, -87.7138367, 90.6611328
3: -55.7257500, 28.3380394, -64.9524078, 32.9353905, -88.6611404, 93.2904510
4: -50.7357826, 36.1751251, -59.1408005, 42.1750679, -92.9108505, 95.3159180
5: -42.7092781, 35.2359581, -49.7831230, 40.9691582, -83.6784363, 85.0190811
6: -42.4419250, 40.1068497, -49.4900360, 46.6972733, -89.1391983, 89.5968857
7: -48.3326912, 38.4957657, -56.3284149, 44.7882500, -93.1209412, 94.8241806
8: -58.6805878, 34.9683151, -68.3187714, 40.7189064, -99.3994904, 103.2870865
9: -40.2164536, 41.6944046, -46.8588867, 48.5862427, -88.8026962, 88.5532837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=48, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=108, inp2_unstable=117, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=225, inp2_unstable=239, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1173749, upper bound: 82.1173053
time: 11.25 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1173749, upper bound: 82.1173359
time: 10.44 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -45.5600395, 37.9077644, -52.1858940, 43.3445663, -88.9045944, 90.0936584
1: -40.3956985, 30.4059143, -46.1319199, 34.8104744, -75.2061691, 76.5378342
2: -51.3412361, 33.5050888, -58.7300606, 38.2489281, -89.5901566, 92.2351532
3: -57.3642464, 29.1663399, -65.6097565, 33.2715836, -90.6358337, 94.7760925
4: -52.2419014, 37.2614479, -59.7448578, 42.6190071, -94.8608856, 97.0062943
5: -43.9702110, 36.2731018, -50.2984848, 41.3851280, -85.3553391, 86.5715866
6: -43.6993256, 41.2913971, -49.9974747, 47.1823044, -90.8816223, 91.2888718
7: -49.7698555, 39.6312599, -56.9018326, 45.2515221, -95.0213470, 96.5330963
8: -60.3865128, 36.0130997, -68.9981308, 41.1503754, -101.5368881, 105.0112152
9: -41.4281921, 42.9307899, -47.3558731, 49.0875740, -90.5157623, 90.2866592

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=48, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=117, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=240, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1172755, upper bound: 82.1172709
time: 10.49 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1172755, upper bound: 82.1173010
time: 11.41 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -49.1792641, 40.8133698, -51.5490723, 42.6506729, -91.8299408, 92.3624420
1: -43.5177383, 32.9680634, -45.4942970, 34.7229309, -78.2406693, 78.4623566
2: -55.3701477, 36.1173515, -57.9519196, 37.8343658, -93.2045059, 94.0692749
3: -61.7418861, 31.4068623, -64.3789825, 32.8489075, -94.5907898, 95.7858429
4: -56.2025986, 40.2632446, -58.6591721, 42.2419128, -98.4445114, 98.9224014
5: -47.4304199, 39.0260735, -49.7008247, 40.7515106, -88.1819153, 88.7268982
6: -47.0796127, 44.5302353, -49.2098465, 46.6170731, -93.6966858, 93.7400665
7: -53.5226555, 42.7265549, -55.8067207, 44.7181320, -98.2407761, 98.5332794
8: -64.8939743, 38.9365883, -67.6469498, 40.9658546, -105.8598251, 106.5835419
9: -44.7548294, 46.2781143, -46.9337845, 48.3873749, -93.1421890, 93.2118988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=48, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=112, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=236, inp2_unstable=242, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 94

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1199935, upper bound: 82.1210074
time: 13.57 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1199935, upper bound: 82.1210409
time: 14.47 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -50.1318665, 41.7601280, -51.5490723, 42.6506729, -92.7825317, 93.3091888
1: -44.4311523, 33.2874374, -45.4942970, 34.7229309, -79.1540833, 78.7817383
2: -56.5092926, 36.7654228, -57.9519196, 37.8343658, -94.3436508, 94.7173386
3: -63.3537903, 32.0259361, -64.3789825, 32.8489075, -96.2026978, 96.4049149
4: -57.6271362, 40.9063034, -58.6591721, 42.2419128, -99.8690491, 99.5654678
5: -48.3309784, 39.9019165, -49.7008247, 40.7515106, -89.0824814, 89.6027374
6: -48.1643677, 45.3783760, -49.2098465, 46.6170731, -94.7814331, 94.5882187
7: -54.9535789, 43.5358047, -55.8067207, 44.7181320, -99.6717072, 99.3425140
8: -66.6306534, 39.3732300, -67.6469498, 40.9658546, -107.5965042, 107.0201721
9: -45.4693336, 47.2774239, -46.9337845, 48.3873749, -93.8567047, 94.2112122

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=48, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=119, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=232, inp2_unstable=242, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 94

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1199935, upper bound: 82.1210074
time: 14.61 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1199935, upper bound: 82.1210409
time: 12.88 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -45.1335030, 37.5450859, -50.8626976, 42.2768898, -87.4103928, 88.4077835
1: -39.9919205, 30.0846291, -44.9893532, 33.8792229, -73.8711395, 75.0739746
2: -50.8550186, 33.1701889, -57.2611160, 37.2934685, -88.1484756, 90.4312973
3: -56.8506279, 28.8769569, -64.0220718, 32.4521828, -89.3028030, 92.8990173
4: -51.7456055, 36.9038544, -58.2857285, 41.5302582, -93.2758636, 95.1895828
5: -43.5546722, 35.9207497, -49.0276031, 40.3779449, -83.9326096, 84.9483490
6: -43.2893753, 40.9034042, -48.7653198, 45.9987793, -89.2881470, 89.6687241
7: -49.2898941, 39.2487755, -55.5314445, 44.1237450, -93.4136353, 94.7801971
8: -59.8437881, 35.6609993, -67.3545914, 40.0768204, -99.9206085, 103.0155792
9: -41.0092354, 42.5155830, -46.1389275, 47.8692894, -88.8785248, 88.6545105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=48, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=110, inp2_unstable=117, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=225, inp2_unstable=235, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1173998, upper bound: 82.1173132
time: 10.98 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1173998, upper bound: 82.1174053
time: 11.62 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -46.4914856, 38.6563148, -51.4017181, 42.7153740, -89.2068481, 90.0580292
1: -41.1966400, 31.0233650, -45.4654160, 34.2589951, -75.4556274, 76.4887848
2: -52.3795891, 34.1689644, -57.8639984, 37.6870346, -90.0666199, 92.0329590
3: -58.5346184, 29.7305088, -64.6810684, 32.7890396, -91.3236389, 94.4115753
4: -53.2932472, 38.0204391, -58.8914337, 41.9752426, -95.2684937, 96.9118652
5: -44.8522415, 36.9864502, -49.5442238, 40.7949219, -85.6471634, 86.5306702
6: -44.5815811, 42.1225929, -49.2736244, 46.4849358, -91.0665131, 91.3962021
7: -50.7629700, 40.4177933, -56.1059074, 44.5878448, -95.3508072, 96.5236893
8: -61.5970268, 36.7358131, -68.0358047, 40.5088005, -102.1058197, 104.7716064
9: -42.2571297, 43.7858887, -46.6368790, 48.3715363, -90.6286469, 90.4227524

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=48, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=111, inp2_unstable=117, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=237, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1173010, upper bound: 82.1172831
time: 9.67 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1173010, upper bound: 82.1173651
time: 11.39 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 22.45 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.45
Output dim: 1, lower bound: -82.1198188, upper bound: 82.1209419
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.45
Output dim: 1, lower bound: -82.1198188, upper bound: 82.1209419
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.45
Output dim: 1, lower bound: -82.1198188, upper bound: 82.1209419
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.45
Output dim: 1, lower bound: -82.1198188, upper bound: 82.1209419
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.45
Output dim: 1, lower bound: -82.1173749, upper bound: 82.1173053
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.45
Output dim: 1, lower bound: -82.1173749, upper bound: 82.1173359
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.45
Output dim: 1, lower bound: -82.1172755, upper bound: 82.1172709
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.45
Output dim: 1, lower bound: -82.1172755, upper bound: 82.1173010
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.45
Output dim: 1, lower bound: -82.1199935, upper bound: 82.1210074
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.45
Output dim: 1, lower bound: -82.1199935, upper bound: 82.1210409
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.45
Output dim: 1, lower bound: -82.1199935, upper bound: 82.1210074
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.45
Output dim: 1, lower bound: -82.1199935, upper bound: 82.1210409
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.45
Output dim: 1, lower bound: -82.1173998, upper bound: 82.1173132
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.45
Output dim: 1, lower bound: -82.1173998, upper bound: 82.1174053
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.45
Output dim: 1, lower bound: -82.1173010, upper bound: 82.1172831
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.45
Output dim: 1, lower bound: -82.1173010, upper bound: 82.1173651

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -48.4329872, 40.2072258, -48.4329872, 40.2072258, -88.6402054, 88.6402054
1: -42.8717613, 32.4849548, -42.8717613, 32.4849548, -75.3566971, 75.3566971
2: -54.5342445, 35.5874519, -54.5342445, 35.5874519, -90.1216965, 90.1216965
3: -60.7848053, 30.9539413, -60.7848053, 30.9539413, -91.7387390, 91.7387314
4: -55.3499374, 39.6578026, -55.3499374, 39.6578026, -95.0077286, 95.0077362
5: -46.7239456, 38.4505692, -46.7239456, 38.4505692, -85.1745148, 85.1745148
6: -46.3644981, 43.8635902, -46.3644981, 43.8635902, -90.2280884, 90.2280884
7: -52.7160759, 42.0955429, -52.7160759, 42.0955429, -94.8116150, 94.8116150
8: -63.9041672, 38.3690300, -63.9041672, 38.3690300, -102.2731934, 102.2731934
9: -44.0978394, 45.5886230, -44.0978394, 45.5886230, -89.6864624, 89.6864624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=47, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=235, inp2_unstable=235, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 94

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1268610, upper bound: 82.1268115
time: 11.52 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1270689, upper bound: 82.1269763
time: 12.93 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -48.4329872, 40.2072258, -49.1792641, 40.8133698, -89.2463531, 89.3864899
1: -42.8717613, 32.4849548, -43.5177383, 32.9680634, -75.8398132, 76.0026855
2: -54.5342445, 35.5874519, -55.3701477, 36.1173515, -90.6515961, 90.9575958
3: -60.7848053, 30.9539413, -61.7418861, 31.4068623, -92.1916580, 92.6958237
4: -55.3499374, 39.6578026, -56.2025986, 40.2632446, -95.6131592, 95.8603973
5: -46.7239456, 38.4505692, -47.4304199, 39.0260735, -85.7500153, 85.8809814
6: -46.3644981, 43.8635902, -47.0796127, 44.5302353, -90.8947067, 90.9432068
7: -52.7160759, 42.0955429, -53.5226555, 42.7265549, -95.4426270, 95.6181870
8: -63.9041672, 38.3690300, -64.8939743, 38.9365883, -102.8407593, 103.2630005
9: -44.0978394, 45.5886230, -44.7548294, 46.2781143, -90.3759537, 90.3434448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=47, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=112, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=235, inp2_unstable=236, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 94

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1268610, upper bound: 82.1268115
time: 10.65 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1270689, upper bound: 82.1269763
time: 12.73 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -49.0971451, 40.9261818, -48.4329872, 40.2072258, -89.3043594, 89.3591690
1: -43.5386963, 32.5983849, -42.8717613, 32.4849548, -76.0236359, 75.4701385
2: -55.3518600, 36.0223236, -54.5342445, 35.5874519, -90.9393005, 90.5565567
3: -62.0587654, 31.4003963, -60.7848053, 30.9539413, -93.0126953, 92.1851883
4: -56.4579201, 40.0593567, -55.3499374, 39.6578026, -96.1157150, 95.4092865
5: -47.3515167, 39.1028900, -46.7239456, 38.4505692, -85.8020782, 85.8268356
6: -47.1814308, 44.4539680, -46.3644981, 43.8635902, -91.0450211, 90.8184586
7: -53.8425102, 42.6591682, -52.7160759, 42.0955429, -95.9380493, 95.3752441
8: -65.2805557, 38.5690727, -63.9041672, 38.3690300, -103.6495819, 102.4732361
9: -44.5461807, 46.3238754, -44.0978394, 45.5886230, -90.1347961, 90.4217072

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=47, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=117, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=231, inp2_unstable=235, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 238

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1185558, upper bound: 82.1197369
time: 15.44 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1182947, upper bound: 82.1195960
time: 13.90 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -49.0971451, 40.9261818, -49.1792641, 40.8133698, -89.9105148, 90.1054459
1: -43.5386963, 32.5983849, -43.5177383, 32.9680634, -76.5067444, 76.1161194
2: -55.3518600, 36.0223236, -55.3701477, 36.1173515, -91.4692078, 91.3924637
3: -62.0587654, 31.4003963, -61.7418861, 31.4068623, -93.4656219, 93.1422729
4: -56.4579201, 40.0593567, -56.2025986, 40.2632446, -96.7211456, 96.2619553
5: -47.3515167, 39.1028900, -47.4304199, 39.0260735, -86.3775940, 86.5333099
6: -47.1814308, 44.4539680, -47.0796127, 44.5302353, -91.7116623, 91.5335846
7: -53.8425102, 42.6591682, -53.5226555, 42.7265549, -96.5690613, 96.1818161
8: -65.2805557, 38.5690727, -64.8939743, 38.9365883, -104.2171478, 103.4630432
9: -44.5461807, 46.3238754, -44.7548294, 46.2781143, -90.8242874, 91.0786896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=47, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=117, inp2_unstable=112, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=231, inp2_unstable=236, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 238

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1185558, upper bound: 82.1197370
time: 14.25 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1182947, upper bound: 82.1195960
time: 9.71 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -44.2396851, 36.8277473, -47.7301788, 39.8262939, -84.0659714, 84.5579224
1: -39.2223816, 29.4948196, -42.3533974, 31.6225376, -70.8449020, 71.8482208
2: -49.8582115, 32.5338402, -53.8307304, 35.0324631, -84.8906708, 86.3645706
3: -55.7257500, 28.3380394, -60.4163589, 30.5476189, -86.2733688, 88.7543945
4: -50.7357826, 36.1751251, -54.9593239, 38.9276581, -89.6634369, 91.1344376
5: -42.7092781, 35.2359581, -46.0442543, 38.0573578, -80.7666321, 81.2802124
6: -42.4419250, 40.1068497, -45.9075928, 43.2312889, -85.6732025, 86.0144424
7: -48.3326912, 38.4957657, -52.4174042, 41.4875679, -89.8202515, 90.9131699
8: -58.6805878, 34.9683151, -63.5834732, 37.4668922, -96.1474762, 98.5517883
9: -40.2164536, 41.6944046, -43.2860107, 45.0565567, -85.2730103, 84.9804153

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=47, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=108, inp2_unstable=117, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=225, inp2_unstable=228, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 238

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1125892, upper bound: 82.1120765
time: 12.29 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1113568, upper bound: 82.1113757
time: 11.12 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -44.2396851, 36.8277473, -48.7517624, 40.6502037, -84.8898849, 85.5795059
1: -39.2223816, 29.4948196, -43.2345848, 32.3029060, -71.5252686, 72.7294006
2: -49.8582115, 32.5338402, -54.9744415, 35.7660904, -85.6242981, 87.5082855
3: -55.7257500, 28.3380394, -61.6961365, 31.1656170, -86.8913574, 90.0341797
4: -50.7357826, 36.1751251, -56.1147308, 39.7642670, -90.5000458, 92.2898483
5: -42.7092781, 35.2359581, -47.0118141, 38.8474045, -81.5566864, 82.2477646
6: -42.4419250, 40.1068497, -46.8783722, 44.1442604, -86.5861816, 86.9852219
7: -48.3326912, 38.4957657, -53.5156937, 42.3535080, -90.6862030, 92.0114594
8: -58.6805878, 34.9683151, -64.9162292, 38.2610855, -96.9416733, 99.8845367
9: -40.2164536, 41.6944046, -44.1982231, 45.9987526, -86.2152100, 85.8926239

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=47, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=108, inp2_unstable=118, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=225, inp2_unstable=228, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 238

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1125892, upper bound: 82.1120765
time: 11.72 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1113568, upper bound: 82.1113757
time: 12.26 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -45.5600395, 37.9077644, -48.2467766, 40.2461319, -85.8061676, 86.1545410
1: -40.3956985, 30.4059143, -42.8093948, 31.9854813, -72.3811646, 73.2153015
2: -51.3412361, 33.5050888, -54.4083176, 35.4089241, -86.7501602, 87.9134064
3: -57.3642464, 29.1663399, -61.0483055, 30.8702507, -88.2344971, 90.2146378
4: -52.2419014, 37.2614479, -55.5400848, 39.3536682, -91.5955658, 92.8015289
5: -43.9702110, 36.2731018, -46.5390167, 38.4581833, -82.4283905, 82.8121185
6: -43.6993256, 41.2913971, -46.3949661, 43.6965370, -87.3958588, 87.6863480
7: -49.7698555, 39.6312599, -52.9698067, 41.9315033, -91.7013397, 92.6010666
8: -60.3865128, 36.0130997, -64.2371979, 37.8794174, -98.2659149, 100.2502823
9: -41.4281921, 42.9307899, -43.7624969, 45.5383301, -86.9665222, 86.6932831

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=47, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=117, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=230, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 94

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1126222, upper bound: 82.1121055
time: 12.32 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1117267, upper bound: 82.1116681
time: 13.59 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -45.5600395, 37.9077644, -49.2843704, 41.0822258, -86.6422577, 87.1921387
1: -40.3956985, 30.4059143, -43.7036705, 32.6768723, -73.0725555, 74.1095886
2: -51.3412361, 33.5050888, -55.5693207, 36.1538200, -87.4950562, 89.0744095
3: -57.3642464, 29.1663399, -62.3466492, 31.4981804, -88.8624268, 91.5129852
4: -52.2419014, 37.2614479, -56.7124214, 40.2027779, -92.4446640, 93.9738541
5: -43.9702110, 36.2731018, -47.5211754, 39.2594299, -83.2296371, 83.7942810
6: -43.6993256, 41.2913971, -47.3799438, 44.6234055, -88.3227310, 88.6713409
7: -49.7698555, 39.6312599, -54.0835571, 42.8106613, -92.5805054, 93.7148132
8: -60.3865128, 36.0130997, -65.5906448, 38.6858177, -99.0723190, 101.6037140
9: -41.4281921, 42.9307899, -44.6883621, 46.4945526, -87.9227371, 87.6191483

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=47, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=119, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=231, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 94

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1126222, upper bound: 82.1121055
time: 13.19 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1117267, upper bound: 82.1116681
time: 12.19 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -49.1792641, 40.8133698, -48.4329872, 40.2072258, -89.3864899, 89.2463531
1: -43.5177383, 32.9680634, -42.8717613, 32.4849548, -76.0026855, 75.8398132
2: -55.3701477, 36.1173515, -54.5342445, 35.5874519, -90.9575958, 90.6515961
3: -61.7418861, 31.4068623, -60.7848053, 30.9539413, -92.6958237, 92.1916580
4: -56.2025986, 40.2632446, -55.3499374, 39.6578026, -95.8603973, 95.6131592
5: -47.4304199, 39.0260735, -46.7239456, 38.4505692, -85.8809814, 85.7500153
6: -47.0796127, 44.5302353, -46.3644981, 43.8635902, -90.9432068, 90.8947067
7: -53.5226555, 42.7265549, -52.7160759, 42.0955429, -95.6181870, 95.4426270
8: -64.8939743, 38.9365883, -63.9041672, 38.3690300, -103.2630005, 102.8407593
9: -44.7548294, 46.2781143, -44.0978394, 45.5886230, -90.3434525, 90.3759537

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=47, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=112, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=236, inp2_unstable=235, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 94

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1266599, upper bound: 82.1267265
time: 12.05 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1269082, upper bound: 82.1269082
time: 10.48 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 23.91 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.91
Output dim: 1, lower bound: -82.1268610, upper bound: 82.1268115
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.91
Output dim: 1, lower bound: -82.1270689, upper bound: 82.1269763
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.91
Output dim: 1, lower bound: -82.1268610, upper bound: 82.1268115
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.91
Output dim: 1, lower bound: -82.1270689, upper bound: 82.1269763
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.91
Output dim: 1, lower bound: -82.1185558, upper bound: 82.1197369
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.91
Output dim: 1, lower bound: -82.1182947, upper bound: 82.1195960
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.91
Output dim: 1, lower bound: -82.1185558, upper bound: 82.1197370
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.91
Output dim: 1, lower bound: -82.1182947, upper bound: 82.1195960
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.91
Output dim: 1, lower bound: -82.1125892, upper bound: 82.1120765
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.91
Output dim: 1, lower bound: -82.1113568, upper bound: 82.1113757
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.91
Output dim: 1, lower bound: -82.1125892, upper bound: 82.1120765
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.91
Output dim: 1, lower bound: -82.1113568, upper bound: 82.1113757
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.91
Output dim: 1, lower bound: -82.1126222, upper bound: 82.1121055
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.91
Output dim: 1, lower bound: -82.1117267, upper bound: 82.1116681
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.91
Output dim: 1, lower bound: -82.1126222, upper bound: 82.1121055
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.91
Output dim: 1, lower bound: -82.1117267, upper bound: 82.1116681
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.91
Output dim: 1, lower bound: -82.1266599, upper bound: 82.1267265
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.91
Output dim: 1, lower bound: -82.1269082, upper bound: 82.1269082
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.91
Output dim: 1, lower bound: -82.1199935, upper bound: 82.1210409
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.91
Output dim: 1, lower bound: -82.1199935, upper bound: 82.1210074
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.91
Output dim: 1, lower bound: -82.1199935, upper bound: 82.1210409
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.91
Output dim: 1, lower bound: -82.1173998, upper bound: 82.1173132
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.91
Output dim: 1, lower bound: -82.1173998, upper bound: 82.1174053
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.91
Output dim: 1, lower bound: -82.1173010, upper bound: 82.1172831
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.91
Output dim: 1, lower bound: -82.1173010, upper bound: 82.1173651

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 15.33 + 606.26 = 621.59 seconds
