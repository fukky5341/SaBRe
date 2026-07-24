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
execution time: IAR + RelationalAnalysis = 2.12 + 2.64 = 4.76 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956
time: 6.14 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956
time: 1.37 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.72 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.72
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956
NS_A2, status: Status.UNKNOWN, split count: 1, time: 7.72
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.7116623, 0.7772864, -0.7141591, 0.7795098, -1.4911721, 1.4914455
1: -0.2632843, 1.1259298, -0.2653555, 1.1262589, -1.3895431, 1.3912853
2: -0.4987734, 0.7406372, -0.5000965, 0.7434788, -1.2422522, 1.2407336
3: -0.3984092, 0.5532498, -0.4006073, 0.5551579, -0.9535671, 0.9538572
4: -0.5998507, 0.6216936, -0.6025041, 0.6237597, -1.2236104, 1.2241976
5: -0.6201006, 0.6730317, -0.6217980, 0.6758969, -1.2959974, 1.2948297
6: -0.5681255, 0.7391180, -0.5698846, 0.7409250, -1.3090506, 1.3090026
7: -0.5258098, 0.7957379, -0.5282102, 0.7967823, -1.3225920, 1.3239481
8: -0.6689795, 0.7601168, -0.6727314, 0.7619013, -1.4308808, 1.4328482
9: -0.6038283, 0.7350418, -0.6058326, 0.7372477, -1.3410760, 1.3408744

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3309396, upper bound: 1.3252106
time: 1.45 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3309396, upper bound: 1.3309396
time: 1.38 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.7098854, 0.7757605, -0.7487078, 0.8067603, -1.5166457, 1.5244683
1: -0.2597983, 1.1257753, -0.3093177, 1.1299214, -1.3897197, 1.4350930
2: -0.4977460, 0.7387493, -0.5193037, 0.7782588, -1.2760048, 1.2580529
3: -0.3968386, 0.5519456, -0.4298161, 0.5796028, -0.9764415, 0.9817617
4: -0.5980328, 0.6200411, -0.6391937, 0.6508259, -1.2488587, 1.2592348
5: -0.6187797, 0.6709954, -0.6436503, 0.7122319, -1.3310115, 1.3146456
6: -0.5668839, 0.7378871, -0.5920236, 0.7630900, -1.3299738, 1.3299108
7: -0.5241609, 0.7923995, -0.5592597, 0.8337017, -1.3578626, 1.3516593
8: -0.6664008, 0.7588431, -0.7199079, 0.7840109, -1.4504118, 1.4787509
9: -0.6022526, 0.7335892, -0.6345332, 0.7649568, -1.3672094, 1.3681225

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956
time: 1.41 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956
time: 7.38 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 11.11 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 11.11
Output dim: 1, lower bound: -1.3309396, upper bound: 1.3252106
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 11.11
Output dim: 1, lower bound: -1.3309396, upper bound: 1.3309396
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 11.11
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 11.11
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.6505494, 0.7192606, -0.5393963, 0.5859495, -1.2364988, 1.2586569
1: -0.1728082, 1.1187764, 0.0239049, 1.1038517, -1.2766598, 1.0948715
2: -0.4636380, 0.6756869, -0.3927249, 0.5403991, -1.0040370, 1.0684118
3: -0.3451241, 0.5063941, -0.2759416, 0.4027903, -0.7479144, 0.7823357
4: -0.5414346, 0.5670408, -0.4349633, 0.4743219, -1.0157566, 1.0020041
5: -0.5728725, 0.6077783, -0.4651313, 0.5031176, -1.0759901, 1.0729096
6: -0.5241617, 0.6914493, -0.4366596, 0.5750873, -1.0992490, 1.1281089
7: -0.4687819, 0.7302395, -0.3878939, 0.5932065, -1.0619884, 1.1181333
8: -0.5837868, 0.7122921, -0.4463747, 0.6050748, -1.1888616, 1.1586668
9: -0.5542934, 0.6778267, -0.4614004, 0.5453287, -1.0996221, 1.1392272

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 230

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3179355, upper bound: 1.2940208
time: 2.32 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3029367, upper bound: 1.2940208
time: 1.59 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.6862639, 0.7546907, -0.5979552, 0.6566006, -1.3428645, 1.3526459
1: -0.2278329, 1.1230559, -0.0768867, 1.1118935, -1.3397264, 1.1999426
2: -0.4846532, 0.7123000, -0.4302110, 0.6113129, -1.0959661, 1.1425110
3: -0.3760829, 0.5341202, -0.3122006, 0.4583267, -0.8344097, 0.8463209
4: -0.5732702, 0.5993659, -0.4915020, 0.5240541, -1.0973243, 1.0908679
5: -0.6019226, 0.6438608, -0.5204816, 0.5590751, -1.1609976, 1.1643424
6: -0.5504732, 0.7207976, -0.4817268, 0.6376112, -1.1880844, 1.2025244
7: -0.5018914, 0.7670676, -0.4310268, 0.6628337, -1.1647251, 1.1980945
8: -0.6314259, 0.7417296, -0.5195850, 0.6600894, -1.2915154, 1.2613146
9: -0.5823532, 0.7129908, -0.5112691, 0.6138146, -1.1961678, 1.2242599

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3252106, upper bound: 1.3309396
time: 1.66 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3252106, upper bound: 1.3309396
time: 1.80 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.7098854, 0.7757605, -0.7116623, 0.7772864, -1.4871718, 1.4874228
1: -0.2597983, 1.1257753, -0.2632843, 1.1259298, -1.3857281, 1.3890595
2: -0.4977460, 0.7387493, -0.4987734, 0.7406372, -1.2383832, 1.2375227
3: -0.3968386, 0.5519456, -0.3984092, 0.5532498, -0.9500885, 0.9503548
4: -0.5980328, 0.6200411, -0.5998507, 0.6216936, -1.2197263, 1.2198918
5: -0.6187797, 0.6709954, -0.6201006, 0.6730317, -1.2918115, 1.2910960
6: -0.5668839, 0.7378871, -0.5681255, 0.7391180, -1.3060019, 1.3060126
7: -0.5241609, 0.7923995, -0.5258098, 0.7957379, -1.3198988, 1.3182093
8: -0.6664008, 0.7588431, -0.6689795, 0.7601168, -1.4265177, 1.4278226
9: -0.6022526, 0.7335892, -0.6038283, 0.7350418, -1.3372943, 1.3374176

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3252106, upper bound: 1.3309396
time: 1.28 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3309396, upper bound: 1.3309396
time: 1.38 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.7098854, 0.7757605, -0.7098854, 0.7757605, -1.4856459, 1.4856459
1: -0.2597983, 1.1257753, -0.2597983, 1.1257753, -1.3855736, 1.3855736
2: -0.4977460, 0.7387493, -0.4977460, 0.7387493, -1.2364953, 1.2364953
3: -0.3968386, 0.5519456, -0.3968386, 0.5519456, -0.9487842, 0.9487842
4: -0.5980328, 0.6200411, -0.5980328, 0.6200411, -1.2180738, 1.2180738
5: -0.6187797, 0.6709954, -0.6187797, 0.6709954, -1.2897750, 1.2897750
6: -0.5668839, 0.7378871, -0.5668839, 0.7378871, -1.3047709, 1.3047709
7: -0.5241609, 0.7923995, -0.5241609, 0.7923995, -1.3165605, 1.3165605
8: -0.6664008, 0.7588431, -0.6664008, 0.7588431, -1.4252439, 1.4252439
9: -0.6022526, 0.7335892, -0.6022526, 0.7335892, -1.3358419, 1.3358419

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3252106, upper bound: 1.3309396
time: 1.24 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3309396, upper bound: 1.3309396
time: 1.29 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.28 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.28
Output dim: 1, lower bound: -1.3179355, upper bound: 1.2940208
NS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 4.28
Output dim: 1, lower bound: -1.3029367, upper bound: 1.2940208
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.28
Output dim: 1, lower bound: -1.3252106, upper bound: 1.3309396
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.28
Output dim: 1, lower bound: -1.3252106, upper bound: 1.3309396
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.28
Output dim: 1, lower bound: -1.3252106, upper bound: 1.3309396
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.28
Output dim: 1, lower bound: -1.3309396, upper bound: 1.3309396
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.28
Output dim: 1, lower bound: -1.3252106, upper bound: 1.3309396
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.28
Output dim: 1, lower bound: -1.3309396, upper bound: 1.3309396

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.5604268, 0.6121538, -0.5359248, 0.5823253, -1.1427522, 1.1480786
1: -0.0292169, 1.1069236, 0.0288265, 1.1033713, -1.1325881, 1.0780971
2: -0.4073571, 0.5659471, -0.3904791, 0.5367792, -0.9441363, 0.9564262
3: -0.2884191, 0.4229092, -0.2741256, 0.3994395, -0.6878586, 0.6970348
4: -0.4554585, 0.4937464, -0.4315558, 0.4713601, -0.9268187, 0.9253022
5: -0.4856943, 0.5237085, -0.4622352, 0.4997769, -0.9854712, 0.9859437
6: -0.4523171, 0.5988972, -0.4341749, 0.5713304, -1.0236475, 1.0330721
7: -0.4030039, 0.6335519, -0.3852647, 0.5899265, -0.9929304, 1.0188166
8: -0.4718915, 0.6259407, -0.4425697, 0.6017658, -1.0736573, 1.0685104
9: -0.4806108, 0.5715756, -0.4584090, 0.5418674, -1.0224782, 1.0299846

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3163970, upper bound: 1.2934728
time: 1.59 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3163970, upper bound: 1.2940208
time: 1.72 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.5395022, 0.5861006, -0.5979552, 0.6566006, -1.1961029, 1.1840558
1: 0.0219786, 1.1038339, -0.0768867, 1.1118935, -1.0899149, 1.1807206
2: -0.3929306, 0.5404775, -0.4302110, 0.6113129, -1.0042436, 0.9706885
3: -0.2756673, 0.4028298, -0.3122006, 0.4583267, -0.7339940, 0.7150304
4: -0.4350157, 0.4746124, -0.4915020, 0.5240541, -0.9590697, 0.9661144
5: -0.4653722, 0.5032192, -0.5204816, 0.5590751, -1.0244472, 1.0237010
6: -0.4367166, 0.5751978, -0.4817268, 0.6376112, -1.0743278, 1.0569246
7: -0.3878735, 0.5956192, -0.4310268, 0.6628337, -1.0507072, 1.0266460
8: -0.4464239, 0.6052052, -0.5195850, 0.6600894, -1.1065133, 1.1247902
9: -0.4616500, 0.5455585, -0.5112691, 0.6138146, -1.0754646, 1.0568275

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 230

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2934728, upper bound: 1.3163970
time: 1.29 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2934728, upper bound: 1.3037407
time: 1.32 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.5962329, 0.6546596, -0.5979552, 0.6566006, -1.2528335, 1.2526147
1: -0.0766336, 1.1116383, -0.0768867, 1.1118935, -1.1885271, 1.1885250
2: -0.4292608, 0.6092262, -0.4302110, 0.6113129, -1.0405738, 1.0394372
3: -0.3107070, 0.4567171, -0.3122006, 0.4583267, -0.7690338, 0.7689177
4: -0.4898412, 0.5228214, -0.4915020, 0.5240541, -1.0138953, 1.0143234
5: -0.5190048, 0.5574895, -0.5204816, 0.5590751, -1.0780799, 1.0779711
6: -0.4803589, 0.6359069, -0.4817268, 0.6376112, -1.1179701, 1.1176337
7: -0.4297242, 0.6633276, -0.4310268, 0.6628337, -1.0925579, 1.0943544
8: -0.5173828, 0.6586046, -0.5195850, 0.6600894, -1.1774721, 1.1781896
9: -0.5099916, 0.6119857, -0.5112691, 0.6138146, -1.1238062, 1.1232548

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 230

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2934728, upper bound: 1.3194749
time: 1.35 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2934728, upper bound: 1.3038354
time: 1.38 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.5372214, 0.5836759, -0.6505494, 0.7192606, -1.2564820, 1.2342253
1: 0.0269102, 1.1035477, -0.1728082, 1.1187764, -1.0918663, 1.2763559
2: -0.3913255, 0.5381303, -0.4636380, 0.6756869, -1.0670123, 1.0017682
3: -0.2748289, 0.4006792, -0.3451241, 0.5063941, -0.7812231, 0.7458032
4: -0.4328260, 0.4724636, -0.5414346, 0.5670408, -0.9998668, 1.0138983
5: -0.4633254, 0.5010158, -0.5728725, 0.6077783, -1.0711037, 1.0738883
6: -0.4350964, 0.5727384, -0.5241617, 0.6914493, -1.1265457, 1.0969001
7: -0.3862402, 0.5911703, -0.4687819, 0.7302395, -1.1164796, 1.0599523
8: -0.4439559, 0.6030060, -0.5837868, 0.7122921, -1.1562481, 1.1867929
9: -0.4595290, 0.5431805, -0.5542934, 0.6778267, -1.1373558, 1.0974739

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 230

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3179355
time: 1.49 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3029367
time: 1.19 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.5949709, 0.6530871, -0.6862639, 0.7546907, -1.3496616, 1.3393511
1: -0.0722129, 1.1114962, -0.2278329, 1.1230559, -1.1952689, 1.3393290
2: -0.4283874, 0.6076438, -0.4846532, 0.7123000, -1.1406875, 1.0922970
3: -0.3103572, 0.4555840, -0.3760829, 0.5341202, -0.8444774, 0.8316669
4: -0.4886894, 0.5216336, -0.5732702, 0.5993659, -1.0880554, 1.0949037
5: -0.5177320, 0.5562869, -0.6019226, 0.6438608, -1.1615927, 1.1582096
6: -0.4793907, 0.6345522, -0.5504732, 0.7207976, -1.2001883, 1.1850255
7: -0.4288820, 0.6595699, -0.5018914, 0.7670676, -1.1959496, 1.1614614
8: -0.5158799, 0.6573967, -0.6314259, 0.7417296, -1.2576096, 1.2888227
9: -0.5088333, 0.6103997, -0.5823532, 0.7129908, -1.2218242, 1.1927528

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3309396, upper bound: 1.3252106
time: 1.44 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3309396, upper bound: 1.3309396
time: 1.46 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.5372214, 0.5836759, -0.6488283, 0.7173219, -1.2545433, 1.2325041
1: 0.0269102, 1.1035477, -0.1687904, 1.1186004, -1.0916902, 1.2723382
2: -0.3913255, 0.5381303, -0.4624999, 0.6737460, -1.0650715, 1.0006301
3: -0.2748289, 0.4006792, -0.3440472, 0.5048873, -0.7797162, 0.7447264
4: -0.4328260, 0.4724636, -0.5398529, 0.5655717, -0.9983976, 1.0123165
5: -0.4633254, 0.5010158, -0.5711122, 0.6062278, -1.0695531, 1.0721279
6: -0.4350964, 0.5727384, -0.5227888, 0.6897980, -1.1248944, 1.0955272
7: -0.3862402, 0.5911703, -0.4676037, 0.7267393, -1.1129795, 1.0587740
8: -0.4439559, 0.6030060, -0.5817788, 0.7105892, -1.1545451, 1.1847848
9: -0.4595290, 0.5431805, -0.5528386, 0.6758583, -1.1353873, 1.0960190

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 230

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3179355
time: 1.38 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3038354
time: 1.33 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.5949709, 0.6530871, -0.6842257, 0.7528957, -1.3478665, 1.3373129
1: -0.0722129, 1.1114962, -0.2237591, 1.1228677, -1.1950805, 1.3352554
2: -0.4283874, 0.6076438, -0.4834857, 0.7100822, -1.1384696, 1.0911295
3: -0.3103572, 0.4555840, -0.3742837, 0.5326059, -0.8429631, 0.8298677
4: -0.4886894, 0.5216336, -0.5711671, 0.5974880, -1.0861773, 1.0928006
5: -0.5177320, 0.5562869, -0.6004142, 0.6415122, -1.1592442, 1.1567011
6: -0.4793907, 0.6345522, -0.5490625, 0.7193520, -1.1987426, 1.1836147
7: -0.4288820, 0.6595699, -0.4999868, 0.7633905, -1.1922724, 1.1595567
8: -0.5158799, 0.6573967, -0.6284469, 0.7402461, -1.2561260, 1.2858436
9: -0.5088333, 0.6103997, -0.5805563, 0.7112955, -1.2201288, 1.1909559

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3309396, upper bound: 1.3252106
time: 5.15 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3309396, upper bound: 1.3309396
time: 1.51 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 8.70 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 8.70
Output dim: 1, lower bound: -1.3163970, upper bound: 1.2934728
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 8.70
Output dim: 1, lower bound: -1.3163970, upper bound: 1.2940208
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 8.70
Output dim: 1, lower bound: -1.2934728, upper bound: 1.3163970
NS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 8.70
Output dim: 1, lower bound: -1.2934728, upper bound: 1.3037407
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 8.70
Output dim: 1, lower bound: -1.2934728, upper bound: 1.3194749
NS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 8.70
Output dim: 1, lower bound: -1.2934728, upper bound: 1.3038354
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 8.70
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3179355
NS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 8.70
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3029367
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 8.70
Output dim: 1, lower bound: -1.3309396, upper bound: 1.3252106
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 8.70
Output dim: 1, lower bound: -1.3309396, upper bound: 1.3309396
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 8.70
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3179355
NS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 8.70
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3038354
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 8.70
Output dim: 1, lower bound: -1.3309396, upper bound: 1.3252106
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 8.70
Output dim: 1, lower bound: -1.3309396, upper bound: 1.3309396

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.5604268, 0.6121538, -0.5359708, 0.5824189, -1.1428457, 1.1481246
1: -0.0292169, 1.1069236, 0.0269089, 1.1033458, -1.1325626, 1.0800147
2: -0.4073571, 0.5659471, -0.3906489, 0.5367982, -0.9441553, 0.9565960
3: -0.2884191, 0.4229092, -0.2738248, 0.3994227, -0.6878418, 0.6967340
4: -0.4554585, 0.4937464, -0.4315488, 0.4716068, -0.9270653, 0.9252952
5: -0.4856943, 0.5237085, -0.4624305, 0.4998228, -0.9855170, 0.9861389
6: -0.4523171, 0.5988972, -0.4341901, 0.5713786, -1.0236957, 1.0330873
7: -0.4030039, 0.6335519, -0.3851972, 0.5923215, -0.9953254, 1.0187490
8: -0.4718915, 0.6259407, -0.4425591, 0.6018406, -1.0737321, 1.0684998
9: -0.4806108, 0.5715756, -0.4586116, 0.5420443, -1.0226550, 1.0301872

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3124087, upper bound: 1.2934728
time: 1.48 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3124087, upper bound: 1.2934728
time: 1.50 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.5604268, 0.6121538, -0.5337406, 0.5801033, -1.1405301, 1.1458944
1: -0.0292169, 1.1069236, 0.0317519, 1.1030662, -1.1322831, 1.0751717
2: -0.4073571, 0.5659471, -0.3890726, 0.5345331, -0.9418902, 0.9550197
3: -0.2884191, 0.4229092, -0.2730072, 0.3973796, -0.6857986, 0.6959164
4: -0.4554585, 0.4937464, -0.4294085, 0.4695599, -0.9250184, 0.9231549
5: -0.4856943, 0.5237085, -0.4604208, 0.4977116, -0.9834059, 0.9841293
6: -0.4523171, 0.5988972, -0.4326363, 0.5689713, -1.0212884, 1.0315335
7: -0.4030039, 0.6335519, -0.3836042, 0.5880114, -0.9910153, 1.0171561
8: -0.4718915, 0.6259407, -0.4403208, 0.5996875, -1.0715790, 1.0662615
9: -0.4806108, 0.5715756, -0.4565496, 0.5397090, -1.0203198, 1.0281253

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3124087, upper bound: 1.2940208
time: 1.77 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3124087, upper bound: 1.2940208
time: 1.57 seconds

## BFS NS instance: NS_A1_B2_A1_B1

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

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 230

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2934728, upper bound: 1.3037407
time: 1.44 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2934728, upper bound: 1.3037407
time: 1.48 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.5925236, 0.6502910, -0.5111784, 0.5577636, -1.1502872, 1.1614695
1: -0.0708640, 1.1111516, 0.0544443, 1.1000473, -1.1709113, 1.0567073
2: -0.4269921, 0.6046688, -0.3749264, 0.5118566, -0.9388487, 0.9795952
3: -0.3083956, 0.4533082, -0.2613841, 0.3760073, -0.6844029, 0.7146922
4: -0.4863481, 0.5198125, -0.4072045, 0.4514893, -0.9378375, 0.9270171
5: -0.5155872, 0.5540248, -0.4421038, 0.4766369, -0.9922241, 0.9961286
6: -0.4774542, 0.6321067, -0.4166759, 0.5450221, -1.0224763, 1.0487826
7: -0.4270603, 0.6593188, -0.3661570, 0.5756422, -1.0027026, 1.0254759
8: -0.5127807, 0.6552563, -0.4170480, 0.5785314, -1.0913121, 1.0723042
9: -0.5069644, 0.6077380, -0.4380647, 0.5184230, -1.0253875, 1.0458027

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 230

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3029367, upper bound: 1.3038354
time: 1.23 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3029367, upper bound: 1.3038354
time: 1.22 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.5337406, 0.5801033, -0.5604268, 0.6121538, -1.1458944, 1.1405301
1: 0.0317519, 1.1030662, -0.0292169, 1.1069236, -1.0751717, 1.1322831
2: -0.3890726, 0.5345331, -0.4073571, 0.5659471, -0.9550197, 0.9418902
3: -0.2730072, 0.3973796, -0.2884191, 0.4229092, -0.6959164, 0.6857986
4: -0.4294085, 0.4695599, -0.4554585, 0.4937464, -0.9231549, 0.9250184
5: -0.4604208, 0.4977116, -0.4856943, 0.5237085, -0.9841293, 0.9834059
6: -0.4326363, 0.5689713, -0.4523171, 0.5988972, -1.0315335, 1.0212884
7: -0.3836042, 0.5880114, -0.4030039, 0.6335519, -1.0171561, 0.9910153
8: -0.4403208, 0.5996875, -0.4718915, 0.6259407, -1.0662615, 1.0715790
9: -0.4565496, 0.5397090, -0.4806108, 0.5715756, -1.0281253, 1.0203198

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 230

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3029367
time: 1.67 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3029367
time: 1.30 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.5949709, 0.6530871, -0.5395022, 0.5861006, -1.1810715, 1.1925893
1: -0.0722129, 1.1114962, 0.0219786, 1.1038339, -1.1760468, 1.0895176
2: -0.4283874, 0.6076438, -0.3929306, 0.5404775, -0.9688649, 1.0005744
3: -0.3103572, 0.4555840, -0.2756673, 0.4028298, -0.7131871, 0.7312514
4: -0.4886894, 0.5216336, -0.4350157, 0.4746124, -0.9633018, 0.9566492
5: -0.5177320, 0.5562869, -0.4653722, 0.5032192, -1.0209513, 1.0216591
6: -0.4793907, 0.6345522, -0.4367166, 0.5751978, -1.0545886, 1.0712688
7: -0.4288820, 0.6595699, -0.3878735, 0.5956192, -1.0245012, 1.0474434
8: -0.5158799, 0.6573967, -0.4464239, 0.6052052, -1.1210852, 1.1038206
9: -0.5088333, 0.6103997, -0.4616500, 0.5455585, -1.0543917, 1.0720497

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 230

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3163970, upper bound: 1.2934728
time: 1.78 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3037407, upper bound: 1.2934728
time: 1.55 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.5949709, 0.6530871, -0.5962329, 0.6546596, -1.2496305, 1.2493200
1: -0.0722129, 1.1114962, -0.0766336, 1.1116383, -1.1838512, 1.1881299
2: -0.4283874, 0.6076438, -0.4292608, 0.6092262, -1.0376136, 1.0369046
3: -0.3103572, 0.4555840, -0.3107070, 0.4567171, -0.7670743, 0.7662911
4: -0.4886894, 0.5216336, -0.4898412, 0.5228214, -1.0115108, 1.0114747
5: -0.5177320, 0.5562869, -0.5190048, 0.5574895, -1.0752215, 1.0752918
6: -0.4793907, 0.6345522, -0.4803589, 0.6359069, -1.1152976, 1.1149111
7: -0.4288820, 0.6595699, -0.4297242, 0.6633276, -1.0922096, 1.0892941
8: -0.5158799, 0.6573967, -0.5173828, 0.6586046, -1.1744845, 1.1747794
9: -0.5088333, 0.6103997, -0.5099916, 0.6119857, -1.1208191, 1.1203912

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 230

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3163970, upper bound: 1.3029367
time: 1.79 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3037407, upper bound: 1.3029367
time: 1.50 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.5337406, 0.5801033, -0.5593967, 0.6108779, -1.1446185, 1.1395001
1: 0.0317519, 1.1030662, -0.0253220, 1.1068200, -1.0750681, 1.1283882
2: -0.3890726, 0.5345331, -0.4065611, 0.5647615, -0.9538341, 0.9410943
3: -0.2730072, 0.3973796, -0.2881831, 0.4219745, -0.6949817, 0.6855626
4: -0.4294085, 0.4695599, -0.4544911, 0.4926745, -0.9220830, 0.9240510
5: -0.4604208, 0.4977116, -0.4845688, 0.5227120, -0.9831328, 0.9822804
6: -0.4326363, 0.5689713, -0.4515466, 0.5977985, -1.0304347, 1.0205179
7: -0.3836042, 0.5880114, -0.4023205, 0.6299915, -1.0135957, 0.9903319
8: -0.4403208, 0.5996875, -0.4706472, 0.6249368, -1.0652575, 1.0703347
9: -0.4565496, 0.5397090, -0.4799919, 0.5702910, -1.0268407, 1.0197009

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 230

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3038354
time: 1.33 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3038354
time: 1.48 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.5949709, 0.6530871, -0.5372214, 0.5836759, -1.1786468, 1.1903086
1: -0.0722129, 1.1114962, 0.0269102, 1.1035477, -1.1757605, 1.0845860
2: -0.4283874, 0.6076438, -0.3913255, 0.5381303, -0.9665177, 0.9989693
3: -0.3103572, 0.4555840, -0.2748289, 0.4006792, -0.7110364, 0.7304130
4: -0.4886894, 0.5216336, -0.4328260, 0.4724636, -0.9611530, 0.9544595
5: -0.5177320, 0.5562869, -0.4633254, 0.5010158, -1.0187478, 1.0196123
6: -0.4793907, 0.6345522, -0.4350964, 0.5727384, -1.0521290, 1.0696487
7: -0.4288820, 0.6595699, -0.3862402, 0.5911703, -1.0200523, 1.0458101
8: -0.5158799, 0.6573967, -0.4439559, 0.6030060, -1.1188860, 1.1013526
9: -0.5088333, 0.6103997, -0.4595290, 0.5431805, -1.0520138, 1.0699286

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 230

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3164307, upper bound: 1.2940208
time: 1.59 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3038354, upper bound: 1.2940208
time: 2.79 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.5949709, 0.6530871, -0.5949709, 0.6530871, -1.2480581, 1.2480581
1: -0.0722129, 1.1114962, -0.0722129, 1.1114962, -1.1837091, 1.1837091
2: -0.4283874, 0.6076438, -0.4283874, 0.6076438, -1.0360312, 1.0360312
3: -0.3103572, 0.4555840, -0.3103572, 0.4555840, -0.7659413, 0.7659413
4: -0.4886894, 0.5216336, -0.4886894, 0.5216336, -1.0103230, 1.0103230
5: -0.5177320, 0.5562869, -0.5177320, 0.5562869, -1.0740190, 1.0740190
6: -0.4793907, 0.6345522, -0.4793907, 0.6345522, -1.1139429, 1.1139429
7: -0.4288820, 0.6595699, -0.4288820, 0.6595699, -1.0884519, 1.0884519
8: -0.5158799, 0.6573967, -0.5158799, 0.6573967, -1.1732767, 1.1732767
9: -0.5088333, 0.6103997, -0.5088333, 0.6103997, -1.1192329, 1.1192329

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 230

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3164307, upper bound: 1.3038354
time: 1.84 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3038354, upper bound: 1.3038354
time: 2.28 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.87 seconds
NS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.87
Output dim: 1, lower bound: -1.3124087, upper bound: 1.2934728
NS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.87
Output dim: 1, lower bound: -1.3124087, upper bound: 1.2934728
NS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.87
Output dim: 1, lower bound: -1.3124087, upper bound: 1.2940208
NS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 5.87
Output dim: 1, lower bound: -1.3124087, upper bound: 1.2940208
NS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.87
Output dim: 1, lower bound: -1.2934728, upper bound: 1.3037407
NS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.87
Output dim: 1, lower bound: -1.2934728, upper bound: 1.3037407
NS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.87
Output dim: 1, lower bound: -1.3029367, upper bound: 1.3038354
NS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.87
Output dim: 1, lower bound: -1.3029367, upper bound: 1.3038354
NS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.87
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3029367
NS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.87
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3029367
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.87
Output dim: 1, lower bound: -1.3163970, upper bound: 1.2934728
NS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.87
Output dim: 1, lower bound: -1.3037407, upper bound: 1.2934728
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.87
Output dim: 1, lower bound: -1.3163970, upper bound: 1.3029367
NS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 5.87
Output dim: 1, lower bound: -1.3037407, upper bound: 1.3029367
NS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.87
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3038354
NS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.87
Output dim: 1, lower bound: -1.2940208, upper bound: 1.3038354
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.87
Output dim: 1, lower bound: -1.3164307, upper bound: 1.2940208
NS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.87
Output dim: 1, lower bound: -1.3038354, upper bound: 1.2940208
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.87
Output dim: 1, lower bound: -1.3164307, upper bound: 1.3038354
NS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 5.87
Output dim: 1, lower bound: -1.3038354, upper bound: 1.3038354

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.5080959, 0.5546181, -0.5359708, 0.5824189, -1.0905149, 1.0905888
1: 0.0586885, 1.0996149, 0.0269089, 1.1033458, -1.0446572, 1.0727060
2: -0.3729362, 0.5086761, -0.3906489, 0.5367982, -0.9097344, 0.8993250
3: -0.2597930, 0.3731009, -0.2738248, 0.3994227, -0.6592157, 0.6469257
4: -0.4041750, 0.4489416, -0.4315488, 0.4716068, -0.8757818, 0.8804904
5: -0.4395361, 0.4738726, -0.4624305, 0.4998228, -0.9393588, 0.9363031
6: -0.4145058, 0.5416850, -0.4341901, 0.5713786, -0.9858844, 0.9758751
7: -0.3638188, 0.5728714, -0.3851972, 0.5923215, -0.9561403, 0.9580685
8: -0.4138752, 0.5755936, -0.4425591, 0.6018406, -1.0157158, 1.0181527
9: -0.4355216, 0.5153548, -0.4586116, 0.5420443, -0.9775659, 0.9739664

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 230

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3037407, upper bound: 1.2934728
time: 1.86 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3037407, upper bound: 1.2934728
time: 1.64 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.5080959, 0.5546181, -0.5925236, 0.6502910, -1.1583869, 1.1471417
1: 0.0586885, 1.0996149, -0.0708640, 1.1111516, -1.0524631, 1.1704788
2: -0.3729362, 0.5086761, -0.4269921, 0.6046688, -0.9776050, 0.9356682
3: -0.2597930, 0.3731009, -0.3083956, 0.4533082, -0.7131011, 0.6814965
4: -0.4041750, 0.4489416, -0.4863481, 0.5198125, -0.9239875, 0.9352897
5: -0.4395361, 0.4738726, -0.5155872, 0.5540248, -0.9935609, 0.9894598
6: -0.4145058, 0.5416850, -0.4774542, 0.6321067, -1.0466125, 1.0191392
7: -0.3638188, 0.5728714, -0.4270603, 0.6593188, -1.0231376, 0.9999317
8: -0.4138752, 0.5755936, -0.5127807, 0.6552563, -1.0691314, 1.0883743
9: -0.4355216, 0.5153548, -0.5069644, 0.6077380, -1.0432596, 1.0223192

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 230

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3038354, upper bound: 1.3029367
time: 1.29 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3038354, upper bound: 1.3029367
time: 1.55 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.5080959, 0.5546181, -0.5337406, 0.5801033, -1.0881993, 1.0883586
1: 0.0586885, 1.0996149, 0.0317519, 1.1030662, -1.0443778, 1.0678630
2: -0.3729362, 0.5086761, -0.3890726, 0.5345331, -0.9074693, 0.8977486
3: -0.2597930, 0.3731009, -0.2730072, 0.3973796, -0.6571726, 0.6461080
4: -0.4041750, 0.4489416, -0.4294085, 0.4695599, -0.8737350, 0.8783500
5: -0.4395361, 0.4738726, -0.4604208, 0.4977116, -0.9372477, 0.9342934
6: -0.4145058, 0.5416850, -0.4326363, 0.5689713, -0.9834771, 0.9743212
7: -0.3638188, 0.5728714, -0.3836042, 0.5880114, -0.9518302, 0.9564756
8: -0.4138752, 0.5755936, -0.4403208, 0.5996875, -1.0135627, 1.0159144
9: -0.4355216, 0.5153548, -0.4565496, 0.5397090, -0.9752306, 0.9719044

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 230

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3038354, upper bound: 1.2940208
time: 1.38 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3038354, upper bound: 1.2940208
time: 1.54 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.5080959, 0.5546181, -0.5912800, 0.6487374, -1.1568333, 1.1458981
1: 0.0586885, 1.0996149, -0.0664384, 1.1110115, -1.0523231, 1.1660533
2: -0.3729362, 0.5086761, -0.4261284, 0.6031063, -0.9760424, 0.9348044
3: -0.2597930, 0.3731009, -0.3080536, 0.4521914, -0.7119844, 0.6811545
4: -0.4041750, 0.4489416, -0.4852138, 0.5186365, -0.9228115, 0.9341553
5: -0.4395361, 0.4738726, -0.5143283, 0.5528393, -0.9923754, 0.9882009
6: -0.4145058, 0.5416850, -0.4765008, 0.6307679, -1.0452738, 1.0181859
7: -0.3638188, 0.5728714, -0.4262320, 0.6555527, -1.0193715, 0.9991034
8: -0.4138752, 0.5755936, -0.5113012, 0.6540631, -1.0679383, 1.0868948
9: -0.4355216, 0.5153548, -0.5058185, 0.6061674, -1.0416890, 1.0211732

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 230

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3038354, upper bound: 1.3038354
time: 1.55 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.3038354, upper bound: 1.3038354
time: 1.43 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.77 seconds
NS_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.77
Output dim: 1, lower bound: -1.3037407, upper bound: 1.2934728
NS_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.77
Output dim: 1, lower bound: -1.3037407, upper bound: 1.2934728
NS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.77
Output dim: 1, lower bound: -1.3038354, upper bound: 1.3029367
NS_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.77
Output dim: 1, lower bound: -1.3038354, upper bound: 1.3029367
NS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.77
Output dim: 1, lower bound: -1.3038354, upper bound: 1.2940208
NS_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.77
Output dim: 1, lower bound: -1.3038354, upper bound: 1.2940208
NS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.77
Output dim: 1, lower bound: -1.3038354, upper bound: 1.3038354
NS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.77
Output dim: 1, lower bound: -1.3038354, upper bound: 1.3038354

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 4.76 + 151.03 = 155.79 seconds
