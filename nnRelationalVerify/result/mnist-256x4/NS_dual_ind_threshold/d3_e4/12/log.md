## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 12)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.6947652919999998


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548)
1: (-0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906)
2: (-0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625)
3: (-1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869)
4: (-1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024)
5: (-0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151)
6: (-0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975)
7: (-0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587)
8: (-1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157)
9: (-1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.89 + 2.82 = 3.71 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.09 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.02 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.17 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.17
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.17
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.9291230, 0.8731711, -1.1150692, 0.9756478, -1.9047709, 1.9882402
1: -0.6306670, 0.6588103, -0.7526883, 0.7897614, -1.4204284, 1.4114985
2: -0.6679621, 0.9325247, -0.8338743, 1.0586368, -1.7265989, 1.7663989
3: -0.7208436, 0.6743109, -0.9486150, 0.7423046, -1.4631481, 1.6229258
4: -0.8423574, 0.8144670, -1.0167658, 0.9576391, -1.7999965, 1.8312328
5: -0.4102694, 1.1948277, -0.6321793, 1.2493834, -1.6596528, 1.8270069
6: -0.6797165, 0.8051720, -0.8098221, 0.9446126, -1.6243291, 1.6149942
7: -0.7240871, 0.8383743, -0.8612908, 0.9862304, -1.7103176, 1.6996651
8: -0.9204464, 0.8898656, -1.1080486, 0.9887329, -1.9091793, 1.9979142
9: -0.7656624, 0.8241870, -0.9156295, 0.9489626, -1.7146249, 1.7398164

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 233

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.08 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.15 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -1.7070554, 1.3179296, -1.1500030, 0.9956961, -2.7027516, 2.4679327
1: -1.1559484, 1.2182931, -0.7763243, 0.8151693, -1.9711177, 1.9946175
2: -1.3862776, 1.4558029, -0.8659626, 1.0822275, -2.4685049, 2.3217654
3: -1.6779869, 0.9697326, -0.9920524, 0.7554038, -2.4333906, 1.9617851
4: -1.5864877, 1.4423213, -1.0505077, 0.9852051, -2.5716927, 2.4928288
5: -1.3531636, 1.4330713, -0.6744611, 1.2601525, -2.6133161, 2.1075325
6: -1.2378416, 1.4116676, -0.8351583, 0.9720090, -2.2098505, 2.2468259
7: -1.3253558, 1.4619617, -0.8881586, 1.0144982, -2.3398540, 2.3501203
8: -1.7115006, 1.3314877, -1.1437284, 1.0076457, -2.7191463, 2.4752159
9: -1.3925310, 1.3635345, -0.9442345, 0.9727091, -2.3652401, 2.3077691

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 233

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.21 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.19 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.22 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.7064028, 0.7555192, -0.5081837, 0.5572852, -1.2636881, 1.2637029
1: -0.5005909, 0.5192268, -0.3407900, 0.3593200, -0.8599110, 0.8600169
2: -0.4891996, 0.7815093, -0.3418081, 0.5176862, -1.0068858, 1.1233175
3: -0.4757721, 0.5964766, -0.3056164, 0.4780991, -0.9538713, 0.9020929
4: -0.6589320, 0.6446846, -0.4769357, 0.4582465, -1.1171784, 1.1216203
5: -0.1575242, 1.1509418, 0.1528127, 1.1149539, -1.2724781, 0.9981292
6: -0.5280892, 0.6685014, -0.3876513, 0.5124586, -1.0405477, 1.0561527
7: -0.5729398, 0.6663014, -0.4124650, 0.4399714, -1.0129112, 1.0787665
8: -0.7132203, 0.7811729, -0.5146397, 0.6047332, -1.3179536, 1.2958126
9: -0.6110113, 0.6785419, -0.4260985, 0.4881891, -1.0992004, 1.1046404

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.19 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.7562957, 0.7841209, -0.6996906, 0.7513720, -1.5076678, 1.4838114
1: -0.5290498, 0.5506238, -0.4966092, 0.5150505, -1.0441003, 1.0472330
2: -0.5274935, 0.8178868, -0.4843162, 0.7761741, -1.3036677, 1.3022031
3: -0.5285816, 0.6142946, -0.4690045, 0.5938416, -1.1224232, 1.0832992
4: -0.6984204, 0.6835862, -0.6535628, 0.6395178, -1.3379383, 1.3371489
5: -0.2143769, 1.1590568, -0.1499867, 1.1497498, -1.3641267, 1.3090435
6: -0.5632732, 0.6979800, -0.5237254, 0.6644152, -1.2276884, 1.2217054
7: -0.6065952, 0.7073761, -0.5681044, 0.6606312, -1.2672265, 1.2754805
8: -0.7583811, 0.8067011, -0.7073933, 0.7774400, -1.5358210, 1.5140945
9: -0.6454954, 0.7123297, -0.6061994, 0.6739493, -1.3194447, 1.3185291

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.31 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.17 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -1.4576763, 1.1736879, -0.5153341, 0.5671250, -2.0248013, 1.6890221
1: -0.9859343, 1.0378565, -0.3477625, 0.3663108, -1.3522451, 1.3856189
2: -1.1532286, 1.2886560, -0.3476287, 0.5304046, -1.6836332, 1.6362846
3: -1.3710208, 0.8737665, -0.3116727, 0.4834523, -1.8544731, 1.1854392
4: -1.3467046, 1.2372899, -0.4842540, 0.4667098, -1.8134143, 1.7215439
5: -1.0494375, 1.3555958, 0.1382566, 1.1163298, -2.1657672, 1.2173393
6: -1.0575300, 1.2149556, -0.3938037, 0.5191348, -1.5766649, 1.6087593
7: -1.1294930, 1.2617803, -0.4190799, 0.4506558, -1.5801487, 1.6808602
8: -1.4575286, 1.1861448, -0.5237523, 0.6118366, -2.0693650, 1.7098970
9: -1.1920063, 1.1884141, -0.4345473, 0.4967596, -1.6887659, 1.6229614

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.08 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.17 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -1.5204215, 1.2099807, -0.7272260, 0.7678093, -2.2882309, 1.9372067
1: -1.0287120, 1.0832561, -0.5125968, 0.5323060, -1.5610180, 1.5958530
2: -1.2118659, 1.3307122, -0.5049729, 0.7971470, -2.0090129, 1.8356850
3: -1.4482560, 0.8979126, -0.4978243, 0.6042462, -2.0525022, 1.3957369
4: -1.4070363, 1.2888777, -0.6754791, 0.6609776, -2.0680139, 1.9643568
5: -1.1258585, 1.3750894, -0.1809075, 1.1544119, -2.2802706, 1.5559969
6: -1.1028985, 1.2644504, -0.5426522, 0.6808333, -1.7837317, 1.8071027
7: -1.1787747, 1.3121483, -0.5871301, 0.6839427, -1.8627174, 1.8992784
8: -1.5214304, 1.2227147, -0.7318447, 0.7921542, -2.3135846, 1.9545593
9: -1.2424607, 1.2324764, -0.6259992, 0.6928905, -1.9353513, 1.8584756

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.20 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.22 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.43 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.4720453, 0.5090779, -0.4699855, 0.5070248, -0.9790701, 0.9790635
1: -0.3053531, 0.3234923, -0.3038034, 0.3214746, -0.6268277, 0.6272957
2: -0.3110486, 0.4600570, -0.3092161, 0.4580934, -0.7691420, 0.7692731
3: -0.2771786, 0.4503048, -0.2760035, 0.4486668, -0.7258453, 0.7263083
4: -0.4409319, 0.4139147, -0.4389631, 0.4113264, -0.8522583, 0.8528779
5: 0.2256563, 1.1079088, 0.2292556, 1.1074898, -0.8818335, 0.8786532
6: -0.3567996, 0.4775645, -0.3552210, 0.4755301, -0.8323298, 0.8327855
7: -0.3804163, 0.3868075, -0.3788981, 0.3841016, -0.7645179, 0.7657056
8: -0.4691589, 0.5671948, -0.4670615, 0.5649582, -1.0341171, 1.0342563
9: -0.3825231, 0.4457902, -0.3799716, 0.4438775, -0.8264006, 0.8257618

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.30 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.25 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.6307308, 0.7042207, -0.4985882, 0.5439158, -1.1746466, 1.2028089
1: -0.4549764, 0.4687619, -0.3313310, 0.3498225, -0.8047989, 0.8000929
2: -0.4327844, 0.7128482, -0.3339003, 0.5004390, -0.9332235, 1.0467484
3: -0.3977718, 0.5647414, -0.2973884, 0.4708985, -0.8686703, 0.8621297
4: -0.6016312, 0.5837743, -0.4671682, 0.4467481, -1.0483793, 1.0509424
5: -0.0722234, 1.1367196, 0.1725892, 1.1131312, -1.1853545, 0.9641304
6: -0.4842878, 0.6208997, -0.3793133, 0.5033878, -0.9876756, 1.0002131
7: -0.5145020, 0.6021583, -0.4034774, 0.4256161, -0.9401181, 1.0056357
8: -0.6497869, 0.7331098, -0.5022994, 0.5950827, -1.2448696, 1.2354093
9: -0.5544902, 0.6239954, -0.4147649, 0.4765453, -1.0310354, 1.0387603

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.21 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.10 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.4894910, 0.5311881, -0.6188331, 0.6920048, -1.1814958, 1.1500211
1: -0.3223299, 0.3407800, -0.4449416, 0.4590856, -0.7814155, 0.7857216
2: -0.3263716, 0.4840308, -0.4237789, 0.6963691, -1.0227406, 0.9078096
3: -0.2895550, 0.4640663, -0.3881962, 0.5572788, -0.8468338, 0.8522625
4: -0.4579252, 0.4358011, -0.5902216, 0.5726596, -1.0305848, 1.0260228
5: 0.1914172, 1.1114105, -0.0534118, 1.1345851, -0.9431679, 1.1648223
6: -0.3713820, 0.4947526, -0.4759214, 0.6112159, -0.9825979, 0.9706740
7: -0.3949209, 0.4119996, -0.5046276, 0.5885114, -0.9834324, 0.9166272
8: -0.4905641, 0.5858948, -0.6380610, 0.7218440, -1.2124081, 1.2239559
9: -0.4040212, 0.4654597, -0.5436339, 0.6120206, -1.0160418, 1.0090935

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.31 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.48 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.6643344, 0.7277970, -0.6763017, 0.7353708, -1.3997052, 1.4040987
1: -0.4763325, 0.4921241, -0.4831549, 0.4997497, -0.9760821, 0.9752790
2: -0.4574634, 0.7462483, -0.4663334, 0.7563549, -1.2138183, 1.2125818
3: -0.4325648, 0.5796445, -0.4448836, 0.5842918, -1.0168566, 1.0245281
4: -0.6263273, 0.6119225, -0.6351837, 0.6211826, -1.2475100, 1.2471062
5: -0.1112646, 1.1432925, -0.1240295, 1.1454337, -1.2566983, 1.2673221
6: -0.5031152, 0.6428936, -0.5098640, 0.6500578, -1.1531730, 1.1527576
7: -0.5403631, 0.6314607, -0.5495643, 0.6411294, -1.1814924, 1.1810250
8: -0.6782587, 0.7558283, -0.6879534, 0.7630590, -1.4413177, 1.4437817
9: -0.5797680, 0.6491210, -0.5885868, 0.6573757, -1.2371438, 1.2377079

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.39 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.30 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.7901074, 0.8023716, -0.4769820, 0.5141569, -1.3042643, 1.2793536
1: -0.5489501, 0.5710795, -0.3092610, 0.3283285, -0.8772786, 0.8803405
2: -0.5538315, 0.8412025, -0.3154408, 0.4649020, -1.0187335, 1.1566434
3: -0.5644913, 0.6253015, -0.2800862, 0.4542310, -1.0187223, 0.9053878
4: -0.7248625, 0.7101637, -0.4456512, 0.4201180, -1.1449805, 1.1558149
5: -0.2518473, 1.1649351, 0.2169161, 1.1089126, -1.3607600, 0.9480190
6: -0.5863768, 0.7175549, -0.3606287, 0.4824417, -1.0688186, 1.0781837
7: -0.6300384, 0.7332414, -0.3840551, 0.3933879, -1.0234263, 1.1172965
8: -0.7890651, 0.8232433, -0.4743352, 0.5725550, -1.3616202, 1.2975785
9: -0.6675976, 0.7342957, -0.3886380, 0.4505307, -1.1181283, 1.1229336

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.12 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.29 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.3227335, 1.0956383, -0.5056342, 0.5537741, -1.8765075, 1.6012726
1: -0.8939382, 0.9402199, -0.3383028, 0.3568260, -1.2507641, 1.2785226
2: -1.0271237, 1.1982112, -0.3397315, 0.5131482, -1.5402720, 1.5379428
3: -1.2049174, 0.8218383, -0.3034557, 0.4761904, -1.6811079, 1.1252940
4: -1.2169548, 1.1263443, -0.4743273, 0.4552270, -1.6721818, 1.6006715
5: -0.8850887, 1.3136723, 0.1580062, 1.1144639, -1.9995526, 1.1556661
6: -0.9599615, 1.1085126, -0.3854564, 0.5100765, -1.4700379, 1.4939691
7: -1.0235103, 1.1534593, -0.4101047, 0.4361621, -1.4596725, 1.5635641
8: -1.3201020, 1.1074979, -0.5113891, 0.6021991, -1.9223011, 1.6188871
9: -1.0835001, 1.0936546, -0.4230863, 0.4851313, -1.5686314, 1.5167410

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.15 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.25 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.8652657, 0.8406322, -0.6325976, 0.7059920, -1.5712577, 1.4732298
1: -0.5926979, 0.6165829, -0.4564160, 0.4702997, -1.0629976, 1.0729989
2: -0.6149014, 0.8903191, -0.4344882, 0.7152823, -1.3301837, 1.3248073
3: -0.6480597, 0.6514266, -0.3996641, 0.5658125, -1.2138722, 1.0510907
4: -0.7874547, 0.7669701, -0.6033102, 0.5856311, -1.3730859, 1.3702803
5: -0.3370465, 1.1788664, -0.0749583, 1.1371310, -1.4741775, 1.2538247
6: -0.6362377, 0.7637224, -0.4855594, 0.6224011, -1.2586389, 1.2492819
7: -0.6806440, 0.7892181, -0.5162011, 0.6041596, -1.2848036, 1.3054192
8: -0.8592476, 0.8586000, -0.6517085, 0.7347263, -1.5939739, 1.5103085
9: -0.7190076, 0.7820684, -0.5560699, 0.6257687, -1.3447763, 1.3381383

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.20 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.16 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.3866040, 1.1325805, -0.6996017, 0.7513126, -2.1379166, 1.8321822
1: -0.9374815, 0.9864324, -0.4965552, 0.5149907, -1.4524722, 1.4829876
2: -1.0868111, 1.2410204, -0.4842461, 0.7760994, -1.8629105, 1.7252666
3: -1.2835370, 0.8464167, -0.4689142, 0.5938038, -1.8773408, 1.3153309
4: -1.2783673, 1.1788566, -0.6534883, 0.6394488, -1.9178160, 1.8323450
5: -0.9628779, 1.3335155, -0.1498835, 1.1497326, -2.1126103, 1.4833990
6: -1.0061424, 1.1588937, -0.5236720, 0.6643571, -1.6704994, 1.6825657
7: -1.0736737, 1.2047298, -0.5680360, 0.6605529, -1.7342266, 1.7727659
8: -1.3851482, 1.1447226, -0.7073152, 0.7773868, -2.1625350, 1.8520378
9: -1.1348579, 1.1385058, -0.6061339, 0.6738840, -1.8087419, 1.7446396

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.21 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.21 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.25 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.4094637, 0.4466353, -0.3232799, 0.3516357, -0.7610994, 0.7699151
1: -0.2570738, 0.2672760, -0.1952764, 0.2071083, -0.4641821, 0.4625524
2: -0.2573270, 0.3988759, -0.1949374, 0.3040727, -0.5613998, 0.5938133
3: -0.2418091, 0.3992911, -0.1875787, 0.3155593, -0.5573685, 0.5868698
4: -0.3795898, 0.3371333, -0.2928843, 0.2608871, -0.6404769, 0.6300176
5: 0.3314744, 1.0948592, 0.4667180, 1.0775721, -0.7460977, 0.6281413
6: -0.3098156, 0.4141678, -0.2444725, 0.3251897, -0.6350052, 0.6586403
7: -0.3345687, 0.3067918, -0.2704576, 0.2324532, -0.5670218, 0.5772494
8: -0.4051532, 0.4980196, -0.3095611, 0.4050322, -0.8101854, 0.8075807
9: -0.3045104, 0.3915410, -0.2280151, 0.3086435, -0.6131539, 0.6195561

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8016169, upper bound: 1.7950476
time: 1.57 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7951458, upper bound: 1.7950476
time: 1.31 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.4395635, 0.4765918, -0.3805370, 0.4188166, -0.8583801, 0.8571288
1: -0.2808373, 0.2918003, -0.2363005, 0.2468990, -0.5277363, 0.5281008
2: -0.2822752, 0.4289898, -0.2362446, 0.3689908, -0.6512660, 0.6652344
3: -0.2586686, 0.4243880, -0.2255192, 0.3744156, -0.6330843, 0.6499072
4: -0.4097831, 0.3732170, -0.3496261, 0.3090927, -0.7188758, 0.7228432
5: 0.2822298, 1.1012824, 0.3752429, 1.0884850, -0.8062552, 0.7260395
6: -0.3318940, 0.4453725, -0.2893034, 0.3853347, -0.7172287, 0.7346759
7: -0.3563963, 0.3442885, -0.3153832, 0.2742773, -0.6306736, 0.6596717
8: -0.4359738, 0.5318444, -0.3762441, 0.4670632, -0.9030370, 0.9080885
9: -0.3421574, 0.4158814, -0.2733175, 0.3673857, -0.7095431, 0.6891990

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8018200, upper bound: 1.7950476
time: 1.68 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7951458, upper bound: 1.7950476
time: 1.49 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.5445094, 0.6070908, -0.3430289, 0.3750614, -0.9195708, 0.9501197
1: -0.3782772, 0.3945977, -0.2096823, 0.2208635, -0.5991406, 0.6042800
2: -0.3715734, 0.5817394, -0.2091264, 0.3261657, -0.6977391, 0.7908659
3: -0.3359661, 0.5053861, -0.2009296, 0.3359460, -0.6719121, 0.7063156
4: -0.5147041, 0.5006588, -0.3114652, 0.2778502, -0.7925543, 0.8121240
5: 0.0774121, 1.1218927, 0.4345305, 1.0803182, -1.0029061, 0.6873622
6: -0.4195905, 0.5463092, -0.2597140, 0.3463441, -0.7659346, 0.8060232
7: -0.4461887, 0.4936359, -0.2858505, 0.2464518, -0.6926405, 0.7794864
8: -0.5603071, 0.6435633, -0.3319127, 0.4267254, -0.9870325, 0.9754760
9: -0.4685940, 0.5331862, -0.2432316, 0.3293142, -0.7979083, 0.7764177

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7926400, upper bound: 1.7775361
time: 1.44 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7780016, upper bound: 1.7775361
time: 1.23 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.5820345, 0.6504152, -0.4052329, 0.4426949, -1.0247294, 1.0556482
1: -0.4120714, 0.4275006, -0.2537335, 0.2643900, -0.6764614, 0.6812341
2: -0.3978744, 0.6402259, -0.2538288, 0.3946430, -0.7925174, 0.8940547
3: -0.3622414, 0.5318530, -0.2394394, 0.3957677, -0.7580091, 0.7712924
4: -0.5529859, 0.5373777, -0.3753459, 0.3320718, -0.8850577, 0.9127235
5: 0.0106793, 1.1283318, 0.3382687, 1.0939566, -1.0832773, 0.7900630
6: -0.4481824, 0.5794259, -0.3069103, 0.4097816, -0.8579640, 0.8863361
7: -0.4756233, 0.5420165, -0.3317574, 0.3015214, -0.7771447, 0.8737739
8: -0.5998434, 0.6834599, -0.4010586, 0.4932649, -1.0931083, 1.0845184
9: -0.5068808, 0.5731062, -0.2994797, 0.3881198, -0.8950006, 0.8725859

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7926723, upper bound: 1.7774695
time: 1.44 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7780016, upper bound: 1.7774695
time: 1.11 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.4259882, 0.4623893, -0.4359990, 0.4728629, -0.8988511, 0.8983883
1: -0.2701198, 0.2793058, -0.2780234, 0.2885199, -0.5586396, 0.5573292
2: -0.2710012, 0.4154081, -0.2793152, 0.4254240, -0.6964252, 0.6947233
3: -0.2510649, 0.4130578, -0.2566722, 0.4214132, -0.6724781, 0.6697300
4: -0.3961655, 0.3569150, -0.4062077, 0.3689366, -0.7651021, 0.7631227
5: 0.3047656, 1.0983856, 0.2881469, 1.1005217, -0.7957560, 0.8102386
6: -0.3214302, 0.4312989, -0.3291466, 0.4416776, -0.7631078, 0.7604455
7: -0.3458953, 0.3273772, -0.3536394, 0.3398482, -0.6857435, 0.6810166
8: -0.4214661, 0.5165890, -0.4321650, 0.5278389, -0.9493051, 0.9487540
9: -0.3245105, 0.4049039, -0.3375241, 0.4129992, -0.7375098, 0.7424279

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8016771, upper bound: 1.7950476
time: 1.57 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7951458, upper bound: 1.7950476
time: 1.44 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.4567713, 0.4938516, -0.5033525, 0.5505818, -1.0073531, 0.9972041
1: -0.2938622, 0.3085303, -0.3360451, 0.3545579, -0.6484201, 0.6445754
2: -0.2974588, 0.4454960, -0.3378431, 0.5090330, -0.8064917, 0.7833391
3: -0.2684642, 0.4381573, -0.3014911, 0.4744764, -0.7429406, 0.7396485
4: -0.4263321, 0.3947212, -0.4720088, 0.4524812, -0.8788133, 0.8667300
5: 0.2523470, 1.1048030, 0.1627285, 1.1140324, -0.8616854, 0.9420745
6: -0.3450925, 0.4624762, -0.3834671, 0.5079105, -0.8530031, 0.8459433
7: -0.3691580, 0.3667454, -0.4079587, 0.4327469, -0.8019049, 0.7747041
8: -0.4536051, 0.5506099, -0.5084456, 0.5998945, -1.0534997, 1.0590556
9: -0.3636033, 0.4316057, -0.4203914, 0.4823508, -0.8459541, 0.8519971

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8018863, upper bound: 1.7950476
time: 1.51 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7951458, upper bound: 1.7950476
time: 1.33 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.5643247, 0.6300236, -0.4623511, 0.4994138, -1.0637385, 1.0923748
1: -0.3961554, 0.4120139, -0.2980600, 0.3139960, -0.7101515, 0.7100739
2: -0.3854850, 0.6126971, -0.3024232, 0.4508152, -0.8363001, 0.9151204
3: -0.3498739, 0.5193863, -0.2716476, 0.4425951, -0.7924690, 0.7910339
4: -0.5349531, 0.5200950, -0.4316654, 0.4017328, -0.9366859, 0.9517604
5: 0.0421045, 1.1253008, 0.2425967, 1.1059376, -1.0638331, 0.8827041
6: -0.4347078, 0.5638385, -0.3493692, 0.4679882, -0.9026959, 0.9132078
7: -0.4617690, 0.5192186, -0.3732708, 0.3740739, -0.8358430, 0.8924894
8: -0.5812346, 0.6646393, -0.4592870, 0.5566684, -1.1379030, 1.1239263
9: -0.4888602, 0.5543125, -0.3705148, 0.4367874, -0.9256477, 0.9248272

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7926848, upper bound: 1.7775361
time: 1.53 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7780016, upper bound: 1.7775361
time: 1.39 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.6020528, 0.6734658, -0.5326177, 0.5911700, -1.1932228, 1.2060835
1: -0.4300621, 0.4450061, -0.3659614, 0.3832078, -0.8132700, 0.8109674
2: -0.4118789, 0.6713426, -0.3620127, 0.5611474, -0.9730263, 1.0333552
3: -0.3762205, 0.5459449, -0.3263106, 0.4964978, -0.8727183, 0.8722555
4: -0.5733695, 0.5569133, -0.5022505, 0.4871659, -1.0605354, 1.0591638
5: -0.0248424, 1.1317575, 0.1017759, 1.1196547, -1.1444972, 1.0299816
6: -0.4634137, 0.5970453, -0.4092599, 0.5352714, -0.9986851, 1.0063052
7: -0.4912835, 0.5677858, -0.4353722, 0.4764792, -0.9677626, 1.0031580
8: -0.6208780, 0.7047341, -0.5457785, 0.6305003, -1.2513783, 1.2505126
9: -0.5272510, 0.5943489, -0.4549688, 0.5185573, -1.0458083, 1.0493177

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7927181, upper bound: 1.7774631
time: 1.33 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7780016, upper bound: 1.7774631
time: 1.18 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.6155389, 0.6884990, -0.3284750, 0.3577980, -0.9733369, 1.0169740
1: -0.4420598, 0.4564231, -0.1990659, 0.2107268, -0.6527867, 0.6554891
2: -0.4214164, 0.6916367, -0.1986699, 0.3098844, -0.7313008, 0.8903067
3: -0.3858016, 0.5551357, -0.1910908, 0.3209223, -0.7067239, 0.7462264
4: -0.5869505, 0.5696778, -0.2977721, 0.2653495, -0.8522999, 0.8674499
5: -0.0480093, 1.1340386, 0.4582508, 1.0782944, -1.1263037, 0.6757877
6: -0.4735233, 0.6085362, -0.2484820, 0.3307541, -0.8042774, 0.8570181
7: -0.5019674, 0.5845925, -0.2745068, 0.2361357, -0.7381031, 0.8590993
8: -0.6347641, 0.7186084, -0.3154407, 0.4107389, -1.0455030, 1.0340490
9: -0.5405359, 0.6085874, -0.2320179, 0.3140811, -0.8546170, 0.8406053

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8015889, upper bound: 1.7950476
time: 1.66 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7950476
time: 1.36 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.6817942, 0.7390859, -0.3864287, 0.4246745, -1.1064687, 1.1255146
1: -0.4862936, 0.5032791, -0.2398647, 0.2511901, -0.7374837, 0.7431438
2: -0.4704898, 0.7610059, -0.2400113, 0.3752842, -0.8457741, 1.0010172
3: -0.4505670, 0.5864624, -0.2288048, 0.3796536, -0.8302206, 0.8152672
4: -0.6394422, 0.6254325, -0.3559358, 0.3132669, -0.9527091, 0.9813683
5: -0.1300237, 1.1464164, 0.3672610, 1.0898274, -1.2198511, 0.7791555
6: -0.5130427, 0.6533911, -0.2936227, 0.3909570, -0.9039997, 0.9470139
7: -0.5539147, 0.6456238, -0.3193523, 0.2789029, -0.8328176, 0.9649761
8: -0.6924608, 0.7664170, -0.3823319, 0.4731517, -1.1656125, 1.1487489
9: -0.5927367, 0.6611639, -0.2785848, 0.3724725, -0.9652092, 0.9397488

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8018115, upper bound: 1.7950476
time: 1.57 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7950476
time: 1.26 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.0480480, 0.9369985, -0.3479572, 0.3809120, -1.4289601, 1.2849557
1: -0.7073420, 0.7410644, -0.2132772, 0.2242981, -0.9316401, 0.9543417
2: -0.7724633, 1.0131363, -0.2126746, 0.3316792, -1.1041424, 1.2258109
3: -0.8653100, 0.7171969, -0.2042612, 0.3410369, -1.2063469, 0.9214581
4: -0.9525789, 0.9045132, -0.3161123, 0.2820832, -1.2346622, 1.2206256
5: -0.5513446, 1.2286127, 0.4264980, 1.0810163, -1.6323609, 0.8021147
6: -0.7619789, 0.8922853, -0.2635257, 0.3516233, -1.1136022, 1.1558111
7: -0.8096220, 0.9317777, -0.2897005, 0.2499451, -1.0595672, 1.2214782
8: -1.0396798, 0.9524772, -0.3375037, 0.4321387, -1.4718184, 1.2899809
9: -0.8605561, 0.9035474, -0.2470323, 0.3344726, -1.1950288, 1.1505797

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8015889, upper bound: 1.7950476
time: 1.44 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7950476
time: 1.25 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.1678042, 1.0059202, -0.4111356, 0.4481920, -1.6159962, 1.4170557
1: -0.7884552, 0.8280671, -0.2583936, 0.2684165, -1.0568718, 1.0864607
2: -0.8824911, 1.0942245, -0.2587093, 0.4005485, -1.2830396, 1.3529338
3: -1.0141201, 0.7622012, -0.2427457, 0.4006831, -1.4148033, 1.0049468
4: -1.0678105, 0.9993209, -0.3812667, 0.3391332, -1.4069438, 1.3805876
5: -0.6962514, 1.2656285, 0.3287896, 1.0952160, -1.7914674, 0.9368389
6: -0.8480138, 0.9861324, -0.3109636, 0.4159012, -1.2639149, 1.2970960
7: -0.9019103, 1.0289493, -0.3356795, 0.3088743, -1.2107846, 1.3646288
8: -1.1620984, 1.0173841, -0.4067712, 0.4998981, -1.6619966, 1.4241552
9: -0.9587484, 0.9849373, -0.3064981, 0.3928932, -1.3516415, 1.2914354

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8018115, upper bound: 1.7950476
time: 1.44 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7950476
time: 1.28 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.6480587, 0.7173388, -0.4446996, 0.4818172, -1.1298759, 1.1620384
1: -0.4663815, 0.4813106, -0.2847807, 0.2967052, -0.7630866, 0.7660913
2: -0.4455707, 0.7314603, -0.2867184, 0.4339872, -0.8795579, 1.0181787
3: -0.4154661, 0.5728444, -0.2615768, 0.4285569, -0.8440230, 0.8344212
4: -0.6143914, 0.5985863, -0.4147937, 0.3795519, -0.9939433, 1.0133801
5: -0.0930684, 1.1402817, 0.2734414, 1.1023481, -1.1954165, 0.8668403
6: -0.4939497, 0.6327514, -0.3358400, 0.4505506, -0.9445002, 0.9685915
7: -0.5277466, 0.6177949, -0.3602600, 0.3508900, -0.8786365, 0.9780549
8: -0.6646273, 0.7454485, -0.4413117, 0.5375024, -1.2021296, 1.1867602
9: -0.5675139, 0.6375864, -0.3486501, 0.4203952, -0.9879091, 0.9862365

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8016420, upper bound: 1.7950476
time: 1.60 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7950476
time: 1.40 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.7385353, 0.7743014, -0.5131695, 0.5641457, -1.3026810, 1.2874708
1: -0.5190767, 0.5394835, -0.3456516, 0.3641945, -0.8832712, 0.8851352
2: -0.5137510, 0.8053597, -0.3458665, 0.5265536, -1.0403047, 1.1512263
3: -0.5099493, 0.6082935, -0.3098391, 0.4818318, -0.9917811, 0.9181325
4: -0.6844654, 0.6698216, -0.4820383, 0.4641477, -1.1486131, 1.1518599
5: -0.1941026, 1.1562042, 0.1426633, 1.1159130, -1.3100157, 1.0135410
6: -0.5508813, 0.6875555, -0.3919411, 0.5171132, -1.0679945, 1.0794966
7: -0.5946941, 0.6933734, -0.4170772, 0.4474212, -1.0421153, 1.1104505
8: -0.7422763, 0.7979666, -0.5209934, 0.6096861, -1.3519624, 1.3189600
9: -0.6338108, 0.7005634, -0.4319895, 0.4941649, -1.1279757, 1.1325529

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8018581, upper bound: 1.7950476
time: 1.62 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7950476
time: 1.37 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.1090240, 0.9721782, -0.4710350, 0.5080711, -1.6170951, 1.4432132
1: -0.7486025, 0.7853636, -0.3045930, 0.3225029, -1.0711055, 1.0899566
2: -0.8283271, 1.0545540, -0.3101497, 0.4590942, -1.2874212, 1.3647037
3: -0.9410976, 0.7400413, -0.2766023, 0.4495014, -1.3905990, 1.0166435
4: -1.0109348, 0.9528694, -0.4399664, 0.4126452, -1.4235799, 1.3928359
5: -0.6248754, 1.2475197, 0.2274216, 1.1077032, -1.7325786, 1.0200981
6: -0.8054366, 0.9398800, -0.3560255, 0.4765669, -1.2820034, 1.2959055
7: -0.8566403, 0.9813437, -0.3796717, 0.3854803, -1.2421206, 1.3610154
8: -1.1018828, 0.9854598, -0.4681305, 0.5660976, -1.6679804, 1.4535903
9: -0.9106783, 0.9448593, -0.3812717, 0.4448520, -1.3555303, 1.3261310

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7925520, upper bound: 1.7775361
time: 1.44 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7774631, upper bound: 1.7775361
time: 1.10 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.2315834, 1.0429163, -0.5421761, 0.6042982, -1.8358816, 1.5850924
1: -0.8317963, 0.8742685, -0.3761173, 0.3924769, -1.2242732, 1.2503858
2: -0.9419419, 1.1371169, -0.3698966, 0.5779693, -1.5199113, 1.5070136
3: -1.0927190, 0.7867616, -0.3342725, 0.5036970, -1.5964160, 1.1210341
4: -1.1293111, 1.0514042, -0.5122619, 0.4982924, -1.6276035, 1.5636661
5: -0.7740743, 1.2853544, 0.0816854, 1.1214775, -1.8955518, 1.2036690
6: -0.8940562, 1.0366126, -0.4177784, 0.5441750, -1.4382312, 1.4543910
7: -0.9519204, 1.0802913, -0.4442913, 0.4905647, -1.4424851, 1.5245826
8: -1.2272726, 1.0543733, -0.5577587, 0.6410672, -1.8683398, 1.6121320
9: -1.0102066, 1.0296462, -0.4661261, 0.5306202, -1.5408268, 1.4957722

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7925801, upper bound: 1.7774631
time: 1.33 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7774631, upper bound: 1.7774631
time: 1.15 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.33 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 5, lower bound: -1.8016169, upper bound: 1.7950476
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 5, lower bound: -1.7951458, upper bound: 1.7950476
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 5, lower bound: -1.8018200, upper bound: 1.7950476
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 5, lower bound: -1.7951458, upper bound: 1.7950476
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 5, lower bound: -1.7926400, upper bound: 1.7775361
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 5, lower bound: -1.7780016, upper bound: 1.7775361
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 5, lower bound: -1.7926723, upper bound: 1.7774695
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 5, lower bound: -1.7780016, upper bound: 1.7774695
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 5, lower bound: -1.8016771, upper bound: 1.7950476
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 5, lower bound: -1.7951458, upper bound: 1.7950476
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 5, lower bound: -1.8018863, upper bound: 1.7950476
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 5, lower bound: -1.7951458, upper bound: 1.7950476
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 5, lower bound: -1.7926848, upper bound: 1.7775361
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 5, lower bound: -1.7780016, upper bound: 1.7775361
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 5, lower bound: -1.7927181, upper bound: 1.7774631
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 5, lower bound: -1.7780016, upper bound: 1.7774631
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 5, lower bound: -1.8015889, upper bound: 1.7950476
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7950476
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 5, lower bound: -1.8018115, upper bound: 1.7950476
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7950476
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 5, lower bound: -1.8015889, upper bound: 1.7950476
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7950476
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 5, lower bound: -1.8018115, upper bound: 1.7950476
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7950476
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 5, lower bound: -1.8016420, upper bound: 1.7950476
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7950476
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 5, lower bound: -1.8018581, upper bound: 1.7950476
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 5, lower bound: -1.7950476, upper bound: 1.7950476
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 5, lower bound: -1.7925520, upper bound: 1.7775361
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 5, lower bound: -1.7774631, upper bound: 1.7775361
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 5, lower bound: -1.7925801, upper bound: 1.7774631
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 5, lower bound: -1.7774631, upper bound: 1.7774631

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.3508882, 0.3844449, -0.3091519, 0.3348772, -0.6857654, 0.6935968
1: -0.2154152, 0.2263631, -0.1849708, 0.1972680, -0.4126832, 0.4113339
2: -0.2148707, 0.3349580, -0.1847868, 0.2882676, -0.5031383, 0.5197448
3: -0.2062427, 0.3441043, -0.1780278, 0.3009754, -0.5072181, 0.5221321
4: -0.3189982, 0.2846007, -0.2795920, 0.2487521, -0.5677503, 0.5641928
5: 0.4217211, 1.0815845, 0.4897444, 1.0756074, -0.6538863, 0.5918400
6: -0.2658909, 0.3547627, -0.2335691, 0.3100560, -0.5759469, 0.5883319
7: -0.2920942, 0.2520226, -0.2594460, 0.2224389, -0.5145332, 0.5114685
8: -0.3409821, 0.4353583, -0.2935713, 0.3895136, -0.7304957, 0.7289295
9: -0.2493337, 0.3375403, -0.2171295, 0.2938561, -0.5431899, 0.5546699

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7715819, upper bound: 1.7756925
time: 1.50 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7695456, upper bound: 1.7512964
time: 1.40 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.5909439, 0.6156465, -0.2849509, 0.3057152, -0.8966590, 0.9005974
1: -0.4003515, 0.3910757, -0.1670374, 0.1801465, -0.5804980, 0.5581131
2: -0.4073772, 0.5804428, -0.1671231, 0.2613201, -0.6686972, 0.7475659
3: -0.3434597, 0.5504204, -0.1615888, 0.2755966, -0.6190563, 0.7120092
4: -0.5616342, 0.5542256, -0.2565427, 0.2276353, -0.7892695, 0.8107684
5: 0.0400368, 1.1335855, 0.5298139, 1.0723257, -1.0322889, 0.6037715
6: -0.4344372, 0.6023102, -0.2149570, 0.2837209, -0.7181581, 0.8172672
7: -0.4551561, 0.5328707, -0.2402837, 0.2055734, -0.6607295, 0.7731544
8: -0.5807894, 0.7019594, -0.2658487, 0.3625085, -0.9432980, 0.9678081
9: -0.5202910, 0.5382967, -0.1981869, 0.2682703, -0.7885613, 0.7364836

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513144, upper bound: 1.7738855
time: 1.33 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513144, upper bound: 1.7512964
time: 1.18 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.3743168, 0.4122189, -0.3655240, 0.4018694, -0.7761861, 0.7777430
1: -0.2322864, 0.2425980, -0.2259893, 0.2365480, -0.4688344, 0.4685872
2: -0.2321351, 0.3620004, -0.2257016, 0.3517200, -0.5838552, 0.5877020
3: -0.2218193, 0.3685541, -0.2160148, 0.3593894, -0.5812086, 0.5845689
4: -0.3429738, 0.3043916, -0.3338326, 0.2970166, -0.6399903, 0.6382242
5: 0.3841674, 1.0869969, 0.3981616, 1.0848279, -0.7006605, 0.6888354
6: -0.2845179, 0.3794444, -0.2775711, 0.3702468, -0.6547647, 0.6570155
7: -0.3109131, 0.2696380, -0.3039002, 0.2628679, -0.5737810, 0.5735382
8: -0.3693883, 0.4606678, -0.3586327, 0.4512362, -0.8206245, 0.8193004
9: -0.2680846, 0.3616570, -0.2609916, 0.3526700, -0.6207546, 0.6226486

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7721633, upper bound: 1.7756925
time: 1.48 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7702454, upper bound: 1.7512964
time: 1.52 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.6209666, 0.6663730, -0.3416038, 0.3733713, -0.9943379, 1.0079768
1: -0.4240541, 0.4587610, -0.2086428, 0.2198711, -0.6439251, 0.6674038
2: -0.4329260, 0.6104799, -0.2081027, 0.3245714, -0.7574974, 0.8185826
3: -0.3602760, 0.5757906, -0.1999663, 0.3344750, -0.6947510, 0.7757570
4: -0.5917502, 0.5910549, -0.3101245, 0.2766262, -0.8683764, 0.9011793
5: -0.0189092, 1.1399924, 0.4368528, 1.0801201, -1.0990293, 0.7031395
6: -0.4717181, 0.6334351, -0.2586142, 0.3448176, -0.8165357, 0.8920493
7: -0.4967178, 0.5702713, -0.2847398, 0.2454418, -0.7421596, 0.8550111
8: -0.6298366, 0.7356977, -0.3302999, 0.4251600, -1.0549966, 1.0659976
9: -0.5779667, 0.5625747, -0.2421336, 0.3278227, -0.9057895, 0.8047084

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513144, upper bound: 1.7738855
time: 1.37 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513144, upper bound: 1.7512964
time: 1.09 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.4959342, 0.5402031, -0.3430289, 0.3750614, -0.8709955, 0.8832320
1: -0.3287051, 0.3471846, -0.2096823, 0.2208635, -0.5495686, 0.5568669
2: -0.3317039, 0.4956530, -0.2091264, 0.3261657, -0.6578696, 0.7047794
3: -0.2951032, 0.4689053, -0.2009296, 0.3359460, -0.6310492, 0.6698349
4: -0.4644717, 0.4435544, -0.3114652, 0.2778502, -0.7423219, 0.7550196
5: 0.1780818, 1.1126292, 0.4345305, 1.0803182, -0.9022364, 0.6780987
6: -0.3769997, 0.5008683, -0.2597140, 0.3463441, -0.7233438, 0.7605823
7: -0.4009813, 0.4216433, -0.2858505, 0.2464518, -0.6474332, 0.7074938
8: -0.4988760, 0.5924021, -0.3319127, 0.4267254, -0.9256014, 0.9243148
9: -0.4116304, 0.4733117, -0.2432316, 0.3293142, -0.7409446, 0.7165433

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7780016, upper bound: 1.7775361
time: 1.18 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7780016, upper bound: 1.7775361
time: 1.17 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.9423161, 1.0831909, -0.3116730, 0.3378682, -1.2801843, 1.3948640
1: -0.7465785, 0.7561719, -0.1868099, 0.1990242, -0.9456027, 0.9429817
2: -0.6574868, 1.2244534, -0.1865983, 0.2910883, -0.9485751, 1.4110518
3: -0.6247057, 0.7933679, -0.1797323, 0.3035779, -0.9282836, 0.9731001
4: -0.9310387, 0.9041635, -0.2819642, 0.2509177, -1.1819564, 1.1861277
5: -0.6511618, 1.1926546, 0.4856351, 1.0759580, -1.7271198, 0.7070195
6: -0.7285162, 0.9102314, -0.2355150, 0.3127566, -1.0412728, 1.1457464
7: -0.7696493, 1.0172546, -0.2614110, 0.2242260, -0.9938753, 1.2786655
8: -0.9947755, 1.0691026, -0.2964246, 0.3922831, -1.3870586, 1.3655272
9: -0.8893326, 0.9706485, -0.2190722, 0.2964951, -1.1858277, 1.1897206

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7703235
time: 1.17 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7512964
time: 1.11 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.5271960, 0.5835984, -0.4052329, 0.4426949, -0.9698910, 0.9888314
1: -0.3601044, 0.3779074, -0.2537335, 0.2643900, -0.6244943, 0.6316409
2: -0.3574657, 0.5515032, -0.2538288, 0.3946430, -0.7521087, 0.8053319
3: -0.3217188, 0.4923938, -0.2394394, 0.3957677, -0.7174866, 0.7318332
4: -0.4965713, 0.4807489, -0.3753459, 0.3320718, -0.8286431, 0.8560947
5: 0.1133628, 1.1186116, 0.3382687, 1.0939566, -0.9805938, 0.7803428
6: -0.4043469, 0.5302096, -0.3069103, 0.4097816, -0.8141285, 0.8371199
7: -0.4302280, 0.4683783, -0.3317574, 0.3015214, -0.7317493, 0.8001357
8: -0.5388690, 0.6244809, -0.4010586, 0.4932649, -1.0321338, 1.0255394
9: -0.4485627, 0.5116004, -0.2994797, 0.3881198, -0.8366824, 0.8110801

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7780016, upper bound: 1.7774695
time: 1.18 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7780016, upper bound: 1.7774695
time: 1.12 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.9848642, 1.1142602, -0.3631538, 0.3990789, -1.3839431, 1.4774139
1: -0.7741024, 0.7797675, -0.2242915, 0.2349173, -1.0090196, 1.0040591
2: -0.6796869, 1.2663953, -0.2239674, 0.3489486, -1.0286355, 1.4903628
3: -0.6435483, 0.8154272, -0.2144499, 0.3569188, -1.0004671, 1.0298772
4: -0.9631641, 0.9304953, -0.3313684, 0.2950285, -1.2581925, 1.2618637
5: -0.7041301, 1.1972724, 0.4019340, 1.0842433, -1.7883734, 0.7953384
6: -0.7546844, 0.9339802, -0.2756986, 0.3677671, -1.1224515, 1.2096788
7: -0.7907576, 1.0605798, -0.3020098, 0.2610430, -1.0518006, 1.3625896
8: -1.0231280, 1.1115565, -0.3557330, 0.4486938, -1.4718218, 1.4672896
9: -0.9167892, 1.0005858, -0.2590795, 0.3502475, -1.2670367, 1.2596653

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7703235
time: 1.21 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7512964
time: 1.10 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.3619591, 0.3976728, -0.4146304, 0.4514467, -0.8134058, 0.8123032
1: -0.2234360, 0.2340952, -0.2611528, 0.2708007, -0.4942367, 0.4952480
2: -0.2230932, 0.3475518, -0.2615986, 0.4040450, -0.6271382, 0.6091504
3: -0.2136613, 0.3556734, -0.2447031, 0.4035937, -0.6172550, 0.6003765
4: -0.3301264, 0.2940264, -0.3847725, 0.3433137, -0.6734401, 0.6787990
5: 0.4038356, 1.0839486, 0.3231773, 1.0959620, -0.6921265, 0.7607713
6: -0.2747546, 0.3665177, -0.3133635, 0.4195240, -0.6942786, 0.6798812
7: -0.3010570, 0.2601231, -0.3380018, 0.3132284, -0.6142855, 0.5981249
8: -0.3542716, 0.4474124, -0.4101534, 0.5038256, -0.8580972, 0.8575658
9: -0.2581157, 0.3490264, -0.3106536, 0.3957191, -0.6538348, 0.6596800

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7716859, upper bound: 1.7756925
time: 1.52 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7698134, upper bound: 1.7512964
time: 1.42 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.6045916, 0.6492416, -0.3826769, 0.4209443, -1.0255358, 1.0319185
1: -0.4111264, 0.4436896, -0.2375952, 0.2484577, -0.6595841, 0.6812848
2: -0.4193271, 0.5940970, -0.2376127, 0.3712766, -0.7906038, 0.8317097
3: -0.3511042, 0.5621237, -0.2267127, 0.3763182, -0.7274224, 0.7888364
4: -0.5753244, 0.5713913, -0.3519179, 0.3106090, -0.8859334, 0.9233092
5: 0.0082737, 1.1364982, 0.3723435, 1.0889726, -1.0806990, 0.7641547
6: -0.4590962, 0.6164593, -0.2908723, 0.3873767, -0.8464729, 0.9073316
7: -0.4840512, 0.5498729, -0.3168249, 0.2759575, -0.7600087, 0.8666978
8: -0.6123370, 0.7172965, -0.3784554, 0.4692746, -1.0816116, 1.0957519
9: -0.5566810, 0.5493329, -0.2752306, 0.3692335, -0.9259145, 0.8245634

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513144, upper bound: 1.7738855
time: 1.27 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513144, upper bound: 1.7512964
time: 1.09 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.3871132, 0.4253555, -0.4807519, 0.5188665, -0.9059797, 0.9061074
1: -0.2402790, 0.2516888, -0.3132715, 0.3320214, -0.5723004, 0.5649602
2: -0.2404491, 0.3760153, -0.3187950, 0.4693333, -0.7097824, 0.6948103
3: -0.2291864, 0.3802629, -0.2827863, 0.4572293, -0.6864157, 0.6630491
4: -0.3566692, 0.3137521, -0.4492571, 0.4248555, -0.7815247, 0.7630092
5: 0.3663333, 1.0899835, 0.2096424, 1.1096791, -0.7433459, 0.8803411
6: -0.2941248, 0.3916099, -0.3637905, 0.4861657, -0.7802905, 0.7554004
7: -0.3198137, 0.2794404, -0.3868340, 0.3989212, -0.7187349, 0.6662743
8: -0.3830392, 0.4738593, -0.4790740, 0.5766485, -0.9596877, 0.9529332
9: -0.2791969, 0.3730634, -0.3933076, 0.4549744, -0.7341713, 0.7663710

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7722822, upper bound: 1.7756925
time: 1.44 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7704867, upper bound: 1.7512964
time: 1.35 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.6437041, 0.6802050, -0.4474347, 0.4845437, -1.1282477, 1.1276398
1: -0.4344925, 0.4916461, -0.2868384, 0.2993843, -0.7338768, 0.7784845
2: -0.4637784, 0.6237079, -0.2891518, 0.4365947, -0.9003731, 0.9128597
3: -0.3751175, 0.5868256, -0.2631372, 0.4307321, -0.8058496, 0.8499628
4: -0.6050132, 0.6296225, -0.4174078, 0.3829888, -0.9880021, 1.0470303
5: -0.0743088, 1.1428138, 0.2686621, 1.1029046, -1.1772134, 0.8741517
6: -0.4883725, 0.6471420, -0.3379363, 0.4532527, -0.9416252, 0.9850783
7: -0.5069451, 0.6122730, -0.3622761, 0.3544823, -0.8614274, 0.9745492
8: -0.6439663, 0.7535844, -0.4440972, 0.5404719, -1.1844382, 1.1976817
9: -0.5951543, 0.6052061, -0.3520381, 0.4229348, -1.0180891, 0.9572443

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513144, upper bound: 1.7738855
time: 1.30 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513144, upper bound: 1.7512964
time: 1.14 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.5097553, 0.5594477, -0.4623511, 0.4994138, -1.0091691, 1.0217988
1: -0.3423225, 0.3608564, -0.2980600, 0.3139960, -0.6563185, 0.6589164
2: -0.3430872, 0.5204811, -0.3024232, 0.4508152, -0.7939023, 0.8229043
3: -0.3069474, 0.4792758, -0.2716476, 0.4425951, -0.7495425, 0.7509234
4: -0.4785438, 0.4601063, -0.4316654, 0.4017328, -0.8802766, 0.8917717
5: 0.1496136, 1.1152563, 0.2425967, 1.1059376, -0.9563240, 0.8726596
6: -0.3890035, 0.5139256, -0.3493692, 0.4679882, -0.8569916, 0.8632948
7: -0.4139187, 0.4423194, -0.3732708, 0.3740739, -0.7879927, 0.8155902
8: -0.5166426, 0.6062943, -0.4592870, 0.5566684, -1.0733110, 1.0655813
9: -0.4279551, 0.4900727, -0.3705148, 0.4367874, -0.8647425, 0.8605875

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7780016, upper bound: 1.7775361
time: 1.18 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7780016, upper bound: 1.7775361
time: 1.16 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.9752443, 1.1031835, -0.4178612, 0.4544555, -1.4296998, 1.5210447
1: -0.7654569, 0.7713550, -0.2637036, 0.2730045, -1.0384614, 1.0350586
2: -0.6729569, 1.2514423, -0.2642701, 0.4072773, -1.0802343, 1.5157125
3: -0.6368302, 0.8086551, -0.2465127, 0.4062840, -1.0431142, 1.0551679
4: -0.9533690, 0.9211073, -0.3880132, 0.3471788, -1.3005478, 1.3091205
5: -0.6870593, 1.1956261, 0.3179888, 1.0966511, -1.7837104, 0.8776373
6: -0.7473649, 0.9255130, -0.3155820, 0.4228735, -1.1702384, 1.2410951
7: -0.7832319, 1.0481956, -0.3401485, 0.3172532, -1.1004851, 1.3883442
8: -1.0130198, 1.1013329, -0.4132802, 0.5074562, -1.5204760, 1.5146132
9: -0.9070002, 0.9903771, -0.3144950, 0.3983316, -1.3053319, 1.3048722

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7702354
time: 1.11 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7512964
time: 1.22 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.5409900, 0.6028623, -0.5326177, 0.5911700, -1.1321599, 1.1354799
1: -0.3750064, 0.3913931, -0.3659614, 0.3832078, -0.7582142, 0.7573545
2: -0.3690341, 0.5760397, -0.3620127, 0.5611474, -0.9301814, 0.9380524
3: -0.3334014, 0.5028355, -0.3263106, 0.4964978, -0.8298993, 0.8291461
4: -0.5110207, 0.4970751, -0.5022505, 0.4871659, -0.9981866, 0.9993256
5: 0.0838835, 1.1212653, 0.1017759, 1.1196547, -1.0357711, 1.0194894
6: -0.4168465, 0.5430885, -0.4092599, 0.5352714, -0.9521179, 0.9523484
7: -0.4433157, 0.4889889, -0.4353722, 0.4764792, -0.9197948, 0.9243611
8: -0.5564481, 0.6397955, -0.5457785, 0.6305003, -1.1869483, 1.1855741
9: -0.4648615, 0.5293006, -0.4549688, 0.5185573, -0.9834188, 0.9842694

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7780016, upper bound: 1.7774631
time: 1.24 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7780016, upper bound: 1.7774631
time: 1.21 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.0022366, 1.1342638, -0.4799513, 0.5178663, -1.5201030, 1.6142150
1: -0.7897150, 0.7949597, -0.3124199, 0.3312368, -1.1209519, 1.1073796
2: -0.6918403, 1.2933996, -0.3180828, 0.4683922, -1.1602325, 1.6114824
3: -0.6556796, 0.8276564, -0.2822129, 0.4565924, -1.1122720, 1.1098694
4: -0.9808538, 0.9474490, -0.4484912, 0.4238497, -1.4047035, 1.3959402
5: -0.7349563, 1.2002455, 0.2111871, 1.1095164, -1.8444726, 0.9890584
6: -0.7679026, 0.9492706, -0.3631189, 0.4853749, -1.2532775, 1.3123895
7: -0.8043480, 1.0829434, -0.3862438, 0.3977464, -1.2020943, 1.4691873
8: -1.0413826, 1.1300182, -0.4780673, 0.5757791, -1.6171618, 1.6080855
9: -0.9344670, 1.0190208, -0.3923160, 0.4540306, -1.3884976, 1.4113368

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7702354
time: 1.24 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7512964
time: 1.14 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.5218445, 0.5761253, -0.3143490, 0.3410422, -0.8628867, 0.8904743
1: -0.3543231, 0.3726758, -0.1887618, 0.2008879, -0.5552109, 0.5614376
2: -0.3529776, 0.5419850, -0.1885208, 0.2940819, -0.6470596, 0.7305058
3: -0.3171866, 0.4883430, -0.1815413, 0.3063402, -0.6235268, 0.6698843
4: -0.4909656, 0.4744149, -0.2844818, 0.2532163, -0.7441819, 0.7588967
5: 0.1247990, 1.1175821, 0.4812738, 1.0763301, -0.9515311, 0.6363083
6: -0.3994979, 0.5252129, -0.2375803, 0.3156230, -0.7151209, 0.7627932
7: -0.4251507, 0.4603827, -0.2634968, 0.2261228, -0.6512734, 0.7238795
8: -0.5320493, 0.6185396, -0.2994533, 0.3952223, -0.9272716, 0.9179929
9: -0.4422396, 0.5047337, -0.2211339, 0.2992961, -0.7415357, 0.7258676

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7713540, upper bound: 1.7757114
time: 1.51 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7694138, upper bound: 1.7512964
time: 1.45 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.8945588, 0.9854352, -0.2900001, 0.3118876, -1.2064464, 1.2754353
1: -0.6861589, 0.6819316, -0.1708330, 0.1837700, -0.8699289, 0.8527646
2: -0.6215180, 1.0924877, -0.1708618, 0.2669179, -0.8884358, 1.2633494
3: -0.5886378, 0.7366695, -0.1650338, 0.2809680, -0.8696058, 0.9017032
4: -0.8640226, 0.8222350, -0.2614057, 0.2321049, -1.0961275, 1.0836407
5: -0.5056041, 1.1803207, 0.5213332, 1.0729945, -1.5785986, 0.6589875
6: -0.6766376, 0.8355091, -0.2188275, 0.2892951, -0.9659327, 1.0543367
7: -0.7272871, 0.9165579, -0.2443395, 0.2090362, -0.9363232, 1.1608974
8: -0.9140369, 0.9926599, -0.2716968, 0.3682243, -1.2822613, 1.2643567
9: -0.8029442, 0.8994082, -0.2021962, 0.2736578, -1.0766020, 1.1016045

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7740229
time: 1.29 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7512964
time: 1.14 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.5596417, 0.6246314, -0.3705169, 0.4077465, -0.9673882, 0.9951483
1: -0.3919467, 0.4079184, -0.2295652, 0.2399835, -0.6319301, 0.6374836
2: -0.3822089, 0.6054178, -0.2293549, 0.3575578, -0.7397667, 0.8347727
3: -0.3466039, 0.5160896, -0.2193109, 0.3645936, -0.7111975, 0.7354004
4: -0.5301847, 0.5155250, -0.3390234, 0.3012046, -0.8313894, 0.8545485
5: 0.0504143, 1.1244993, 0.3902150, 1.0860596, -1.0356452, 0.7342843
6: -0.4311443, 0.5597171, -0.2815160, 0.3754695, -0.8066138, 0.8412330
7: -0.4581053, 0.5131902, -0.3078825, 0.2667123, -0.7248176, 0.8210727
8: -0.5763135, 0.6596626, -0.3647402, 0.4565918, -1.0329053, 1.0244029
9: -0.4840948, 0.5493428, -0.2650194, 0.3577735, -0.8418683, 0.8143622

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7719791, upper bound: 1.7757114
time: 1.50 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7700814, upper bound: 1.7512964
time: 1.44 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.1937680, 1.0878980, -0.3465663, 0.3792574, -1.5730255, 1.4344642
1: -0.7788723, 0.8330752, -0.2122626, 0.2233273, -1.0021995, 1.0453378
2: -0.8579524, 1.1948962, -0.2116680, 0.3301230, -1.1880754, 1.4065642
3: -0.9803172, 0.7898743, -0.2033210, 0.3395975, -1.3199147, 0.9931952
4: -1.0378428, 1.0215905, -0.3147933, 0.2808885, -1.3187313, 1.3363838
5: -0.6911156, 1.2380170, 0.4287650, 1.0808100, -1.7719256, 0.8092520
6: -0.8093370, 0.9653975, -0.2624438, 0.3501333, -1.1594703, 1.2278414
7: -0.9594234, 1.0662134, -0.2886075, 0.2489593, -1.2083827, 1.3548210
8: -1.1126075, 1.0805607, -0.3359163, 0.4306109, -1.5432184, 1.4164770
9: -0.9796579, 1.0142932, -0.2459570, 0.3330168, -1.3126746, 1.2602502

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7740229
time: 1.42 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7512964
time: 1.19 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.7500989, 0.7807307, -0.3337642, 0.3640719, -1.1141708, 1.1144949
1: -0.5254763, 0.5468257, -0.2029241, 0.2144106, -0.7398869, 0.7497498
2: -0.5226768, 0.8135972, -0.2024701, 0.3158013, -0.8384781, 1.0160673
3: -0.5220075, 0.6122502, -0.1946663, 0.3263823, -0.8483899, 0.8069165
4: -0.6936269, 0.6786708, -0.3027484, 0.2698923, -0.9635193, 0.9814193
5: -0.2074164, 1.1580364, 0.4496303, 1.0790298, -1.2864462, 0.7084060
6: -0.5589824, 0.6943934, -0.2525638, 0.3364200, -0.8954023, 0.9469572
7: -0.6023631, 0.7025900, -0.2786293, 0.2398848, -0.8422480, 0.9812193
8: -0.7528441, 0.8036382, -0.3214272, 0.4165486, -1.1693926, 1.1250653
9: -0.6414750, 0.7082493, -0.2360932, 0.3196171, -0.9610921, 0.9443425

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 159

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7713540, upper bound: 1.7771158
time: 1.36 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7694138, upper bound: 1.7512964
time: 1.47 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.9028841, 1.4312276, -0.3092656, 0.3350127, -2.2378969, 1.7404932
1: -1.2873967, 1.3514810, -0.1850538, 0.1973473, -1.4847440, 1.5365348
2: -1.5503548, 1.5956163, -0.1848687, 0.2883952, -1.8387500, 1.7804850
3: -1.9238698, 1.0336332, -0.1781048, 0.3010929, -2.2249627, 1.2117381
4: -1.7712983, 1.5834508, -0.2796992, 0.2488500, -2.0201483, 1.8631500
5: -1.5825011, 1.4945148, 0.4895588, 1.0756234, -2.6581244, 1.0049560
6: -1.3719459, 1.5446436, -0.2336571, 0.3101781, -1.6821239, 1.7783008
7: -1.4653777, 1.6288482, -0.2595347, 0.2225197, -1.6878974, 1.8883829
8: -1.9140856, 1.4084264, -0.2937001, 0.3896388, -2.3037245, 1.7021265
9: -1.5627673, 1.4752576, -0.2172174, 0.2939752, -1.8567424, 1.6924751

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7512964, upper bound: 1.7764832
time: 1.28 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7512964, upper bound: 1.7512964
time: 1.22 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.8596202, 0.8378190, -0.3931517, 0.4312390, -1.2908592, 1.2309707
1: -0.5893329, 0.6132132, -0.2444847, 0.2559983, -0.8453313, 0.8576978
2: -0.6102188, 0.8867373, -0.2445835, 0.3823360, -0.9925548, 1.1313207
3: -0.6418445, 0.6494372, -0.2326061, 0.3855237, -1.0273683, 0.8820434
4: -0.7827438, 0.7627804, -0.3630062, 0.3190593, -1.1018031, 1.1257867
5: -0.3306012, 1.1776814, 0.3573518, 1.0913317, -1.4219329, 0.8203297
6: -0.6324370, 0.7602199, -0.2984631, 0.3975284, -1.0299654, 1.0586830
7: -0.6768394, 0.7849897, -0.3238001, 0.2861972, -0.9630367, 1.1087898
8: -0.8538550, 0.8560370, -0.3891534, 0.4801596, -1.3340147, 1.2451904
9: -0.7150136, 0.7785510, -0.2855172, 0.3781722, -1.0931859, 1.0640682

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7719791, upper bound: 1.7771158
time: 1.56 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7700814, upper bound: 1.7512964
time: 1.38 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2.0336108, 1.5118757, -0.3662709, 0.4027479, -2.4363587, 1.8781466
1: -1.3785752, 1.4560708, -0.2265240, 0.2370619, -1.6156371, 1.6825948
2: -1.6871980, 1.6822412, -0.2262480, 0.3525931, -2.0397911, 1.9084892
3: -2.0867009, 1.0945420, -0.2165077, 0.3601677, -2.4468687, 1.3110497
4: -1.9088242, 1.7013348, -0.3346090, 0.2976431, -2.2064674, 2.0359437
5: -1.7559741, 1.5338664, 0.3969731, 1.0850122, -2.8409863, 1.1368933
6: -1.4728479, 1.6781583, -0.2781612, 0.3710278, -1.8438756, 1.9563195
7: -1.5829787, 1.7313566, -0.3044958, 0.2634431, -1.8464218, 2.0358524
8: -2.0555625, 1.5152626, -0.3595460, 0.4520371, -2.5075996, 1.8748085
9: -1.6641866, 1.5851002, -0.2615939, 0.3534334, -2.0176201, 1.8466940

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7512964, upper bound: 1.7764832
time: 1.24 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7512964, upper bound: 1.7512964
time: 1.23 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.5367367, 0.5969215, -0.4230097, 0.4592735, -0.9960101, 1.0199312
1: -0.3704110, 0.3872343, -0.2677683, 0.2765648, -0.6469758, 0.6550026
2: -0.3654668, 0.5684731, -0.2685277, 0.4124283, -0.7778951, 0.8370008
3: -0.3297989, 0.4996155, -0.2493965, 0.4105718, -0.7403707, 0.7490120
4: -0.5065650, 0.4920408, -0.3931778, 0.3533386, -0.8599036, 0.8852186
5: 0.0929738, 1.1204469, 0.3097101, 1.0977501, -1.0047762, 0.8107368
6: -0.4129919, 0.5391169, -0.3191345, 0.4282110, -0.8412029, 0.8582514
7: -0.4392794, 0.4826332, -0.3435916, 0.3236669, -0.7629463, 0.8262248
8: -0.5510271, 0.6350729, -0.4182833, 0.5132419, -1.0642691, 1.0533562
9: -0.4598354, 0.5238424, -0.3206391, 0.4024951, -0.8623306, 0.8444815

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7714389, upper bound: 1.7757114
time: 1.68 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7696675, upper bound: 1.7512964
time: 1.42 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1.0740392, 1.0149161, -0.3889841, 0.4272155, -1.5012547, 1.4039001
1: -0.7499120, 0.7717568, -0.2414109, 0.2530513, -1.0029633, 1.0131677
2: -0.7614541, 1.1527984, -0.2416452, 0.3780139, -1.1394680, 1.3944436
3: -0.8606691, 0.7665939, -0.2302297, 0.3819261, -1.2425952, 0.9968237
4: -0.9355633, 0.9716128, -0.3586729, 0.3150776, -1.2506410, 1.3302855
5: -0.5913951, 1.2235649, 0.3637986, 1.0904096, -1.6818048, 0.8597663
6: -0.7211946, 0.9145343, -0.2954965, 0.3933952, -1.1145898, 1.2100308
7: -0.8665524, 0.9910374, -0.3210739, 0.2809092, -1.1474617, 1.3121114
8: -1.0319613, 1.0362898, -0.3849725, 0.4757928, -1.5077541, 1.4212623
9: -0.9094788, 0.9528443, -0.2808694, 0.3746789, -1.2841576, 1.2337136

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7740229
time: 1.29 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7512964
time: 1.28 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.5777271, 0.6454557, -0.4887983, 0.5302190, -1.1079462, 1.1342540
1: -0.4082001, 0.4237340, -0.3216444, 0.3400916, -0.7482917, 0.7453784
2: -0.3948611, 0.6335298, -0.3257982, 0.4827814, -0.8776425, 0.9593280
3: -0.3592333, 0.5288207, -0.2889587, 0.4635460, -0.8227794, 0.8177794
4: -0.5486000, 0.5331741, -0.4572213, 0.4349675, -0.9835675, 0.9903954
5: 0.0183229, 1.1275945, 0.1928509, 1.1112797, -1.0929569, 0.9347436
6: -0.4449049, 0.5756347, -0.3707782, 0.4940950, -0.9390000, 0.9464129
7: -0.4722534, 0.5364712, -0.3942692, 0.4109626, -0.8832160, 0.9307405
8: -0.5953174, 0.6788825, -0.4896705, 0.5851949, -1.1805123, 1.1685529
9: -0.5024979, 0.5685350, -0.4032031, 0.4646156, -0.9671134, 0.9717381

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7720406, upper bound: 1.7757114
time: 1.48 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7702719, upper bound: 1.7512964
time: 1.49 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.3992282, 1.1363866, -0.4545387, 0.4916260, -1.8908541, 1.5909253
1: -0.8818427, 0.9647343, -0.2921827, 0.3063434, -1.1881860, 1.2569170
2: -1.0091600, 1.2793995, -0.2954726, 0.4433674, -1.4525274, 1.5748721
3: -1.1861728, 0.8378342, -0.2671905, 0.4363818, -1.6225547, 1.1050247
4: -1.2120326, 1.1662004, -0.4241983, 0.3919157, -1.6039484, 1.5903987
5: -0.9585391, 1.2560433, 0.2562482, 1.1043490, -2.0628881, 0.9997951
6: -1.0109549, 1.0762472, -0.3433813, 0.4602707, -1.4712256, 1.4196285
7: -1.0298045, 1.2240785, -0.3675123, 0.3638133, -1.3936177, 1.5915909
8: -1.3506910, 1.1217434, -0.4513313, 0.5481859, -1.8988769, 1.5730748
9: -1.0727117, 1.1331306, -0.3608379, 0.4295322, -1.5022439, 1.4939685

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7740229
time: 1.29 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7512964
time: 1.18 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.9165044, 0.8667363, -0.4710350, 0.5080711, -1.4245756, 1.3377713
1: -0.6231701, 0.6501805, -0.3045930, 0.3225029, -0.9456731, 0.9547735
2: -0.6574708, 0.9239666, -0.3101497, 0.4590942, -1.1165650, 1.2341163
3: -0.7061045, 0.6697813, -0.2766023, 0.4495014, -1.1556059, 0.9463835
4: -0.8313135, 0.8050882, -0.4399664, 0.4126452, -1.2439587, 1.2450546
5: -0.3957884, 1.1912724, 0.2274216, 1.1077032, -1.5034916, 0.9638507
6: -0.6710997, 0.7966892, -0.3560255, 0.4765669, -1.1476667, 1.1527147
7: -0.7154753, 0.8285961, -0.3796717, 0.3854803, -1.1009557, 1.2082677
8: -0.9083625, 0.8833411, -0.4681305, 0.5660976, -1.4744601, 1.3514715
9: -0.7560694, 0.8158063, -0.3812717, 0.4448520, -1.2009214, 1.1970780

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7774631, upper bound: 1.7775361
time: 1.10 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7774631, upper bound: 1.7775361
time: 1.17 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2.4195089, 1.7244682, -0.4260397, 0.4624432, -2.8819520, 2.1505079
1: -1.6343555, 1.7387809, -0.2701603, 0.2793533, -1.9137088, 2.0089412
2: -2.0308948, 1.9397511, -0.2710439, 0.4154595, -2.4463542, 2.2107952
3: -2.5707858, 1.2306234, -0.2510935, 0.4131009, -2.9838867, 1.4817169
4: -2.2750106, 1.9869006, -0.3962170, 0.3569769, -2.6319876, 2.3831177
5: -2.2083044, 1.6516125, 0.3046802, 1.0983965, -3.3067009, 1.3469323
6: -1.7561767, 1.9659433, -0.3214699, 0.4313520, -2.1875286, 2.2874131
7: -1.8648443, 2.0408621, -0.3459352, 0.3274413, -2.1922855, 2.3867972
8: -2.4384792, 1.6951520, -0.4215212, 0.5166469, -2.9551260, 2.1166732
9: -1.9840641, 1.8343470, -0.3245776, 0.4049454, -2.3890095, 2.1589246

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7512964, upper bound: 1.7702719
time: 1.24 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7512964, upper bound: 1.7512964
time: 1.23 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1.0371654, 0.9307392, -0.5421761, 0.6042982, -1.6414636, 1.4729154
1: -0.6999609, 0.7333237, -0.3761173, 0.3924769, -1.0924377, 1.1094410
2: -0.7625598, 1.0057282, -0.3698966, 0.5779693, -1.3405292, 1.3756249
3: -0.8518399, 0.7131721, -0.3342725, 0.5036970, -1.3555369, 1.0474446
4: -0.9422083, 0.8958738, -0.5122619, 0.4982924, -1.4405007, 1.4081357
5: -0.5382136, 1.2253131, 0.0816854, 1.1214775, -1.6596911, 1.1436276
6: -0.7542307, 0.8839969, -0.4177784, 0.5441750, -1.2984056, 1.3017752
7: -0.8013637, 0.9229009, -0.4442913, 0.4905647, -1.2919284, 1.3671921
8: -1.0285857, 0.9466708, -0.5577587, 0.6410672, -1.6696529, 1.5044295
9: -0.8516646, 0.8962675, -0.4661261, 0.5306202, -1.3822849, 1.3623936

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7774631, upper bound: 1.7774631
time: 1.13 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7774631, upper bound: 1.7774631
time: 1.18 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2.5298834, 1.7938523, -0.4884289, 0.5297024, -3.0595858, 2.2822812
1: -1.7169087, 1.8136460, -0.3212790, 0.3397247, -2.0566335, 2.1349249
2: -2.1552198, 2.0073040, -0.3254927, 0.4821153, -2.6373351, 2.3327966
3: -2.6908209, 1.2863729, -0.2886407, 0.4632686, -3.1540895, 1.5750136
4: -2.3776546, 2.1188223, -0.4568463, 0.4345232, -2.8121777, 2.5756686
5: -2.3553047, 1.6887033, 0.1936152, 1.1112096, -3.4665143, 1.4950881
6: -1.8327789, 2.0607190, -0.3704563, 0.4937442, -2.3265231, 2.4311752
7: -1.9716034, 2.1224618, -0.3939221, 0.4104100, -2.3820133, 2.5163839
8: -2.5494804, 1.8110478, -0.4891942, 0.5848222, -3.1343026, 2.3002419
9: -2.0541620, 1.9413445, -0.4027669, 0.4641659, -2.5183280, 2.3441114

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 96

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7512964, upper bound: 1.7702719
time: 1.31 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7512964, upper bound: 1.7512964
time: 1.40 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.74 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7715819, upper bound: 1.7756925
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7695456, upper bound: 1.7512964
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7513144, upper bound: 1.7738855
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7513144, upper bound: 1.7512964
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7721633, upper bound: 1.7756925
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7702454, upper bound: 1.7512964
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7513144, upper bound: 1.7738855
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7513144, upper bound: 1.7512964
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7780016, upper bound: 1.7775361
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7780016, upper bound: 1.7775361
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7703235
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7512964
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7780016, upper bound: 1.7774695
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7780016, upper bound: 1.7774695
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7703235
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7512964
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7716859, upper bound: 1.7756925
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7698134, upper bound: 1.7512964
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7513144, upper bound: 1.7738855
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7513144, upper bound: 1.7512964
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7722822, upper bound: 1.7756925
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7704867, upper bound: 1.7512964
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7513144, upper bound: 1.7738855
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7513144, upper bound: 1.7512964
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7780016, upper bound: 1.7775361
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7780016, upper bound: 1.7775361
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7702354
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7512964
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7780016, upper bound: 1.7774631
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7780016, upper bound: 1.7774631
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7702354
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7513879, upper bound: 1.7512964
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7713540, upper bound: 1.7757114
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7694138, upper bound: 1.7512964
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7740229
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7512964
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7719791, upper bound: 1.7757114
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7700814, upper bound: 1.7512964
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7740229
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7512964
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7713540, upper bound: 1.7771158
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7694138, upper bound: 1.7512964
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7512964, upper bound: 1.7764832
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7512964, upper bound: 1.7512964
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7719791, upper bound: 1.7771158
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7700814, upper bound: 1.7512964
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7512964, upper bound: 1.7764832
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7512964, upper bound: 1.7512964
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7714389, upper bound: 1.7757114
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7696675, upper bound: 1.7512964
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7740229
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7512964
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7720406, upper bound: 1.7757114
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7702719, upper bound: 1.7512964
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7740229
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7512964
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7774631, upper bound: 1.7775361
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7774631, upper bound: 1.7775361
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7512964, upper bound: 1.7702719
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7512964, upper bound: 1.7512964
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7774631, upper bound: 1.7774631
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7774631, upper bound: 1.7774631
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7512964, upper bound: 1.7702719
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 5, lower bound: -1.7512964, upper bound: 1.7512964

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.3508882, 0.3844449, -0.2841525, 0.3047394, -0.6556276, 0.6685975
1: -0.2154152, 0.2263631, -0.1664372, 0.1795737, -0.3949890, 0.3928003
2: -0.2148707, 0.3349580, -0.1665321, 0.2604349, -0.4753056, 0.5014901
3: -0.2062427, 0.3441043, -0.1610441, 0.2747474, -0.4809901, 0.5051484
4: -0.3189982, 0.2846007, -0.2557738, 0.2269287, -0.5459269, 0.5403746
5: 0.4217211, 1.0815845, 0.5311549, 1.0722203, -0.6504992, 0.5504296
6: -0.2658909, 0.3547627, -0.2143452, 0.2828397, -0.5487306, 0.5691079
7: -0.2920942, 0.2520226, -0.2396425, 0.2050259, -0.4971201, 0.4916651
8: -0.3409821, 0.4353583, -0.2649239, 0.3616051, -0.7025872, 0.7002822
9: -0.2493337, 0.3375403, -0.1975531, 0.2674185, -0.5167522, 0.5350935

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7695456, upper bound: 1.7512964
time: 1.67 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7695456, upper bound: 1.7512964
time: 1.60 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.3115253, 0.3376928, -0.5674018, 0.6412069, -0.9527322, 0.9050945
1: -0.1867021, 0.1989212, -0.3733499, 0.3771398, -0.5638419, 0.5722711
2: -0.1864921, 0.2909231, -0.3703318, 0.5771703, -0.7636623, 0.6612549
3: -0.1796324, 0.3034254, -0.3526114, 0.5675619, -0.7471943, 0.6560369
4: -0.2818252, 0.2507909, -0.5225657, 0.4705705, -0.7523956, 0.7733566
5: 0.4858759, 1.0759376, 0.0688395, 1.1115172, -0.6256413, 1.0070981
6: -0.2354010, 0.3125981, -0.4328741, 0.5866866, -0.8220876, 0.7454722
7: -0.2612958, 0.2241213, -0.4607314, 0.4054933, -0.6667891, 0.6848528
8: -0.2962573, 0.3921209, -0.5858553, 0.6731839, -0.9694412, 0.9779762
9: -0.2189584, 0.2963404, -0.4161094, 0.5641587, -0.7831171, 0.7124498

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6566031, upper bound: 1.6155543
time: 1.63 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7424588, upper bound: 1.7139215
time: 1.74 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7368981, upper bound: 1.7139215
time: 1.73 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.5909439, 0.6156465, -0.2602990, 0.2755789, -0.8665228, 0.8759454
1: -0.4003515, 0.3910757, -0.1485049, 0.1624553, -0.5628067, 0.5395806
2: -0.4073772, 0.5804428, -0.1488694, 0.2339886, -0.6413658, 0.7293122
3: -0.3434597, 0.5504204, -0.1447690, 0.2493703, -0.5928300, 0.6951895
4: -0.5616342, 0.5542256, -0.2327991, 0.2058132, -0.7674474, 0.7870247
5: 0.0400368, 1.1335855, 0.5712220, 1.0690620, -1.0290252, 0.5623634
6: -0.4344372, 0.6023102, -0.1960593, 0.2565064, -0.6909436, 0.7983695
7: -0.4551561, 0.5328707, -0.2204816, 0.1886661, -0.6438222, 0.7533523
8: -0.5807894, 0.7019594, -0.2372956, 0.3346015, -0.9153910, 0.9392551
9: -0.5202910, 0.5382967, -0.1786115, 0.2419662, -0.7622572, 0.7169082

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513144, upper bound: 1.7512964
time: 1.34 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513144, upper bound: 1.7512964
time: 1.31 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.5376922, 0.5660532, -0.5350873, 0.6114997, -1.1491919, 1.1011405
1: -0.3583094, 0.3547491, -0.3550812, 0.3596562, -0.7179656, 0.7098304
2: -0.3633478, 0.5271657, -0.3523380, 0.5386430, -0.9019908, 0.8795037
3: -0.3136322, 0.5060745, -0.3322536, 0.5417086, -0.8553408, 0.8383281
4: -0.5082170, 0.4905244, -0.4974619, 0.4490590, -0.9572760, 0.9879863
5: 0.1255535, 1.1222223, 0.1096582, 1.1054441, -0.9798906, 1.0125641
6: -0.3978696, 0.5471033, -0.4067073, 0.5598595, -0.9577291, 0.9538106
7: -0.4197721, 0.4665323, -0.4412111, 0.3771258, -0.7968979, 0.9077435
8: -0.5292524, 0.6421173, -0.5555682, 0.6456740, -1.1749264, 1.1976855
9: -0.4569745, 0.4952339, -0.3968126, 0.5351710, -0.9921454, 0.8920465

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7328974, upper bound: 1.7139215
time: 1.62 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7139773, upper bound: 1.7139215
time: 1.36 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.3743168, 0.4122189, -0.3354237, 0.3660403, -0.7403570, 0.7476426
1: -0.2322864, 0.2425980, -0.2041347, 0.2155663, -0.4478528, 0.4467326
2: -0.2321351, 0.3620004, -0.2036622, 0.3176579, -0.5497931, 0.5656626
3: -0.2218193, 0.3685541, -0.1957882, 0.3280952, -0.5499145, 0.5643423
4: -0.3429738, 0.3043916, -0.3043098, 0.2713179, -0.6142917, 0.6087015
5: 0.3841674, 1.0869969, 0.4469257, 1.0792607, -0.6950933, 0.6400712
6: -0.2845179, 0.3794444, -0.2538445, 0.3381976, -0.6227155, 0.6332889
7: -0.3109131, 0.2696380, -0.2799227, 0.2410610, -0.5519741, 0.5495607
8: -0.3693883, 0.4606678, -0.3233053, 0.4183714, -0.7877597, 0.7839731
9: -0.2680846, 0.3616570, -0.2373718, 0.3213540, -0.5894387, 0.5990288

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7702454, upper bound: 1.7512964
time: 1.66 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7702454, upper bound: 1.7512964
time: 1.62 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.3318160, 0.3617608, -0.6141759, 0.6945617, -1.0263777, 0.9759367
1: -0.2015030, 0.2130537, -0.4040734, 0.4076353, -0.6091384, 0.6171271
2: -0.2010704, 0.3136221, -0.4076398, 0.6424439, -0.8435143, 0.7212619
3: -0.1933494, 0.3243709, -0.3801654, 0.6185678, -0.8119172, 0.7045363
4: -0.3009156, 0.2682189, -0.5923413, 0.5055794, -0.8064950, 0.8605602
5: 0.4528058, 1.0787590, 0.0024096, 1.1461641, -0.6933582, 1.0763494
6: -0.2510602, 0.3343329, -0.4740242, 0.6303463, -0.8814065, 0.8083571
7: -0.2771108, 0.2385039, -0.5022173, 0.4543238, -0.7314346, 0.7407212
8: -0.3192221, 0.4144087, -0.6627978, 0.7179548, -1.0371768, 1.0772066
9: -0.2345921, 0.3175781, -0.4615797, 0.6068197, -0.8414118, 0.7791579

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6603299, upper bound: 1.6164912
time: 1.42 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7448230, upper bound: 1.7143948
time: 1.49 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7383351, upper bound: 1.7143948
time: 1.57 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.6209666, 0.6663730, -0.3116378, 0.3378260, -0.9587926, 0.9780108
1: -0.4240541, 0.4587610, -0.1867842, 0.1989995, -0.6230536, 0.6455452
2: -0.4329260, 0.6104799, -0.1865729, 0.2910487, -0.7239747, 0.7970528
3: -0.3602760, 0.5757906, -0.1797084, 0.3035414, -0.6638174, 0.7554991
4: -0.5917502, 0.5910549, -0.2819309, 0.2508874, -0.8426377, 0.8729857
5: -0.0189092, 1.1399924, 0.4856927, 1.0759530, -1.0948622, 0.6542997
6: -0.4717181, 0.6334351, -0.2354878, 0.3127186, -0.7844367, 0.8689228
7: -0.4967178, 0.5702713, -0.2613835, 0.2242010, -0.7209188, 0.8316548
8: -0.6298366, 0.7356977, -0.2963846, 0.3922443, -1.0220809, 1.0320823
9: -0.5779667, 0.5625747, -0.2190449, 0.2964582, -0.8744249, 0.7816197

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513144, upper bound: 1.7512964
time: 1.20 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513144, upper bound: 1.7512964
time: 1.35 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.5657925, 0.6086506, -0.5841228, 0.6610410, -1.2268336, 1.1927735
1: -0.3804945, 0.4079793, -0.3855470, 0.3887864, -0.7692809, 0.7935263
2: -0.3871053, 0.5552794, -0.3823454, 0.5958760, -0.9829812, 0.9376248
3: -0.3293719, 0.5297411, -0.3639152, 0.5848228, -0.9141947, 0.8936564
4: -0.5364046, 0.5247991, -0.5382978, 0.4849329, -1.0213375, 1.0630969
5: 0.0726829, 1.1282187, 0.0415870, 1.1138424, -1.0411595, 1.0866317
6: -0.4291901, 0.5762357, -0.4457787, 0.6045977, -1.0337877, 1.0220144
7: -0.4540386, 0.5015384, -0.4737643, 0.4173455, -0.8713841, 0.9753027
8: -0.5708728, 0.6736953, -0.6047802, 0.6915510, -1.2624238, 1.2784755
9: -0.5062451, 0.5179576, -0.4289930, 0.5816603, -1.0879054, 0.9469506

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7335013, upper bound: 1.7143948
time: 1.42 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7139773, upper bound: 1.7143948
time: 1.30 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.4959342, 0.5402031, -0.3178241, 0.3451643, -0.8410985, 0.8580272
1: -0.3287051, 0.3471846, -0.1912968, 0.2033082, -0.5320133, 0.5384814
2: -0.3317039, 0.4956530, -0.1910176, 0.2979694, -0.6296732, 0.6866706
3: -0.2951032, 0.4689053, -0.1838905, 0.3099275, -0.6050307, 0.6527959
4: -0.4644717, 0.4435544, -0.2877514, 0.2562012, -0.7206729, 0.7313058
5: 0.1780818, 1.1126292, 0.4756098, 1.0768135, -0.8987316, 0.6370194
6: -0.3769997, 0.5008683, -0.2402621, 0.3193452, -0.6963450, 0.7411305
7: -0.4009813, 0.4216433, -0.2662053, 0.2285861, -0.6295674, 0.6878486
8: -0.4988760, 0.5924021, -0.3033864, 0.3990395, -0.8979155, 0.8957884
9: -0.4116304, 0.4733117, -0.2238115, 0.3029333, -0.7145637, 0.6971232

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7836622, upper bound: 1.7512964
time: 1.63 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7759568, upper bound: 1.7512964
time: 1.51 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.4959342, 0.5402031, -0.6022165, 0.6825030, -1.1784372, 1.1424196
1: -0.3287051, 0.3471846, -0.3987453, 0.4013883, -0.7300934, 0.7459298
2: -0.3317039, 0.4956530, -0.3953450, 0.6161171, -0.9478210, 0.8909980
3: -0.2951032, 0.4689053, -0.3761469, 0.6035005, -0.8986037, 0.8450522
4: -0.4644717, 0.4435544, -0.5553211, 0.5004739, -0.9649456, 0.9988755
5: 0.1780818, 1.1126292, 0.0120977, 1.1163583, -0.9382765, 1.1005315
6: -0.3769997, 0.5008683, -0.4597423, 0.6239791, -1.0009788, 0.9606106
7: -0.4009813, 0.4216433, -0.4878667, 0.4301706, -0.8311519, 0.9095101
8: -0.4988760, 0.5924021, -0.6252583, 0.7114255, -1.2103015, 1.2176604
9: -0.4116304, 0.4733117, -0.4429338, 0.6005982, -1.0122286, 0.9162455

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7836622, upper bound: 1.7512964
time: 1.44 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7759568, upper bound: 1.7512964
time: 1.51 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.9176760, 1.0537012, -0.2609904, 0.2764242, -1.1941001, 1.3146915
1: -0.7237659, 0.7337766, -0.1490247, 0.1629512, -0.8867171, 0.8828012
2: -0.6397777, 1.1846439, -0.1493814, 0.2347552, -0.8745329, 1.3340253
3: -0.6068212, 0.7755302, -0.1452408, 0.2501057, -0.8569269, 0.9207710
4: -0.9052511, 0.8791707, -0.2334650, 0.2064253, -1.1116765, 1.1126357
5: -0.6060343, 1.1882716, 0.5700608, 1.0691535, -1.6751878, 0.6182109
6: -0.7093816, 0.8876902, -0.1965893, 0.2572696, -0.9666513, 1.0842795
7: -0.7496143, 0.9848217, -0.2210369, 0.1891405, -0.9387547, 1.2058586
8: -0.9678649, 1.0427448, -0.2380964, 0.3353844, -1.3032494, 1.2808411
9: -0.8632721, 0.9435523, -0.1791607, 0.2427039, -1.1059761, 1.1227130

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7350633, upper bound: 1.7398275
time: 1.46 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7388026
time: 1.30 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.8740649, 1.0015066, -0.4447153, 0.4956794, -1.3697443, 1.4462218
1: -0.6833891, 0.6941371, -0.2838568, 0.2916883, -0.9750774, 0.9779938
2: -0.6084332, 1.1141831, -0.2821851, 0.4399217, -1.0483549, 1.3963683
3: -0.5751667, 0.7439590, -0.2696722, 0.4409148, -1.0160815, 1.0136312
4: -0.8596088, 0.8349347, -0.4071365, 0.3651916, -1.2248003, 1.2420712
5: -0.5261607, 1.1805140, 0.2687983, 1.0944575, -1.6206181, 0.9117157
6: -0.6755140, 0.8477937, -0.3381905, 0.4552682, -1.1307822, 1.1859841
7: -0.7141535, 0.9274174, -0.3651070, 0.3185297, -1.0326831, 1.2925243
8: -0.9202341, 0.9960932, -0.4470004, 0.5384209, -1.4586550, 1.4430935
9: -0.8171470, 0.8955932, -0.3215802, 0.4357462, -1.2528931, 1.2171733

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7329714, upper bound: 1.7139215
time: 1.50 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7139215
time: 1.18 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.5271960, 0.5835984, -0.3696549, 0.4067313, -0.9339274, 0.9532533
1: -0.3601044, 0.3779074, -0.2289475, 0.2393903, -0.5994947, 0.6068549
2: -0.3574657, 0.5515032, -0.2287241, 0.3565496, -0.7140152, 0.7802273
3: -0.3217188, 0.4923938, -0.2187416, 0.3636948, -0.6854136, 0.7111354
4: -0.4965713, 0.4807489, -0.3381271, 0.3004814, -0.7970527, 0.8188760
5: 0.1133628, 1.1186116, 0.3915872, 1.0858469, -0.9724841, 0.7270244
6: -0.4043469, 0.5302096, -0.2808347, 0.3745675, -0.7789145, 0.8110443
7: -0.4302280, 0.4683783, -0.3071948, 0.2660486, -0.6962765, 0.7755731
8: -0.5388690, 0.6244809, -0.3636855, 0.4556672, -0.9945362, 0.9881664
9: -0.4485627, 0.5116004, -0.2643239, 0.3568921, -0.8054548, 0.7759242

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7842880, upper bound: 1.7512964
time: 1.49 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7761746, upper bound: 1.7512964
time: 1.45 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.5271960, 0.5835984, -0.7207143, 0.7365009, -1.2636969, 1.3043127
1: -0.3601044, 0.3779074, -0.5028043, 0.4796008, -0.8397051, 0.8807117
2: -0.3574657, 0.5515032, -0.5146731, 0.7102754, -1.0677412, 1.0661762
3: -0.3217188, 0.4923938, -0.4161467, 0.6584882, -0.9802070, 0.9085404
4: -0.4965713, 0.4807489, -0.6918080, 0.7094610, -1.2060323, 1.1725569
5: 0.1133628, 1.1186116, -0.1683608, 1.1612775, -1.0479147, 1.2869723
6: -0.4043469, 0.5302096, -0.5235501, 0.7368447, -1.1411916, 1.0537597
7: -0.4302280, 0.4683783, -0.5413843, 0.6945321, -1.1247600, 1.0097626
8: -0.5388690, 0.6244809, -0.7063810, 0.8477901, -1.3866591, 1.3308618
9: -0.4485627, 0.5116004, -0.6745886, 0.6432363, -1.0917990, 1.1861889

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7842880, upper bound: 1.7512964
time: 1.54 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7761746, upper bound: 1.7512964
time: 1.34 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.9591941, 1.0847015, -0.3113879, 0.3375298, -1.2967238, 1.3960894
1: -0.7510318, 0.7573190, -0.1866019, 0.1988255, -0.9498572, 0.9439210
2: -0.6617283, 1.2264924, -0.1863935, 0.2907691, -0.9524975, 1.4128859
3: -0.6256217, 0.7973564, -0.1795395, 0.3032836, -0.9289054, 0.9768959
4: -0.9370254, 0.9054439, -0.2816959, 0.2506728, -1.1876981, 1.1871399
5: -0.6585783, 1.1928791, 0.4860999, 1.0759186, -1.7344968, 0.7067792
6: -0.7351525, 0.9113865, -0.2352948, 0.3124512, -1.0476037, 1.1466813
7: -0.7706757, 1.0275344, -0.2611888, 0.2240240, -0.9946997, 1.2887231
8: -0.9961539, 1.0842760, -0.2961020, 0.3919699, -1.3881238, 1.3803779
9: -0.8906679, 0.9733443, -0.2188524, 0.2961967, -1.1868646, 1.1921967

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7355126, upper bound: 1.7398275
time: 1.41 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7388026
time: 1.32 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.9144822, 1.0332174, -0.5067768, 0.5681397, -1.4826219, 1.5399942
1: -0.7108489, 0.7182199, -0.3271542, 0.3337384, -1.0445873, 1.0453740
2: -0.6304489, 1.1569912, -0.3290560, 0.5168725, -1.1473215, 1.4860473
3: -0.5943982, 0.7658815, -0.3092645, 0.5066217, -1.1010199, 1.0751460
4: -0.8914980, 0.8618101, -0.4806844, 0.4154957, -1.3069937, 1.3424945
5: -0.5792392, 1.1852275, 0.1733454, 1.1196713, -1.6989105, 1.0118821
6: -0.7011328, 0.8720330, -0.3891708, 0.5180025, -1.2191353, 1.2612038
7: -0.7356977, 0.9699770, -0.4165590, 0.3716288, -1.1073265, 1.3865359
8: -0.9491723, 1.0367600, -0.5314209, 0.6027520, -1.5519242, 1.5681808
9: -0.8451704, 0.9258969, -0.3749404, 0.4970459, -1.3422163, 1.3008373

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7335611, upper bound: 1.7143948
time: 1.59 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7143948
time: 1.30 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.3619591, 0.3976728, -0.3825096, 0.4207779, -0.7827370, 0.7801825
1: -0.2234360, 0.2340952, -0.2374940, 0.2483358, -0.4717718, 0.4715892
2: -0.2230932, 0.3475518, -0.2375058, 0.3710979, -0.5941911, 0.5850576
3: -0.2136613, 0.3556734, -0.2266194, 0.3761694, -0.5898307, 0.5822928
4: -0.3301264, 0.2940264, -0.3517387, 0.3104904, -0.6406168, 0.6457652
5: 0.4038356, 1.0839486, 0.3725702, 1.0889344, -0.6850989, 0.7113784
6: -0.2747546, 0.3665177, -0.2907495, 0.3872171, -0.6619717, 0.6572672
7: -0.3010570, 0.2601231, -0.3167121, 0.2758262, -0.5768833, 0.5768352
8: -0.3542716, 0.4474124, -0.3782824, 0.4691016, -0.8233732, 0.8256947
9: -0.2581157, 0.3490264, -0.2750812, 0.3690889, -0.6272045, 0.6241076

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7698134, upper bound: 1.7512964
time: 1.44 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7698134, upper bound: 1.7512964
time: 1.46 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.3223304, 0.3505093, -0.7441741, 0.7583489, -1.0806793, 1.0946834
1: -0.1945838, 0.2064469, -0.5213258, 0.4956044, -0.6901882, 0.7277727
2: -0.1942551, 0.3030105, -0.5340701, 0.7337461, -0.9280012, 0.8370806
3: -0.1869369, 0.3145791, -0.4292870, 0.6780247, -0.8649616, 0.7438660
4: -0.2919910, 0.2600716, -0.7153410, 0.7375246, -1.0295156, 0.9754125
5: 0.4682656, 1.0774399, -0.2060347, 1.1662840, -0.6980184, 1.2834747
6: -0.2437398, 0.3241723, -0.5396599, 0.7611658, -1.0049056, 0.8638322
7: -0.2697175, 0.2317801, -0.5569726, 0.7237576, -0.9934750, 0.7887528
8: -0.3084863, 0.4039895, -0.7290855, 0.8741531, -1.1826394, 1.1330750
9: -0.2272834, 0.3076497, -0.7024825, 0.6622075, -0.8894910, 1.0101322

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6590741, upper bound: 1.6155543
time: 1.40 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7433118, upper bound: 1.7139215
time: 1.44 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7376053, upper bound: 1.7139215
time: 1.48 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.6045916, 0.6492416, -0.3556281, 0.3901585, -0.9947501, 1.0048697
1: -0.4111264, 0.4436896, -0.2188727, 0.2297027, -0.6408291, 0.6625623
2: -0.4193271, 0.5940970, -0.2184223, 0.3402604, -0.7595875, 0.8125193
3: -0.3511042, 0.5621237, -0.2094470, 0.3490650, -0.7001692, 0.7715707
4: -0.5753244, 0.5713913, -0.3236651, 0.2886720, -0.8639964, 0.8950564
5: 0.0082737, 1.1364982, 0.4139957, 1.0825028, -1.0742292, 0.7225025
6: -0.4590962, 0.6164593, -0.2697157, 0.3598400, -0.8189362, 0.8861750
7: -0.4840512, 0.5498729, -0.2959655, 0.2553826, -0.7394338, 0.8458384
8: -0.6123370, 0.7172965, -0.3466073, 0.4405648, -1.0529017, 1.0639038
9: -0.5566810, 0.5493329, -0.2530557, 0.3425015, -0.8991826, 0.8023885

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513144, upper bound: 1.7512964
time: 1.18 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513144, upper bound: 1.7512964
time: 1.26 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.5513415, 0.5935322, -0.6909270, 0.7274431, -1.2787846, 1.2844591
1: -0.3690857, 0.3946794, -0.4240801, 0.4729663, -0.8420520, 0.8187596
2: -0.3751042, 0.5408216, -0.4346919, 0.7005451, -1.0756494, 0.9755135
3: -0.3212777, 0.5176803, -0.3986064, 0.6503890, -0.9716667, 0.9162867
4: -0.5219088, 0.5074457, -0.6820523, 0.5290099, -1.0509187, 1.1894979
5: 0.0966721, 1.1251349, -0.0452811, 1.1592022, -1.0625302, 1.1704161
6: -0.4180516, 0.5612541, -0.5168716, 0.6815209, -1.0995724, 1.0781257
7: -0.4428605, 0.4835362, -0.5244969, 0.5179635, -0.9608241, 1.0080332
8: -0.5554296, 0.6574561, -0.6969686, 0.7878358, -1.3432653, 1.3544247
9: -0.4874600, 0.5062721, -0.5508147, 0.6353714, -1.1228313, 1.0570867

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7332664, upper bound: 1.7139215
time: 1.38 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7139773, upper bound: 1.7139215
time: 1.17 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.3871132, 0.4253555, -0.4426551, 0.4797791, -0.8668922, 0.8680106
1: -0.2402790, 0.2516888, -0.2832426, 0.2947023, -0.5349813, 0.5349314
2: -0.2404491, 0.3760153, -0.2848994, 0.4320380, -0.6724871, 0.6609147
3: -0.2291864, 0.3802629, -0.2604102, 0.4269307, -0.6561171, 0.6406731
4: -0.3566692, 0.3137521, -0.4128391, 0.3769829, -0.7336521, 0.7265911
5: 0.3663333, 1.0899835, 0.2770140, 1.1019325, -0.7355993, 0.8129694
6: -0.2941248, 0.3916099, -0.3342728, 0.4485309, -0.7426558, 0.7258827
7: -0.3198137, 0.2794404, -0.3587529, 0.3482047, -0.6680184, 0.6381932
8: -0.3830392, 0.4738593, -0.4392297, 0.5352824, -0.9183217, 0.9130890
9: -0.2791969, 0.3730634, -0.3461178, 0.4184963, -0.6976932, 0.7191812

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7704867, upper bound: 1.7512964
time: 1.46 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7704867, upper bound: 1.7512964
time: 1.52 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.3426237, 0.3745809, -0.8078421, 0.9274827, -1.2701063, 1.1824229
1: -0.2093868, 0.2205812, -0.6612132, 0.6524317, -0.8618184, 0.8817945
2: -0.2088353, 0.3257124, -0.6098162, 0.8537886, -1.0626239, 0.9355286
3: -0.2006556, 0.3355278, -0.5170543, 0.7173648, -0.9180204, 0.8525822
4: -0.3110839, 0.2775022, -0.7621205, 0.8358794, -1.1469634, 1.0396228
5: 0.4351907, 1.0802618, -0.4214540, 1.1761894, -0.7409987, 1.5017159
6: -0.2594011, 0.3459101, -0.6381137, 0.8092898, -1.0686909, 0.9840238
7: -0.2855346, 0.2461646, -0.6279302, 0.8790547, -1.1645893, 0.8740948
8: -0.3314542, 0.4262803, -0.8902119, 0.9318078, -1.2632620, 1.3164923
9: -0.2429194, 0.3288901, -0.7984694, 0.8404989, -1.0834183, 1.1273595

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6618854, upper bound: 1.6166291
time: 1.46 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7457908, upper bound: 1.7143948
time: 1.56 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7388940, upper bound: 1.7143948
time: 1.49 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.6437041, 0.6802050, -0.4106886, 0.4477759, -1.0914800, 1.0908936
1: -0.4344925, 0.4916461, -0.2580408, 0.2681115, -0.7026041, 0.7496868
2: -0.4637784, 0.6237079, -0.2583397, 0.4001013, -0.8638797, 0.8820475
3: -0.3751175, 0.5868256, -0.2424951, 0.4003111, -0.7754286, 0.8293208
4: -0.6050132, 0.6296225, -0.3808184, 0.3385983, -0.9436115, 1.0104408
5: -0.0743088, 1.1428138, 0.3295075, 1.0951207, -1.1694295, 0.8133063
6: -0.4883725, 0.6471420, -0.3106567, 0.4154376, -0.9038101, 0.9577987
7: -0.5069451, 0.6122730, -0.3353824, 0.3083176, -0.8152627, 0.9476554
8: -0.6439663, 0.7535844, -0.4063387, 0.4993960, -1.1433623, 1.1599231
9: -0.5951543, 0.6052061, -0.3059666, 0.3925315, -0.9876858, 0.9111727

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513144, upper bound: 1.7512964
time: 1.21 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7513144, upper bound: 1.7512964
time: 1.19 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.5858453, 0.6225258, -0.7743340, 0.8104302, -1.3962755, 1.3968598
1: -0.3909653, 0.4349688, -0.5327661, 0.6196083, -1.0105736, 0.9677349
2: -0.4122997, 0.5685483, -0.5800033, 0.7482439, -1.1605436, 1.1485516
3: -0.3421066, 0.5408105, -0.4496475, 0.6907162, -1.0328228, 0.9904580
4: -0.5497085, 0.5569168, -0.7298764, 0.7937730, -1.3434815, 1.2867932
5: 0.0267966, 1.1310488, -0.3025778, 1.1693759, -1.1425793, 1.4336267
6: -0.4440250, 0.5899850, -0.5884976, 0.7761880, -1.2202129, 1.1784825
7: -0.4642978, 0.5362781, -0.6032317, 0.7838490, -1.2481468, 1.1395098
8: -0.5850465, 0.6907606, -0.7769921, 0.8954242, -1.4804708, 1.4677527
9: -0.5234857, 0.5514740, -0.7569634, 0.7265193, -1.2500050, 1.3084373

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7338453, upper bound: 1.7143948
time: 1.44 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7139773, upper bound: 1.7143948
time: 1.39 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.5097553, 0.5594477, -0.4264027, 0.4628231, -0.9725783, 0.9858504
1: -0.3423225, 0.3608564, -0.2704470, 0.2796873, -0.6220098, 0.6313033
2: -0.3430872, 0.5204811, -0.2713455, 0.4158229, -0.7589101, 0.7918266
3: -0.3069474, 0.4792758, -0.2512970, 0.4134037, -0.7203512, 0.7305728
4: -0.4785438, 0.4601063, -0.3965814, 0.3574127, -0.8359566, 0.8566877
5: 0.1496136, 1.1152563, 0.3040776, 1.0984740, -0.9488605, 0.8111787
6: -0.3890035, 0.5139256, -0.3217496, 0.4317287, -0.8207322, 0.8356752
7: -0.4139187, 0.4423194, -0.3462161, 0.3278936, -0.7418123, 0.7885355
8: -0.5166426, 0.6062943, -0.4219094, 0.5170547, -1.0336972, 1.0282037
9: -0.4279551, 0.4900727, -0.3250495, 0.4052389, -0.8331940, 0.8151222

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7839076, upper bound: 1.7512964
time: 1.45 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7763118, upper bound: 1.7512964
time: 1.49 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.5097553, 0.5594477, -0.8061013, 0.8420991, -1.3518543, 1.3655491
1: -0.3423225, 0.3608564, -0.5566647, 0.6507269, -0.9930494, 0.9175211
2: -0.3430872, 0.5204811, -0.6082678, 0.7785293, -1.1216165, 1.1287489
3: -0.3069474, 0.4792758, -0.4677725, 0.7159806, -1.0229280, 0.9470482
4: -0.4785438, 0.4601063, -0.7602416, 0.8336925, -1.3122363, 1.2203479
5: 0.1496136, 1.1152563, -0.3580900, 1.1758354, -1.0262218, 1.4733464
6: -0.3890035, 0.5139256, -0.6128463, 0.8075704, -1.1965739, 1.1267719
7: -0.4139187, 0.4423194, -0.6266472, 0.8255742, -1.2394929, 1.0689666
8: -0.5166426, 0.6062943, -0.8093421, 0.9299178, -1.4465604, 1.4156365
9: -0.4279551, 0.4900727, -0.7963133, 0.7560210, -1.1839762, 1.2863860

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7839076, upper bound: 1.7512964
time: 1.49 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7763118, upper bound: 1.7512964
time: 1.45 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.9495661, 1.0736156, -0.3567375, 0.3914954, -1.3410615, 1.4303530
1: -0.7423793, 0.7488998, -0.2196819, 0.2304844, -0.9728637, 0.9685817
2: -0.6549929, 1.2115266, -0.2192534, 0.3415014, -0.9964944, 1.4307799
3: -0.6188986, 0.7905791, -0.2101968, 0.3502261, -0.9691247, 1.0007759
4: -0.9272223, 0.8960485, -0.3247574, 0.2896248, -1.2168471, 1.2208059
5: -0.6414946, 1.1912315, 0.4121876, 1.0827178, -1.7242124, 0.7790439
6: -0.7278271, 0.9029122, -0.2706108, 0.3610283, -1.0888554, 1.1735231
7: -0.7631440, 1.0151405, -0.2968716, 0.2561687, -1.0193126, 1.3120121
8: -0.9860375, 1.0740442, -0.3479236, 0.4417834, -1.4278209, 1.4219677
9: -0.8808709, 0.9631276, -0.2539266, 0.3436626, -1.2245336, 1.2170541

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7353591, upper bound: 1.7398275
time: 1.45 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7388026
time: 1.24 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.9044836, 1.0217044, -0.5951828, 0.6195940, -1.5240777, 1.6168872
1: -0.7018625, 0.7094761, -0.4036978, 0.3939674, -1.0958298, 1.1131738
2: -0.6234539, 1.1414496, -0.4108820, 0.5846835, -1.2081374, 1.5523316
3: -0.5874159, 0.7588428, -0.3458340, 0.5539503, -1.1413662, 1.1046767
4: -0.8813171, 0.8520525, -0.5658860, 0.5592963, -1.4406134, 1.4179385
5: -0.5614965, 1.1835158, 0.0332297, 1.1344901, -1.6959866, 1.1502861
6: -0.6935247, 0.8632326, -0.4373479, 0.6067046, -1.3002293, 1.3005805
7: -0.7278756, 0.9571054, -0.4579726, 0.5381513, -1.2660270, 1.4150779
8: -0.9386657, 1.0261339, -0.5848916, 0.7067229, -1.6453886, 1.6110255
9: -0.8349960, 0.9152859, -0.5253309, 0.5417243, -1.3767203, 1.4406168

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7333309, upper bound: 1.7139215
time: 1.43 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7139215
time: 1.14 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.5409900, 0.6028623, -0.4888084, 0.5302328, -1.0712228, 1.0916708
1: -0.3750064, 0.3913931, -0.3216540, 0.3401017, -0.7151080, 0.7130471
2: -0.3690341, 0.5760397, -0.3258067, 0.4827994, -0.8518335, 0.9018465
3: -0.3334014, 0.5028355, -0.2889671, 0.4635537, -0.7969551, 0.7918026
4: -0.5110207, 0.4970751, -0.4572318, 0.4349796, -0.9460003, 0.9543070
5: 0.0838835, 1.1212653, 0.1928304, 1.1112815, -1.0273981, 0.9284349
6: -0.4168465, 0.5430885, -0.3707868, 0.4941043, -0.9109508, 0.9138753
7: -0.4433157, 0.4889889, -0.3942788, 0.4109778, -0.8542935, 0.8832677
8: -0.5564481, 0.6397955, -0.4896831, 0.5852053, -1.1416534, 1.1294787
9: -0.4648615, 0.5293006, -0.4032150, 0.4646277, -0.9294893, 0.9325156

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7845807, upper bound: 1.7512964
time: 1.58 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7766255, upper bound: 1.7512964
time: 1.52 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.5409900, 0.6028623, -0.8890504, 1.0889295, -1.6299195, 1.4919127
1: -0.3750064, 0.3913931, -0.7510180, 0.7316698, -1.1066761, 1.1424112
2: -0.3690341, 0.5760397, -0.6609331, 1.1951432, -1.5641773, 1.2369728
3: -0.3334014, 0.5028355, -0.6281861, 0.7663066, -1.0997081, 1.1310216
4: -0.5110207, 0.4970751, -0.8756167, 0.9090275, -1.4200482, 1.3726918
5: 0.0838835, 1.1212653, -0.6599441, 1.1882242, -1.1043408, 1.7812093
6: -0.4168465, 0.5430885, -0.7322397, 0.8680571, -1.2849036, 1.2753282
7: -0.4433157, 0.4889889, -0.7735481, 1.0090373, -1.4523529, 1.2625370
8: -0.5564481, 0.6397955, -1.0000124, 1.0262154, -1.5826635, 1.6398079
9: -0.4648615, 0.5293006, -0.8761179, 0.9759216, -1.4407831, 1.4054185

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7845807, upper bound: 1.7512964
time: 1.52 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7766255, upper bound: 1.7512964
time: 1.64 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.9765504, 1.1046867, -0.4112335, 0.4482836, -1.4248340, 1.5159203
1: -0.7666306, 0.7724969, -0.2584712, 0.2684833, -1.0351138, 1.0309681
2: -0.6738707, 1.2534714, -0.2587905, 0.4006465, -1.0745173, 1.5122619
3: -0.6377420, 0.8095747, -0.2428006, 0.4007649, -1.0385070, 1.0523753
4: -0.9546986, 0.9223818, -0.3813652, 0.3392506, -1.2939492, 1.3037469
5: -0.6893762, 1.1958497, 0.3286320, 1.0952371, -1.7846134, 0.8672177
6: -0.7483584, 0.9266630, -0.3110309, 0.4160028, -1.1643612, 1.2376939
7: -0.7842534, 1.0498767, -0.3357448, 0.3089971, -1.0932505, 1.3856214
8: -1.0143920, 1.1027211, -0.4068661, 0.5000084, -1.5144005, 1.5095873
9: -0.9083291, 0.9917626, -0.3066148, 0.3929723, -1.3013014, 1.2983774

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7357365, upper bound: 1.7398275
time: 1.38 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7388026
time: 1.31 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.9319482, 1.0533290, -0.6729281, 0.7589422, -1.6908903, 1.7262571
1: -0.7265457, 0.7334936, -0.5176986, 0.5202730, -1.2468187, 1.2511923
2: -0.6426677, 1.1841409, -0.4897795, 0.6952133, -1.3378811, 1.6739204
3: -0.6065955, 0.7781765, -0.4204265, 0.6100676, -1.2166631, 1.1986030
4: -0.9092829, 0.8788552, -0.6330748, 0.6663455, -1.5756284, 1.5119300
5: -0.6102315, 1.1882163, -0.1611474, 1.1487560, -1.7589875, 1.3493637
6: -0.7144220, 0.8874058, -0.5249644, 0.6760117, -1.3904338, 1.4123702
7: -0.7493610, 0.9924608, -0.5284860, 0.6810154, -1.4303765, 1.5209467
8: -0.9675246, 1.0553212, -0.7206312, 0.7853162, -1.7528408, 1.7759523
9: -0.8629434, 0.9444315, -0.6313534, 0.6814829, -1.5444263, 1.5757849

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7339313, upper bound: 1.7143948
time: 1.49 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7143948
time: 1.21 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.5218445, 0.5761253, -0.2892788, 0.3110055, -0.8328500, 0.8654041
1: -0.3543231, 0.3726758, -0.1702908, 0.1832523, -0.5375754, 0.5429666
2: -0.3529776, 0.5419850, -0.1703276, 0.2661181, -0.6190958, 0.7123126
3: -0.3171866, 0.4883430, -0.1645415, 0.2802007, -0.5973873, 0.6528845
4: -0.4909656, 0.4744149, -0.2607109, 0.2314664, -0.7224320, 0.7351258
5: 0.1247990, 1.1175821, 0.5225447, 1.0728989, -0.9480999, 0.5950373
6: -0.3994979, 0.5252129, -0.2182746, 0.2884985, -0.6879964, 0.7434875
7: -0.4251507, 0.4603827, -0.2437601, 0.2085414, -0.6336921, 0.7041428
8: -0.5320493, 0.6185396, -0.2708611, 0.3674080, -0.8994573, 0.8894008
9: -0.4422396, 0.5047337, -0.2016234, 0.2728880, -0.7151276, 0.7063571

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7694138, upper bound: 1.7512964
time: 1.42 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7694138, upper bound: 1.7512964
time: 1.59 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.4660488, 0.5030999, -0.5729296, 0.6477635, -1.1138123, 1.0760295
1: -0.3008417, 0.3176182, -0.3773821, 0.3809898, -0.6818316, 0.6950003
2: -0.3057132, 0.4543405, -0.3743033, 0.5833541, -0.8890673, 0.8286438
3: -0.2737572, 0.4455358, -0.3563483, 0.5732678, -0.8470250, 0.8018841
4: -0.4351999, 0.4063793, -0.5277663, 0.4753184, -0.9105183, 0.9341456
5: 0.2361352, 1.1066892, 0.0598306, 1.1122859, -0.8761507, 1.0468585
6: -0.3522035, 0.4716408, -0.4371400, 0.5926076, -0.9448111, 0.9087808
7: -0.3759962, 0.3789307, -0.4650398, 0.4094114, -0.7854076, 0.8439704
8: -0.4630526, 0.5606835, -0.5921116, 0.6792557, -1.1423082, 1.1527951
9: -0.3750949, 0.4402212, -0.4203682, 0.5699443, -0.9450392, 0.8605894

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6566031, upper bound: 1.6155599
time: 1.47 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7423393, upper bound: 1.7139215
time: 1.45 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7368606, upper bound: 1.7139215
time: 1.50 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.8945588, 0.9854352, -0.2654024, 0.2818178, -1.1763766, 1.2508376
1: -0.6861589, 0.6819316, -0.1523415, 0.1661177, -0.8522766, 0.8342731
2: -0.6215180, 1.0924877, -0.1526484, 0.2396468, -0.8611648, 1.2451361
3: -0.5886378, 0.7366695, -0.1482511, 0.2547997, -0.8434375, 0.8849206
4: -0.8640226, 0.8222350, -0.2377145, 0.2103308, -1.0743535, 1.0599495
5: -0.5056041, 1.1803207, 0.5626496, 1.0697376, -1.5753417, 0.6176711
6: -0.6766376, 0.8355091, -0.1999715, 0.2621403, -0.9387779, 1.0354806
7: -0.7272871, 0.9165579, -0.2245810, 0.1921664, -0.9194534, 1.1411389
8: -0.9140369, 0.9926599, -0.2432066, 0.3403791, -1.2544160, 1.2358665
9: -0.8029442, 0.8994082, -0.1826641, 0.2474117, -1.0503559, 1.0820723

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7512964
time: 1.15 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7512964
time: 1.29 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.8148264, 0.9005836, -0.5404627, 0.6180710, -1.4328973, 1.4410462
1: -0.6164058, 0.6174906, -0.3591223, 0.3635138, -0.9799196, 0.9766130
2: -0.5643374, 0.9779413, -0.3563181, 0.5446025, -1.1089399, 1.3342595
3: -0.5306756, 0.6847947, -0.3359210, 0.5474274, -1.0781031, 1.0207157
4: -0.7848467, 0.7500646, -0.5026394, 0.4538175, -1.2386642, 1.2527039
5: -0.3748425, 1.1670952, 0.1006292, 1.1061558, -1.4809983, 1.0664660
6: -0.6185961, 0.7706494, -0.4108280, 0.5657936, -1.1843898, 1.1814774
7: -0.6629001, 0.8216958, -0.4455291, 0.3808124, -1.0437126, 1.2672249
8: -0.8342324, 0.9143473, -0.5617940, 0.6517591, -1.4859915, 1.4761412
9: -0.7279589, 0.8163040, -0.4010811, 0.5409067, -1.2688656, 1.2173851

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7328974, upper bound: 1.7139215
time: 1.50 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7139773, upper bound: 1.7139215
time: 1.11 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.5596417, 0.6246314, -0.3404328, 0.3719823, -0.9316241, 0.9650642
1: -0.3919467, 0.4079184, -0.2077886, 0.2190552, -0.6110019, 0.6157070
2: -0.3822089, 0.6054178, -0.2072613, 0.3232616, -0.7054705, 0.8126792
3: -0.3466039, 0.5160896, -0.1991745, 0.3332663, -0.6798702, 0.7152641
4: -0.5301847, 0.5155250, -0.3090228, 0.2756203, -0.8058050, 0.8245478
5: 0.0504143, 1.1244993, 0.4387616, 1.0799572, -1.0295429, 0.6857377
6: -0.4311443, 0.5597171, -0.2577103, 0.3435632, -0.7747076, 0.8174274
7: -0.4581053, 0.5131902, -0.2838270, 0.2446118, -0.7027171, 0.7970172
8: -0.5763135, 0.6596626, -0.3289746, 0.4238737, -1.0001872, 0.9886372
9: -0.4840948, 0.5493428, -0.2412313, 0.3265969, -0.8106917, 0.7905741

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7700814, upper bound: 1.7512964
time: 1.46 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7700814, upper bound: 1.7512964
time: 1.46 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.4942945, 0.5379089, -0.6196997, 0.7010634, -1.1953579, 1.1576086
1: -0.3270826, 0.3455547, -0.4080295, 0.4114360, -0.7385186, 0.7535841
2: -0.3303468, 0.4926952, -0.4116816, 0.6489021, -0.9792489, 0.9043767
3: -0.2936912, 0.4676740, -0.3838119, 0.6243254, -0.9180167, 0.8514859
4: -0.4628057, 0.4415810, -0.5980838, 0.5102126, -0.9730183, 1.0396647
5: 0.1814755, 1.1123190, -0.0063818, 1.1475264, -0.9660509, 1.1187007
6: -0.3755701, 0.4993122, -0.4783882, 0.6361244, -1.0116944, 0.9777004
7: -0.3994390, 0.4191890, -0.5066228, 0.4585769, -0.8580159, 0.9258118
8: -0.4967609, 0.5907461, -0.6695546, 0.7238797, -1.2206407, 1.2603006
9: -0.4096938, 0.4713134, -0.4660357, 0.6124657, -1.0221595, 0.9373491

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6603299, upper bound: 1.6164932
time: 1.36 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7447926, upper bound: 1.7143948
time: 1.59 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7383228, upper bound: 1.7143948
time: 1.56 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.1937680, 1.0878980, -0.3167219, 0.3438568, -1.5376248, 1.4046199
1: -0.7788723, 0.8330752, -0.1904928, 0.2025405, -0.9814127, 1.0235680
2: -0.8579524, 1.1948962, -0.1902256, 0.2967364, -1.1546888, 1.3851218
3: -0.9803172, 0.7898743, -0.1831454, 0.3087896, -1.2891068, 0.9730196
4: -1.0378428, 1.0215905, -0.2867143, 0.2552544, -1.2930971, 1.3083048
5: -0.6911156, 1.2380170, 0.4774063, 1.0766599, -1.7677755, 0.7606106
6: -0.8093370, 0.9653975, -0.2394114, 0.3181649, -1.1275018, 1.2048090
7: -0.9594234, 1.0662134, -0.2653463, 0.2278048, -1.1872282, 1.3315597
8: -1.1126075, 1.0805607, -0.3021389, 0.3978288, -1.5104363, 1.3826995
9: -0.9796579, 1.0142932, -0.2229621, 0.3017797, -1.2814376, 1.2372553

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7512964
time: 1.33 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7512964
time: 1.34 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.0409985, 0.9838146, -0.5895771, 0.6675106, -1.7085091, 1.5733917
1: -0.6915687, 0.7346662, -0.3895256, 0.3925848, -1.0841535, 1.1241918
2: -0.7423364, 1.0654256, -0.3862641, 0.6019776, -1.3443140, 1.4516897
3: -0.8222431, 0.7291775, -0.3676024, 0.5904529, -1.4126960, 1.0967798
4: -0.9189626, 0.9033795, -0.5434293, 0.4896175, -1.4085801, 1.4468088
5: -0.5236895, 1.2106838, 0.0326979, 1.1146007, -1.6382902, 1.1779859
6: -0.7209247, 0.8722969, -0.4499878, 0.6104401, -1.3313649, 1.3222847
7: -0.8384220, 0.9407127, -0.4780153, 0.4212116, -1.2596337, 1.4187281
8: -0.9872380, 0.9868219, -0.6109532, 0.6975417, -1.6847798, 1.5977751
9: -0.8642035, 0.9089217, -0.4331950, 0.5873687, -1.4515722, 1.3421167

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7335013, upper bound: 1.7143948
time: 1.68 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7139773, upper bound: 1.7143948
time: 1.32 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.7500989, 0.7807307, -0.3085934, 0.3342153, -1.0843143, 1.0893241
1: -0.5254763, 0.5468257, -0.1845636, 0.1968792, -0.7223555, 0.7313893
2: -0.5226768, 0.8135972, -0.1843857, 0.2876432, -0.8103200, 0.9979829
3: -0.5220075, 0.6122502, -0.1776505, 0.3003990, -0.8224065, 0.7899007
4: -0.6936269, 0.6786708, -0.2790668, 0.2482728, -0.9418997, 0.9577376
5: -0.2074164, 1.1580364, 0.4906542, 1.0755302, -1.2829466, 0.6673821
6: -0.5589824, 0.6943934, -0.2331384, 0.3094579, -0.8684403, 0.9275318
7: -0.6023631, 0.7025900, -0.2590108, 0.2220432, -0.8244063, 0.9616008
8: -0.7528441, 0.8036382, -0.2929393, 0.3889004, -1.1417445, 1.0965775
9: -0.6414750, 0.7082493, -0.2166994, 0.2932720, -0.9347469, 0.9249486

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7694138, upper bound: 1.7512964
time: 1.45 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7694138, upper bound: 1.7512964
time: 1.40 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.6098116, 0.6823879, -0.5931847, 0.6717899, -1.2816015, 1.2755725
1: -0.4370434, 0.4517820, -0.3921571, 0.3950979, -0.8321413, 0.8439391
2: -0.4173119, 0.6833867, -0.3888560, 0.6060137, -1.0233256, 1.0722427
3: -0.3816430, 0.5513994, -0.3700414, 0.5941771, -0.9758201, 0.9214407
4: -0.5812590, 0.5644804, -0.5468236, 0.4927163, -1.0739753, 1.1113040
5: -0.0385918, 1.1330872, 0.0268177, 1.1151024, -1.1536942, 1.1062694
6: -0.4693440, 0.6038651, -0.4527721, 0.6143044, -1.0836484, 1.0566373
7: -0.4973481, 0.5777604, -0.4808272, 0.4237688, -0.9211169, 1.0585876
8: -0.6290224, 0.7129679, -0.6150363, 0.7015046, -1.3305269, 1.3280042
9: -0.5351355, 0.6026101, -0.4359747, 0.5911451, -1.1262805, 1.0385848

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6651781, upper bound: 1.6160434
time: 1.36 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6271045, upper bound: 1.6160434
time: 1.28 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.9028841, 1.4312276, -0.2844579, 0.3051125, -2.2079966, 1.7156854
1: -1.2873967, 1.3514810, -0.1666667, 0.1797926, -1.4671892, 1.5181477
2: -1.5503548, 1.5956163, -0.1667580, 0.2607733, -1.8111281, 1.7623743
3: -1.9238698, 1.0336332, -0.1612523, 0.2750719, -2.1989417, 1.1948856
4: -1.7712983, 1.5834508, -0.2560679, 0.2271987, -1.9984970, 1.8395187
5: -1.5825011, 1.4945148, 0.5306423, 1.0722605, -2.6547616, 0.9638726
6: -1.3719459, 1.5446436, -0.2145790, 0.2831766, -1.6551224, 1.7592226
7: -1.4653777, 1.6288482, -0.2398876, 0.2052352, -1.6706129, 1.8687358
8: -1.9140856, 1.4084264, -0.2652775, 0.3619505, -2.2760360, 1.6737039
9: -1.5627673, 1.4752576, -0.1977953, 0.2677442, -1.8305116, 1.6730530

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7512964, upper bound: 1.7512964
time: 1.21 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7512964, upper bound: 1.7512964
time: 1.12 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.6551428, 1.2879941, -0.5682681, 0.6422342, -2.2973771, 1.8562622
1: -1.1192894, 1.1745753, -0.3739817, 0.3777431, -1.4970325, 1.5485570
2: -1.3249124, 1.4268072, -0.3709542, 0.5781393, -1.9030517, 1.7977613
3: -1.6170869, 0.9419262, -0.3531969, 0.5684559, -2.1855428, 1.2951231
4: -1.5340233, 1.3866860, -0.5233806, 0.4713144, -2.0053377, 1.9100666
5: -1.2836592, 1.4174531, 0.0674281, 1.1116376, -2.3952968, 1.3500249
6: -1.1951702, 1.3555814, -0.4335425, 0.5876142, -1.7827843, 1.7891239
7: -1.2753316, 1.4268287, -0.4614066, 0.4061072, -1.6814388, 1.8882353
8: -1.6606725, 1.2762867, -0.5868357, 0.6741353, -2.3348079, 1.8631225
9: -1.3592577, 1.3095694, -0.4167767, 0.5650653, -1.9243231, 1.7263460

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7329528, upper bound: 1.7139215
time: 1.42 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7139215
time: 1.21 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.8596202, 0.8378190, -0.3600215, 0.3953921, -1.2550124, 1.1978405
1: -0.5893329, 0.6132132, -0.2220483, 0.2327618, -0.8220948, 0.8352615
2: -0.6102188, 0.8867373, -0.2216754, 0.3452864, -0.9555051, 1.1084126
3: -0.6418445, 0.6494372, -0.2123821, 0.3536537, -0.9954982, 0.8618193
4: -0.7827438, 0.7627804, -0.3281120, 0.2924013, -1.0751451, 1.0908924
5: -0.3306012, 1.1776814, 0.4069194, 1.0834706, -1.4140718, 0.7707620
6: -0.6324370, 0.7602199, -0.2732238, 0.3644908, -0.9969279, 1.0334437
7: -0.6768394, 0.7849897, -0.2995116, 0.2586312, -0.9354706, 1.0845013
8: -0.8538550, 0.8560370, -0.3519016, 0.4453338, -1.2991889, 1.2079386
9: -0.7150136, 0.7785510, -0.2565525, 0.3470458, -1.0620594, 1.0351036

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7700814, upper bound: 1.7512964
time: 1.48 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7700814, upper bound: 1.7512964
time: 1.49 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.6601056, 0.7252666, -0.6981118, 0.7252518, -1.3853574, 1.4233783
1: -0.4739293, 0.4894292, -0.4711414, 0.4713609, -0.9452902, 0.9605706
2: -0.4544124, 0.7426769, -0.4604424, 0.6981907, -1.1526031, 1.2031193
3: -0.4282404, 0.5780022, -0.4066600, 0.6484295, -1.0766699, 0.9846622
4: -0.6233371, 0.6086501, -0.6796918, 0.6136352, -1.2369723, 1.2883419
5: -0.1068048, 1.1425359, -0.1168782, 1.1587000, -1.2655048, 1.2594141
6: -0.5008093, 0.6403625, -0.5152556, 0.7004398, -1.2012491, 1.1556181
7: -0.5372350, 0.6280438, -0.5230120, 0.6794854, -1.2167205, 1.1510558
8: -0.6748893, 0.7532738, -0.6946912, 0.7998799, -1.4747692, 1.4479649
9: -0.5767488, 0.6462045, -0.6285053, 0.6334686, -1.2102175, 1.2747098

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6674094, upper bound: 1.6172222
time: 1.55 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6287729, upper bound: 1.6172222
time: 1.20 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -2.0336108, 1.5118757, -0.3362331, 0.3670006, -2.4006114, 1.8481088
1: -1.3785752, 1.4560708, -0.2047251, 0.2161302, -1.5947055, 1.6607958
2: -1.6871980, 1.6822412, -0.2042439, 0.3185633, -2.0057614, 1.8864851
3: -2.0867009, 1.0945420, -0.1963354, 0.3289308, -2.4156318, 1.2908775
4: -1.9088242, 1.7013348, -0.3050714, 0.2720131, -2.1808372, 2.0064063
5: -1.7559741, 1.5338664, 0.4456063, 1.0793731, -2.8353472, 1.0882602
6: -1.4728479, 1.6781583, -0.2544693, 0.3390646, -1.8119125, 1.9326276
7: -1.5829787, 1.7313566, -0.2805537, 0.2416348, -1.8246135, 2.0119104
8: -2.0555625, 1.5152626, -0.3242213, 0.4192609, -2.4748235, 1.8394840
9: -1.6641866, 1.5851002, -0.2379955, 0.3222013, -1.9863878, 1.8230957

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7512964, upper bound: 1.7512964
time: 1.16 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7512964, upper bound: 1.7512964
time: 1.10 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.7821460, 1.3649260, -0.6156922, 0.6963462, -2.4784923, 1.9806182
1: -1.2071813, 1.2736737, -0.4051595, 0.4086783, -1.6158596, 1.6788331
2: -1.4534799, 1.5114577, -0.4087492, 0.6442163, -2.0976963, 1.9202068
3: -1.7751818, 0.9980172, -0.3811661, 0.6201481, -2.3953300, 1.3791833
4: -1.6645608, 1.4974422, -0.5939173, 0.5068510, -2.1714118, 2.0913596
5: -1.4481883, 1.4559596, -0.0000033, 1.1465380, -2.5947263, 1.4559629
6: -1.2913716, 1.4771669, -0.4752219, 0.6319321, -1.9233037, 1.9523888
7: -1.3851694, 1.5273497, -0.5034264, 0.4554913, -1.8406607, 2.0307760
8: -1.7960651, 1.3706589, -0.6646521, 0.7195809, -2.5156460, 2.0353110
9: -1.4592994, 1.4107891, -0.4628026, 0.6083694, -2.0676689, 1.8735918

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7335601, upper bound: 1.7143948
time: 1.46 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7143948
time: 1.28 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.5367367, 0.5969215, -0.3890232, 0.4272543, -0.9639909, 0.9859447
1: -0.3704110, 0.3872343, -0.2414345, 0.2530799, -0.6234909, 0.6286688
2: -0.3654668, 0.5684731, -0.2416702, 0.3780553, -0.7435221, 0.8101433
3: -0.3297989, 0.4996155, -0.2302514, 0.3819608, -0.7117597, 0.7298669
4: -0.5065650, 0.4920408, -0.3587145, 0.3151053, -0.8216704, 0.8507553
5: 0.0929738, 1.1204469, 0.3637456, 1.0904186, -0.9974447, 0.7567014
6: -0.4129919, 0.5391169, -0.2955251, 0.3934327, -0.8064246, 0.8346419
7: -0.4392794, 0.4826332, -0.3211004, 0.2809399, -0.7202193, 0.8037336
8: -0.5510271, 0.6350729, -0.3850127, 0.4758333, -1.0268604, 1.0200857
9: -0.4598354, 0.5238424, -0.2809045, 0.3747123, -0.8345476, 0.8047469

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7696675, upper bound: 1.7512964
time: 1.57 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7696675, upper bound: 1.7512964
time: 1.63 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.4793331, 0.5170941, -0.7513874, 0.8028172, -1.2821504, 1.2684815
1: -0.3117623, 0.3306313, -0.5270208, 0.5787984, -0.8905606, 0.8576521
2: -0.3175328, 0.4676658, -0.5412372, 0.7409633, -1.0584961, 1.0089030
3: -0.2817701, 0.4561007, -0.4333273, 0.6846423, -0.9664124, 0.8894280
4: -0.4479001, 0.4230728, -0.7225766, 0.7476709, -1.1955709, 1.1456494
5: 0.2123798, 1.1093907, -0.2354150, 1.1678231, -0.9554433, 1.3448057
6: -0.3626004, 0.4847643, -0.5722451, 0.7686440, -1.1312444, 1.0570095
7: -0.3857882, 0.3968393, -0.5976027, 0.7327434, -1.1185316, 0.9944420
8: -0.4772908, 0.5751079, -0.7692153, 0.8822594, -1.3595502, 1.3443232
9: -0.3915505, 0.4533020, -0.7475038, 0.6680404, -1.0595908, 1.2008059

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6590741, upper bound: 1.6155599
time: 1.55 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7694138, upper bound: 1.7512964
time: 1.74 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7694138, upper bound: 1.7512964
time: 1.54 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.0740392, 1.0149161, -0.3610687, 0.3966247, -1.4706639, 1.3759848
1: -0.7499120, 0.7717568, -0.2227984, 0.2334824, -0.9833944, 0.9945551
2: -0.7614541, 1.1527984, -0.2224416, 0.3465108, -1.1079650, 1.3752400
3: -0.8606691, 0.7665939, -0.2130734, 0.3547453, -1.2154143, 0.9796673
4: -0.9355633, 0.9716128, -0.3292005, 0.2932797, -1.2288430, 1.3008132
5: -0.5913951, 1.2235649, 0.4052526, 1.0837290, -1.6751242, 0.8183122
6: -0.7211946, 0.9145343, -0.2740511, 0.3655860, -1.0867807, 1.1885854
7: -0.8665524, 0.9910374, -0.3003467, 0.2594374, -1.1259899, 1.2913841
8: -1.0319613, 1.0362898, -0.3531823, 0.4464573, -1.4784186, 1.3894721
9: -0.9094788, 0.9528443, -0.2573974, 0.3481162, -1.2575949, 1.2102416

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7512964
time: 1.22 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7512964
time: 1.22 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.9484716, 0.9271985, -0.6976992, 0.7341775, -1.6826491, 1.6248977
1: -0.6663350, 0.6861410, -0.4281773, 0.4778989, -1.1442338, 1.1143183
2: -0.6683402, 1.0286002, -0.4390219, 0.7077794, -1.3761196, 1.4676222
3: -0.7294354, 0.7094818, -0.4023831, 0.6564106, -1.3858460, 1.1118650
4: -0.8408902, 0.8616544, -0.6893057, 0.5338082, -1.3746984, 1.5509601
5: -0.4445019, 1.1990155, -0.0544565, 1.1607451, -1.6052470, 1.2534721
6: -0.6542090, 0.8314723, -0.5218369, 0.6879833, -1.3421923, 1.3533092
7: -0.7666821, 0.8810155, -0.5290595, 0.5232809, -1.2899630, 1.4100749
8: -0.9236817, 0.9505574, -0.7039664, 0.7948344, -1.7185161, 1.6545237
9: -0.8086766, 0.8599146, -0.5568695, 0.6412189, -1.4498955, 1.4167840

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7332664, upper bound: 1.7139215
time: 1.51 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7139773, upper bound: 1.7139215
time: 1.41 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.5777271, 0.6454557, -0.4500548, 0.4871559, -1.0648830, 1.0955105
1: -0.4082001, 0.4237340, -0.2888094, 0.3019510, -0.7101511, 0.7125434
2: -0.3948611, 0.6335298, -0.2914829, 0.4390927, -0.8339539, 0.9250126
3: -0.3592333, 0.5288207, -0.2646321, 0.4328157, -0.7920491, 0.7934527
4: -0.5486000, 0.5331741, -0.4199123, 0.3862811, -0.9348811, 0.9530864
5: 0.0183229, 1.1275945, 0.2640836, 1.1034373, -1.0851145, 0.8635108
6: -0.4449049, 0.5756347, -0.3399446, 0.4558409, -0.9007457, 0.9155793
7: -0.4722534, 0.5364712, -0.3642073, 0.3579234, -0.8301768, 0.9006785
8: -0.5953174, 0.6788825, -0.4467654, 0.5433170, -1.1386343, 1.1256479
9: -0.5024979, 0.5685350, -0.3552836, 0.4253681, -0.9278660, 0.9238186

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7702719, upper bound: 1.7512964
time: 1.51 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7702719, upper bound: 1.7512964
time: 1.54 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.5087428, 0.5580543, -0.8403907, 1.0221305, -1.5308733, 1.3984450
1: -0.3413355, 0.3598669, -0.6695250, 0.6895621, -1.0308976, 1.0293919
2: -0.3422632, 0.5186803, -0.6167685, 1.1169431, -1.4592063, 1.1354489
3: -0.3060901, 0.4785177, -0.5917076, 0.7276000, -1.0336901, 1.0702252
4: -0.4775078, 0.4589083, -0.8144474, 0.8580497, -1.3355575, 1.2733557
5: 0.1516745, 1.1150615, -0.5348240, 1.1777781, -1.0261036, 1.6498854
6: -0.3881324, 0.5129803, -0.6773108, 0.8278437, -1.2159761, 1.1902912
7: -0.4129821, 0.4408068, -0.7249638, 0.9372095, -1.3501916, 1.1657706
8: -0.5153525, 0.6052886, -0.9432293, 0.9402919, -1.4556444, 1.5485179
9: -0.4267589, 0.4888596, -0.8184288, 0.8930537, -1.3198127, 1.3072884

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6618854, upper bound: 1.6166290
time: 1.46 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7700814, upper bound: 1.7512964
time: 1.39 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7700814, upper bound: 1.7512964
time: 1.45 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.3992282, 1.1363866, -0.4177500, 0.4543522, -1.8535805, 1.5541366
1: -0.8818427, 0.9647343, -0.2636158, 0.2729289, -1.1547716, 1.2283502
2: -1.0091600, 1.2793995, -0.2641783, 0.4071663, -1.4163263, 1.5435778
3: -1.1861728, 0.8378342, -0.2464507, 0.4061916, -1.5923644, 1.0842849
4: -1.2120326, 1.1662004, -0.3879020, 0.3470456, -1.5590782, 1.5541024
5: -0.9585391, 1.2560433, 0.3181673, 1.0966275, -2.0551665, 0.9378760
6: -1.0109549, 1.0762472, -0.3155059, 0.4227584, -1.4337133, 1.3917531
7: -1.0298045, 1.2240785, -0.3400748, 0.3171143, -1.3469187, 1.5641534
8: -1.3506910, 1.1217434, -0.4131727, 0.5073315, -1.8580225, 1.5349162
9: -1.0727117, 1.1331306, -0.3143627, 0.3982422, -1.4709539, 1.4474933

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7512964
time: 1.16 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7512964
time: 1.20 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.1956108, 1.0247967, -0.7820492, 0.8181220, -2.0137329, 1.8068459
1: -0.7700427, 0.8336775, -0.5385702, 0.6271665, -1.3972092, 1.3722477
2: -0.8564810, 1.1333063, -0.5868679, 0.7555995, -1.6120805, 1.7201742
3: -0.9777690, 0.7670926, -0.4540498, 0.6968522, -1.6746211, 1.2211423
4: -1.0494428, 1.0132229, -0.7372514, 0.8034682, -1.8529110, 1.7504743
5: -0.7229491, 1.2252742, -0.3160604, 1.1709449, -1.8938941, 1.5413346
6: -0.8691661, 0.9564576, -0.5944111, 0.7838102, -1.6529763, 1.5508687
7: -0.8957089, 1.0605216, -0.6089189, 0.7939831, -1.6896920, 1.6694405
8: -1.1631852, 1.0219591, -0.7848493, 0.9038017, -2.0669870, 1.8068084
9: -0.9374480, 0.9998186, -0.7665206, 0.7336847, -1.6711326, 1.7663391

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7338453, upper bound: 1.7143948
time: 1.52 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7139773, upper bound: 1.7143948
time: 1.14 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.9165044, 0.8667363, -0.4348741, 0.4716856, -1.3881900, 1.3016104
1: -0.6231701, 0.6501805, -0.2771350, 0.2874843, -0.9106544, 0.9273155
2: -0.6574708, 0.9239666, -0.2783806, 0.4242984, -1.0817692, 1.2023473
3: -0.7061045, 0.6697813, -0.2560419, 0.4204741, -1.1265787, 0.9258232
4: -0.8313135, 0.8050882, -0.4050790, 0.3675855, -1.1988989, 1.2101672
5: -0.3957884, 1.1912724, 0.2900147, 1.1002817, -1.4960701, 0.9012576
6: -0.6710997, 0.7966892, -0.3282793, 0.4405108, -1.1116105, 1.1249685
7: -0.7154753, 0.8285961, -0.3527689, 0.3384465, -1.0539218, 1.1813650
8: -0.9083625, 0.8833411, -0.4309625, 0.5265746, -1.4349371, 1.3143036
9: -0.7560694, 0.8158063, -0.3360614, 0.4120893, -1.1681588, 1.1518676

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7837333, upper bound: 1.7512964
time: 1.47 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7762218, upper bound: 1.7512964
time: 1.64 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.9165044, 0.8667363, -0.8131564, 0.8491322, -1.7656367, 1.6798928
1: -0.6231701, 0.6501805, -0.5619724, 0.6576376, -1.2808077, 1.2121528
2: -0.6574708, 0.9239666, -0.6145447, 0.7852552, -1.4427260, 1.5385113
3: -0.7061045, 0.6697813, -0.4717974, 0.7215915, -1.4276960, 1.1415787
4: -0.8313135, 0.8050882, -0.7669849, 0.8425576, -1.6738710, 1.5720731
5: -0.3957884, 1.1912724, -0.3704182, 1.1772702, -1.5730586, 1.5616906
6: -0.6710997, 0.7966892, -0.6182539, 0.8145400, -1.4856398, 1.4149432
7: -0.7154753, 0.8285961, -0.6318475, 0.8348404, -1.5503157, 1.4604435
8: -0.9083625, 0.8833411, -0.8165267, 0.9375783, -1.8459408, 1.6998677
9: -0.7560694, 0.8158063, -0.8050525, 0.7625726, -1.5186421, 1.6208587

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7837333, upper bound: 1.7512964
time: 1.47 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7762218, upper bound: 1.7512964
time: 1.52 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -2.3329704, 1.6747909, -0.3616624, 0.3973238, -2.7302942, 2.0364532
1: -1.5758646, 1.6758221, -0.2232236, 0.2338910, -1.8097556, 1.8990457
2: -1.9514822, 1.8812972, -0.2228762, 0.3472049, -2.2986870, 2.1041734
3: -2.4631681, 1.1982279, -0.2134655, 0.3553641, -2.8185322, 1.4116933
4: -2.1915371, 1.9186187, -0.3298180, 0.2937776, -2.4853148, 2.2484367
5: -2.1037424, 1.6249282, 0.4043077, 1.0838754, -3.1876178, 1.2206206
6: -1.6933944, 1.8981869, -0.2745202, 0.3662076, -2.0596020, 2.1727071
7: -1.7982666, 1.9708970, -0.3008205, 0.2598947, -2.0581613, 2.2717175
8: -2.3502164, 1.6482879, -0.3539089, 0.4470941, -2.7973106, 2.0021968
9: -1.9131836, 1.7756097, -0.2578763, 0.3487231, -2.2619066, 2.0334861

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6594685, upper bound: 1.6353639
time: 1.67 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6173412, upper bound: 1.6288816
time: 1.13 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2.1862803, 1.5905819, -0.6020399, 0.6465716, -2.8328519, 2.1926217
1: -1.4767163, 1.5691009, -0.4091116, 0.4413412, -1.9180574, 1.9782124
2: -1.8168724, 1.7822115, -0.4172078, 0.5915440, -2.4084163, 2.1994193
3: -2.2807474, 1.1433141, -0.3496748, 0.5599939, -2.8407412, 1.4929889
4: -2.0500419, 1.8028733, -0.5727646, 0.5683267, -2.6183686, 2.3756378
5: -1.9264994, 1.5796956, 0.0125103, 1.1359535, -3.0624528, 1.5671853
6: -1.5869720, 1.7833333, -0.4571294, 0.6138134, -2.2007854, 2.2404628
7: -1.6854125, 1.8522979, -0.4820774, 0.5466938, -2.2321062, 2.3343754
8: -2.2006030, 1.5688472, -0.6096100, 0.7144287, -2.9150317, 2.1784573
9: -1.7930329, 1.6760440, -0.5533638, 0.5472695, -2.3403025, 2.2294078

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6583088, upper bound: 1.6160434
time: 1.46 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6173412, upper bound: 1.6160434
time: 1.25 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1.0371654, 0.9307392, -0.4983442, 0.5435746, -1.5807400, 1.4290833
1: -0.6999609, 0.7333237, -0.3310897, 0.3495798, -1.0495406, 1.0644133
2: -0.7625598, 1.0057282, -0.3336982, 0.4999993, -1.2625592, 1.3394264
3: -0.8518399, 0.7131721, -0.2971787, 0.4707153, -1.3225552, 1.0103507
4: -0.9422083, 0.8958738, -0.4669205, 0.4464545, -1.3886628, 1.3627943
5: -0.5382136, 1.2253131, 0.1730944, 1.1130850, -1.6512985, 1.0522187
6: -0.7542307, 0.8839969, -0.3791008, 0.5031562, -1.2573869, 1.2630978
7: -0.8013637, 0.9229009, -0.4032480, 0.4252506, -1.2266144, 1.3261487
8: -1.0285857, 0.9466708, -0.5019848, 0.5948360, -1.6234217, 1.4486556
9: -0.8516646, 0.8962675, -0.4144765, 0.4762483, -1.3279129, 1.3107440

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7843920, upper bound: 1.7512964
time: 1.45 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7764832, upper bound: 1.7512964
time: 1.53 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.0371654, 0.9307392, -0.9562910, 1.0999165, -2.1370819, 1.8870301
1: -0.6999609, 0.7333237, -0.7595167, 0.7688740, -1.4688349, 1.4928404
2: -0.7625598, 1.0057282, -0.6675311, 1.2470318, -2.0095916, 1.6732594
3: -0.8518399, 0.7131721, -0.6348488, 0.8034844, -1.6553243, 1.3480209
4: -0.9422083, 0.8958738, -0.9456645, 0.9183384, -1.8605468, 1.8415383
5: -0.5382136, 1.2253131, -0.6767565, 1.1951406, -1.7333541, 1.9020696
6: -0.7542307, 0.8839969, -0.7393686, 0.9230158, -1.6772466, 1.6233655
7: -0.8013637, 0.9229009, -0.7810123, 1.0356488, -1.8370125, 1.7039132
8: -1.0285857, 0.9466708, -1.0100383, 1.0840518, -2.1126375, 1.9567090
9: -0.8516646, 0.8962675, -0.9041129, 0.9860164, -1.8376811, 1.8003805

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7843920, upper bound: 1.7512964
time: 1.53 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7764832, upper bound: 1.7512964
time: 1.42 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -2.4414310, 1.7426924, -0.4177955, 0.4543944, -2.8958254, 2.1604879
1: -1.6566067, 1.7496475, -0.2636518, 0.2729596, -1.9295663, 2.0132992
2: -2.0725601, 1.9480190, -0.2642159, 0.4072118, -2.4797719, 2.2122350
3: -2.5819435, 1.2523350, -0.2464760, 0.4062294, -2.9881730, 1.4988110
4: -2.2926061, 2.0461009, -0.3879476, 0.3471000, -2.6397061, 2.4340484
5: -2.2475772, 1.6612236, 0.3180945, 1.0966372, -3.3442144, 1.3431292
6: -1.7688253, 1.9909477, -0.3155370, 0.4228052, -2.1916306, 2.3064847
7: -1.9021331, 2.0514598, -0.3401050, 0.3171713, -2.2193043, 2.3915648
8: -2.4593999, 1.7594962, -0.4132167, 0.5073823, -2.9667823, 2.1727128
9: -1.9830389, 1.8792311, -0.3144169, 0.3982788, -2.3813176, 2.1936479

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 96

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6604711, upper bound: 1.6354306
time: 1.26 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6173412, upper bound: 1.6291783
time: 2.37 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2.2938447, 1.6573281, -0.6945968, 0.8181510, -3.1119957, 2.3519249
1: -1.5559899, 1.6428618, -0.5252703, 0.5446478, -2.1006377, 2.1681323
2: -1.9346390, 1.8490987, -0.4961126, 0.8539770, -2.7886159, 2.3452113
3: -2.4002762, 1.1955409, -0.4661674, 0.6181058, -3.0183821, 1.6617084
4: -2.1506982, 1.9247606, -0.6663173, 0.6826116, -2.8333097, 2.5910778
5: -2.0678282, 1.6153717, -0.2330813, 1.1502033, -3.2180314, 1.8484530
6: -1.6621143, 1.8745307, -0.5502017, 0.6894489, -2.3515632, 2.4247322
7: -1.7862190, 1.9329890, -0.5878357, 0.7189922, -2.5052114, 2.5208247
8: -2.3090949, 1.6734796, -0.7551533, 0.7930447, -3.1021395, 2.4286327
9: -1.8643649, 1.7755923, -0.6462481, 0.7153944, -2.5797591, 2.4218404

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6600046, upper bound: 1.6173412
time: 1.77 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6173412, upper bound: 1.6173412
time: 1.24 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.93 seconds
NS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7695456, upper bound: 1.7512964
NS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7695456, upper bound: 1.7512964
NS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7424588, upper bound: 1.7139215
NS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7368981, upper bound: 1.7139215
NS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7513144, upper bound: 1.7512964
NS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7513144, upper bound: 1.7512964
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7328974, upper bound: 1.7139215
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7139773, upper bound: 1.7139215
NS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7702454, upper bound: 1.7512964
NS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7702454, upper bound: 1.7512964
NS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7448230, upper bound: 1.7143948
NS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7383351, upper bound: 1.7143948
NS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7513144, upper bound: 1.7512964
NS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7513144, upper bound: 1.7512964
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7335013, upper bound: 1.7143948
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7139773, upper bound: 1.7143948
NS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7836622, upper bound: 1.7512964
NS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7759568, upper bound: 1.7512964
NS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7836622, upper bound: 1.7512964
NS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7759568, upper bound: 1.7512964
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7350633, upper bound: 1.7398275
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7388026
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7329714, upper bound: 1.7139215
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7139215
NS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7842880, upper bound: 1.7512964
NS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7761746, upper bound: 1.7512964
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7842880, upper bound: 1.7512964
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7761746, upper bound: 1.7512964
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7355126, upper bound: 1.7398275
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7388026
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7335611, upper bound: 1.7143948
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7143948
NS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7698134, upper bound: 1.7512964
NS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7698134, upper bound: 1.7512964
NS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7433118, upper bound: 1.7139215
NS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7376053, upper bound: 1.7139215
NS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7513144, upper bound: 1.7512964
NS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7513144, upper bound: 1.7512964
NS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7332664, upper bound: 1.7139215
NS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7139773, upper bound: 1.7139215
NS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7704867, upper bound: 1.7512964
NS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7704867, upper bound: 1.7512964
NS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7457908, upper bound: 1.7143948
NS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7388940, upper bound: 1.7143948
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7513144, upper bound: 1.7512964
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7513144, upper bound: 1.7512964
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7338453, upper bound: 1.7143948
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7139773, upper bound: 1.7143948
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7839076, upper bound: 1.7512964
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7763118, upper bound: 1.7512964
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7839076, upper bound: 1.7512964
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7763118, upper bound: 1.7512964
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7353591, upper bound: 1.7398275
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7388026
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7333309, upper bound: 1.7139215
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7139215
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7845807, upper bound: 1.7512964
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7766255, upper bound: 1.7512964
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7845807, upper bound: 1.7512964
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7766255, upper bound: 1.7512964
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7357365, upper bound: 1.7398275
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7388026
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7339313, upper bound: 1.7143948
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7143948
NS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7694138, upper bound: 1.7512964
NS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7694138, upper bound: 1.7512964
NS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7423393, upper bound: 1.7139215
NS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7368606, upper bound: 1.7139215
NS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7512964
NS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7512964
NS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7328974, upper bound: 1.7139215
NS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7139773, upper bound: 1.7139215
NS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7700814, upper bound: 1.7512964
NS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7700814, upper bound: 1.7512964
NS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7447926, upper bound: 1.7143948
NS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7383228, upper bound: 1.7143948
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7512964
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7512964
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7335013, upper bound: 1.7143948
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7139773, upper bound: 1.7143948
NS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7694138, upper bound: 1.7512964
NS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7694138, upper bound: 1.7512964
NS_A2_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.6651781, upper bound: 1.6160434
NS_A2_B1_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.6271045, upper bound: 1.6160434
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7512964, upper bound: 1.7512964
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7512964, upper bound: 1.7512964
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7329528, upper bound: 1.7139215
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7139215
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7700814, upper bound: 1.7512964
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7700814, upper bound: 1.7512964
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.6674094, upper bound: 1.6172222
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.6287729, upper bound: 1.6172222
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7512964, upper bound: 1.7512964
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7512964, upper bound: 1.7512964
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7335601, upper bound: 1.7143948
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7143948, upper bound: 1.7143948
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7696675, upper bound: 1.7512964
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7696675, upper bound: 1.7512964
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7694138, upper bound: 1.7512964
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7694138, upper bound: 1.7512964
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7512964
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7512964
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7332664, upper bound: 1.7139215
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7139773, upper bound: 1.7139215
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7702719, upper bound: 1.7512964
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7702719, upper bound: 1.7512964
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7700814, upper bound: 1.7512964
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7700814, upper bound: 1.7512964
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7512964
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7511661, upper bound: 1.7512964
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7338453, upper bound: 1.7143948
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7139773, upper bound: 1.7143948
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7837333, upper bound: 1.7512964
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7762218, upper bound: 1.7512964
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7837333, upper bound: 1.7512964
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7762218, upper bound: 1.7512964
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.6594685, upper bound: 1.6353639
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.6173412, upper bound: 1.6288816
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.6583088, upper bound: 1.6160434
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.6173412, upper bound: 1.6160434
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7843920, upper bound: 1.7512964
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7764832, upper bound: 1.7512964
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7843920, upper bound: 1.7512964
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.7764832, upper bound: 1.7512964
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.6604711, upper bound: 1.6354306
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.6173412, upper bound: 1.6291783
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.6600046, upper bound: 1.6173412
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.93
Output dim: 5, lower bound: -1.6173412, upper bound: 1.6173412

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.3186191, 0.3461073, -0.2841525, 0.3047394, -0.6233585, 0.6302598
1: -0.1918767, 0.2038621, -0.1664372, 0.1795737, -0.3714504, 0.3702993
2: -0.1915888, 0.2988588, -0.1665321, 0.2604349, -0.4520236, 0.4653909
3: -0.1844280, 0.3107483, -0.1610441, 0.2747474, -0.4591754, 0.4717924
4: -0.2884993, 0.2568841, -0.2557738, 0.2269287, -0.5154281, 0.5126579
5: 0.4743142, 1.0769240, 0.5311549, 1.0722203, -0.5979061, 0.5457691
6: -0.2408757, 0.3201970, -0.2143452, 0.2828397, -0.5237154, 0.5345422
7: -0.2668250, 0.2291495, -0.2396425, 0.2050259, -0.4718508, 0.4687920
8: -0.3042862, 0.3999128, -0.2649239, 0.3616051, -0.6658913, 0.6648368
9: -0.2244241, 0.3037654, -0.1975531, 0.2674185, -0.4918426, 0.5013185

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6025224, upper bound: 1.6168659
time: 1.36 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7566303, upper bound: 1.7589030
time: 1.40 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7625071, upper bound: 1.7669439
time: 1.34 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.5889295, 0.6713803, -0.2841525, 0.3047394, -0.8936689, 0.9555328
1: -0.3890533, 0.3940852, -0.1664372, 0.1795737, -0.5686270, 0.5605224
2: -0.3932304, 0.6012533, -0.1665321, 0.2604349, -0.6536653, 0.7677854
3: -0.3671647, 0.5932363, -0.1610441, 0.2747474, -0.6419121, 0.7542804
4: -0.5533757, 0.4890614, -0.2557738, 0.2269287, -0.7803044, 0.7448353
5: 0.0337530, 1.1277156, 0.5311549, 1.0722203, -1.0384674, 0.5965607
6: -0.4579781, 0.6097464, -0.2143452, 0.2828397, -0.7408178, 0.8240916
7: -0.4865106, 0.4207528, -0.2396425, 0.2050259, -0.6915364, 0.6603953
8: -0.6234852, 0.6968307, -0.2649239, 0.3616051, -0.9850903, 0.9617547
9: -0.4362478, 0.5866913, -0.1975531, 0.2674185, -0.7036663, 0.7842444

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6025224, upper bound: 1.6168659
time: 1.57 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7566303, upper bound: 1.7589030
time: 1.27 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7625071, upper bound: 1.7669439
time: 1.42 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.2065610, 0.2223649, -0.5309615, 0.5979823, -0.8045433, 0.7533264
1: -0.1124054, 0.1242189, -0.3467688, 0.3517590, -0.4641645, 0.4709877
2: -0.1131593, 0.1744100, -0.3441505, 0.5364050, -0.6495643, 0.5185605
3: -0.1096248, 0.2022735, -0.3279769, 0.5299452, -0.6395699, 0.5302504
4: -0.1866502, 0.1582439, -0.4882812, 0.4392710, -0.6259212, 0.6465251
5: 0.6551478, 1.0619470, 0.1282315, 1.1064503, -0.4513025, 0.9337155
6: -0.1575987, 0.2012568, -0.4047512, 0.5476527, -0.7052513, 0.6060079
7: -0.1809687, 0.1525272, -0.4323291, 0.3796634, -0.5606321, 0.5848563
8: -0.1793091, 0.2737679, -0.5446129, 0.6331566, -0.8124657, 0.8183808
9: -0.1372844, 0.1878947, -0.3880322, 0.5260178, -0.6633022, 0.5759269

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7424588, upper bound: 1.7139215
time: 1.51 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7424588, upper bound: 1.7139215
time: 1.35 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.2307049, 0.2435391, -0.4844916, 0.5428609, -0.7735658, 0.7280307
1: -0.1284354, 0.1412170, -0.3128715, 0.3193929, -0.4478283, 0.4540884
2: -0.1282806, 0.2011781, -0.3107634, 0.4844193, -0.6126999, 0.5119414
3: -0.1245774, 0.2229905, -0.2965620, 0.4819753, -0.6065527, 0.5195525
4: -0.2069103, 0.1796161, -0.4445601, 0.3993565, -0.6062668, 0.6241762
5: 0.6188188, 1.0651437, 0.2039695, 1.0999887, -0.4811699, 0.8611742
6: -0.1747583, 0.2238359, -0.3688880, 0.4978753, -0.6726336, 0.5927240
7: -0.1985607, 0.1683695, -0.3961094, 0.3467243, -0.5452850, 0.5644789
8: -0.2041771, 0.3010995, -0.4920187, 0.5821127, -0.7862897, 0.7931182
9: -0.1551118, 0.2120445, -0.3522276, 0.4773791, -0.6324909, 0.5642720

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7368980, upper bound: 1.7139215
time: 1.46 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7368980, upper bound: 1.7139215
time: 1.50 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.5495033, 0.5770528, -0.2602990, 0.2755789, -0.8250822, 0.8373518
1: -0.3676340, 0.3628061, -0.1485049, 0.1624553, -0.5300893, 0.5113110
2: -0.3731135, 0.5389824, -0.1488694, 0.2339886, -0.6071020, 0.6878518
3: -0.3202479, 0.5159102, -0.1447690, 0.2493703, -0.5696182, 0.6606792
4: -0.5200647, 0.5046529, -0.2327991, 0.2058132, -0.7258779, 0.7374520
5: 0.1065861, 1.1247424, 0.5712220, 1.0690620, -0.9624759, 0.5535204
6: -0.4059800, 0.5593482, -0.1960593, 0.2565064, -0.6624863, 0.7554075
7: -0.4276201, 0.4812457, -0.2204816, 0.1886661, -0.6162862, 0.7017273
8: -0.5406832, 0.6553902, -0.2372956, 0.3346015, -0.8752847, 0.8926858
9: -0.4710177, 0.5047852, -0.1786115, 0.2419662, -0.7129839, 0.6833967

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7415464, upper bound: 1.7571397
time: 1.53 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7416587, upper bound: 1.7650170
time: 1.26 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.9019167, 0.9052532, -0.2602990, 0.2755789, -1.1774956, 1.1655521
1: -0.6458625, 0.6032109, -0.1485049, 0.1624553, -0.8083178, 0.7517157
2: -0.6644934, 0.8915642, -0.1488694, 0.2339886, -0.8984821, 1.0404336
3: -0.5176414, 0.8093861, -0.1447690, 0.2493703, -0.7670117, 0.9541552
4: -0.8735736, 0.9262209, -0.2327991, 0.2058132, -1.0793867, 1.1590199
5: -0.4593520, 1.1999446, 0.5712220, 1.0690620, -1.5284140, 0.6287226
6: -0.6479807, 0.9246990, -0.1960593, 0.2565064, -0.9044871, 1.1207583
7: -0.6617872, 0.9202651, -0.2204816, 0.1886661, -0.8504533, 1.1407467
8: -0.8817483, 1.0514174, -0.2372956, 0.3346015, -1.2163498, 1.2887130
9: -0.8900387, 0.7897670, -0.1786115, 0.2419662, -1.1320050, 0.9683785

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7415464, upper bound: 1.7571397
time: 1.45 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7416587, upper bound: 1.7650170
time: 1.33 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.3965878, 0.4345518, -0.4998342, 0.5684035, -0.9649913, 0.9343860
1: -0.2470385, 0.2584251, -0.3285791, 0.3343570, -0.5813954, 0.5870043
2: -0.2470157, 0.3858946, -0.3262347, 0.4995582, -0.7465739, 0.7121292
3: -0.2345674, 0.3884861, -0.3082007, 0.5042041, -0.7387714, 0.6966869
4: -0.3665747, 0.3223782, -0.4635076, 0.4178526, -0.7844273, 0.7858859
5: 0.3520085, 1.0920904, 0.1688732, 1.1007766, -0.7487680, 0.9232172
6: -0.3009058, 0.4009414, -0.3796828, 0.5209416, -0.8218474, 0.7806243
7: -0.3260447, 0.2906284, -0.4128934, 0.3529478, -0.6789925, 0.7035218
8: -0.3925961, 0.4837622, -0.5147362, 0.6057661, -0.9983621, 0.9984984
9: -0.2893819, 0.3810487, -0.3688192, 0.4975553, -0.7869371, 0.7498679

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7328936, upper bound: 1.7139215
time: 1.47 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7328936, upper bound: 1.7139215
time: 1.61 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.4279771, 0.4638764, -0.4544701, 0.5129472, -0.9409243, 0.9183465
1: -0.2716899, 0.2799053, -0.2944761, 0.3018015, -0.5734913, 0.5743814
2: -0.2726341, 0.4173981, -0.2926445, 0.4492637, -0.7218977, 0.7100426
3: -0.2521788, 0.4147081, -0.2772495, 0.4559428, -0.7081217, 0.6919576
4: -0.3981605, 0.3592796, -0.4198154, 0.3776959, -0.7758564, 0.7790949
5: 0.3017437, 1.0988098, 0.2450715, 1.0947703, -0.7930266, 0.8537384
6: -0.3225286, 0.4333607, -0.3449076, 0.4708619, -0.7933905, 0.7782682
7: -0.3468702, 0.3298548, -0.3764538, 0.3218356, -0.6687058, 0.7063085
8: -0.4230703, 0.5188240, -0.4621935, 0.5544119, -0.9774821, 0.9810176
9: -0.3265229, 0.4065121, -0.3327970, 0.4491508, -0.7756737, 0.7393091

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7139773, upper bound: 1.7139215
time: 1.27 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7139773, upper bound: 1.7139215
time: 1.25 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.3417125, 0.3735000, -0.3354237, 0.3660403, -0.7077528, 0.7089237
1: -0.2087220, 0.2199466, -0.2041347, 0.2155663, -0.4242884, 0.4240812
2: -0.2081806, 0.3246932, -0.2036622, 0.3176579, -0.5258386, 0.5283554
3: -0.2000396, 0.3345872, -0.1957882, 0.3280952, -0.5281348, 0.5303754
4: -0.3102267, 0.2767195, -0.3043098, 0.2713179, -0.5815446, 0.5810294
5: 0.4366760, 1.0801353, 0.4469257, 1.0792607, -0.6425847, 0.6332096
6: -0.2586981, 0.3449339, -0.2538445, 0.3381976, -0.5968957, 0.5987784
7: -0.2848243, 0.2455187, -0.2799227, 0.2410610, -0.5258853, 0.5254414
8: -0.3304229, 0.4252793, -0.3233053, 0.4183714, -0.7487943, 0.7485846
9: -0.2422172, 0.3279364, -0.2373718, 0.3213540, -0.5635712, 0.5653082

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6055159, upper bound: 1.6216735
time: 1.44 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5926848, upper bound: 1.5822255
time: 1.31 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.6111618, 0.6910132, -0.3354237, 0.3660403, -0.9772022, 1.0264369
1: -0.4019148, 0.4055612, -0.2041347, 0.2155663, -0.6174811, 0.6096959
2: -0.4054343, 0.6389198, -0.2036622, 0.3176579, -0.7230922, 0.8425820
3: -0.3781755, 0.6154261, -0.1957882, 0.3280952, -0.7062707, 0.8112142
4: -0.5892073, 0.5030512, -0.3043098, 0.2713179, -0.8605251, 0.8073611
5: 0.0072072, 1.1454206, 0.4469257, 1.0792607, -1.0720536, 0.6984949
6: -0.4716428, 0.6271932, -0.2538445, 0.3381976, -0.8098404, 0.8810377
7: -0.4998133, 0.4520029, -0.2799227, 0.2410610, -0.7408743, 0.7319256
8: -0.6591105, 0.7147214, -0.3233053, 0.4183714, -1.0774820, 1.0380267
9: -0.4591482, 0.6037389, -0.2373718, 0.3213540, -0.7805022, 0.8411106

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6055159, upper bound: 1.6216735
time: 1.75 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5926848, upper bound: 1.5822255
time: 1.19 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.2252192, 0.2377484, -0.5771497, 0.6509775, -0.8761967, 0.8148982
1: -0.1247933, 0.1372805, -0.3775555, 0.3821591, -0.5069524, 0.5148360
2: -0.1245117, 0.1950963, -0.3805478, 0.5991530, -0.7236648, 0.5756441
3: -0.1208347, 0.2182835, -0.3557221, 0.5799745, -0.7008092, 0.5740056
4: -0.2022053, 0.1747603, -0.5538473, 0.4745228, -0.6767281, 0.7286075
5: 0.6275652, 1.0644174, 0.0613400, 1.1370306, -0.5094654, 1.0030773
6: -0.1708596, 0.2177801, -0.4447709, 0.5916156, -0.7624753, 0.6625509
7: -0.1945639, 0.1646073, -0.4726864, 0.4258147, -0.6203786, 0.6372936
8: -0.1980795, 0.2948899, -0.6175054, 0.6782383, -0.8763177, 0.9123953
9: -0.1507559, 0.2065577, -0.4317108, 0.5689750, -0.7197309, 0.6382685

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7448230, upper bound: 1.7143948
time: 1.50 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7448230, upper bound: 1.7143948
time: 1.55 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.2464728, 0.2601844, -0.5292673, 0.5946137, -0.8410865, 0.7894516
1: -0.1389045, 0.1525329, -0.3432618, 0.3492129, -0.4881173, 0.4957948
2: -0.1391143, 0.2186599, -0.3455122, 0.5431684, -0.6822827, 0.5641721
3: -0.1353357, 0.2365206, -0.3241116, 0.5300643, -0.6654000, 0.5606322
4: -0.2204350, 0.1935742, -0.5040665, 0.4343600, -0.6547950, 0.6976408
5: 0.5936760, 1.0672314, 0.1375498, 1.1252193, -0.5315433, 0.9296815
6: -0.1859652, 0.2412429, -0.4069400, 0.5415282, -0.7274934, 0.6481829
7: -0.2100499, 0.1791838, -0.4344966, 0.3889462, -0.5989960, 0.6136805
8: -0.2217036, 0.3189497, -0.5589325, 0.6268765, -0.8485801, 0.8778822
9: -0.1676327, 0.2278166, -0.3930835, 0.5200336, -0.6876663, 0.6209001

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7383351, upper bound: 1.7143948
time: 1.53 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7383351, upper bound: 1.7143948
time: 1.46 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.5795444, 0.6230376, -0.3116378, 0.3378260, -0.9173703, 0.9346753
1: -0.3913515, 0.4206364, -0.1867842, 0.1989995, -0.5903510, 0.6074206
2: -0.3985258, 0.5690379, -0.1865729, 0.2910487, -0.6895745, 0.7556108
3: -0.3370746, 0.5412187, -0.1797084, 0.3035414, -0.6406159, 0.7209271
4: -0.5501992, 0.5413132, -0.2819309, 0.2508874, -0.8010867, 0.8232440
5: 0.0498539, 1.1311532, 0.4856927, 1.0759530, -1.0260992, 0.6454606
6: -0.4397900, 0.5904924, -0.2354878, 0.3127186, -0.7525086, 0.8259801
7: -0.4646762, 0.5186697, -0.2613835, 0.2242010, -0.6888771, 0.7800533
8: -0.5855692, 0.6891490, -0.2963846, 0.3922443, -0.9778135, 0.9855336
9: -0.5241214, 0.5290782, -0.2190449, 0.2964582, -0.8205796, 0.7481232

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7415882, upper bound: 1.7571397
time: 1.36 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7416587, upper bound: 1.7650170
time: 1.28 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.9208385, 0.9800938, -0.3116378, 0.3378260, -1.2586645, 1.2917316
1: -0.6608014, 0.7347579, -0.1867842, 0.1989995, -0.8598009, 0.9215420
2: -0.6819620, 0.9104951, -0.1865729, 0.2910487, -0.9730107, 1.0970681
3: -0.5282401, 0.8260694, -0.1797084, 0.3035414, -0.8317815, 1.0057778
4: -0.8925545, 0.9511563, -0.2819309, 0.2508874, -1.1434419, 1.2330872
5: -0.5167128, 1.2039824, 0.4856927, 1.0759530, -1.5926659, 0.7182897
6: -0.7028564, 0.9443155, -0.2354878, 0.3127186, -1.0155751, 1.1798033
7: -0.7286785, 0.9438372, -0.2613835, 0.2242010, -0.9528795, 1.2052207
8: -0.9503047, 1.0726813, -0.2963846, 0.3922443, -1.3425491, 1.3690659
9: -0.9677765, 0.8050684, -0.2190449, 0.2964582, -1.2642348, 1.0241133

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7415882, upper bound: 1.7571397
time: 1.49 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7416587, upper bound: 1.7650170
time: 1.36 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.4232077, 0.4594809, -0.5479532, 0.6181374, -1.0413451, 1.0074341
1: -0.2679245, 0.2767471, -0.3591632, 0.3635938, -0.6315182, 0.6359103
2: -0.2686922, 0.4126267, -0.3563586, 0.5554133, -0.8241055, 0.7689852
3: -0.2495075, 0.4107373, -0.3394636, 0.5474854, -0.7969928, 0.7502009
4: -0.3933766, 0.3535760, -0.5042675, 0.4538656, -0.8472421, 0.8578434
5: 0.3093814, 1.0977920, 0.1005378, 1.1088128, -0.7994314, 0.9972543
6: -0.3192872, 0.4284164, -0.4178645, 0.5658538, -0.8851410, 0.8462809
7: -0.3437447, 0.3239130, -0.4455726, 0.3917075, -0.7354522, 0.7694857
8: -0.4184950, 0.5134647, -0.5638439, 0.6518206, -1.0703156, 1.0773087
9: -0.3208964, 0.4026552, -0.4011241, 0.5438025, -0.8646989, 0.8037794

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7335013, upper bound: 1.7143948
time: 1.60 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7335013, upper bound: 1.7143948
time: 1.66 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.4532987, 0.4909616, -0.5009768, 0.5624152, -1.0157139, 0.9919384
1: -0.2916813, 0.3044422, -0.3248965, 0.3308746, -0.6225560, 0.6293387
2: -0.2936820, 0.4427321, -0.3226073, 0.5028613, -0.7965432, 0.7653394
3: -0.2663621, 0.4358518, -0.3077064, 0.4989926, -0.7653546, 0.7435582
4: -0.4235611, 0.3897109, -0.4600699, 0.4135162, -0.8370773, 0.8497808
5: 0.2594286, 1.1042132, 0.1771017, 1.1022809, -0.8428522, 0.9271115
6: -0.3424811, 0.4596120, -0.3816103, 0.5155337, -0.8580148, 0.8412223
7: -0.3670212, 0.3613995, -0.4089583, 0.3584093, -0.7254305, 0.7703577
8: -0.4506528, 0.5472794, -0.5106764, 0.6002205, -1.0508733, 1.0579557
9: -0.3600122, 0.4269886, -0.3649292, 0.4946336, -0.8546457, 0.7919178

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7139773, upper bound: 1.7143948
time: 1.29 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7139773, upper bound: 1.7143948
time: 1.28 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.4260807, 0.4624867, -0.3035233, 0.3282011, -0.7542818, 0.7660099
1: -0.2701930, 0.2793913, -0.1808651, 0.1933476, -0.4635406, 0.4602565
2: -0.2710783, 0.4155010, -0.1807429, 0.2819711, -0.5530493, 0.5962439
3: -0.2511169, 0.4131350, -0.1742228, 0.2951651, -0.5462819, 0.5873578
4: -0.3962587, 0.3570263, -0.2742964, 0.2439177, -0.6401764, 0.6313227
5: 0.3046118, 1.0984052, 0.4989178, 1.0748248, -0.7702130, 0.5994874
6: -0.3215017, 0.4313949, -0.2292254, 0.3040268, -0.6255286, 0.6606203
7: -0.3459671, 0.3274928, -0.2550589, 0.2184494, -0.5644165, 0.5825517
8: -0.4215654, 0.5166932, -0.2872008, 0.3833311, -0.8048965, 0.8038940
9: -0.3246312, 0.4049788, -0.2127928, 0.2879649, -0.6125962, 0.6177715

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7404727, upper bound: 1.7545820
time: 1.31 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7378455, upper bound: 1.7254857
time: 1.49 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.7067714, 0.8351841, -0.2795037, 0.2990565, -1.0058279, 1.1146878
1: -0.5373161, 0.5567493, -0.1629424, 0.1762375, -0.7135535, 0.7196916
2: -0.5061879, 0.8759362, -0.1630899, 0.2552808, -0.7614686, 1.0390260
3: -0.4766508, 0.6272486, -0.1578724, 0.2698016, -0.7464524, 0.7851210
4: -0.6786870, 0.6972612, -0.2512964, 0.2228135, -0.9015005, 0.9485576
5: -0.2582780, 1.1525059, 0.5389634, 1.0716047, -1.3298826, 0.6135425
6: -0.5608160, 0.7010055, -0.2107814, 0.2777076, -0.8385237, 0.9117869
7: -0.5992866, 0.7372140, -0.2359083, 0.2018375, -0.8011241, 0.9731224
8: -0.7708586, 0.8053408, -0.2595397, 0.3563423, -1.1272008, 1.0648804
9: -0.6606261, 0.7302300, -0.1938616, 0.2624581, -0.9230843, 0.9240916

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 159

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7259185, upper bound: 1.7545039
time: 1.25 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7259185, upper bound: 1.7254857
time: 1.15 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.4260807, 0.4624867, -0.5880560, 0.6657064, -1.0917871, 1.0505427
1: -0.2701930, 0.2793913, -0.3884159, 0.3915254, -0.6617184, 0.6678072
2: -0.2710783, 0.4155010, -0.3851712, 0.6002763, -0.8713546, 0.8006722
3: -0.2511169, 0.4131350, -0.3665742, 0.5888826, -0.8399994, 0.7797092
4: -0.3962587, 0.3570263, -0.5419983, 0.4883110, -0.8845696, 0.8990246
5: 0.3046118, 1.0984052, 0.0351769, 1.1143891, -0.8097773, 1.0632284
6: -0.3215017, 0.4313949, -0.4488139, 0.6088108, -0.9303125, 0.8802088
7: -0.3459671, 0.3274928, -0.4768298, 0.4201335, -0.7661006, 0.8043226
8: -0.4215654, 0.5166932, -0.6092316, 0.6958710, -1.1174364, 1.1259248
9: -0.3246312, 0.4049788, -0.4320231, 0.5857768, -0.9104080, 0.8370019

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7812957, upper bound: 1.7511661
time: 1.49 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7812957, upper bound: 1.7512964
time: 1.57 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.7067714, 0.8351841, -0.5630671, 0.6360650, -1.3428364, 1.3982513
1: -0.5373161, 0.5567493, -0.3701880, 0.3741207, -0.9114368, 0.9269373
2: -0.5061879, 0.8759362, -0.3672175, 0.5723210, -1.0785089, 1.2431537
3: -0.4766508, 0.6272486, -0.3496810, 0.5630873, -1.0397382, 0.9769296
4: -0.6786870, 0.6972612, -0.5184876, 0.4668474, -1.1455344, 1.2157488
5: -0.2582780, 1.1525059, 0.0759046, 1.1109143, -1.3691924, 1.0766013
6: -0.5608160, 0.7010055, -0.4295287, 0.5820433, -1.1428593, 1.1305342
7: -0.5992866, 0.7372140, -0.4573529, 0.4024206, -1.0017072, 1.1945670
8: -0.7708586, 0.8053408, -0.5809494, 0.6684226, -1.4392812, 1.3862902
9: -0.6606261, 0.7302300, -0.4127695, 0.5596216, -1.2202477, 1.1429994

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7740251, upper bound: 1.7511661
time: 1.52 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7740251, upper bound: 1.7512964
time: 1.64 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.7249247, 0.8230141, -0.2265528, 0.2391559, -0.9640806, 1.0495669
1: -0.5453113, 0.5585808, -0.1256787, 0.1382372, -0.6835485, 0.6842594
2: -0.5012425, 0.8732260, -0.1254278, 0.1965747, -0.6978172, 0.9986538
3: -0.4669168, 0.6359927, -0.1217445, 0.2194276, -0.6863444, 0.7577372
4: -0.7035221, 0.6836587, -0.2033489, 0.1759407, -0.8794628, 0.8870076
5: -0.2530142, 1.1539848, 0.6254390, 1.0645941, -1.3176084, 0.5285457
6: -0.5596961, 0.7113572, -0.1718073, 0.2192521, -0.7789482, 0.8831645
7: -0.5928862, 0.7311102, -0.1955353, 0.1655217, -0.7584079, 0.9266455
8: -0.7573491, 0.8365558, -0.1995617, 0.2963992, -1.0537484, 1.0361176
9: -0.6594097, 0.7315862, -0.1518147, 0.2078913, -0.8673010, 0.8834008

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5751149, upper bound: 1.5897868
time: 1.36 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7180277, upper bound: 1.7207169
time: 1.55 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7256804, upper bound: 1.7307847
time: 1.44 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.7473945, 0.8499061, -0.1853635, 0.2073216, -0.9547161, 1.0352696
1: -0.5661144, 0.5790037, -0.0987827, 0.1089145, -0.6750289, 0.6777864
2: -0.5173921, 0.9095290, -0.1007548, 0.1508120, -0.6682041, 1.0102838
3: -0.4832259, 0.6522591, -0.0981273, 0.1832085, -0.6664344, 0.7503863
4: -0.7270383, 0.7064502, -0.1684681, 0.1411347, -0.8681731, 0.8749182
5: -0.2941671, 1.1579818, 0.6840756, 1.0590051, -1.3531723, 0.4739062
6: -0.5771453, 0.7319130, -0.1429003, 0.1846752, -0.7618205, 0.8748133
7: -0.6111565, 0.7606863, -0.1648513, 0.1386863, -0.7498427, 0.9255376
8: -0.7818900, 0.8605924, -0.1584528, 0.2497331, -1.0316231, 1.0190451
9: -0.6831746, 0.7562957, -0.1222637, 0.1678665, -0.8510411, 0.8785595

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7017910, upper bound: 1.7126651
time: 1.68 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7058700, upper bound: 1.7329208
time: 1.29 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.6815262, 0.7710739, -0.4074079, 0.4514261, -1.1329523, 1.1784818
1: -0.5051315, 0.5191349, -0.2566433, 0.2657038, -0.7708353, 0.7757782
2: -0.4700508, 0.8031089, -0.2553810, 0.3981863, -0.8682371, 1.0584899
3: -0.4354166, 0.6045758, -0.2444515, 0.4024032, -0.8378198, 0.8490272
4: -0.6581025, 0.6396380, -0.3720361, 0.3331473, -0.9912498, 1.0116740
5: -0.1735306, 1.1462651, 0.3296032, 1.0892701, -1.2628007, 0.8166620
6: -0.5259938, 0.6716554, -0.3093986, 0.4153052, -0.9412990, 0.9810539
7: -0.5575982, 0.6739861, -0.3360289, 0.2920854, -0.8496836, 1.0100150
8: -0.7099513, 0.7901318, -0.4047763, 0.4974414, -1.2073927, 1.1949081
9: -0.6135091, 0.6838615, -0.2928351, 0.3966980, -1.0102072, 0.9766966

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7156535, upper bound: 1.6920847
time: 1.51 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7235660, upper bound: 1.7023960
time: 1.60 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.71 + 598.99 = 602.70 seconds
