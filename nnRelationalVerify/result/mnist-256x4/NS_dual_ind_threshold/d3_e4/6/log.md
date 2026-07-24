## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 107.2381207338


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414)
1: (-55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092)
2: (-70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666)
3: (-81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354)
4: (-72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363)
5: (-62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670)
6: (-60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196)
7: (-69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519)
8: (-77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454)
9: (-60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.81 + 13.06 = 13.88 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -107.3454662, upper bound: 107.3454661

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3417672, upper bound: 107.3417143
time: 9.90 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3420384, upper bound: 107.3420384
time: 9.93 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 19.88 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 19.88
Output dim: 7, lower bound: -107.3417672, upper bound: 107.3417143
NS_A2, status: Status.UNKNOWN, split count: 1, time: 19.88
Output dim: 7, lower bound: -107.3420384, upper bound: 107.3420384

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -54.8241081, 43.0569534, -62.3754387, 48.9225540, -103.7466583, 105.4323883
1: -45.4302597, 38.5692978, -51.6440125, 43.8149452, -89.2451935, 90.2133102
2: -57.7149849, 35.3450928, -65.9461365, 40.6444244, -98.3594055, 101.2912292
3: -67.1298523, 31.9761009, -76.0626755, 36.6205292, -103.7503815, 108.0387650
4: -59.6581039, 45.5115700, -67.7385406, 51.6597900, -111.3178940, 113.2501068
5: -51.4531670, 39.7658691, -58.5638390, 45.3444405, -96.7975922, 98.3297119
6: -49.0709686, 50.1650696, -56.0134315, 56.7901382, -105.8611069, 106.1784897
7: -57.5313797, 40.3114319, -65.0433731, 46.6732025, -104.2045822, 105.3547974
8: -63.1841316, 42.7644730, -72.2916870, 48.8856812, -112.0698013, 115.0561600
9: -49.1056328, 49.1908188, -55.9541321, 55.9477806, -105.0534058, 105.1449509

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 186

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3376072, upper bound: 107.3375540
time: 12.07 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3374019, upper bound: 107.3373860
time: 8.96 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -59.6072922, 46.7624054, -61.8887863, 48.5410309, -108.1483231, 108.6511841
1: -49.3685532, 41.8875961, -51.2407303, 43.4870682, -92.8555984, 93.1283264
2: -62.8696785, 38.5685043, -65.4233093, 40.3090744, -103.1787567, 103.9918137
3: -72.8384476, 34.8074989, -75.4736023, 36.3298569, -109.1682968, 110.2810898
4: -64.8151855, 49.4113007, -67.2047501, 51.2609634, -116.0761337, 116.6160507
5: -55.9513435, 43.2402878, -58.1087341, 44.9828186, -100.9341583, 101.3490219
6: -53.4172859, 54.4014511, -55.5685501, 56.3615913, -109.7788696, 109.9700012
7: -62.3693047, 44.1060638, -64.5543137, 46.2743759, -108.6436768, 108.6603622
8: -68.8457184, 46.5383682, -71.7171097, 48.4900665, -117.3357849, 118.2554779
9: -53.4014587, 53.4184113, -55.5101318, 55.5003510, -108.9018097, 108.9285355

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 186

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3380958, upper bound: 107.3380639
time: 10.82 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3378541, upper bound: 107.3378541
time: 9.11 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 20.79 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 20.79
Output dim: 7, lower bound: -107.3376072, upper bound: 107.3375540
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 20.79
Output dim: 7, lower bound: -107.3374019, upper bound: 107.3373860
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 20.79
Output dim: 7, lower bound: -107.3380958, upper bound: 107.3380639
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 20.79
Output dim: 7, lower bound: -107.3378541, upper bound: 107.3378541

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -52.6905518, 41.4118538, -54.6219101, 42.9051781, -95.5957336, 96.0337677
1: -43.6704826, 37.0869293, -45.2573166, 38.4228401, -82.0933228, 82.3442459
2: -55.3603706, 33.7887115, -57.3757133, 35.0246086, -90.3849792, 91.1644211
3: -64.5969925, 30.6501904, -66.9068832, 31.7829323, -96.3799286, 97.5570602
4: -57.3458176, 43.7822838, -59.3403587, 45.3502693, -102.6960754, 103.1226349
5: -49.4605865, 38.1654396, -51.2983627, 39.5164604, -88.9770508, 89.4638062
6: -47.0748940, 48.3138199, -48.7769814, 50.0400314, -97.1149292, 97.0907822
7: -55.3987045, 38.4087181, -57.3142471, 39.8105240, -95.2092133, 95.7229538
8: -60.5871353, 41.0398254, -62.8441887, 42.5440636, -103.1311951, 103.8840179
9: -47.1273727, 47.2282677, -48.7850418, 48.8315811, -95.9589539, 96.0133057

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3340431, upper bound: 107.3339188
time: 11.58 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3341511, upper bound: 107.3340495
time: 10.71 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -51.1851387, 40.2538605, -56.7380714, 44.5696487, -95.7547684, 96.9919205
1: -42.4329605, 36.0440750, -47.0269127, 39.8604164, -82.2933731, 83.0709839
2: -53.7044792, 32.6984406, -59.5528107, 36.2725449, -89.9770203, 92.2512512
3: -62.8117332, 29.7094460, -69.5094070, 32.8987808, -95.7105103, 99.2188568
4: -55.7282257, 42.5663605, -61.6746216, 47.0832825, -102.8115082, 104.2409668
5: -48.0562744, 37.0401459, -53.2637100, 41.0070610, -89.0633392, 90.3038483
6: -45.6705399, 47.0133133, -50.6196404, 51.9884338, -97.6589737, 97.6329422
7: -53.9044685, 37.0769081, -59.5587616, 41.1660118, -95.0704803, 96.6356659
8: -58.7438278, 39.8295174, -65.1972275, 44.1343002, -102.8781281, 105.0267334
9: -45.7381973, 45.8624039, -50.6218033, 50.7077560, -96.4459534, 96.4841919

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3338760, upper bound: 107.3337831
time: 10.94 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3339796, upper bound: 107.3339023
time: 10.74 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -57.3415871, 45.0135689, -54.1888733, 42.5705223, -99.9120941, 99.2024384
1: -47.5052605, 40.3153954, -44.8996162, 38.1322594, -85.6375198, 85.2149963
2: -60.3663826, 36.9273415, -56.9133911, 34.7306137, -95.0970001, 93.8407288
3: -70.1589890, 33.4007454, -66.3815536, 31.5263767, -101.6853638, 99.7822952
4: -62.3586578, 47.5712204, -58.8679848, 45.0000343, -107.3586884, 106.4392090
5: -53.8304977, 41.5476456, -50.8984985, 39.2000084, -93.0305023, 92.4461441
6: -51.3014030, 52.4328156, -48.3817787, 49.6599503, -100.9613495, 100.8145905
7: -60.1086197, 42.1031799, -56.8758621, 39.4640274, -99.5726471, 98.9790344
8: -66.0932922, 44.6978951, -62.3362961, 42.2042427, -108.2975311, 107.0341949
9: -51.3131485, 51.3442154, -48.3921432, 48.4364891, -99.7496338, 99.7363586

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3245440, upper bound: 107.3234832
time: 12.09 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3355507, upper bound: 107.3355118
time: 11.34 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -55.7474937, 43.7890472, -56.3005142, 44.2303543, -99.9778442, 100.0895386
1: -46.1977310, 39.2137756, -46.6636848, 39.5637703, -85.7614899, 85.8774567
2: -58.6101570, 35.7745285, -59.0832672, 35.9720421, -94.5821838, 94.8577957
3: -68.2775879, 32.4036942, -68.9785995, 32.6393661, -100.9169464, 101.3822937
4: -60.6419258, 46.2825699, -61.1959229, 46.7266998, -107.3686218, 107.4784851
5: -52.3376350, 40.3575172, -52.8594971, 40.6844521, -93.0220871, 93.2170105
6: -49.8162346, 51.0577469, -50.2185631, 51.6035995, -101.4198303, 101.2763062
7: -58.5284157, 40.6985855, -59.1125145, 40.8113823, -99.3397903, 99.8110962
8: -64.1545715, 43.4093933, -64.6853714, 43.7879753, -107.9425507, 108.0947647
9: -49.8511887, 49.8970451, -50.2203064, 50.3061028, -100.1572876, 100.1173401

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3244045, upper bound: 107.3233709
time: 12.19 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3353131, upper bound: 107.3353131
time: 9.02 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 22.08 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.08
Output dim: 7, lower bound: -107.3340431, upper bound: 107.3339188
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.08
Output dim: 7, lower bound: -107.3341511, upper bound: 107.3340495
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.08
Output dim: 7, lower bound: -107.3338760, upper bound: 107.3337831
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.08
Output dim: 7, lower bound: -107.3339796, upper bound: 107.3339023
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.08
Output dim: 7, lower bound: -107.3245440, upper bound: 107.3234832
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.08
Output dim: 7, lower bound: -107.3355507, upper bound: 107.3355118
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.08
Output dim: 7, lower bound: -107.3244045, upper bound: 107.3233709
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.08
Output dim: 7, lower bound: -107.3353131, upper bound: 107.3353131

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -46.6159401, 36.7049217, -52.1708908, 41.0076027, -87.6235428, 88.8758087
1: -38.6145554, 32.8321266, -43.2261887, 36.7082939, -75.3228455, 76.0583038
2: -48.7213326, 29.4107475, -54.6926880, 33.2620239, -81.9833527, 84.1034393
3: -57.3054276, 26.8203430, -63.9776268, 30.2414761, -87.5469055, 90.7979660
4: -50.8449326, 38.8189316, -56.7176704, 43.3556824, -94.2006073, 95.5366058
5: -43.7398224, 33.6577110, -48.9964027, 37.6973495, -81.4371719, 82.6541138
6: -41.3974876, 42.9887314, -46.4930725, 47.8987617, -89.2962494, 89.4818039
7: -49.2833710, 33.1276932, -54.8502693, 37.6943359, -86.9776917, 87.9779663
8: -53.1887207, 36.1656380, -59.8693047, 40.5771103, -93.7658310, 96.0349350
9: -41.5366592, 41.6779327, -46.5395393, 46.6058655, -88.1425171, 88.2174683

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 186

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3287268, upper bound: 107.3285513
time: 12.66 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3278434, upper bound: 107.3277289
time: 11.40 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -47.5023804, 37.4042892, -51.6503563, 40.6064796, -88.1088562, 89.0546417
1: -39.3576546, 33.4578209, -42.7931900, 36.3474274, -75.7050705, 76.2510071
2: -49.6898308, 30.0215454, -54.1263504, 32.8929901, -82.5828094, 84.1478806
3: -58.3882256, 27.3430653, -63.3555527, 29.9225883, -88.3108139, 90.6986160
4: -51.8170547, 39.5421333, -56.1591988, 42.9309959, -94.7480469, 95.7013245
5: -44.5819283, 34.3070564, -48.5108566, 37.3118668, -81.8937988, 82.8179016
6: -42.2112885, 43.7845497, -46.0122032, 47.4446373, -89.6558990, 89.7967529
7: -50.2005844, 33.8522911, -54.3314476, 37.2524986, -87.4530792, 88.1837387
8: -54.2520142, 36.8589439, -59.2416573, 40.1611862, -94.4132004, 96.1005936
9: -42.3446922, 42.4713936, -46.0631943, 46.1322517, -88.4769363, 88.5345917

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 186

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3286003, upper bound: 107.3284089
time: 13.39 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3277898, upper bound: 107.3276740
time: 9.46 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -45.2929611, 35.6820488, -54.2515411, 42.6461372, -87.9390945, 89.9335861
1: -37.5158463, 31.9077377, -44.9673004, 38.1201477, -75.6359940, 76.8750381
2: -47.2686691, 28.4479141, -56.8293953, 34.4790916, -81.7477417, 85.2773132
3: -55.7199554, 25.9844303, -66.5405350, 31.3287773, -87.0487366, 92.5249634
4: -49.4177017, 37.7403030, -59.0132408, 45.0617256, -94.4794235, 96.7535248
5: -42.5001488, 32.6657448, -50.9303436, 39.1559410, -81.6560822, 83.5960846
6: -40.1540794, 41.8406219, -48.3007050, 49.8171349, -89.9712143, 90.1413116
7: -47.9719009, 31.9340782, -57.0587463, 39.0055008, -86.9774017, 88.9928284
8: -51.5629196, 35.1039314, -62.1747055, 42.1395073, -93.7024231, 97.2786407
9: -40.3141632, 40.4633064, -48.3375969, 48.4476624, -88.7618103, 88.8009033

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 186

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3188326, upper bound: 107.3199667
time: 12.12 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3313106, upper bound: 107.3311843
time: 10.16 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -46.3019791, 36.4776077, -53.8877716, 42.3655930, -88.6675720, 90.3653717
1: -38.3613205, 32.6207962, -44.6634636, 37.8695984, -76.2309113, 77.2842560
2: -48.3730850, 29.1508617, -56.4374924, 34.2264175, -82.5994949, 85.5883255
3: -56.9504356, 26.5854568, -66.1045685, 31.1081333, -88.0585632, 92.6900177
4: -50.5228195, 38.5648804, -58.6226044, 44.7650909, -95.2879028, 97.1874695
5: -43.4570999, 33.4082642, -50.5909805, 38.8882446, -82.3453369, 83.9992218
6: -41.0852127, 42.7439613, -47.9671097, 49.4983482, -90.5835571, 90.7110748
7: -49.0118790, 32.7738113, -56.6962204, 38.7046471, -87.7165070, 89.4700317
8: -52.7787018, 35.8977814, -61.7420349, 41.8511009, -94.6297913, 97.6398163
9: -41.2377815, 41.3724594, -48.0070953, 48.1183128, -89.3560791, 89.3795471

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 186

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3191590, upper bound: 107.3202745
time: 10.25 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3313694, upper bound: 107.3312500
time: 10.59 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -46.1606331, 36.3935318, -49.1048584, 38.6495056, -84.8101349, 85.4983902
1: -38.2842941, 32.5118637, -40.7075882, 34.5882492, -72.8725433, 73.2194519
2: -48.1523399, 28.8948212, -51.3619041, 31.0899105, -79.2422485, 80.2567215
3: -56.8357162, 26.3407917, -60.3201675, 28.3458023, -85.1815033, 86.6609573
4: -50.4780846, 38.5077324, -53.4691467, 40.8820534, -91.3601379, 91.9768753
5: -43.2924728, 33.3075829, -46.1187515, 35.4534836, -78.7459564, 79.4263306
6: -40.9435768, 42.6813354, -43.6826096, 45.2269516, -86.1705170, 86.3639450
7: -49.0005569, 32.4101181, -51.8334846, 35.1003609, -84.1009216, 84.2435989
8: -52.4952087, 35.7554932, -56.1504402, 38.1371956, -90.6323853, 91.9059219
9: -41.1057587, 41.3310814, -43.7716446, 43.8936806, -84.9994354, 85.1027145

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 186

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3204749, upper bound: 107.3192582
time: 13.18 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3208176, upper bound: 107.3196198
time: 12.27 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -53.3702126, 41.9539909, -52.7984924, 41.4998474, -94.8700562, 94.7524719
1: -44.2439804, 37.5680733, -43.7559700, 37.1671143, -81.4110870, 81.3240433
2: -56.0246544, 34.0927773, -55.3936081, 33.7414703, -89.7661133, 89.4863739
3: -65.4543991, 30.9364853, -64.7302628, 30.6667442, -96.1211395, 95.6667480
4: -58.1290474, 44.3756866, -57.3881989, 43.8769493, -102.0059967, 101.7638702
5: -50.1066895, 38.6247406, -49.5969086, 38.1777191, -88.2843933, 88.2216339
6: -47.6446228, 48.9809418, -47.0989914, 48.4519844, -96.0966034, 96.0799332
7: -56.1646309, 38.7044487, -55.4927177, 38.2764130, -94.4410400, 94.1971588
8: -61.2942886, 41.5304832, -60.6540604, 41.0967369, -102.3910217, 102.1845398
9: -47.7078705, 47.7975845, -47.1305084, 47.1968307, -94.9047012, 94.9280930

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 186

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3319214, upper bound: 107.3319054
time: 10.06 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3321178, upper bound: 107.3320834
time: 9.16 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -44.9741287, 35.4797783, -51.2203751, 40.3147774, -85.2888947, 86.7001343
1: -37.3006325, 31.6806831, -42.4726562, 36.0219803, -73.3226013, 74.1533279
2: -46.8557434, 28.0346470, -53.5379486, 32.3333206, -79.1890640, 81.5725937
3: -55.4067268, 25.5913372, -62.9228973, 29.4558487, -84.8625793, 88.5142365
4: -49.1895676, 37.5376701, -55.7940903, 42.6155128, -91.8050766, 93.3317566
5: -42.1807404, 32.4150658, -48.0878601, 36.9335632, -79.1142960, 80.5029221
6: -39.8169479, 41.6545334, -45.5225639, 47.1729660, -86.9899139, 87.1770935
7: -47.8134003, 31.3334770, -54.0705528, 36.4386673, -84.2520676, 85.4040222
8: -51.0460663, 34.8065948, -58.5012436, 39.7273903, -90.7734528, 93.3078308
9: -40.0037766, 40.2383041, -45.5935860, 45.7659111, -85.7696838, 85.8318787

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 186

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3202722, upper bound: 107.3190891
time: 11.77 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3207150, upper bound: 107.3195401
time: 12.31 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -51.9406776, 40.8569107, -54.9177170, 43.1654129, -95.1060791, 95.7746277
1: -43.0644035, 36.5756569, -45.5276184, 38.6044655, -81.6688690, 82.1032562
2: -54.4530525, 33.0552216, -57.5731544, 34.9872704, -89.4403229, 90.6283722
3: -63.7565155, 30.0460663, -67.3376312, 31.7811890, -95.5377045, 97.3836975
4: -56.5918121, 43.2159615, -59.7242165, 45.6117935, -102.2035980, 102.9401779
5: -48.7742691, 37.5544357, -51.5665169, 39.6655045, -88.4397583, 89.1209564
6: -46.3084946, 47.7450943, -48.9441185, 50.4020004, -96.7104950, 96.6892090
7: -54.7437286, 37.4338913, -57.7366295, 39.6286125, -94.3723373, 95.1705017
8: -59.5409088, 40.3826065, -63.0130386, 42.6888466, -102.2297516, 103.3956451
9: -46.3841362, 46.4981079, -48.9642220, 49.0731239, -95.4572601, 95.4623260

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 186

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3316386, upper bound: 107.3316698
time: 11.12 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3319138, upper bound: 107.3319138
time: 10.20 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 22.13 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 7, lower bound: -107.3287268, upper bound: 107.3285513
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 7, lower bound: -107.3278434, upper bound: 107.3277289
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 7, lower bound: -107.3286003, upper bound: 107.3284089
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 7, lower bound: -107.3277898, upper bound: 107.3276740
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 7, lower bound: -107.3188326, upper bound: 107.3199667
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 7, lower bound: -107.3313106, upper bound: 107.3311843
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 7, lower bound: -107.3191590, upper bound: 107.3202745
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 7, lower bound: -107.3313694, upper bound: 107.3312500
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 7, lower bound: -107.3204749, upper bound: 107.3192582
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 7, lower bound: -107.3208176, upper bound: 107.3196198
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 7, lower bound: -107.3319214, upper bound: 107.3319054
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 7, lower bound: -107.3321178, upper bound: 107.3320834
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 7, lower bound: -107.3202722, upper bound: 107.3190891
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 7, lower bound: -107.3207150, upper bound: 107.3195401
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 7, lower bound: -107.3316386, upper bound: 107.3316698
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 7, lower bound: -107.3319138, upper bound: 107.3319138

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -45.6203308, 35.9376602, -47.8888931, 37.7126732, -83.3330002, 83.8265457
1: -37.7865295, 32.1383629, -39.6736641, 33.7277946, -71.5143127, 71.8120270
2: -47.6476021, 28.6996059, -50.0614929, 30.1871738, -77.8347778, 78.7610931
3: -56.1221962, 26.2116337, -58.8984451, 27.6229668, -83.7451553, 85.1100769
4: -49.7749634, 38.0139236, -52.1186943, 39.9006577, -89.6756210, 90.1326141
5: -42.8104286, 32.9225922, -45.0031548, 34.5286484, -77.3390808, 77.9257431
6: -40.4758759, 42.1187172, -42.5327911, 44.1655502, -84.6414261, 84.6515045
7: -48.2802048, 32.2691154, -50.5382233, 34.0046043, -82.2848053, 82.8073425
8: -51.9929466, 35.3917694, -54.7031593, 37.2383614, -89.2312927, 90.0949020
9: -40.6266632, 40.7736969, -42.6176758, 42.7304840, -83.3571472, 83.3913651

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 43

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3156451, upper bound: 107.3144044
time: 12.31 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3257562, upper bound: 107.3255470
time: 13.23 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -43.8356094, 34.5548592, -49.2212944, 38.7534828, -82.5890808, 83.7761536
1: -36.2961769, 30.8897820, -40.7615738, 34.6096344, -70.9058075, 71.6513519
2: -45.7187881, 27.4080582, -51.4224815, 30.8469582, -76.5657501, 78.8305359
3: -53.9911804, 25.1048717, -60.5708771, 28.2702599, -82.2614288, 85.6757507
4: -47.8565331, 36.5636215, -53.5906754, 40.9888382, -88.8453674, 90.1542892
5: -41.1336212, 31.6003227, -46.2264709, 35.4527092, -76.5863342, 77.8267899
6: -38.8172302, 40.5573616, -43.6466026, 45.3998680, -84.2170944, 84.2039642
7: -46.4826431, 30.7069283, -51.9543381, 34.6991501, -81.1817932, 82.6612549
8: -49.8308334, 33.9908867, -56.0974197, 38.2478333, -88.0786667, 90.0882950
9: -38.9889793, 39.1429977, -43.7321663, 43.8269196, -82.8158875, 82.8751678

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3148344, upper bound: 107.3136291
time: 13.65 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3248304, upper bound: 107.3246839
time: 10.84 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -46.5081635, 36.6385078, -47.4464073, 37.3719330, -83.8800964, 84.0849152
1: -38.5307884, 32.7646904, -39.3055458, 33.4208221, -71.9516068, 72.0702362
2: -48.6175003, 29.3114109, -49.5828133, 29.8746452, -78.4921417, 78.8942261
3: -57.2063522, 26.7348175, -58.3648567, 27.3506165, -84.5569611, 85.0996704
4: -50.7483788, 38.7379456, -51.6439171, 39.5368271, -90.2852020, 90.3818665
5: -43.6536598, 33.5732079, -44.5875397, 34.2002258, -77.8538818, 78.1607285
6: -41.2911072, 42.9156227, -42.1209106, 43.7780037, -85.0690994, 85.0365295
7: -49.1985855, 32.9956856, -50.0999947, 33.6258926, -82.8244781, 83.0956726
8: -53.0577469, 36.0859489, -54.1682510, 36.8844376, -89.9421844, 90.2541809
9: -41.4360428, 41.5684853, -42.2123337, 42.3261566, -83.7621994, 83.7808228

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 43

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3155547, upper bound: 107.3142840
time: 11.26 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3256289, upper bound: 107.3253983
time: 10.27 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -44.6834488, 35.2253876, -48.7421265, 38.3852005, -83.0686493, 83.9675064
1: -37.0072021, 31.4870186, -40.3620491, 34.2777557, -71.2849579, 71.8490677
2: -46.6429787, 27.9913597, -50.9045105, 30.5062981, -77.1492691, 78.8958740
3: -55.0298347, 25.6028900, -59.9960709, 27.9761696, -83.0060043, 85.5989532
4: -48.7877312, 37.2558022, -53.0776062, 40.5960770, -89.3837967, 90.3334045
5: -41.9400978, 32.2208862, -45.7774048, 35.0964203, -77.0365067, 77.9982910
6: -39.5954590, 41.3190536, -43.1996841, 44.9817505, -84.5772095, 84.5187378
7: -47.3602524, 31.3988838, -51.4782867, 34.2863312, -81.6465759, 82.8771667
8: -50.8460083, 34.6537514, -55.5193405, 37.8657761, -88.7117844, 90.1730957
9: -39.7621002, 39.9023056, -43.2943497, 43.3892670, -83.1513672, 83.1966553

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3147691, upper bound: 107.3135584
time: 12.01 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3247543, upper bound: 107.3246020
time: 10.04 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -41.2642212, 32.5553474, -44.2682076, 34.9310837, -76.1953049, 76.8235550
1: -34.1737709, 29.0734558, -36.7060165, 31.1171875, -65.2909546, 65.7794571
2: -42.8851967, 25.5699387, -45.9436684, 27.2888412, -70.1740265, 71.5136032
3: -50.8562698, 23.4308586, -54.5689278, 24.9825020, -75.8387756, 77.9997864
4: -45.1218987, 34.4535828, -48.3895302, 36.9346466, -82.0565338, 82.8431091
5: -38.6808701, 29.6967049, -41.4935760, 31.7806950, -70.4615631, 71.1902695
6: -36.3935471, 38.3099594, -38.9891205, 41.0872574, -77.4808044, 77.2990799
7: -43.9298897, 28.4127274, -47.1071320, 30.2544212, -74.1843109, 75.5198441
8: -46.6862030, 31.8963394, -50.0125237, 34.1670494, -80.8532333, 81.9088593
9: -36.6348724, 36.8234138, -39.1925163, 39.4668617, -76.1017227, 76.0159302

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3142094, upper bound: 107.3151615
time: 11.92 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3135036, upper bound: 107.3143587
time: 12.54 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -44.1712570, 34.8138771, -50.6310997, 39.8568726, -84.0281067, 85.4449768
1: -36.5849342, 31.1253548, -41.9716187, 35.6023254, -72.1872482, 73.0969696
2: -46.0498352, 27.6492786, -52.8880844, 31.9003735, -77.9502106, 80.5373306
3: -54.3706436, 25.2854958, -62.2082062, 29.0892220, -83.4598618, 87.4936981
4: -48.2192497, 36.8295860, -55.1462135, 42.1288948, -90.3481369, 91.9757996
5: -41.4441757, 31.8401966, -47.5328712, 36.4836922, -77.9278641, 79.3730621
6: -39.1138840, 40.8592339, -44.9510841, 46.6560898, -85.7699585, 85.8103027
7: -46.8513451, 30.9635468, -53.4532738, 35.9027748, -82.7540970, 84.4167938
8: -50.2015152, 34.2154808, -57.7707062, 39.2600327, -89.4615479, 91.9861908
9: -39.2910004, 39.4544830, -45.0376740, 45.2056503, -84.4966354, 84.4921570

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3254852, upper bound: 107.3254682
time: 10.78 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3246102, upper bound: 107.3245083
time: 9.76 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -42.2145309, 33.3074379, -44.0355072, 34.7506104, -76.9651413, 77.3429413
1: -34.9664345, 29.7508125, -36.5113106, 30.9538059, -65.9202423, 66.2621231
2: -43.9257164, 26.2247696, -45.6938133, 27.1267509, -71.0524597, 71.9185715
3: -52.0219650, 23.9934521, -54.2858582, 24.8388119, -76.8607788, 78.2793045
4: -46.1625977, 35.2333031, -48.1398964, 36.7410622, -82.9036407, 83.3731995
5: -39.5854034, 30.3977013, -41.2742004, 31.6072807, -71.1926880, 71.6719055
6: -37.2749252, 39.1625404, -38.7717934, 40.8819046, -78.1568298, 77.9343338
7: -44.9156036, 29.2020988, -46.8711662, 30.0577469, -74.9733505, 76.0732651
8: -47.8239403, 32.6439590, -49.7375717, 33.9833794, -81.8073196, 82.3815308
9: -37.5037766, 37.6846848, -38.9788170, 39.2508965, -76.7546692, 76.6634979

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3141910, upper bound: 107.3151925
time: 9.82 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3134174, upper bound: 107.3143058
time: 12.13 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -45.1594009, 35.5937881, -50.2939987, 39.5972710, -84.7566681, 85.8877716
1: -37.4131851, 31.8236885, -41.6904182, 35.3692703, -72.7824554, 73.5140991
2: -47.1299286, 28.3355637, -52.5265350, 31.6671524, -78.7970657, 80.8620987
3: -55.5788460, 25.8713551, -61.8039856, 28.8816242, -84.4604721, 87.6753387
4: -49.3031006, 37.6371765, -54.7855110, 41.8518944, -91.1549835, 92.4226837
5: -42.3820381, 32.5663757, -47.2177277, 36.2351837, -78.6172180, 79.7840958
6: -40.0252457, 41.7448044, -44.6415138, 46.3593330, -86.3845825, 86.3863220
7: -47.8716965, 31.7838001, -53.1174355, 35.6234016, -83.4951019, 84.9012222
8: -51.3909912, 34.9912605, -57.3696213, 38.9937325, -90.3847122, 92.3608856
9: -40.1945877, 40.3445625, -44.7296333, 44.8998871, -85.0944748, 85.0741959

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3254429, upper bound: 107.3254529
time: 12.20 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3244903, upper bound: 107.3243879
time: 11.58 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -44.2100868, 34.8790855, -43.5694046, 34.3371925, -78.5472794, 78.4484863
1: -36.6558495, 31.1339684, -36.0802307, 30.6806717, -67.3365173, 67.2142029
2: -46.0291290, 27.4962120, -45.3215637, 27.0945072, -73.1236343, 72.8177719
3: -54.4759636, 25.1001663, -53.6301346, 24.8233490, -79.2993011, 78.7303009
4: -48.3792496, 36.9062347, -47.5305099, 36.3311844, -84.7104263, 84.4367447
5: -41.4497375, 31.8544807, -40.8757591, 31.3314266, -72.7811584, 72.7302322
6: -39.0992928, 40.9681129, -38.4752312, 40.3465424, -79.4458313, 79.4433441
7: -47.0177078, 30.6946144, -46.2422295, 30.2409172, -77.2586136, 76.9368439
8: -50.1374207, 34.1965866, -49.3941460, 33.7051888, -83.8425980, 83.5907288
9: -39.3073349, 39.5314713, -38.6641655, 38.8202286, -78.1275635, 78.1956329

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3159477, upper bound: 107.3149479
time: 11.66 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3151372, upper bound: 107.3141071
time: 14.58 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -43.9346924, 34.6680908, -44.6014633, 35.1542931, -79.0889816, 79.2695465
1: -36.4271774, 30.9444790, -36.9487495, 31.4131413, -67.8403168, 67.8932266
2: -45.7352943, 27.3105030, -46.4523964, 27.8190994, -73.5543976, 73.7629013
3: -54.1435852, 24.9353371, -54.8882256, 25.4434395, -79.5870209, 79.8235550
4: -48.0828972, 36.6779518, -48.6565781, 37.1767006, -85.2595978, 85.3345337
5: -41.1930695, 31.6514454, -41.8597031, 32.0902214, -73.2832794, 73.5111465
6: -38.8432198, 40.7285347, -39.4304504, 41.2734146, -80.1166382, 80.1589737
7: -46.7415657, 30.4658642, -47.3096428, 31.1060486, -77.8476105, 77.7754974
8: -49.8132439, 33.9816971, -50.6375313, 34.5177956, -84.3310394, 84.6192169
9: -39.0575676, 39.2782135, -39.6104965, 39.7457733, -78.8033295, 78.8887100

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3159456, upper bound: 107.3149442
time: 12.60 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3151631, upper bound: 107.3140836
time: 10.70 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -50.9959183, 40.1184883, -46.6982574, 36.7662468, -87.7621613, 86.8167343
1: -42.2708092, 35.9054146, -38.6739235, 32.8782272, -75.1490326, 74.5793381
2: -53.4267731, 32.3808594, -48.7264786, 29.3425026, -82.7692566, 81.1073380
3: -62.6099358, 29.4443512, -57.3921127, 26.8126602, -89.4225922, 86.8364639
4: -55.5873947, 42.4362183, -50.8512306, 38.8863068, -94.4736862, 93.2874451
5: -47.8776894, 36.8561783, -43.8432465, 33.6355476, -81.5132370, 80.6994247
6: -45.4274902, 46.9010620, -41.3830643, 43.0927505, -88.5202408, 88.2841263
7: -53.7727432, 36.6394920, -49.3576393, 32.9731598, -86.7459030, 85.9971313
8: -58.3960571, 39.6275673, -53.2033958, 36.2044411, -94.6004639, 92.8309631
9: -45.5165329, 45.6325607, -41.5241241, 41.6248741, -87.1414032, 87.1566849

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3258709, upper bound: 107.3266340
time: 12.41 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3255874, upper bound: 107.3255505
time: 9.42 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 13.88 + 606.10 = 619.98 seconds
