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
execution time: IAR + RelationalAnalysis = 1.03 + 13.48 = 14.51 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -107.3454662, upper bound: 107.3454661

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3241953, upper bound: 107.3248325
time: 13.34 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3453684, upper bound: 107.3453684
time: 10.75 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 24.16 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 24.16
Output dim: 7, lower bound: -107.3241953, upper bound: 107.3248325
NS_B2, status: Status.UNKNOWN, split count: 1, time: 24.16
Output dim: 7, lower bound: -107.3453684, upper bound: 107.3453684

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -64.6237183, 50.7071915, -64.0898895, 50.2896042, -114.9133224, 114.7970734
1: -53.5045700, 45.3951416, -53.0554504, 44.9065208, -98.4110870, 98.4505920
2: -68.4125214, 42.2013855, -67.5403671, 41.2985992, -109.7111206, 109.7417526
3: -78.7144165, 37.9930382, -78.3133545, 37.2079010, -115.9223175, 116.3063965
4: -70.1204071, 53.5102806, -69.6880188, 53.0828209, -123.2032318, 123.1983032
5: -60.7049866, 47.0303459, -60.1112175, 46.4402046, -107.1451874, 107.1415558
6: -58.0755424, 58.7820549, -57.3589096, 58.4169083, -116.4924316, 116.1409607
7: -67.2641296, 48.5393410, -66.8935089, 47.2144890, -114.4786224, 115.4328461
8: -75.0211639, 50.7537689, -73.9750671, 50.1169930, -125.1381378, 124.7288361
9: -57.9863510, 57.9476929, -57.3111382, 57.2904510, -115.2768021, 115.2588348

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 216

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2740920, upper bound: 107.2737537
time: 13.18 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3234419, upper bound: 107.3240246
time: 10.90 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -66.9629440, 52.5338936, -66.0929947, 51.8533020, -118.8162460, 118.6268921
1: -55.4344978, 47.0252304, -54.7155418, 46.4171753, -101.8516541, 101.7407532
2: -70.9811935, 43.8698730, -70.0227280, 43.2433357, -114.2245255, 113.8926010
3: -81.4659729, 39.4653587, -80.4447098, 38.9130745, -120.3790436, 119.9100647
4: -72.6269455, 55.4208984, -71.6942825, 54.7091408, -127.3360901, 127.1151810
5: -62.9200974, 48.7844696, -62.0950432, 48.1275444, -111.0476379, 110.8795166
6: -60.2321777, 60.8447495, -59.4262390, 60.0777016, -120.3098755, 120.2709885
7: -69.6045456, 50.5524025, -68.7342606, 49.7926636, -119.3972015, 119.2866669
8: -77.8515930, 52.6643562, -76.7948151, 51.9506645, -129.8022461, 129.4591675
9: -60.1245384, 60.0685272, -59.3262596, 59.2746925, -119.3992310, 119.3947601

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 174

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3200546, upper bound: 107.3187724
time: 11.35 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3447910, upper bound: 107.3447911
time: 10.85 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 23.20 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 23.20
Output dim: 7, lower bound: -107.2740920, upper bound: 107.2737537
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 23.20
Output dim: 7, lower bound: -107.3234419, upper bound: 107.3240246
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 23.20
Output dim: 7, lower bound: -107.3200546, upper bound: 107.3187724
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 23.20
Output dim: 7, lower bound: -107.3447910, upper bound: 107.3447911

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -53.5391731, 42.1203423, -59.5507774, 46.7659111, -100.3050842, 101.6711197
1: -44.3740349, 37.6215553, -49.3179817, 41.7450447, -86.1190796, 86.9395370
2: -56.0859833, 33.9338531, -62.5318451, 38.0151443, -94.1011276, 96.4656982
3: -65.7847443, 30.7790642, -72.9798584, 34.3405380, -100.1252747, 103.7589188
4: -58.3692551, 44.4997597, -64.8476639, 49.3855057, -107.7547607, 109.3474121
5: -50.2370682, 38.6084862, -55.8361015, 43.0304718, -93.2675400, 94.4445877
6: -47.6998291, 49.1702347, -53.1514702, 54.4616661, -102.1614990, 102.3217010
7: -56.3584328, 38.3885193, -62.3917923, 43.2292976, -99.5877228, 100.7803116
8: -61.2847900, 41.6106949, -68.4241180, 46.3814850, -107.6662750, 110.0348129
9: -47.7137146, 47.8003654, -53.1436043, 53.1605835, -100.8742981, 100.9439697

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 166

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of NS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of NS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 52

## Relational analysis of NS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 52

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of NS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_B1_A1_B1

### Relational analysis result of NS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2579369, upper bound: 107.2576650
time: 12.72 seconds

## Relational analysis of NS_B1_A1_B2

### Relational analysis result of NS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2550427, upper bound: 107.2546676
time: 12.14 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -62.8776627, 49.3370132, -64.0898895, 50.2896042, -113.1672668, 113.4269028
1: -52.0572395, 44.1745987, -53.0554504, 44.9065208, -96.9637604, 97.2300491
2: -66.4781494, 40.9402580, -67.5403671, 41.2985992, -107.7767487, 108.4806213
3: -76.6658783, 36.8865433, -78.3133545, 37.2079010, -113.8737793, 115.1998901
4: -68.2523270, 52.0759048, -69.6880188, 53.0828209, -121.3351440, 121.7639160
5: -59.0433197, 45.7090378, -60.1112175, 46.4402046, -105.4835205, 105.8202438
6: -56.4564171, 57.2446365, -57.3589096, 58.4169083, -114.8733215, 114.6035385
7: -65.5315933, 47.0089111, -66.8935089, 47.2144890, -112.7460785, 113.9024200
8: -72.8826294, 49.3036499, -73.9750671, 50.1169930, -122.9996033, 123.2787018
9: -56.3835754, 56.3552704, -57.3111382, 57.2904510, -113.6740265, 113.6663895

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 216

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_B1_A2_A1

### Relational analysis result of NS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3183282, upper bound: 107.3187605
time: 12.55 seconds

## Relational analysis of NS_B1_A2_A2

### Relational analysis result of NS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3181498, upper bound: 107.3185540
time: 14.67 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -55.4400406, 43.5753212, -61.0351028, 47.8798599, -103.3198929, 104.6104279
1: -45.9425201, 38.9366989, -50.5320473, 42.8757095, -88.8182144, 89.4687347
2: -58.1381912, 35.2882538, -64.4145279, 39.5853996, -97.7235870, 99.7027740
3: -68.0392075, 31.9533882, -74.5025253, 35.6972809, -103.7364883, 106.4559097
4: -60.3995552, 46.0324249, -66.2943497, 50.5535736, -110.9531174, 112.3267746
5: -52.0120239, 40.0087280, -57.2852669, 44.2979431, -96.3099670, 97.2939911
6: -49.4500198, 50.8290100, -54.7430878, 55.6221161, -105.0721359, 105.5720978
7: -58.2671661, 40.0144043, -63.7233238, 45.3617668, -103.6289368, 103.7377319
8: -63.5644531, 43.1024208, -70.5862198, 47.7337456, -111.2982025, 113.6886444
9: -49.4447327, 49.5133820, -54.6842041, 54.6602631, -104.1049805, 104.1975861

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 86

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_B2_A1_B1

### Relational analysis result of NS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3140047, upper bound: 107.3130453
time: 12.81 seconds

## Relational analysis of NS_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3138109, upper bound: 107.3129120
time: 14.18 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -65.1881714, 51.1397667, -66.0929947, 51.8533020, -117.0414734, 117.2327576
1: -53.9649925, 45.7811394, -54.7155418, 46.4171753, -100.3821716, 100.4966812
2: -69.0146408, 42.5899773, -70.0227280, 43.2433357, -112.2579803, 112.6126938
3: -79.3844147, 38.3417397, -80.4447098, 38.9130745, -118.2974854, 118.7864532
4: -70.7281570, 53.9627228, -71.6942825, 54.7091408, -125.4373016, 125.6570053
5: -61.2309990, 47.4403915, -62.0950432, 48.1275444, -109.3585358, 109.5354309
6: -58.5862999, 59.2805290, -59.4262390, 60.0777016, -118.6640015, 118.7067719
7: -67.8424911, 48.9979668, -68.7342606, 49.7926636, -117.6351471, 117.7322235
8: -75.6820450, 51.1857529, -76.7948151, 51.9506645, -127.6327057, 127.9805679
9: -58.4929962, 58.4505005, -59.3262596, 59.2746925, -117.7676849, 117.7767487

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 166

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of NS_B2_A2_B1

### Relational analysis result of NS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3187724, upper bound: 107.3200545
time: 10.34 seconds

## Relational analysis of NS_B2_A2_B2

### Relational analysis result of NS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3187724, upper bound: 107.3447910
time: 13.75 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 24.99 seconds
NS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 24.99
Output dim: 7, lower bound: -107.2579369, upper bound: 107.2576650
NS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 24.99
Output dim: 7, lower bound: -107.2550427, upper bound: 107.2546676
NS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 24.99
Output dim: 7, lower bound: -107.3183282, upper bound: 107.3187605
NS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 24.99
Output dim: 7, lower bound: -107.3181498, upper bound: 107.3185540
NS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 24.99
Output dim: 7, lower bound: -107.3140047, upper bound: 107.3130453
NS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 24.99
Output dim: 7, lower bound: -107.3138109, upper bound: 107.3129120
NS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 24.99
Output dim: 7, lower bound: -107.3187724, upper bound: 107.3200545
NS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 24.99
Output dim: 7, lower bound: -107.3187724, upper bound: 107.3447910

## BFS NS instance: NS_B1_A1_B1

### Backsubstitution after applying NS history:
0: -50.0276489, 39.4127998, -49.3819962, 38.9139671, -88.9416199, 88.7947998
1: -41.4683533, 35.1826515, -40.9076614, 34.6798820, -76.1482315, 76.0903168
2: -52.2912865, 31.4628181, -51.5208740, 30.8556442, -83.1469269, 82.9836807
3: -61.5907745, 28.6297302, -60.8525848, 28.1191196, -89.7098846, 89.4823151
4: -54.6312523, 41.6408997, -53.9873657, 41.1333084, -95.7645569, 95.6282654
5: -46.9321518, 36.0331650, -46.2984886, 35.5409775, -82.4731216, 82.3316498
6: -44.4621964, 46.0926743, -43.7853661, 45.5543671, -90.0165405, 89.8780365
7: -52.8490562, 35.4106216, -52.2397881, 34.6011238, -87.4501801, 87.6504059
8: -57.0591049, 38.8224945, -56.1661568, 38.2845116, -95.3436127, 94.9886398
9: -44.5140648, 44.6539192, -43.8883514, 44.0585060, -88.5725708, 88.5422668

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_B1_A1_B1_A1

### Relational analysis result of NS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2550427, upper bound: 107.2546676
time: 11.45 seconds

## Relational analysis of NS_B1_A1_B1_A2

### Relational analysis result of NS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2550427, upper bound: 107.2546675
time: 13.21 seconds

## BFS NS instance: NS_B1_A1_B2

### Backsubstitution after applying NS history:
0: -49.6825256, 39.1432190, -53.4594879, 42.0734634, -91.7559814, 92.6026840
1: -41.1767044, 34.9473190, -44.2839775, 37.5119019, -78.6886063, 79.2312927
2: -51.9238892, 31.2349396, -55.8829880, 33.5841904, -85.5080795, 87.1179276
3: -61.1632538, 28.4309635, -65.7723694, 30.5239182, -91.6871490, 94.2033310
4: -54.2514420, 41.3556595, -58.3834763, 44.4662552, -98.7176971, 99.7391357
5: -46.6098480, 35.7804070, -50.1366920, 38.4875183, -85.0973663, 85.9170990
6: -44.1459885, 45.7858353, -47.4961777, 49.1758461, -93.3218384, 93.2820129
7: -52.4942513, 35.1438255, -56.3817825, 37.8290520, -90.3233032, 91.5256042
8: -56.6528397, 38.5518036, -60.9906616, 41.4725952, -98.1254349, 99.5424576
9: -44.2027626, 44.3368607, -47.5617218, 47.6706161, -91.8733826, 91.8985748

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 25

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_B1_A1_B2_A1

### Relational analysis result of NS_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2550427, upper bound: 107.2546676
time: 12.71 seconds

## Relational analysis of NS_B1_A1_B2_A2

### Relational analysis result of NS_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2550427, upper bound: 107.2546676
time: 14.69 seconds

## BFS NS instance: NS_B1_A2_A1

### Backsubstitution after applying NS history:
0: -55.0805397, 43.2661705, -61.7262878, 48.4415283, -103.5220642, 104.9924622
1: -45.6307144, 38.7455635, -51.1072388, 43.2606697, -88.8913879, 89.8527985
2: -57.8556366, 35.2947693, -64.9229736, 39.5841904, -97.4398270, 100.2177429
3: -67.4692230, 32.0212402, -75.5256958, 35.7320709, -103.2012939, 107.5469360
4: -59.8202209, 45.7318497, -67.1389236, 51.1586914, -110.9789124, 112.8707504
5: -51.7342415, 39.8341408, -57.8923607, 44.6524887, -96.3867188, 97.7265015
6: -49.1844482, 50.4492493, -55.1557236, 56.3599472, -105.5443878, 105.6049728
7: -57.7582130, 40.1201515, -64.5409241, 45.1181374, -102.8763351, 104.6610718
8: -63.3778915, 42.9164085, -71.0867386, 48.1603813, -111.5382690, 114.0031433
9: -49.1741829, 49.1985130, -55.1209984, 55.1156845, -104.2898560, 104.3195114

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 216

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_B1_A2_A1_B1

### Relational analysis result of NS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3181498, upper bound: 107.3185540
time: 10.51 seconds

## Relational analysis of NS_B1_A2_A1_B2

### Relational analysis result of NS_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3181498, upper bound: 107.3185540
time: 11.06 seconds

## BFS NS instance: NS_B1_A2_A2

### Backsubstitution after applying NS history:
0: -57.1877594, 44.9243965, -60.0677338, 47.1601372, -104.3479004, 104.9921265
1: -47.3940163, 40.1773949, -49.7498970, 42.1107483, -89.5047607, 89.9272919
2: -60.0249596, 36.5409088, -63.0942688, 38.3915825, -98.4165344, 99.6351700
3: -70.0608826, 33.1320763, -73.5752335, 34.6973419, -104.7582169, 106.7073059
4: -62.1451302, 47.4573021, -65.3589172, 49.8145943, -111.9597244, 112.8162079
5: -53.6910744, 41.3198776, -56.3403244, 43.4122086, -97.1032715, 97.6602020
6: -51.0217438, 52.3903999, -53.6124458, 54.9290619, -105.9508057, 106.0028458
7: -59.9924812, 41.4736404, -62.9004860, 43.6638412, -103.6563263, 104.3741302
8: -65.7246094, 44.5008316, -69.0672684, 46.8089638, -112.5335693, 113.5681000
9: -51.0037651, 51.0681381, -53.5991135, 53.6161346, -104.6199036, 104.6672516

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_B1_A2_A2_B1

### Relational analysis result of NS_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3181498, upper bound: 107.3185540
time: 12.32 seconds

## Relational analysis of NS_B1_A2_A2_B2

### Relational analysis result of NS_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3181498, upper bound: 107.3185540
time: 12.90 seconds

## BFS NS instance: NS_B2_A1_B1

### Backsubstitution after applying NS history:
0: -53.4594040, 42.0575905, -53.6024170, 42.1242943, -95.5836792, 95.6599884
1: -44.3083839, 37.5598564, -44.4081001, 37.7041664, -82.0125504, 81.9679489
2: -55.9695435, 33.8524208, -56.2152405, 34.2056999, -90.1752472, 90.0676575
3: -65.6860657, 30.7276421, -65.7274780, 31.0680180, -96.7540817, 96.4551239
4: -58.2664261, 44.4310760, -58.2560844, 44.5273972, -102.7938232, 102.6871643
5: -50.1671524, 38.5275841, -50.3388367, 38.7138901, -88.8810349, 88.8664093
6: -47.6005859, 49.1146393, -47.8055000, 49.1658745, -96.7664642, 96.9201355
7: -56.2915726, 38.2549744, -56.3047485, 38.7850494, -95.0766144, 94.5597153
8: -61.1633759, 41.5232773, -61.5460434, 41.7067795, -102.8701553, 103.0693207
9: -47.6115761, 47.7037697, -47.8107719, 47.8497505, -95.4613113, 95.5145416

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_B2_A1_B1_A1

### Relational analysis result of NS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3138109, upper bound: 107.3129119
time: 10.29 seconds

## Relational analysis of NS_B2_A1_B1_A2

### Relational analysis result of NS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3138109, upper bound: 107.3129119
time: 12.21 seconds

## BFS NS instance: NS_B2_A1_B2

### Backsubstitution after applying NS history:
0: -52.0717468, 40.9922142, -55.7002945, 43.7746315, -95.8463745, 96.6925049
1: -43.1643944, 36.5972443, -46.1646004, 39.1295052, -82.2938995, 82.7618408
2: -54.4513206, 32.8469048, -58.3737106, 35.4433861, -89.8947067, 91.2205811
3: -64.0302887, 29.8596706, -68.3081207, 32.1714134, -96.2017059, 98.1677856
4: -56.7753792, 43.3057709, -60.5714493, 46.2454567, -103.0208359, 103.8772125
5: -48.8648605, 37.4889679, -52.2893524, 40.1897888, -89.0546417, 89.7783051
6: -46.3000412, 47.9144058, -49.6324806, 51.0984268, -97.3984680, 97.5468903
7: -54.9171181, 37.0230522, -58.5302277, 40.1267052, -95.0438232, 95.5532837
8: -59.4663620, 40.4158440, -63.8801079, 43.2821007, -102.7484589, 104.2959366
9: -46.3323021, 46.4471779, -49.6284027, 49.7106934, -96.0429764, 96.0755768

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of NS_B2_A1_B2_B1

### Relational analysis result of NS_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3081613, upper bound: 107.3081613
time: 9.80 seconds

## Relational analysis of NS_B2_A1_B2_B2

### Relational analysis result of NS_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3081613, upper bound: 107.3129120
time: 10.32 seconds

## BFS NS instance: NS_B2_A2_B1

### Backsubstitution after applying NS history:
0: -65.1881714, 51.1397667, -54.7550430, 43.0498543, -108.2380219, 105.8948059
1: -53.9649925, 45.7811394, -45.3764496, 38.4613228, -92.4263153, 91.1575928
2: -69.0146408, 42.5899773, -57.3954163, 34.7940369, -103.8086777, 99.9853973
3: -79.3844147, 38.3417397, -67.2282028, 31.5259132, -110.9103088, 105.5699463
4: -70.7281570, 53.9627228, -59.6685715, 45.4804764, -116.2086334, 113.6312866
5: -61.2309990, 47.4403915, -51.3724365, 39.5012093, -100.7322083, 98.8128281
6: -58.5862999, 59.2805290, -48.8163567, 50.2317276, -108.8180237, 108.0968857
7: -67.8424911, 48.9979668, -57.5789642, 39.4182701, -107.2607574, 106.5769348
8: -75.6820450, 51.1857529, -62.7391891, 42.5629730, -118.2450104, 113.9249420
9: -58.4929962, 58.4505005, -48.8188286, 48.8914185, -107.3843994, 107.2693253

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 86

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_B2_A2_B1_A1

### Relational analysis result of NS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3130453, upper bound: 107.3140047
time: 13.99 seconds

## Relational analysis of NS_B2_A2_B1_A2

### Relational analysis result of NS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3129120, upper bound: 107.3138109
time: 11.91 seconds

## BFS NS instance: NS_B2_A2_B2

### Backsubstitution after applying NS history:
0: -65.1881714, 51.1397667, -64.3235779, 50.4634399, -115.6516113, 115.4633484
1: -53.9649925, 45.7811394, -53.2500916, 45.1778107, -99.1428070, 99.0312347
2: -69.0146408, 42.5899773, -68.0623703, 41.9664879, -110.9811249, 110.6523438
3: -79.3844147, 38.3417397, -78.3687363, 37.7928085, -117.1772232, 116.7104797
4: -70.7281570, 53.9627228, -69.8010712, 53.2558861, -123.9840393, 123.7637863
5: -61.2309990, 47.4403915, -60.4110146, 46.7876472, -108.0186462, 107.8514099
6: -58.5862999, 59.2805290, -57.7851524, 58.5186501, -117.1049500, 117.0656815
7: -67.8424911, 48.9979668, -66.9783173, 48.2418861, -116.0843811, 115.9762878
8: -75.6820450, 51.1857529, -74.6313095, 50.4773140, -126.1593628, 125.8170624
9: -58.4929962, 58.4505005, -57.6994781, 57.6620560, -116.1550522, 116.1499786

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 166

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_B2_A2_B2_A1

### Relational analysis result of NS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3130453, upper bound: 107.3412111
time: 10.94 seconds

## Relational analysis of NS_B2_A2_B2_A2

### Relational analysis result of NS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3129120, upper bound: 107.3410193
time: 11.45 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 23.31 seconds
NS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 23.31
Output dim: 7, lower bound: -107.2550427, upper bound: 107.2546676
NS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 23.31
Output dim: 7, lower bound: -107.2550427, upper bound: 107.2546675
NS_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 23.31
Output dim: 7, lower bound: -107.2550427, upper bound: 107.2546676
NS_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 23.31
Output dim: 7, lower bound: -107.2550427, upper bound: 107.2546676
NS_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.31
Output dim: 7, lower bound: -107.3181498, upper bound: 107.3185540
NS_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.31
Output dim: 7, lower bound: -107.3181498, upper bound: 107.3185540
NS_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.31
Output dim: 7, lower bound: -107.3181498, upper bound: 107.3185540
NS_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.31
Output dim: 7, lower bound: -107.3181498, upper bound: 107.3185540
NS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 23.31
Output dim: 7, lower bound: -107.3138109, upper bound: 107.3129119
NS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 23.31
Output dim: 7, lower bound: -107.3138109, upper bound: 107.3129119
NS_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 23.31
Output dim: 7, lower bound: -107.3081613, upper bound: 107.3081613
NS_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 23.31
Output dim: 7, lower bound: -107.3081613, upper bound: 107.3129120
NS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 23.31
Output dim: 7, lower bound: -107.3130453, upper bound: 107.3140047
NS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 23.31
Output dim: 7, lower bound: -107.3129120, upper bound: 107.3138109
NS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 23.31
Output dim: 7, lower bound: -107.3130453, upper bound: 107.3412111
NS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 23.31
Output dim: 7, lower bound: -107.3129120, upper bound: 107.3410193

## BFS NS instance: NS_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -44.3392677, 34.9923286, -49.3819962, 38.9139671, -83.2532349, 84.3743286
1: -36.7083855, 31.1916733, -40.9076614, 34.6798820, -71.3882523, 72.0993347
2: -46.1251526, 27.4609241, -51.5208740, 30.8556442, -76.9807968, 78.9817886
3: -54.7107964, 25.1574898, -60.8525848, 28.1191196, -82.8299026, 86.0100708
4: -48.5052795, 36.9826813, -53.9873657, 41.1333084, -89.6385880, 90.9700470
5: -41.5625000, 31.8541451, -46.2984886, 35.5409775, -77.1034698, 78.1526337
6: -39.1809349, 41.0779877, -43.7853661, 45.5543671, -84.7352905, 84.8633499
7: -47.1004639, 30.5437088, -52.2397881, 34.6011238, -81.7015686, 82.7834930
8: -50.1820297, 34.2791519, -56.1661568, 38.2845116, -88.4665375, 90.4452972
9: -39.3186569, 39.5061722, -43.8883514, 44.0585060, -83.3771591, 83.3945084

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 105

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 52

### Candidate
type: A, layer: 1, pos: 52

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of NS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of NS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of NS_B1_A1_B1_A1_B1

### Relational analysis result of NS_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2508838, upper bound: 107.2505133
time: 11.48 seconds

## Relational analysis of NS_B1_A1_B1_A1_B2

### Relational analysis result of NS_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2512934, upper bound: 107.2509116
time: 15.60 seconds

## BFS NS instance: NS_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -48.2880096, 38.0674820, -49.3819962, 38.9139671, -87.2019806, 87.4494781
1: -40.0008392, 33.9532394, -40.9076614, 34.6798820, -74.6807175, 74.8609009
2: -50.3583488, 30.1097832, -51.5208740, 30.8556442, -81.2139893, 81.6306534
3: -59.5093460, 27.4788666, -60.8525848, 28.1191196, -87.6284637, 88.3314514
4: -52.7964630, 40.2217064, -53.9873657, 41.1333084, -93.9297714, 94.2090759
5: -45.2866135, 34.7168121, -46.2984886, 35.5409775, -80.8275909, 81.0152969
6: -42.7999115, 44.5997887, -43.7853661, 45.5543671, -88.3542709, 88.3851471
7: -51.1472740, 33.6924286, -52.2397881, 34.6011238, -85.7483978, 85.9322205
8: -54.8588371, 37.3780212, -56.1661568, 38.2845116, -93.1433487, 93.5441742
9: -42.8854942, 43.0283356, -43.8883514, 44.0585060, -86.9440002, 86.9166870

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 105

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 52

### Candidate
type: A, layer: 1, pos: 52

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of NS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 144

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of NS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 50

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

### Candidate
type: A, layer: 1, pos: 94

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of NS_B1_A1_B1_A2_B1

### Relational analysis result of NS_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2508838, upper bound: 107.2505130
time: 12.31 seconds

## Relational analysis of NS_B1_A1_B1_A2_B2

### Relational analysis result of NS_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2512934, upper bound: 107.2509116
time: 12.03 seconds

## BFS NS instance: NS_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -44.3392677, 34.9923286, -53.4594879, 42.0734634, -86.4127350, 88.4517975
1: -36.7083855, 31.1916733, -44.2839775, 37.5119019, -74.2202682, 75.4756470
2: -46.1251526, 27.4609241, -55.8829880, 33.5841904, -79.7093430, 83.3439102
3: -54.7107964, 25.1574898, -65.7723694, 30.5239182, -85.2347107, 90.9298553
4: -48.5052795, 36.9826813, -58.3834763, 44.4662552, -92.9715271, 95.3661575
5: -41.5625000, 31.8541451, -50.1366920, 38.4875183, -80.0500183, 81.9908371
6: -39.1809349, 41.0779877, -47.4961777, 49.1758461, -88.3567810, 88.5741577
7: -47.1004639, 30.5437088, -56.3817825, 37.8290520, -84.9294968, 86.9254837
8: -50.1820297, 34.2791519, -60.9906616, 41.4725952, -91.6546249, 95.2697983
9: -39.3186569, 39.5061722, -47.5617218, 47.6706161, -86.9892578, 87.0678940

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 105

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 52

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of NS_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 52

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of NS_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 14.51 + 586.51 = 601.02 seconds
