## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 105.986448459


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-57.8715248, 46.4339218, -57.8715248, 46.4339218, -104.3054352, 104.3054352)
1: (-48.2762108, 41.0486870, -48.2762108, 41.0486870, -89.3248901, 89.3248901)
2: (-63.9665565, 41.6315002, -63.9665565, 41.6315002, -105.5980453, 105.5980453)
3: (-66.9424515, 36.1956024, -66.9424515, 36.1956024, -103.1380463, 103.1380463)
4: (-61.7093658, 47.3356514, -61.7093658, 47.3356514, -109.0450134, 109.0450134)
5: (-55.6309967, 43.3028641, -55.6309967, 43.3028641, -98.9338531, 98.9338531)
6: (-53.1199455, 51.6415939, -53.1199455, 51.6415939, -104.7615356, 104.7615280)
7: (-57.6861839, 49.2933502, -57.6861839, 49.2933502, -106.9795380, 106.9795380)
8: (-70.8696365, 48.8952446, -70.8696365, 48.8952446, -119.7648773, 119.7648773)
9: (-53.1959190, 52.2425995, -53.1959190, 52.2425995, -105.4385071, 105.4385071)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.30 + 11.34 = 13.64 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -106.0925410, upper bound: 106.0925410

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0868158, upper bound: 106.0871470
time: 8.44 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0861978, upper bound: 106.0861978
time: 9.02 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 17.68 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 17.68
Output dim: 7, lower bound: -106.0868158, upper bound: 106.0871470
NS_A2, status: Status.UNKNOWN, split count: 1, time: 17.68
Output dim: 7, lower bound: -106.0861978, upper bound: 106.0861978

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -50.7896881, 40.7596703, -57.8715248, 46.4339218, -97.2236023, 98.6311951
1: -42.3400688, 35.9941635, -48.2762108, 41.0486870, -83.3887558, 84.2703629
2: -56.1183128, 36.5910606, -63.9665565, 41.6315002, -97.7498169, 100.5576019
3: -58.6062164, 31.8126183, -66.9424515, 36.1956024, -94.8018188, 98.7550659
4: -54.0662346, 41.5098991, -61.7093658, 47.3356514, -101.4018707, 103.2192688
5: -48.7517319, 37.9980659, -55.6309967, 43.3028641, -92.0545959, 93.6290588
6: -46.5708199, 45.3505554, -53.1199455, 51.6415939, -98.2124023, 98.4704971
7: -50.5452423, 43.2948265, -57.6861839, 49.2933502, -99.8385925, 100.9810104
8: -62.3479767, 43.0311890, -70.8696365, 48.8952446, -111.2432251, 113.9008255
9: -46.7031975, 45.8032837, -53.1959190, 52.2425995, -98.9457855, 98.9992065

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0861978, upper bound: 106.0861978
time: 7.74 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0861978, upper bound: 106.0861978
time: 8.42 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -47.7540474, 38.3459854, -53.4782715, 42.9206314, -90.6746826, 91.8242493
1: -39.6950417, 33.7907906, -44.5664330, 37.9109497, -77.6059799, 78.3572159
2: -52.7147102, 34.4151001, -59.0860596, 38.4937668, -91.2084808, 93.5011597
3: -55.0544510, 29.9323940, -61.7681351, 33.4823990, -88.5368423, 91.7005234
4: -50.7346687, 38.9928017, -56.9596901, 43.7132759, -94.4479370, 95.9524689
5: -45.8242645, 35.6927643, -51.3769264, 40.0162468, -85.8405075, 87.0696793
6: -43.7399940, 42.6280899, -49.0540466, 47.7335243, -91.4735031, 91.6821213
7: -47.4326401, 40.6913719, -53.2476692, 45.5720177, -93.0046539, 93.9390411
8: -58.6624794, 40.5025520, -65.5628891, 45.2562752, -103.9187546, 106.0654373
9: -43.8950310, 42.9902229, -49.1725044, 48.2444344, -92.1394653, 92.1627197

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 53

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0810732, upper bound: 106.0811611
time: 8.28 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0824770, upper bound: 106.0824770
time: 9.39 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 20.03 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 20.03
Output dim: 7, lower bound: -106.0861978, upper bound: 106.0861978
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 20.03
Output dim: 7, lower bound: -106.0861978, upper bound: 106.0861978
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 20.03
Output dim: 7, lower bound: -106.0810732, upper bound: 106.0811611
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 20.03
Output dim: 7, lower bound: -106.0824770, upper bound: 106.0824770

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -50.7896881, 40.7596703, -50.7896881, 40.7596703, -91.5493622, 91.5493622
1: -42.3400688, 35.9941635, -42.3400688, 35.9941635, -78.3342285, 78.3342285
2: -56.1183128, 36.5910606, -56.1183128, 36.5910606, -92.7093735, 92.7093735
3: -58.6062164, 31.8126183, -58.6062164, 31.8126183, -90.4188385, 90.4188385
4: -54.0662346, 41.5098991, -54.0662346, 41.5098991, -95.5761185, 95.5761185
5: -48.7517319, 37.9980659, -48.7517319, 37.9980659, -86.7498016, 86.7498016
6: -46.5708199, 45.3505554, -46.5708199, 45.3505554, -91.9213715, 91.9213715
7: -50.5452423, 43.2948265, -50.5452423, 43.2948265, -93.8400726, 93.8400726
8: -62.3479767, 43.0311890, -62.3479767, 43.0311890, -105.3791656, 105.3791656
9: -46.7031975, 45.8032837, -46.7031975, 45.8032837, -92.5064850, 92.5064850

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0818839, upper bound: 106.0822145
time: 8.79 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0831144, upper bound: 106.0835150
time: 7.63 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -50.7896881, 40.7596703, -47.7540474, 38.3459854, -89.1356735, 88.5137177
1: -42.3400688, 35.9941635, -39.6950417, 33.7907906, -76.1308594, 75.6891937
2: -56.1183128, 36.5910606, -52.7147102, 34.4151001, -90.5334167, 89.3057709
3: -58.6062164, 31.8126183, -55.0544510, 29.9323940, -88.5386047, 86.8670654
4: -54.0662346, 41.5098991, -50.7346687, 38.9928017, -93.0590057, 92.2445679
5: -48.7517319, 37.9980659, -45.8242645, 35.6927643, -84.4444885, 83.8223267
6: -46.5708199, 45.3505554, -43.7399940, 42.6280899, -89.1988983, 89.0905457
7: -50.5452423, 43.2948265, -47.4326401, 40.6913719, -91.2366104, 90.7274628
8: -62.3479767, 43.0311890, -58.6624794, 40.5025520, -102.8505249, 101.6936646
9: -46.7031975, 45.8032837, -43.8950310, 42.9902229, -89.6934204, 89.6983185

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0818839, upper bound: 106.0822145
time: 9.78 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0831144, upper bound: 106.0835150
time: 9.05 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -44.7005539, 35.9090233, -41.2521439, 33.1719055, -77.8724594, 77.1611633
1: -37.0884628, 31.6020775, -34.1582413, 29.1661873, -66.2546310, 65.7603149
2: -49.2993622, 32.2168732, -45.4258614, 29.6851254, -78.9844894, 77.6427307
3: -51.4462929, 28.0289040, -47.3360100, 25.8899231, -77.3362045, 75.3649139
4: -47.4127655, 36.4720345, -43.6925888, 33.6226387, -81.0354004, 80.1646118
5: -42.8877411, 33.3981743, -39.6150742, 30.8626671, -73.7504044, 73.0132370
6: -40.8745461, 39.9044037, -37.6113434, 36.8546143, -77.7291489, 77.5157471
7: -44.3055611, 38.0689507, -40.7588463, 35.1037979, -79.4093552, 78.8277969
8: -54.9016151, 37.9737282, -50.5389786, 35.1575203, -90.0591278, 88.5126953
9: -41.0509300, 40.1730728, -37.8229790, 36.9877815, -78.0386810, 77.9960480

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 240

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0806589, upper bound: 106.0806589
time: 8.74 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0806589, upper bound: 106.0811610
time: 8.24 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -45.1224976, 36.2506943, -45.6715088, 36.7102470, -81.8327408, 81.9221954
1: -37.4594460, 31.9128761, -37.9465065, 32.3430862, -69.8025131, 69.8593826
2: -49.7859993, 32.5261497, -50.3997040, 32.8708611, -82.6568527, 82.9258499
3: -51.9544067, 28.2959957, -52.5644722, 28.6510735, -80.6054840, 80.8604507
4: -47.8774452, 36.8268547, -48.5002594, 37.2711487, -85.1485901, 85.3271179
5: -43.2928391, 33.7178230, -43.8661957, 34.1905785, -77.4834137, 77.5840149
6: -41.2789764, 40.2871437, -41.7703476, 40.8056412, -82.0846176, 82.0574951
7: -44.7544746, 38.4389801, -45.3169746, 38.9053535, -83.6598282, 83.7559357
8: -55.4426994, 38.3272667, -56.0227852, 38.8110733, -94.2537613, 94.3500519
9: -41.4548340, 40.5774078, -41.9532852, 41.0860863, -82.5409241, 82.5306931

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 208

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0739076, upper bound: 106.0720237
time: 9.32 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0810338, upper bound: 106.0810338
time: 7.76 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 19.36 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 19.36
Output dim: 7, lower bound: -106.0818839, upper bound: 106.0822145
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 19.36
Output dim: 7, lower bound: -106.0831144, upper bound: 106.0835150
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 19.36
Output dim: 7, lower bound: -106.0818839, upper bound: 106.0822145
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 19.36
Output dim: 7, lower bound: -106.0831144, upper bound: 106.0835150
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 19.36
Output dim: 7, lower bound: -106.0806589, upper bound: 106.0806589
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 19.36
Output dim: 7, lower bound: -106.0806589, upper bound: 106.0811610
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 19.36
Output dim: 7, lower bound: -106.0739076, upper bound: 106.0720237
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 19.36
Output dim: 7, lower bound: -106.0810338, upper bound: 106.0810338

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -38.8772850, 31.2613239, -47.6765327, 38.2795525, -77.1568222, 78.9378433
1: -32.1913948, 27.4707890, -39.6948509, 33.7698822, -65.9612732, 67.1656342
2: -42.8070412, 28.0077896, -52.6452789, 34.3396759, -77.1467133, 80.6530685
3: -44.5495644, 24.4045963, -54.9395523, 29.8824997, -74.4320602, 79.3441467
4: -41.1276894, 31.6824036, -50.6924095, 38.9440079, -80.0717010, 82.3748169
5: -37.3024979, 29.0631142, -45.7644119, 35.6689644, -72.9714661, 74.8275146
6: -35.4128723, 34.7436562, -43.6647873, 42.5855141, -77.9983826, 78.4084473
7: -38.3617210, 33.0808830, -47.3696709, 40.6325073, -78.9942245, 80.4505539
8: -47.6955681, 33.1865578, -58.5245323, 40.4638863, -88.1594543, 91.7110901
9: -35.6262093, 34.8328972, -43.8188019, 42.9412460, -78.5674591, 78.6517029

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0846088, upper bound: 106.0846088
time: 7.25 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0846088, upper bound: 106.0850090
time: 8.33 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -43.0196724, 34.5813866, -48.0211372, 38.5586014, -81.5782700, 82.6025162
1: -35.7369156, 30.4450989, -39.9960823, 34.0233574, -69.7602692, 70.4411697
2: -47.4723358, 30.9982471, -53.0447540, 34.5937462, -82.0660858, 84.0429993
3: -49.4507256, 26.9920750, -55.3498535, 30.0998898, -79.5506058, 82.3419266
4: -45.6297493, 35.1038322, -51.0695419, 39.2304497, -84.8601990, 86.1733704
5: -41.2890892, 32.1810265, -46.0927200, 35.9285393, -77.2176285, 78.2737350
6: -39.3103333, 38.4477692, -43.9946022, 42.8972740, -82.2076111, 82.4423676
7: -42.6416473, 36.6428299, -47.7369194, 40.9323578, -83.5739975, 84.3797379
8: -52.8406258, 36.6023598, -58.9696159, 40.7490540, -93.5896759, 95.5719757
9: -39.4985886, 38.6736450, -44.1462822, 43.2737923, -82.7723770, 82.8199158

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 53

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0850090, upper bound: 106.0851551
time: 9.10 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0850090, upper bound: 106.0864312
time: 9.33 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -38.8772850, 31.2613239, -44.7005539, 35.9090233, -74.7863083, 75.9618683
1: -32.1913948, 27.4707890, -37.0884628, 31.6020775, -63.7934685, 64.5592346
2: -42.8070412, 28.0077896, -49.2993622, 32.2168732, -75.0239105, 77.3071518
3: -44.5495644, 24.4045963, -51.4462929, 28.0289040, -72.5784683, 75.8508759
4: -41.1276894, 31.6824036, -47.4127655, 36.4720345, -77.5997162, 79.0951691
5: -37.3024979, 29.0631142, -42.8877411, 33.3981743, -70.7006683, 71.9508514
6: -35.4128723, 34.7436562, -40.8745461, 39.9044037, -75.3172760, 75.6181870
7: -38.3617210, 33.0808830, -44.3055611, 38.0689507, -76.4306641, 77.3864441
8: -47.6955681, 33.1865578, -54.9016151, 37.9737282, -85.6692886, 88.0881577
9: -35.6262093, 34.8328972, -41.0509300, 40.1730728, -75.7992859, 75.8838196

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 240

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0813473, upper bound: 106.0817327
time: 9.24 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0813473, upper bound: 106.0822145
time: 8.68 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -43.0196724, 34.5813866, -45.1224976, 36.2506943, -79.2703629, 79.7038803
1: -35.7369156, 30.4450989, -37.4594460, 31.9128761, -67.6497955, 67.9045258
2: -47.4723358, 30.9982471, -49.7859993, 32.5261497, -79.9984741, 80.7842331
3: -49.4507256, 26.9920750, -51.9544067, 28.2959957, -77.7467041, 78.9464798
4: -45.6297493, 35.1038322, -47.8774452, 36.8268547, -82.4566040, 82.9812775
5: -41.2890892, 32.1810265, -43.2928391, 33.7178230, -75.0069122, 75.4738617
6: -39.3103333, 38.4477692, -41.2789764, 40.2871437, -79.5974731, 79.7267456
7: -42.6416473, 36.6428299, -44.7544746, 38.4389801, -81.0806046, 81.3973083
8: -52.8406258, 36.6023598, -55.4426994, 38.3272667, -91.1678925, 92.0450439
9: -39.4985886, 38.6736450, -41.4548340, 40.5774078, -80.0759735, 80.1284790

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 208

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0724027, upper bound: 106.0744653
time: 9.16 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0817062, upper bound: 106.0821445
time: 8.96 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -36.0356941, 28.9887333, -41.2521439, 33.1719055, -69.2075958, 70.2408752
1: -29.7058315, 25.4009972, -34.1582413, 29.1661873, -58.8720169, 59.5592384
2: -39.6169357, 25.9604797, -45.4258614, 29.6851254, -69.3020630, 71.3863373
3: -41.2202682, 22.6341400, -47.3360100, 25.8899231, -67.1101913, 69.9701538
4: -37.9989166, 29.3256817, -43.6925888, 33.6226387, -71.6215515, 73.0182571
5: -34.5593605, 26.8846588, -39.6150742, 30.8626671, -65.4220276, 66.4997253
6: -32.7509003, 32.1860046, -37.6113434, 36.8546143, -69.6054993, 69.7973480
7: -35.4232521, 30.6251411, -40.7588463, 35.1037979, -70.5270309, 71.3839874
8: -44.2459068, 30.8343639, -50.5389786, 35.1575203, -79.4034271, 81.3733292
9: -32.9629593, 32.1869965, -37.8229790, 36.9877815, -69.9507294, 70.0099792

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0806589, upper bound: 106.0806589
time: 8.15 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0806589, upper bound: 106.0806589
time: 7.79 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -40.3179855, 32.4200478, -41.2521439, 33.1719055, -73.4898911, 73.6721878
1: -33.3707085, 28.4747887, -34.1582413, 29.1661873, -62.5368958, 62.6330299
2: -44.4276581, 29.0729008, -45.4258614, 29.6851254, -74.1127853, 74.4987564
3: -46.2721939, 25.3028603, -47.3360100, 25.8899231, -72.1621094, 72.6388702
4: -42.6485062, 32.8563194, -43.6925888, 33.6226387, -76.2711334, 76.5488968
5: -38.6677094, 30.1147766, -39.6150742, 30.8626671, -69.5303802, 69.7298355
6: -36.7717171, 36.0077209, -37.6113434, 36.8546143, -73.6263123, 73.6190643
7: -39.8440437, 34.3165054, -40.7588463, 35.1037979, -74.9478302, 75.0753479
8: -49.5568047, 34.3538094, -50.5389786, 35.1575203, -84.7143250, 84.8927765
9: -36.9784622, 36.1488571, -37.8229790, 36.9877815, -73.9662247, 73.9718323

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0806589, upper bound: 106.0811610
time: 6.82 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0806589, upper bound: 106.0811610
time: 8.16 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -30.6780338, 24.6635189, -40.3761673, 32.4794655, -63.1574974, 65.0396881
1: -25.1336918, 21.6248493, -33.4265480, 28.5657597, -53.6994514, 55.0513954
2: -33.6105042, 22.2396297, -44.4748383, 29.1061745, -62.7166786, 66.7144699
3: -35.0010376, 19.3285789, -46.3509064, 25.3727608, -60.3737946, 65.6794891
4: -32.1081734, 24.9012318, -42.7081795, 32.8968697, -65.0050430, 67.6094131
5: -29.2282753, 22.8215122, -38.7060509, 30.2022266, -59.4304962, 61.5275650
6: -27.9234905, 27.2820969, -36.8539467, 36.0463562, -63.9698486, 64.1360397
7: -29.9728107, 25.9615917, -39.9274712, 34.3449135, -64.3177261, 65.8890457
8: -37.9805145, 26.3577156, -49.6014404, 34.4095764, -72.3900833, 75.9591370
9: -27.8378887, 27.3232403, -36.9893303, 36.2330132, -64.0709000, 64.3125687

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0739076, upper bound: 106.0720237
time: 8.98 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0739076, upper bound: 106.0720237
time: 11.30 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -41.6441803, 33.4633942, -45.6412506, 36.6860809, -78.3302612, 79.1046448
1: -34.4892464, 29.4344959, -37.9206543, 32.3214989, -66.8107452, 67.3551483
2: -45.9206238, 30.0472393, -50.3660812, 32.8492813, -78.7698898, 80.4133072
3: -47.8680229, 26.1408367, -52.5289612, 28.6323700, -76.5003738, 78.6697998
4: -44.0930405, 33.9610710, -48.4673195, 37.2462769, -81.3393173, 82.4283905
5: -39.9047585, 31.0959072, -43.8367958, 34.1677818, -74.0725403, 74.9327011
6: -38.0478973, 37.1783333, -41.7422371, 40.7786331, -78.8265305, 78.9205627
7: -41.2237740, 35.4528580, -45.2862968, 38.8793983, -80.1031723, 80.7391510
8: -51.2288055, 35.4568634, -55.9860725, 38.7860641, -90.0148621, 91.4429245
9: -38.2101021, 37.3996811, -41.9251213, 41.0584564, -79.2685547, 79.3247986

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0810338, upper bound: 106.0810340
time: 8.71 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0810338, upper bound: 106.0810338
time: 9.35 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 20.36 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.36
Output dim: 7, lower bound: -106.0846088, upper bound: 106.0846088
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.36
Output dim: 7, lower bound: -106.0846088, upper bound: 106.0850090
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.36
Output dim: 7, lower bound: -106.0850090, upper bound: 106.0851551
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.36
Output dim: 7, lower bound: -106.0850090, upper bound: 106.0864312
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.36
Output dim: 7, lower bound: -106.0813473, upper bound: 106.0817327
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.36
Output dim: 7, lower bound: -106.0813473, upper bound: 106.0822145
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.36
Output dim: 7, lower bound: -106.0724027, upper bound: 106.0744653
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.36
Output dim: 7, lower bound: -106.0817062, upper bound: 106.0821445
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.36
Output dim: 7, lower bound: -106.0806589, upper bound: 106.0806589
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.36
Output dim: 7, lower bound: -106.0806589, upper bound: 106.0806589
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.36
Output dim: 7, lower bound: -106.0806589, upper bound: 106.0811610
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.36
Output dim: 7, lower bound: -106.0806589, upper bound: 106.0811610
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.36
Output dim: 7, lower bound: -106.0739076, upper bound: 106.0720237
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.36
Output dim: 7, lower bound: -106.0739076, upper bound: 106.0720237
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.36
Output dim: 7, lower bound: -106.0810338, upper bound: 106.0810340
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.36
Output dim: 7, lower bound: -106.0810338, upper bound: 106.0810338

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -38.8772850, 31.2613239, -38.8772850, 31.2613239, -70.1385956, 70.1385956
1: -32.1913948, 27.4707890, -32.1913948, 27.4707890, -59.6621819, 59.6621819
2: -42.8070412, 28.0077896, -42.8070412, 28.0077896, -70.8148193, 70.8148193
3: -44.5495644, 24.4045963, -44.5495644, 24.4045963, -68.9541626, 68.9541626
4: -41.1276894, 31.6824036, -41.1276894, 31.6824036, -72.8100891, 72.8100891
5: -37.3024979, 29.0631142, -37.3024979, 29.0631142, -66.3656158, 66.3656158
6: -35.4128723, 34.7436562, -35.4128723, 34.7436562, -70.1565247, 70.1565247
7: -38.3617210, 33.0808830, -38.3617210, 33.0808830, -71.4426041, 71.4426041
8: -47.6955681, 33.1865578, -47.6955681, 33.1865578, -80.8821182, 80.8821182
9: -35.6262093, 34.8328972, -35.6262093, 34.8328972, -70.4590912, 70.4590912

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 240

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 208

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0751179, upper bound: 106.0733916
time: 8.72 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0831402, upper bound: 106.0831402
time: 9.78 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -38.8772850, 31.2613239, -43.0196724, 34.5813866, -73.4586639, 74.2809982
1: -32.1913948, 27.4707890, -35.7369156, 30.4450989, -62.6364861, 63.2077026
2: -42.8070412, 28.0077896, -47.4723358, 30.9982471, -73.8052673, 75.4801254
3: -44.5495644, 24.4045963, -49.4507256, 26.9920750, -71.5416412, 73.8553085
4: -41.1276894, 31.6824036, -45.6297493, 35.1038322, -76.2315216, 77.3121490
5: -37.3024979, 29.0631142, -41.2890892, 32.1810265, -69.4835205, 70.3522034
6: -35.4128723, 34.7436562, -39.3103333, 38.4477692, -73.8606415, 74.0539856
7: -38.3617210, 33.0808830, -42.6416473, 36.6428299, -75.0045395, 75.7225342
8: -47.6955681, 33.1865578, -52.8406258, 36.6023598, -84.2979202, 86.0271835
9: -35.6262093, 34.8328972, -39.4985886, 38.6736450, -74.2998352, 74.3314743

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 240

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 208

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0751179, upper bound: 106.0737645
time: 9.95 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0831402, upper bound: 106.0835710
time: 8.11 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -43.0196724, 34.5813866, -38.8772850, 31.2613239, -74.2809982, 73.4586639
1: -35.7369156, 30.4450989, -32.1913948, 27.4707890, -63.2077026, 62.6364861
2: -47.4723358, 30.9982471, -42.8070412, 28.0077896, -75.4801254, 73.8052673
3: -49.4507256, 26.9920750, -44.5495644, 24.4045963, -73.8553085, 71.5416412
4: -45.6297493, 35.1038322, -41.1276894, 31.6824036, -77.3121490, 76.2315216
5: -41.2890892, 32.1810265, -37.3024979, 29.0631142, -70.3522034, 69.4835205
6: -39.3103333, 38.4477692, -35.4128723, 34.7436562, -74.0539856, 73.8606415
7: -42.6416473, 36.6428299, -38.3617210, 33.0808830, -75.7225342, 75.0045395
8: -52.8406258, 36.6023598, -47.6955681, 33.1865578, -86.0271835, 84.2979202
9: -39.4985886, 38.6736450, -35.6262093, 34.8328972, -74.3314743, 74.2998352

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 208

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0754941, upper bound: 106.0738580
time: 10.28 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0835710, upper bound: 106.0837545
time: 9.72 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -43.0196724, 34.5813866, -43.0196724, 34.5813866, -77.6010590, 77.6010590
1: -35.7369156, 30.4450989, -35.7369156, 30.4450989, -66.1820145, 66.1820145
2: -47.4723358, 30.9982471, -47.4723358, 30.9982471, -78.4705734, 78.4705734
3: -49.4507256, 26.9920750, -49.4507256, 26.9920750, -76.4427948, 76.4427948
4: -45.6297493, 35.1038322, -45.6297493, 35.1038322, -80.7335815, 80.7335815
5: -41.2890892, 32.1810265, -41.2890892, 32.1810265, -73.4701157, 73.4701157
6: -39.3103333, 38.4477692, -39.3103333, 38.4477692, -77.7580948, 77.7580948
7: -42.6416473, 36.6428299, -42.6416473, 36.6428299, -79.2844696, 79.2844696
8: -52.8406258, 36.6023598, -52.8406258, 36.6023598, -89.4429855, 89.4429855
9: -39.4985886, 38.6736450, -39.4985886, 38.6736450, -78.1722183, 78.1722183

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 208

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0754941, upper bound: 106.0738580
time: 9.27 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0835710, upper bound: 106.0850669
time: 8.85 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -38.8772850, 31.2613239, -36.0356941, 28.9887333, -67.8660202, 67.2970123
1: -32.1913948, 27.4707890, -29.7058315, 25.4009972, -57.5923920, 57.1766129
2: -42.8070412, 28.0077896, -39.6169357, 25.9604797, -68.7675171, 67.6247253
3: -44.5495644, 24.4045963, -41.2202682, 22.6341400, -67.1837006, 65.6248627
4: -41.1276894, 31.6824036, -37.9989166, 29.3256817, -70.4533691, 69.6813202
5: -37.3024979, 29.0631142, -34.5593605, 26.8846588, -64.1871567, 63.6224747
6: -35.4128723, 34.7436562, -32.7509003, 32.1860046, -67.5988770, 67.4945374
7: -38.3617210, 33.0808830, -35.4232521, 30.6251411, -68.9868546, 68.5041351
8: -47.6955681, 33.1865578, -44.2459068, 30.8343639, -78.5299225, 77.4324646
9: -35.6262093, 34.8328972, -32.9629593, 32.1869965, -67.8132019, 67.7958527

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 240

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 208

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0722299, upper bound: 106.0707073
time: 9.99 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0797773, upper bound: 106.0801287
time: 9.54 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -38.8772850, 31.2613239, -40.3179855, 32.4200478, -71.2973251, 71.5793076
1: -32.1913948, 27.4707890, -33.3707085, 28.4747887, -60.6661835, 60.8414955
2: -42.8070412, 28.0077896, -44.4276581, 29.0729008, -71.8799210, 72.4354477
3: -44.5495644, 24.4045963, -46.2721939, 25.3028603, -69.8524246, 70.6767807
4: -41.1276894, 31.6824036, -42.6485062, 32.8563194, -73.9840012, 74.3309021
5: -37.3024979, 29.0631142, -38.6677094, 30.1147766, -67.4172668, 67.7308197
6: -35.4128723, 34.7436562, -36.7717171, 36.0077209, -71.4205933, 71.5153580
7: -38.3617210, 33.0808830, -39.8440437, 34.3165054, -72.6782150, 72.9249268
8: -47.6955681, 33.1865578, -49.5568047, 34.3538094, -82.0493698, 82.7433624
9: -35.6262093, 34.8328972, -36.9784622, 36.1488571, -71.7750473, 71.8113556

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 240

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 208

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0722299, upper bound: 106.0711859
time: 10.72 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0797773, upper bound: 106.0806761
time: 9.77 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -37.7689438, 30.3831444, -30.6780338, 24.6635189, -62.4324570, 61.0611763
1: -31.2541656, 26.7030811, -25.1336918, 21.6248493, -52.8790092, 51.8367729
2: -41.6006203, 27.2674942, -33.6105042, 22.2396297, -63.8402443, 60.8779984
3: -43.2837524, 23.7367249, -35.0010376, 19.3285789, -62.6123314, 58.7377625
4: -39.8933334, 30.7602539, -32.1081734, 24.9012318, -64.7945633, 62.8684273
5: -36.1678085, 28.2300720, -29.2282753, 22.8215122, -58.9893188, 57.4583435
6: -34.4433746, 33.7279854, -27.9234905, 27.2820969, -61.7254639, 61.6514740
7: -37.2908669, 32.1203613, -29.9728107, 25.9615917, -63.2524567, 62.0931702
8: -46.4846611, 32.2509766, -37.9805145, 26.3577156, -72.8423691, 70.2314682
9: -34.5694466, 33.8650742, -27.8378887, 27.3232403, -61.8926849, 61.7029648

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0611413, upper bound: 106.0634312
time: 9.33 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0601048, upper bound: 106.0621227
time: 9.52 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -42.9895821, 34.5573540, -41.6441803, 33.4633942, -76.4529724, 76.2015305
1: -35.7112045, 30.4236393, -34.4892464, 29.4344959, -65.1456985, 64.9128876
2: -47.4388885, 30.9767838, -45.9206238, 30.0472393, -77.4861221, 76.8973999
3: -49.4154091, 26.9734821, -47.8680229, 26.1408367, -75.5562439, 74.8414917
4: -45.5969772, 35.0790939, -44.0930405, 33.9610710, -79.5580444, 79.1721344
5: -41.2598495, 32.1583595, -39.9047585, 31.0959072, -72.3557587, 72.0631104
6: -39.2823677, 38.4208984, -38.0478973, 37.1783333, -76.4606781, 76.4687958
7: -42.6111298, 36.6170158, -41.2237740, 35.4528580, -78.0639877, 77.8407822
8: -52.8041039, 36.5774803, -51.2288055, 35.4568634, -88.2609711, 87.8062668
9: -39.4705620, 38.6461525, -38.2101021, 37.3996811, -76.8702393, 76.8562546

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 208

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0743867, upper bound: 106.0726901
time: 11.37 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0743867, upper bound: 106.0821445
time: 9.94 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -36.0356941, 28.9887333, -38.8772850, 31.2613239, -67.2970123, 67.8660202
1: -29.7058315, 25.4009972, -32.1913948, 27.4707890, -57.1766129, 57.5923920
2: -39.6169357, 25.9604797, -42.8070412, 28.0077896, -67.6247253, 68.7675171
3: -41.2202682, 22.6341400, -44.5495644, 24.4045963, -65.6248627, 67.1837006
4: -37.9989166, 29.3256817, -41.1276894, 31.6824036, -69.6813202, 70.4533691
5: -34.5593605, 26.8846588, -37.3024979, 29.0631142, -63.6224747, 64.1871567
6: -32.7509003, 32.1860046, -35.4128723, 34.7436562, -67.4945374, 67.5988770
7: -35.4232521, 30.6251411, -38.3617210, 33.0808830, -68.5041351, 68.9868546
8: -44.2459068, 30.8343639, -47.6955681, 33.1865578, -77.4324646, 78.5299225
9: -32.9629593, 32.1869965, -35.6262093, 34.8328972, -67.7958527, 67.8132019

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 53

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 208

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0715722, upper bound: 106.0696461
time: 9.73 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0790738, upper bound: 106.0790737
time: 7.92 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -36.0356941, 28.9887333, -36.0356941, 28.9887333, -65.0244293, 65.0244293
1: -29.7058315, 25.4009972, -29.7058315, 25.4009972, -55.1068268, 55.1068268
2: -39.6169357, 25.9604797, -39.6169357, 25.9604797, -65.5774155, 65.5774155
3: -41.2202682, 22.6341400, -41.2202682, 22.6341400, -63.8544044, 63.8544044
4: -37.9989166, 29.3256817, -37.9989166, 29.3256817, -67.3246002, 67.3246002
5: -34.5593605, 26.8846588, -34.5593605, 26.8846588, -61.4440193, 61.4440193
6: -32.7509003, 32.1860046, -32.7509003, 32.1860046, -64.9368973, 64.9368973
7: -35.4232521, 30.6251411, -35.4232521, 30.6251411, -66.0483856, 66.0483856
8: -44.2459068, 30.8343639, -44.2459068, 30.8343639, -75.0802689, 75.0802689
9: -32.9629593, 32.1869965, -32.9629593, 32.1869965, -65.1499557, 65.1499557

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 53

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 208

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0715722, upper bound: 106.0696461
time: 9.28 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0715722, upper bound: 106.0790737
time: 9.47 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -40.3179855, 32.4200478, -38.8772850, 31.2613239, -71.5793076, 71.2973251
1: -33.3707085, 28.4747887, -32.1913948, 27.4707890, -60.8414955, 60.6661835
2: -44.4276581, 29.0729008, -42.8070412, 28.0077896, -72.4354477, 71.8799210
3: -46.2721939, 25.3028603, -44.5495644, 24.4045963, -70.6767807, 69.8524246
4: -42.6485062, 32.8563194, -41.1276894, 31.6824036, -74.3309021, 73.9840012
5: -38.6677094, 30.1147766, -37.3024979, 29.0631142, -67.7308197, 67.4172668
6: -36.7717171, 36.0077209, -35.4128723, 34.7436562, -71.5153580, 71.4205933
7: -39.8440437, 34.3165054, -38.3617210, 33.0808830, -72.9249268, 72.6782150
8: -49.5568047, 34.3538094, -47.6955681, 33.1865578, -82.7433624, 82.0493698
9: -36.9784622, 36.1488571, -35.6262093, 34.8328972, -71.8113556, 71.7750473

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 53

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 208

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0720473, upper bound: 106.0703435
time: 10.50 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0794788, upper bound: 106.0795989
time: 9.78 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -40.3179855, 32.4200478, -36.0356941, 28.9887333, -69.3067169, 68.4557419
1: -33.3707085, 28.4747887, -29.7058315, 25.4009972, -58.7717056, 58.1806183
2: -44.4276581, 29.0729008, -39.6169357, 25.9604797, -70.3881378, 68.6898346
3: -46.2721939, 25.3028603, -41.2202682, 22.6341400, -68.9063263, 66.5231323
4: -42.6485062, 32.8563194, -37.9989166, 29.3256817, -71.9741745, 70.8552399
5: -38.6677094, 30.1147766, -34.5593605, 26.8846588, -65.5523682, 64.6741257
6: -36.7717171, 36.0077209, -32.7509003, 32.1860046, -68.9577103, 68.7586060
7: -39.8440437, 34.3165054, -35.4232521, 30.6251411, -70.4691849, 69.7397461
8: -49.5568047, 34.3538094, -44.2459068, 30.8343639, -80.3911591, 78.5997162
9: -36.9784622, 36.1488571, -32.9629593, 32.1869965, -69.1654587, 69.1118164

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 208

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0720473, upper bound: 106.0703435
time: 10.12 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0794788, upper bound: 106.0795989
time: 9.03 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -30.6780338, 24.6635189, -37.7689438, 30.3831444, -61.0611763, 62.4324570
1: -25.1336918, 21.6248493, -31.2541656, 26.7030811, -51.8367729, 52.8790092
2: -33.6105042, 22.2396297, -41.6006203, 27.2674942, -60.8779984, 63.8402405
3: -35.0010376, 19.3285789, -43.2837524, 23.7367249, -58.7377625, 62.6123314
4: -32.1081734, 24.9012318, -39.8933334, 30.7602539, -62.8684273, 64.7945557
5: -29.2282753, 22.8215122, -36.1678085, 28.2300720, -57.4583435, 58.9893188
6: -27.9234905, 27.2820969, -34.4433746, 33.7279854, -61.6514740, 61.7254639
7: -29.9728107, 25.9615917, -37.2908669, 32.1203613, -62.0931702, 63.2524567
8: -37.9805145, 26.3577156, -46.4846611, 32.2509766, -70.2314682, 72.8423691
9: -27.8378887, 27.3232403, -34.5694466, 33.8650742, -61.7029648, 61.8926849

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 240

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0715722, upper bound: 106.0702015
time: 10.35 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0715722, upper bound: 106.0720218
time: 9.92 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -30.6780338, 24.6635189, -35.2919273, 28.3940392, -59.0720673, 59.9554367
1: -25.1336918, 21.6248493, -29.0835361, 24.8953400, -50.0290260, 50.7083778
2: -33.6105042, 22.2396297, -38.8096275, 25.4893379, -59.0998421, 61.0492554
3: -35.0010376, 19.3285789, -40.3788567, 22.1820488, -57.1830864, 59.7074356
4: -32.1081734, 24.9012318, -37.1606445, 28.6993809, -60.8075562, 62.0618744
5: -29.2282753, 22.8215122, -33.7743835, 26.3231678, -55.5514412, 56.5958939
6: -27.9234905, 27.2820969, -32.1180611, 31.4888077, -59.4123001, 59.4001541
7: -29.9728107, 25.9615917, -34.7118721, 29.9782028, -59.9510117, 60.6734619
8: -37.9805145, 26.3577156, -43.4781189, 30.1934357, -68.1739502, 69.8358307
9: -27.8378887, 27.3232403, -32.2496796, 31.5482178, -59.3861084, 59.5729218

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 240

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0715722, upper bound: 106.0702015
time: 9.13 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0715722, upper bound: 106.0720218
time: 9.39 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 13.64 + 596.20 = 609.85 seconds
