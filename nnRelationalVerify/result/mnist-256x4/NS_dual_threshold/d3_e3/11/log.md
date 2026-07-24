## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 73.7928750582


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820)
1: (-34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384)
2: (-47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675)
3: (-53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606)
4: (-47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160)
5: (-42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925)
6: (-40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805)
7: (-44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109)
8: (-54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457)
9: (-42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.81 + 10.59 = 11.40 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -73.8667418, upper bound: 73.8667418

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8660987, upper bound: 73.8661358
time: 7.46 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8667418, upper bound: 73.8667418
time: 7.82 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 15.35 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 15.35
Output dim: 9, lower bound: -73.8660987, upper bound: 73.8661358
NS_B2, status: Status.UNKNOWN, split count: 1, time: 15.35
Output dim: 9, lower bound: -73.8667418, upper bound: 73.8667418

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -43.3486595, 36.0913048, -47.3745499, 39.3558083, -82.7044525, 83.4658432
1: -34.4152489, 31.4890862, -37.7715950, 34.3604851, -68.7757339, 69.2606812
2: -47.0388412, 31.9161606, -51.4815025, 34.8175926, -81.8564301, 83.3976517
3: -52.4679565, 26.5668087, -57.2593613, 29.0056801, -81.4736328, 83.8261642
4: -47.4091682, 35.3890648, -51.7362175, 38.6926384, -86.1017914, 87.1252823
5: -41.7576065, 32.6922760, -45.6506615, 35.6969490, -77.4545364, 78.3429413
6: -40.4538383, 38.9556236, -44.1699219, 42.5525703, -83.0064087, 83.1255341
7: -44.1353149, 36.9684715, -48.2111778, 40.3428993, -84.4781952, 85.1796494
8: -53.7397652, 35.6706352, -58.6446228, 38.9975662, -92.7373352, 94.3152313
9: -42.4229393, 37.1565742, -46.1277008, 40.7373047, -83.1602478, 83.2842712

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 247

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_B1_B1

### Relational analysis result of NS_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8587155, upper bound: 73.8592412
time: 185.72 seconds

## Relational analysis of NS_B1_B2

### Relational analysis result of NS_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8636932, upper bound: 73.8637383
time: 10.89 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -43.8349380, 36.4853439, -43.5510902, 36.2549553, -80.0898895, 80.0364380
1: -34.8081284, 31.8318081, -34.5783997, 31.6313457, -66.4394760, 66.4102020
2: -47.5706978, 32.2634697, -47.2594337, 32.0603905, -79.6310883, 79.5228958
3: -53.0499039, 26.8578606, -52.7100830, 26.6885586, -79.7384567, 79.5679321
4: -47.9338379, 35.7787933, -47.6267815, 35.5511093, -83.4849091, 83.4055710
5: -42.2251434, 33.0509529, -41.9524498, 32.8418007, -75.0669327, 75.0034027
6: -40.8981514, 39.3887291, -40.6384430, 39.1351700, -80.0333252, 80.0271683
7: -44.6253395, 37.3742828, -44.3389549, 37.1370277, -81.7623672, 81.7132263
8: -54.3321075, 36.0625381, -53.9859161, 35.8325958, -90.1647034, 90.0484543
9: -42.8744965, 37.5781250, -42.6103706, 37.3318710, -80.2063675, 80.1884918

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_B2_B1

### Relational analysis result of NS_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8622616, upper bound: 73.8629576
time: 9.74 seconds

## Relational analysis of NS_B2_B2

### Relational analysis result of NS_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8667418, upper bound: 73.8667418
time: 7.17 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 17.75 seconds
NS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 17.75
Output dim: 9, lower bound: -73.8587155, upper bound: 73.8592412
NS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 17.75
Output dim: 9, lower bound: -73.8636932, upper bound: 73.8637383
NS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 17.75
Output dim: 9, lower bound: -73.8622616, upper bound: 73.8629576
NS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 17.75
Output dim: 9, lower bound: -73.8667418, upper bound: 73.8667418

## BFS NS instance: NS_B1_B1

### Backsubstitution after applying NS history:
0: -37.0059624, 30.9711666, -27.6590366, 23.4537086, -60.4596558, 58.6302032
1: -29.0997448, 26.9777470, -21.2788162, 20.3865833, -49.4863281, 48.2565613
2: -40.0508041, 27.3737450, -29.8043823, 20.7244492, -60.7752533, 57.1781235
3: -44.9635925, 22.7092190, -33.9901314, 16.9943161, -61.9579086, 56.6993484
4: -40.5758743, 30.1633434, -30.5171585, 22.4529076, -63.0287819, 60.6804962
5: -35.6059189, 27.9538212, -26.5420074, 20.9888363, -56.5947571, 54.4958229
6: -34.6029282, 33.3047600, -25.9885941, 25.0181484, -59.6210785, 59.2933502
7: -37.7447014, 31.6505184, -28.3653908, 23.8274689, -61.5721703, 60.0159035
8: -46.0487022, 30.4515800, -34.7908707, 22.8096390, -68.8583374, 65.2424469
9: -36.6361771, 31.4625511, -28.1594505, 23.0447540, -59.6809235, 59.6220016

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 153

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_B1_B1_A1

### Relational analysis result of NS_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8575076, upper bound: 73.8575504
time: 11.12 seconds

## Relational analysis of NS_B1_B1_A2

### Relational analysis result of NS_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8575076, upper bound: 73.8592412
time: 9.68 seconds

## BFS NS instance: NS_B1_B2

### Backsubstitution after applying NS history:
0: -41.8170319, 34.8573494, -40.3006439, 33.6579361, -75.4749680, 75.1579895
1: -33.1321564, 30.3986378, -31.8387394, 29.3237686, -62.4559250, 62.2373695
2: -45.3501549, 30.8217793, -43.6743011, 29.7611656, -75.1113129, 74.4960709
3: -50.6538773, 25.6336975, -48.8884125, 24.6897297, -75.3435898, 74.5220947
4: -45.7615128, 34.1307068, -44.1287956, 32.8740463, -78.6355591, 78.2594910
5: -40.2753792, 31.5482712, -38.8030434, 30.4124298, -70.6877975, 70.3513031
6: -39.0422287, 37.5896072, -37.6490936, 36.2383957, -75.2806168, 75.2386780
7: -42.5922356, 35.6861229, -41.0854187, 34.4188766, -77.0110931, 76.7715302
8: -51.8786583, 34.4086914, -50.0432854, 33.1627083, -85.0413666, 84.4519730
9: -41.0253105, 35.7818222, -39.6755676, 34.3808784, -75.4061890, 75.4573898

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 247

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_B1_B2_A1

### Relational analysis result of NS_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8599902, upper bound: 73.8601411
time: 9.59 seconds

## Relational analysis of NS_B1_B2_A2

### Relational analysis result of NS_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8599267, upper bound: 73.8599649
time: 12.06 seconds

## BFS NS instance: NS_B2_B1

### Backsubstitution after applying NS history:
0: -43.2403717, 36.0006943, -47.9377785, 39.7944984, -83.0348663, 83.9384766
1: -34.3293953, 31.4113827, -38.3441620, 34.7856636, -69.1150589, 69.7555466
2: -46.9252243, 31.8380508, -52.1689644, 35.2137604, -82.1389694, 84.0070190
3: -52.3345108, 26.4991722, -57.8509140, 29.3943596, -81.7288666, 84.3500824
4: -47.2927589, 35.3006554, -52.3172569, 39.1840820, -86.4768372, 87.6179047
5: -41.6519318, 32.6076851, -46.1985893, 36.1176987, -77.7696228, 78.8062668
6: -40.3545265, 38.8614235, -44.6974831, 43.0945702, -83.4490967, 83.5588989
7: -44.0260773, 36.8765564, -48.7508965, 40.8209572, -84.8470306, 85.6274414
8: -53.6066818, 35.5791931, -59.3780556, 39.4991379, -93.1058197, 94.9572372
9: -42.3170853, 37.0651627, -46.5710678, 41.3619804, -83.6790619, 83.6362305

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 247

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_B2_B1_B1

### Relational analysis result of NS_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8543451, upper bound: 73.8552462
time: 11.06 seconds

## Relational analysis of NS_B2_B1_B2

### Relational analysis result of NS_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8599258, upper bound: 73.8605152
time: 10.91 seconds

## BFS NS instance: NS_B2_B2

### Backsubstitution after applying NS history:
0: -43.8349380, 36.4853439, -43.2922783, 36.0433884, -79.8783264, 79.7776184
1: -34.8081284, 31.8318081, -34.3717422, 31.4478073, -66.2559280, 66.2035370
2: -47.5706978, 32.2634697, -46.9789658, 31.8750687, -79.4457703, 79.2424316
3: -53.0499039, 26.8578606, -52.3973579, 26.5331783, -79.5830841, 79.2552032
4: -47.9338379, 35.7787933, -47.3473587, 35.3436966, -83.2775116, 83.1261520
5: -42.2251434, 33.0509529, -41.7034073, 32.6491394, -74.8742752, 74.7543640
6: -40.8981514, 39.3887291, -40.4014549, 38.9055710, -79.8037262, 79.7901840
7: -44.6253395, 37.3742828, -44.0783119, 36.9208183, -81.5461578, 81.4525833
8: -54.3321075, 36.0625381, -53.6690063, 35.6216812, -89.9537811, 89.7315445
9: -42.8744965, 37.5781250, -42.3666039, 37.1105919, -79.9850922, 79.9447174

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_B2_A1

### Relational analysis result of NS_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8659414, upper bound: 73.8660987
time: 7.16 seconds

## Relational analysis of NS_B2_B2_A2

### Relational analysis result of NS_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8659414, upper bound: 73.8667400
time: 6.95 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 14.92 seconds
NS_B1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 14.92
Output dim: 9, lower bound: -73.8575076, upper bound: 73.8575504
NS_B1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 14.92
Output dim: 9, lower bound: -73.8575076, upper bound: 73.8592412
NS_B1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 14.92
Output dim: 9, lower bound: -73.8599902, upper bound: 73.8601411
NS_B1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 14.92
Output dim: 9, lower bound: -73.8599267, upper bound: 73.8599649
NS_B2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 14.92
Output dim: 9, lower bound: -73.8543451, upper bound: 73.8552462
NS_B2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 14.92
Output dim: 9, lower bound: -73.8599258, upper bound: 73.8605152
NS_B2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 14.92
Output dim: 9, lower bound: -73.8659414, upper bound: 73.8660987
NS_B2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 14.92
Output dim: 9, lower bound: -73.8659414, upper bound: 73.8667400

## BFS NS instance: NS_B1_B1_A1

### Backsubstitution after applying NS history:
0: -24.6971436, 21.0487061, -27.6590366, 23.4537086, -48.1508484, 48.7077408
1: -18.8347816, 18.2842140, -21.2788162, 20.3865833, -39.2213631, 39.5630264
2: -26.5523014, 18.5971336, -29.8043823, 20.7244492, -47.2767487, 48.4015160
3: -30.4682312, 15.1813040, -33.9901314, 16.9943161, -47.4625435, 49.1714363
4: -27.3315430, 20.0266037, -30.5171585, 22.4529076, -49.7844505, 50.5437622
5: -23.6864433, 18.7937965, -26.5420074, 20.9888363, -44.6752777, 45.3358040
6: -23.2391453, 22.3834190, -25.9885941, 25.0181484, -48.2572899, 48.3720131
7: -25.3723087, 21.3512039, -28.3653908, 23.8274689, -49.1997757, 49.7165947
8: -31.2048569, 20.3848495, -34.7908707, 22.8096390, -54.0144958, 55.1757164
9: -25.4168739, 20.4155769, -28.1594505, 23.0447540, -48.4616203, 48.5750275

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 153

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_B1_B1_A1_B1

### Relational analysis result of NS_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8524473, upper bound: 73.8524797
time: 10.23 seconds

## Relational analysis of NS_B1_B1_A1_B2

### Relational analysis result of NS_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8519102, upper bound: 73.8521561
time: 7.73 seconds

## BFS NS instance: NS_B1_B1_A2

### Backsubstitution after applying NS history:
0: -36.3644638, 30.4599323, -27.6590366, 23.4537086, -59.8181686, 58.1189690
1: -28.5618267, 26.5207062, -21.2788162, 20.3865833, -48.9484062, 47.7995224
2: -39.3470192, 26.9258785, -29.8043823, 20.7244492, -60.0714684, 56.7302628
3: -44.2022552, 22.3071709, -33.9901314, 16.9943161, -61.1965599, 56.2972946
4: -39.8993225, 29.6499767, -30.5171585, 22.4529076, -62.3522186, 60.1671371
5: -35.0038490, 27.4853420, -26.5420074, 20.9888363, -55.9926834, 54.0273438
6: -34.0197029, 32.7366371, -25.9885941, 25.0181484, -59.0378494, 58.7252312
7: -37.1066818, 31.1172886, -28.3653908, 23.8274689, -60.9341507, 59.4826813
8: -45.2498932, 29.9184551, -34.7908707, 22.8096390, -68.0595322, 64.7093201
9: -36.0521965, 30.8794918, -28.1594505, 23.0447540, -59.0969467, 59.0389404

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 153

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_B1_B1_A2_A1

### Relational analysis result of NS_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8519523, upper bound: 73.8542399
time: 10.84 seconds

## Relational analysis of NS_B1_B1_A2_A2

### Relational analysis result of NS_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8519102, upper bound: 73.8541664
time: 8.38 seconds

## BFS NS instance: NS_B1_B2_A1

### Backsubstitution after applying NS history:
0: -32.1935654, 27.0255127, -38.6972847, 32.3550339, -64.5485992, 65.7227783
1: -25.0717297, 23.4998913, -30.4935493, 28.1706066, -53.2423363, 53.9934387
2: -34.7308197, 23.8504543, -41.9026718, 28.5985184, -63.3293381, 65.7531281
3: -39.1944122, 19.7169437, -46.9779472, 23.7040081, -62.8984222, 66.6948853
4: -35.3497124, 26.1698418, -42.3946724, 31.5443974, -66.8941040, 68.5645142
5: -30.9262218, 24.3235912, -37.2447205, 29.2075195, -60.1337433, 61.5682869
6: -30.1167278, 28.9690723, -36.1604843, 34.8009834, -64.9177094, 65.1295471
7: -32.8162766, 27.5417747, -39.4569397, 33.0623055, -65.8785629, 66.9987030
8: -40.1158447, 26.4864273, -48.0851746, 31.8404236, -71.9562683, 74.5716019
9: -32.1323051, 27.1115170, -38.1950645, 32.9313202, -65.0636215, 65.3065796

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 153

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_B1_B2_A1_B1

### Relational analysis result of NS_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8599050, upper bound: 73.8599609
time: 10.71 seconds

## Relational analysis of NS_B1_B2_A1_B2

### Relational analysis result of NS_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8599050, upper bound: 73.8599609
time: 7.77 seconds

## BFS NS instance: NS_B1_B2_A2

### Backsubstitution after applying NS history:
0: -34.0304985, 28.5532322, -36.9252510, 30.9171104, -64.9476089, 65.4784851
1: -26.5093002, 24.8279209, -29.0165329, 26.9104652, -53.4197655, 53.8444481
2: -36.7157707, 25.2115536, -39.9573212, 27.3261967, -64.0419693, 65.1688766
3: -41.4548798, 20.8139439, -44.8743134, 22.6206245, -64.0755005, 65.6882553
4: -37.3563919, 27.6333313, -40.4884720, 30.0916519, -67.4480438, 68.1218033
5: -32.6793022, 25.6930046, -35.5293808, 27.8874474, -60.5667496, 61.2223816
6: -31.8192940, 30.6140423, -34.5263023, 33.2223778, -65.0416718, 65.1403351
7: -34.7145157, 29.1144352, -37.6744843, 31.5739174, -66.2884293, 66.7889023
8: -42.3630333, 27.9673271, -45.9157372, 30.3814945, -72.7445297, 73.8830643
9: -33.9615746, 28.6076794, -36.5725098, 31.3433018, -65.3048782, 65.1801834

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 153

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_B1_B2_A2_B1

### Relational analysis result of NS_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8599267, upper bound: 73.8599649
time: 11.50 seconds

## Relational analysis of NS_B1_B2_A2_B2

### Relational analysis result of NS_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8532678, upper bound: 73.8599649
time: 10.44 seconds

## BFS NS instance: NS_B2_B1_B1

### Backsubstitution after applying NS history:
0: -36.9051781, 30.8862152, -28.1842461, 23.8549480, -60.7601242, 59.0704613
1: -29.0202198, 26.9049511, -21.8000565, 20.7581673, -49.7783890, 48.7049980
2: -39.9454498, 27.3014336, -30.4165955, 21.0744362, -61.0198860, 57.7180252
3: -44.8380814, 22.6462879, -34.5163116, 17.3552647, -62.1933441, 57.1625977
4: -40.4676819, 30.0810604, -31.0358334, 22.9047661, -63.3724480, 61.1168938
5: -35.5074158, 27.8749046, -27.0592308, 21.3658218, -56.8732376, 54.9341354
6: -34.5104828, 33.2172699, -26.4699154, 25.5036831, -60.0141678, 59.6871834
7: -37.6430283, 31.5646381, -28.8473949, 24.2688084, -61.9118195, 60.4120293
8: -45.9252853, 30.3672981, -35.4435501, 23.2729607, -69.1982422, 65.8108521
9: -36.5368538, 31.3781719, -28.5585594, 23.6304035, -60.1672592, 59.9367294

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 153

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_B2_B1_B1_A1

### Relational analysis result of NS_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8532447, upper bound: 73.8538511
time: 6.53 seconds

## Relational analysis of NS_B2_B1_B1_A2

### Relational analysis result of NS_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8532447, upper bound: 73.8552462
time: 9.02 seconds

## BFS NS instance: NS_B2_B1_B2

### Backsubstitution after applying NS history:
0: -41.7156219, 34.7721901, -40.7392540, 33.9986343, -75.7142563, 75.5114441
1: -33.0520439, 30.3258133, -32.3119621, 29.6621094, -62.7141533, 62.6377754
2: -45.2441177, 30.7486458, -44.2261581, 30.0713615, -75.3154755, 74.9748077
3: -50.5282097, 25.5702801, -49.3335876, 25.0055218, -75.5337296, 74.9038696
4: -45.6526070, 34.0481377, -44.5776787, 33.2642593, -78.9168625, 78.6258087
5: -40.1762695, 31.4686565, -39.2349739, 30.7423668, -70.9186401, 70.7036285
6: -38.9493256, 37.5015373, -38.0655136, 36.6665268, -75.6158524, 75.5670395
7: -42.4897728, 35.5997505, -41.5015907, 34.7960091, -77.2857819, 77.1013412
8: -51.7545700, 34.3238831, -50.6229286, 33.5607872, -85.3153534, 84.9468079
9: -40.9256744, 35.6968918, -40.0052185, 34.9077339, -75.8334045, 75.7021103

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_B1_B2_A1

### Relational analysis result of NS_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8576334, upper bound: 73.8593671
time: 11.06 seconds

## Relational analysis of NS_B2_B1_B2_A2

### Relational analysis result of NS_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8576334, upper bound: 73.8604468
time: 10.56 seconds

## BFS NS instance: NS_B2_B2_A1

### Backsubstitution after applying NS history:
0: -47.3745499, 39.3558083, -43.2922783, 36.0433884, -83.4179382, 82.6480865
1: -37.7715950, 34.3604851, -34.3717422, 31.4478073, -69.2194061, 68.7322235
2: -51.4815025, 34.8175926, -46.9789658, 31.8750687, -83.3565674, 81.7965546
3: -57.2593613, 29.0056801, -52.3973579, 26.5331783, -83.7925415, 81.4030304
4: -51.7362175, 38.6926384, -47.3473587, 35.3436966, -87.0799026, 86.0399933
5: -45.6506615, 35.6969490, -41.7034073, 32.6491394, -78.2997971, 77.4003525
6: -44.1699219, 42.5525703, -40.4014549, 38.9055710, -83.0754929, 82.9540253
7: -48.2111778, 40.3428993, -44.0783119, 36.9208183, -85.1319962, 84.4212036
8: -58.6446228, 38.9975662, -53.6690063, 35.6216812, -94.2662964, 92.6665726
9: -46.1277008, 40.7373047, -42.3666039, 37.1105919, -83.2382965, 83.1039124

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 247

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_B2_B2_A1_A1

### Relational analysis result of NS_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8590365, upper bound: 73.8587155
time: 11.47 seconds

## Relational analysis of NS_B2_B2_A1_A2

### Relational analysis result of NS_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8590365, upper bound: 73.8636932
time: 11.37 seconds

## BFS NS instance: NS_B2_B2_A2

### Backsubstitution after applying NS history:
0: -43.5510902, 36.2549553, -43.2922783, 36.0433884, -79.5944824, 79.5472336
1: -34.5783997, 31.6313457, -34.3717422, 31.4478073, -66.0261917, 66.0030899
2: -47.2594337, 32.0603905, -46.9789658, 31.8750687, -79.1344986, 79.0393524
3: -52.7100830, 26.6885586, -52.3973579, 26.5331783, -79.2432556, 79.0858994
4: -47.6267815, 35.5511093, -47.3473587, 35.3436966, -82.9704666, 82.8984604
5: -41.9524498, 32.8418007, -41.7034073, 32.6491394, -74.6015930, 74.5451965
6: -40.6384430, 39.1351700, -40.4014549, 38.9055710, -79.5440063, 79.5366211
7: -44.3389549, 37.1370277, -44.0783119, 36.9208183, -81.2597656, 81.2153397
8: -53.9859161, 35.8325958, -53.6690063, 35.6216812, -89.6075974, 89.5016022
9: -42.6103706, 37.3318710, -42.3666039, 37.1105919, -79.7209625, 79.6984711

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_B2_B2_A2_A1

### Relational analysis result of NS_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8656306, upper bound: 73.8666252
time: 8.60 seconds

## Relational analysis of NS_B2_B2_A2_A2

### Relational analysis result of NS_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8655643, upper bound: 73.8665619
time: 7.71 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 17.15 seconds
NS_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.15
Output dim: 9, lower bound: -73.8524473, upper bound: 73.8524797
NS_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.15
Output dim: 9, lower bound: -73.8519102, upper bound: 73.8521561
NS_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 17.15
Output dim: 9, lower bound: -73.8519523, upper bound: 73.8542399
NS_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 17.15
Output dim: 9, lower bound: -73.8519102, upper bound: 73.8541664
NS_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.15
Output dim: 9, lower bound: -73.8599050, upper bound: 73.8599609
NS_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.15
Output dim: 9, lower bound: -73.8599050, upper bound: 73.8599609
NS_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.15
Output dim: 9, lower bound: -73.8599267, upper bound: 73.8599649
NS_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.15
Output dim: 9, lower bound: -73.8532678, upper bound: 73.8599649
NS_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 17.15
Output dim: 9, lower bound: -73.8532447, upper bound: 73.8538511
NS_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 17.15
Output dim: 9, lower bound: -73.8532447, upper bound: 73.8552462
NS_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 17.15
Output dim: 9, lower bound: -73.8576334, upper bound: 73.8593671
NS_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 17.15
Output dim: 9, lower bound: -73.8576334, upper bound: 73.8604468
NS_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 17.15
Output dim: 9, lower bound: -73.8590365, upper bound: 73.8587155
NS_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 17.15
Output dim: 9, lower bound: -73.8590365, upper bound: 73.8636932
NS_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 17.15
Output dim: 9, lower bound: -73.8656306, upper bound: 73.8666252
NS_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 17.15
Output dim: 9, lower bound: -73.8655643, upper bound: 73.8665619

## BFS NS instance: NS_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -23.4850063, 20.0639572, -20.1802654, 17.3899689, -40.8749771, 40.2442169
1: -17.8260002, 17.4205761, -15.0912743, 15.0657463, -32.8917465, 32.5118484
2: -25.2170296, 17.7199783, -21.5783958, 15.3085737, -40.5256004, 39.2983742
3: -29.0213566, 14.4263697, -25.0732803, 12.3628378, -41.3841934, 39.4996452
4: -26.0183220, 19.0178661, -22.4266376, 16.2366199, -42.2549400, 41.4444962
5: -22.5113506, 17.8933697, -19.3087158, 15.4151659, -37.9265175, 37.2020798
6: -22.1106148, 21.2937851, -19.0033817, 18.3155785, -40.4261894, 40.2971649
7: -24.1448631, 20.3258781, -20.7903156, 17.5268211, -41.6716843, 41.1161957
8: -29.7249565, 19.3930283, -25.6645622, 16.6917419, -46.4166946, 45.0575905
9: -24.2877922, 19.3214626, -21.1834183, 16.3121452, -40.5999374, 40.5048752

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 153

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_B1_B1_A1_B1_A1

### Relational analysis result of NS_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8518758, upper bound: 73.8521066
time: 8.66 seconds

## Relational analysis of NS_B1_B1_A1_B1_A2

### Relational analysis result of NS_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8518758, upper bound: 73.8521561
time: 8.07 seconds

## BFS NS instance: NS_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -22.1173172, 18.9556980, -22.1209641, 18.9947891, -41.1121063, 41.0766602
1: -16.6907215, 16.4527264, -16.5941887, 16.4565277, -33.1472473, 33.0469131
2: -23.7159405, 16.7379189, -23.6769829, 16.7416706, -40.4576073, 40.4148979
3: -27.3879471, 13.5789261, -27.4465332, 13.5239906, -40.9119377, 41.0254517
4: -24.5447578, 17.8881226, -24.5341930, 17.7886810, -42.3334351, 42.4223099
5: -21.1907482, 16.8810081, -21.1479225, 16.8593044, -38.0500488, 38.0289268
6: -20.8362389, 20.0708141, -20.8096771, 20.0539055, -40.8901443, 40.8804893
7: -22.7690220, 19.1782722, -22.7774658, 19.1686840, -41.9377060, 41.9557381
8: -28.0487385, 18.2742596, -28.0513535, 18.2516232, -46.3003578, 46.3256149
9: -23.0199890, 18.0915051, -23.1095123, 17.9032040, -40.9231949, 41.2010155

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_B1_B1_A1_B2_A1

### Relational analysis result of NS_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8517037, upper bound: 73.8519704
time: 7.54 seconds

## Relational analysis of NS_B1_B1_A1_B2_A2

### Relational analysis result of NS_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8517033, upper bound: 73.8519704
time: 8.60 seconds

## BFS NS instance: NS_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -27.3624802, 23.1245575, -26.3169308, 22.3638000, -49.7262764, 49.4414902
1: -21.0458469, 20.0914021, -20.1658726, 19.4276524, -40.4734917, 40.2572746
2: -29.4409218, 20.4217815, -28.3266449, 19.7591858, -49.2001076, 48.7484283
3: -33.5009766, 16.7557716, -32.3953896, 16.1619453, -49.6629219, 49.1511612
4: -30.1656094, 22.2064457, -29.0680275, 21.3395939, -51.5052032, 51.2744751
5: -26.2424622, 20.7362213, -25.2371826, 19.9850521, -46.2275124, 45.9734039
6: -25.6655006, 24.6846676, -24.7375336, 23.8181934, -49.4836960, 49.4221992
7: -27.9739895, 23.5043163, -27.0048027, 22.6988316, -50.6728210, 50.5091171
8: -34.2840195, 22.5185738, -33.1598587, 21.7063179, -55.9903374, 55.6784325
9: -27.7255058, 22.7823277, -26.9137955, 21.8367329, -49.5622406, 49.6961212

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 111

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_B1_B1_A2_A1_A1

### Relational analysis result of NS_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8530727, upper bound: 73.8540340
time: 11.23 seconds

## Relational analysis of NS_B1_B1_A2_A1_A2

### Relational analysis result of NS_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8530502, upper bound: 73.8539973
time: 8.78 seconds

## BFS NS instance: NS_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -28.9980011, 24.4850159, -24.9494438, 21.2561340, -50.2541351, 49.4344521
1: -22.3035221, 21.2748013, -19.0366611, 18.4596691, -40.7631912, 40.3114548
2: -31.1950302, 21.6267014, -26.8287315, 18.7807941, -49.9758224, 48.4554329
3: -35.5228233, 17.7245617, -30.7696857, 15.3174381, -50.8402634, 48.4942474
4: -31.9494133, 23.4993382, -27.5931225, 20.2183819, -52.1677933, 51.0924606
5: -27.8003540, 21.9532471, -23.9172783, 18.9744511, -46.7748032, 45.8705177
6: -27.1819973, 26.1429424, -23.4687309, 22.5979385, -49.7799339, 49.6116714
7: -29.6617126, 24.8935528, -25.6329746, 21.5560646, -51.2177734, 50.5265236
8: -36.2783699, 23.8287773, -31.4895706, 20.5824528, -56.8608246, 55.3183441
9: -29.3706055, 24.0837078, -25.6502647, 20.6144409, -49.9850464, 49.7339706

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 153

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of NS_B1_B1_A2_A2_A1

### Relational analysis result of NS_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8446258, upper bound: 73.8454842
time: 12.24 seconds

## Relational analysis of NS_B1_B1_A2_A2_A2

### Relational analysis result of NS_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8490972, upper bound: 73.8502576
time: 11.25 seconds

## BFS NS instance: NS_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -32.1935654, 27.0255127, -30.7802505, 25.9057426, -58.0993080, 57.8057556
1: -25.0717297, 23.4998913, -23.8618164, 22.5079117, -47.5796318, 47.3617058
2: -34.7308197, 23.8504543, -33.1750412, 22.8666420, -57.5974541, 57.0254898
3: -39.1944122, 19.7169437, -37.5600052, 18.8289051, -58.0233116, 57.2769470
4: -35.3497124, 26.1698418, -33.8276062, 24.9906998, -60.3404045, 59.9974480
5: -30.9262218, 24.3235912, -29.5503254, 23.2699852, -54.1962051, 53.8739128
6: -30.1167278, 28.9690723, -28.8179188, 27.7143021, -57.8310242, 57.7869911
7: -32.8162766, 27.5417747, -31.4167309, 26.3592834, -59.1755524, 58.9585037
8: -40.1158447, 26.4864273, -38.4068184, 25.3276978, -65.4435425, 64.8932419
9: -32.1323051, 27.1115170, -30.8814201, 25.7947598, -57.9270592, 57.9929352

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 153

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_B1_B2_A1_B1_A1

### Relational analysis result of NS_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8519523, upper bound: 73.8537846
time: 9.20 seconds

## Relational analysis of NS_B1_B2_A1_B1_A2

### Relational analysis result of NS_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8519523, upper bound: 73.8522307
time: 9.84 seconds

## BFS NS instance: NS_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -32.1935654, 27.0255127, -32.8515282, 27.6183929, -59.8119545, 59.8770370
1: -25.0717297, 23.4998913, -25.5023289, 24.0011234, -49.0728531, 49.0022202
2: -34.7308197, 23.8504543, -35.4178162, 24.3903923, -59.1212082, 59.2682648
3: -39.1944122, 19.7169437, -40.0886078, 20.0756035, -59.2700157, 59.8055496
4: -35.3497124, 26.1698418, -36.0858727, 26.6541290, -62.0038376, 62.2557144
5: -30.9262218, 24.3235912, -31.5348091, 24.8119946, -55.7382164, 55.8583908
6: -30.1167278, 28.9690723, -30.7409477, 29.5658512, -59.6825790, 59.7100220
7: -32.8162766, 27.5417747, -33.5458450, 28.1228161, -60.9390831, 61.0876198
8: -40.1158447, 26.4864273, -40.9410629, 27.0096531, -67.1254959, 67.4274750
9: -32.1323051, 27.1115170, -32.9177704, 27.5146828, -59.6469841, 60.0292892

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_B1_B2_A1_B2_A1

### Relational analysis result of NS_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8596607, upper bound: 73.8598403
time: 9.53 seconds

## Relational analysis of NS_B1_B2_A1_B2_A2

### Relational analysis result of NS_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8596307, upper bound: 73.8597674
time: 10.10 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 11.40 + 595.51 = 606.92 seconds
