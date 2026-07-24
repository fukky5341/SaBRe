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
execution time: IAR + RelationalAnalysis = 2.32 + 10.95 = 13.27 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -73.8667418, upper bound: 73.8667418

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 108

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8666026, upper bound: 73.8666286
time: 7.84 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8665635, upper bound: 73.8665635
time: 8.12 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 16.21 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 16.21
Output dim: 9, lower bound: -73.8666026, upper bound: 73.8666286
NS_A2, status: Status.UNKNOWN, split count: 1, time: 16.21
Output dim: 9, lower bound: -73.8665635, upper bound: 73.8665635

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -41.9552040, 34.9413033, -43.4184456, 36.1431007, -78.0983047, 78.3597488
1: -33.2997169, 30.4907455, -34.4742126, 31.5347042, -64.8344193, 64.9649582
2: -45.5276489, 30.8985691, -47.1179047, 31.9604969, -77.4881439, 78.0164719
3: -50.7755966, 25.7245560, -52.5462761, 26.6071281, -77.3827133, 78.2708282
4: -45.9074059, 34.2733688, -47.4847603, 35.4454269, -81.3528290, 81.7581329
5: -40.4165268, 31.6543522, -41.8246727, 32.7417221, -73.1582489, 73.4790268
6: -39.1585159, 37.7151985, -40.5129280, 39.0177689, -78.1762848, 78.2281265
7: -42.7177734, 35.7938194, -44.2025414, 37.0240555, -79.7418213, 79.9963531
8: -52.0247536, 34.5292816, -53.8204155, 35.7223511, -87.7470932, 88.3496780
9: -41.0868912, 35.9621048, -42.4777756, 37.2206841, -78.3075714, 78.4398651

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 108

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8665635, upper bound: 73.8665635
time: 7.97 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8665635, upper bound: 73.8665635
time: 7.44 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -45.3043518, 37.6697235, -43.1755028, 35.9439125, -81.2482605, 80.8452301
1: -36.0654221, 32.9002457, -34.2801933, 31.3621979, -67.4276199, 67.1804352
2: -49.2113991, 33.2841911, -46.8541260, 31.7831917, -80.9945908, 80.1383209
3: -54.8335609, 27.8340836, -52.2550468, 26.4630737, -81.2966309, 80.0891266
4: -49.4980164, 36.9965744, -47.2232323, 35.2516479, -84.7496643, 84.2198029
5: -43.6440659, 34.1433525, -41.5917664, 32.5616226, -76.2056885, 75.7350998
6: -42.2432365, 40.7238617, -40.2889748, 38.8027306, -81.0459671, 81.0128326
7: -46.0569763, 38.6111069, -43.9562454, 36.8204918, -82.8774719, 82.5673370
8: -56.1463432, 37.3227654, -53.5230637, 35.5250740, -91.6714172, 90.8458252
9: -44.2005234, 38.9269371, -42.2494469, 37.0131683, -81.2136917, 81.1763840

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 108

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8665635, upper bound: 73.8665635
time: 7.07 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8665635, upper bound: 73.8665635
time: 7.96 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 17.26 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 17.26
Output dim: 9, lower bound: -73.8665635, upper bound: 73.8665635
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 17.26
Output dim: 9, lower bound: -73.8665635, upper bound: 73.8665635
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 17.26
Output dim: 9, lower bound: -73.8665635, upper bound: 73.8665635
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 17.26
Output dim: 9, lower bound: -73.8665635, upper bound: 73.8665635

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -41.9552040, 34.9413033, -41.9552040, 34.9413033, -76.8965073, 76.8965073
1: -33.2997169, 30.4907455, -33.2997169, 30.4907455, -63.7904625, 63.7904625
2: -45.5276489, 30.8985691, -45.5276489, 30.8985691, -76.4262161, 76.4262161
3: -50.7755966, 25.7245560, -50.7755966, 25.7245560, -76.5001526, 76.5001526
4: -45.9074059, 34.2733688, -45.9074059, 34.2733688, -80.1807709, 80.1807709
5: -40.4165268, 31.6543522, -40.4165268, 31.6543522, -72.0708771, 72.0708771
6: -39.1585159, 37.7151985, -39.1585159, 37.7151985, -76.8737183, 76.8737183
7: -42.7177734, 35.7938194, -42.7177734, 35.7938194, -78.5115891, 78.5115891
8: -52.0247536, 34.5292816, -52.0247536, 34.5292816, -86.5540314, 86.5540314
9: -41.0868912, 35.9621048, -41.0868912, 35.9621048, -77.0489807, 77.0489807

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 108

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8630914, upper bound: 73.8636219
time: 10.09 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8624052, upper bound: 73.8624053
time: 6.65 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -41.9552040, 34.9413033, -45.2952805, 37.6639061, -79.6191101, 80.2365875
1: -33.2997169, 30.4907455, -36.0593491, 32.8946686, -66.1943817, 66.5500946
2: -45.5276489, 30.8985691, -49.2011452, 33.2779350, -78.8055878, 80.0997162
3: -50.7755966, 25.7245560, -54.8253746, 27.8292065, -78.6048050, 80.5499268
4: -45.9074059, 34.2733688, -49.4887009, 36.9890366, -82.8964386, 83.7620697
5: -40.4165268, 31.6543522, -43.6364021, 34.1382828, -74.5548096, 75.2907562
6: -39.1585159, 37.7151985, -42.2364922, 40.7150574, -79.8735733, 79.9516907
7: -42.7177734, 35.7938194, -46.0494003, 38.6050415, -81.3228149, 81.8432159
8: -52.0247536, 34.5292816, -56.1340523, 37.3129196, -89.3376770, 90.6633301
9: -41.0868912, 35.9621048, -44.1929092, 38.9204025, -80.0072937, 80.1550064

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 108

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8630914, upper bound: 73.8636289
time: 11.56 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8624052, upper bound: 73.8624058
time: 13.54 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -45.3043518, 37.6697235, -41.9552040, 34.9413033, -80.2456512, 79.6249237
1: -36.0654221, 32.9002457, -33.2997169, 30.4907455, -66.5561676, 66.1999588
2: -49.2113991, 33.2841911, -45.5276489, 30.8985691, -80.1099701, 78.8118439
3: -54.8335609, 27.8340836, -50.7755966, 25.7245560, -80.5581207, 78.6096802
4: -49.4980164, 36.9965744, -45.9074059, 34.2733688, -83.7713852, 82.9039764
5: -43.6440659, 34.1433525, -40.4165268, 31.6543522, -75.2984161, 74.5598755
6: -42.2432365, 40.7238617, -39.1585159, 37.7151985, -79.9584351, 79.8823776
7: -46.0569763, 38.6111069, -42.7177734, 35.7938194, -81.8507996, 81.3288651
8: -56.1463432, 37.3227654, -52.0247536, 34.5292816, -90.6756210, 89.3475113
9: -44.2005234, 38.9269371, -41.0868912, 35.9621048, -80.1626205, 80.0138245

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 108

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8630666, upper bound: 73.8636008
time: 10.10 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8624058, upper bound: 73.8624073
time: 8.00 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -45.3043518, 37.6697235, -45.2952805, 37.6639061, -82.9682541, 82.9650040
1: -36.0654221, 32.9002457, -36.0593491, 32.8946686, -68.9600906, 68.9595947
2: -49.2113991, 33.2841911, -49.2011452, 33.2779350, -82.4893341, 82.4853363
3: -54.8335609, 27.8340836, -54.8253746, 27.8292065, -82.6627655, 82.6594543
4: -49.4980164, 36.9965744, -49.4887009, 36.9890366, -86.4870529, 86.4852753
5: -43.6440659, 34.1433525, -43.6364021, 34.1382828, -77.7823486, 77.7797546
6: -42.2432365, 40.7238617, -42.2364922, 40.7150574, -82.9582977, 82.9603577
7: -46.0569763, 38.6111069, -46.0494003, 38.6050415, -84.6620178, 84.6605072
8: -56.1463432, 37.3227654, -56.1340523, 37.3129196, -93.4592590, 93.4568024
9: -44.2005234, 38.9269371, -44.1929092, 38.9204025, -83.1209259, 83.1198425

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 108

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8630666, upper bound: 73.8636008
time: 11.28 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8624058, upper bound: 73.8624101
time: 7.66 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 21.16 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.16
Output dim: 9, lower bound: -73.8630914, upper bound: 73.8636219
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.16
Output dim: 9, lower bound: -73.8624052, upper bound: 73.8624053
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.16
Output dim: 9, lower bound: -73.8630914, upper bound: 73.8636289
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.16
Output dim: 9, lower bound: -73.8624052, upper bound: 73.8624058
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.16
Output dim: 9, lower bound: -73.8630666, upper bound: 73.8636008
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.16
Output dim: 9, lower bound: -73.8624058, upper bound: 73.8624073
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.16
Output dim: 9, lower bound: -73.8630666, upper bound: 73.8636008
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.16
Output dim: 9, lower bound: -73.8624058, upper bound: 73.8624101

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -40.7637100, 33.9660873, -41.9552040, 34.9413033, -75.7050171, 75.9212952
1: -32.3470917, 29.6526985, -33.2997169, 30.4907455, -62.8378372, 62.9524155
2: -44.2371216, 30.0425682, -45.5276489, 30.8985691, -75.1356888, 75.5702209
3: -49.3580132, 25.0145435, -50.7755966, 25.7245560, -75.0825653, 75.7901382
4: -44.6303940, 33.3231201, -45.9074059, 34.2733688, -78.9037628, 79.2305222
5: -39.2700005, 30.7721176, -40.4165268, 31.6543522, -70.9243546, 71.1886444
6: -38.0686150, 36.6629829, -39.1585159, 37.7151985, -75.7838135, 75.8215027
7: -41.5144806, 34.8010597, -42.7177734, 35.7938194, -77.3082809, 77.5188217
8: -50.5759888, 33.5684929, -52.0247536, 34.5292816, -85.1052704, 85.5932465
9: -39.9688530, 34.9376831, -41.0868912, 35.9621048, -75.9309540, 76.0245667

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 108

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8624053, upper bound: 73.8624053
time: 7.35 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8624053, upper bound: 73.8624053
time: 9.13 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -45.4667549, 37.7858429, -41.0225945, 34.1776161, -79.6443634, 78.8084412
1: -36.2191391, 33.0267029, -32.5537338, 29.8348160, -66.0539551, 65.5804367
2: -49.4070053, 33.3764496, -44.5173912, 30.2277508, -79.6347580, 77.8938370
3: -55.0613365, 27.8935146, -49.6666756, 25.1693687, -80.2307053, 77.5601883
4: -49.6832771, 37.1419678, -44.9067993, 33.5298080, -83.2130737, 82.0487518
5: -43.7761650, 34.2483826, -39.5189400, 30.9638290, -74.7399673, 73.7673187
6: -42.3474121, 40.8670158, -38.3057938, 36.8916092, -79.2390213, 79.1728058
7: -46.2372017, 38.7120819, -41.7755966, 35.0165405, -81.2537384, 80.4876785
8: -56.3297806, 37.5068207, -50.8895721, 33.7769623, -90.1067429, 88.3963776
9: -44.3375549, 39.0121689, -40.2130241, 35.1596909, -79.4972458, 79.2251816

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 108

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8585671, upper bound: 73.8583871
time: 9.64 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8581492, upper bound: 73.8581492
time: 28.77 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -40.7637100, 33.9660873, -45.2952805, 37.6639061, -78.4276123, 79.2613678
1: -32.3470917, 29.6526985, -36.0593491, 32.8946686, -65.2417526, 65.7120514
2: -44.2371216, 30.0425682, -49.2011452, 33.2779350, -77.5150604, 79.2437057
3: -49.3580132, 25.0145435, -54.8253746, 27.8292065, -77.1872177, 79.8399048
4: -44.6303940, 33.3231201, -49.4887009, 36.9890366, -81.6194153, 82.8118210
5: -39.2700005, 30.7721176, -43.6364021, 34.1382828, -73.4082794, 74.4085236
6: -38.0686150, 36.6629829, -42.2364922, 40.7150574, -78.7836685, 78.8994751
7: -41.5144806, 34.8010597, -46.0494003, 38.6050415, -80.1195145, 80.8504486
8: -50.5759888, 33.5684929, -56.1340523, 37.3129196, -87.8889084, 89.7025375
9: -39.9688530, 34.9376831, -44.1929092, 38.9204025, -78.8892517, 79.1305923

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 108

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8624073, upper bound: 73.8624058
time: 74.17 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8624073, upper bound: 73.8624058
time: 8.57 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -45.4667549, 37.7858429, -44.4038925, 36.9355927, -82.4023361, 82.1897278
1: -36.2191391, 33.0267029, -35.3458366, 32.2680893, -68.4872208, 68.3725357
2: -49.4070053, 33.3764496, -48.2357025, 32.6394348, -82.0464401, 81.6121521
3: -55.0613365, 27.8935146, -53.7665291, 27.2973213, -82.3586502, 81.6600342
4: -49.6832771, 37.1419678, -48.5330429, 36.2764893, -85.9597626, 85.6750031
5: -43.7761650, 34.2483826, -42.7776299, 33.4781570, -77.2543182, 77.0260086
6: -42.3474121, 40.8670158, -41.4211388, 39.9272156, -82.2746277, 82.2881546
7: -46.2372017, 38.7120819, -45.1494637, 37.8622093, -84.0994110, 83.8615417
8: -56.3297806, 37.5068207, -55.0497665, 36.5937805, -92.9235611, 92.5565796
9: -44.3375549, 39.0121689, -43.3587341, 38.1522408, -82.4897919, 82.3708878

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 108

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8585690, upper bound: 73.8583968
time: 8.35 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8581492, upper bound: 73.8581493
time: 6.80 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -44.1183624, 36.7016144, -41.9552040, 34.9413033, -79.0596619, 78.6568146
1: -35.1174622, 32.0663567, -33.2997169, 30.4907455, -65.6082077, 65.3660736
2: -47.9273491, 32.4348183, -45.5276489, 30.8985691, -78.8259201, 77.9624634
3: -53.4241257, 27.1260052, -50.7755966, 25.7245560, -79.1486816, 77.9016037
4: -48.2280540, 36.0481262, -45.9074059, 34.2733688, -82.5014191, 81.9555206
5: -42.5021019, 33.2661743, -40.4165268, 31.6543522, -74.1564560, 73.6826935
6: -41.1588783, 39.6750031, -39.1585159, 37.7151985, -78.8740692, 78.8335190
7: -44.8604622, 37.6231155, -42.7177734, 35.7938194, -80.6542740, 80.3408890
8: -54.7047386, 36.3650017, -52.0247536, 34.5292816, -89.2340240, 88.3897552
9: -43.0882721, 37.9062958, -41.0868912, 35.9621048, -79.0503769, 78.9931870

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 108

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8624058, upper bound: 73.8624073
time: 6.74 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8624058, upper bound: 73.8624073
time: 8.46 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -49.3102493, 40.9101562, -41.0225945, 34.1776161, -83.4878693, 81.9327545
1: -39.3794975, 35.7870407, -32.5537338, 29.8348160, -69.2143097, 68.3407669
2: -53.6212311, 36.1153145, -44.5173912, 30.2277508, -83.8489761, 80.6326981
3: -59.7228394, 30.3091335, -49.6666756, 25.1693687, -84.8922043, 79.9758072
4: -53.7988281, 40.2622147, -44.9067993, 33.5298080, -87.3286133, 85.1690140
5: -47.4745178, 37.0958786, -39.5189400, 30.9638290, -78.4383316, 76.6148224
6: -45.8796883, 44.3219872, -38.3057938, 36.8916092, -82.7713013, 82.6277771
7: -50.0628891, 41.9415436, -41.7755966, 35.0165405, -85.0794296, 83.7171402
8: -61.0557899, 40.7115746, -50.8895721, 33.7769623, -94.8327484, 91.6011429
9: -47.9068260, 42.3947563, -40.2130241, 35.1596909, -83.0665131, 82.6077805

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 108

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8585671, upper bound: 73.8583858
time: 8.22 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8581493, upper bound: 73.8581492
time: 7.91 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -44.1183624, 36.7016144, -45.2952805, 37.6639061, -81.7822723, 81.9968872
1: -35.1174622, 32.0663567, -36.0593491, 32.8946686, -68.0121307, 68.1257019
2: -47.9273491, 32.4348183, -49.2011452, 33.2779350, -81.2052841, 81.6359634
3: -53.4241257, 27.1260052, -54.8253746, 27.8292065, -81.2533340, 81.9513779
4: -48.2280540, 36.0481262, -49.4887009, 36.9890366, -85.2170868, 85.5368195
5: -42.5021019, 33.2661743, -43.6364021, 34.1382828, -76.6403809, 76.9025726
6: -41.1588783, 39.6750031, -42.2364922, 40.7150574, -81.8739319, 81.9114990
7: -44.8604622, 37.6231155, -46.0494003, 38.6050415, -83.4654999, 83.6725159
8: -54.7047386, 36.3650017, -56.1340523, 37.3129196, -92.0176544, 92.4990463
9: -43.0882721, 37.9062958, -44.1929092, 38.9204025, -82.0086746, 82.0992050

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 108

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8624102, upper bound: 73.8624101
time: 8.53 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8624102, upper bound: 73.8624101
time: 7.89 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -49.3102493, 40.9101562, -44.4038925, 36.9355927, -86.2458420, 85.3140488
1: -39.3794975, 35.7870407, -35.3458366, 32.2680893, -71.6475830, 71.1328735
2: -53.6212311, 36.1153145, -48.2357025, 32.6394348, -86.2606659, 84.3510132
3: -59.7228394, 30.3091335, -53.7665291, 27.2973213, -87.0201492, 84.0756607
4: -53.7988281, 40.2622147, -48.5330429, 36.2764893, -90.0753098, 88.7952576
5: -47.4745178, 37.0958786, -42.7776299, 33.4781570, -80.9526749, 79.8735046
6: -45.8796883, 44.3219872, -41.4211388, 39.9272156, -85.8069000, 85.7431259
7: -50.0628891, 41.9415436, -45.1494637, 37.8622093, -87.9250946, 87.0910034
8: -61.0557899, 40.7115746, -55.0497665, 36.5937805, -97.6495667, 95.7613373
9: -47.9068260, 42.3947563, -43.3587341, 38.1522408, -86.0590668, 85.7534866

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 108

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8585678, upper bound: 73.8583858
time: 9.74 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8581493, upper bound: 73.8581492
time: 8.05 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 19.99 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 19.99
Output dim: 9, lower bound: -73.8624053, upper bound: 73.8624053
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.99
Output dim: 9, lower bound: -73.8624053, upper bound: 73.8624053
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.99
Output dim: 9, lower bound: -73.8585671, upper bound: 73.8583871
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.99
Output dim: 9, lower bound: -73.8581492, upper bound: 73.8581492
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 19.99
Output dim: 9, lower bound: -73.8624073, upper bound: 73.8624058
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.99
Output dim: 9, lower bound: -73.8624073, upper bound: 73.8624058
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.99
Output dim: 9, lower bound: -73.8585690, upper bound: 73.8583968
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.99
Output dim: 9, lower bound: -73.8581492, upper bound: 73.8581493
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 19.99
Output dim: 9, lower bound: -73.8624058, upper bound: 73.8624073
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.99
Output dim: 9, lower bound: -73.8624058, upper bound: 73.8624073
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.99
Output dim: 9, lower bound: -73.8585671, upper bound: 73.8583858
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.99
Output dim: 9, lower bound: -73.8581493, upper bound: 73.8581492
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 19.99
Output dim: 9, lower bound: -73.8624102, upper bound: 73.8624101
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.99
Output dim: 9, lower bound: -73.8624102, upper bound: 73.8624101
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.99
Output dim: 9, lower bound: -73.8585678, upper bound: 73.8583858
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.99
Output dim: 9, lower bound: -73.8581493, upper bound: 73.8581492

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -40.7637100, 33.9660873, -40.7637100, 33.9660873, -74.7297974, 74.7297974
1: -32.3470917, 29.6526985, -32.3470917, 29.6526985, -61.9997902, 61.9997826
2: -44.2371216, 30.0425682, -44.2371216, 30.0425682, -74.2796783, 74.2796783
3: -49.3580132, 25.0145435, -49.3580132, 25.0145435, -74.3725510, 74.3725510
4: -44.6303940, 33.3231201, -44.6303940, 33.3231201, -77.9535141, 77.9535141
5: -39.2700005, 30.7721176, -39.2700005, 30.7721176, -70.0421143, 70.0421143
6: -38.0686150, 36.6629829, -38.0686150, 36.6629829, -74.7315979, 74.7315979
7: -41.5144806, 34.8010597, -41.5144806, 34.8010597, -76.3155136, 76.3155212
8: -50.5759888, 33.5684929, -50.5759888, 33.5684929, -84.1444855, 84.1444855
9: -39.9688530, 34.9376831, -39.9688530, 34.9376831, -74.9065399, 74.9065399

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 108

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8590376, upper bound: 73.8597008
time: 8.04 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8589063, upper bound: 73.8594753
time: 8.90 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -40.7637100, 33.9660873, -45.4667549, 37.7858429, -78.5495529, 79.4328384
1: -32.3470917, 29.6526985, -36.2191391, 33.0267029, -65.3737946, 65.8718414
2: -44.2371216, 30.0425682, -49.4070053, 33.3764496, -77.6135712, 79.4495697
3: -49.3580132, 25.0145435, -55.0613365, 27.8935146, -77.2515182, 80.0758743
4: -44.6303940, 33.3231201, -49.6832771, 37.1419678, -81.7723389, 83.0063934
5: -39.2700005, 30.7721176, -43.7761650, 34.2483826, -73.5183868, 74.5482788
6: -38.0686150, 36.6629829, -42.3474121, 40.8670158, -78.9356232, 79.0103912
7: -41.5144806, 34.8010597, -46.2372017, 38.7120819, -80.2265625, 81.0382538
8: -50.5759888, 33.5684929, -56.3297806, 37.5068207, -88.0828094, 89.8982697
9: -39.9688530, 34.9376831, -44.3375549, 39.0121689, -78.9810104, 79.2752380

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 108

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8590376, upper bound: 73.8597008
time: 9.53 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8589063, upper bound: 73.8594753
time: 9.36 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -43.9457893, 36.5499535, -33.6176529, 28.1567841, -72.1025696, 70.1676025
1: -34.9372025, 31.9285583, -26.3344917, 24.5059032, -59.4431076, 58.2630463
2: -47.7158241, 32.2807426, -36.3293839, 24.8940887, -72.6099091, 68.6101227
3: -53.2377472, 26.9531612, -40.8191261, 20.5821304, -73.8198776, 67.7722702
4: -48.0470963, 35.8881149, -36.9458618, 27.4499664, -75.4970627, 72.8339767
5: -42.3023796, 33.1072655, -32.3563118, 25.4359074, -67.7382812, 65.4635773
6: -40.9397011, 39.4940376, -31.4586887, 30.2353878, -71.1750793, 70.9527283
7: -44.6938591, 37.4258537, -34.2772713, 28.7581406, -73.4519882, 71.7031250
8: -54.4709167, 36.2473946, -41.8438301, 27.6600208, -82.1309357, 78.0912170
9: -42.9319611, 37.6428871, -33.3706589, 28.5212021, -71.4531631, 71.0135422

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 108

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8565751, upper bound: 73.8562921
time: 9.30 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8585671, upper bound: 73.8583871
time: 7.94 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -42.3093987, 35.2269096, -33.2208290, 27.8656197, -70.1750183, 68.4477310
1: -33.5595741, 30.7542896, -25.9388866, 24.2359447, -57.7955170, 56.6931725
2: -45.9030647, 31.1031494, -35.8840446, 24.6186352, -70.5216980, 66.9871902
3: -51.2833939, 25.9431591, -40.3936310, 20.3031559, -71.5865479, 66.3367920
4: -46.2959595, 34.5416031, -36.5524483, 27.0862350, -73.3821945, 71.0940475
5: -40.7221298, 31.8834953, -31.9783440, 25.1252499, -65.8473816, 63.8618393
6: -39.4311600, 38.0257187, -31.1089745, 29.8824024, -69.3135605, 69.1346893
7: -43.0427971, 36.0499649, -33.8990898, 28.4183979, -71.4611969, 69.9490433
8: -52.4770050, 34.8927269, -41.4056168, 27.3374081, -79.8144150, 76.2983398
9: -41.4300385, 36.1675110, -33.0755272, 28.0795956, -69.5096359, 69.2430420

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 108

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8561611, upper bound: 73.8560282
time: 11.06 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8581492, upper bound: 73.8581492
time: 9.90 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -40.7637100, 33.9660873, -44.1140404, 36.6988182, -77.4625244, 78.0801239
1: -32.3470917, 29.6526985, -35.1145706, 32.0636826, -64.4107742, 64.7672729
2: -44.2371216, 30.0425682, -47.9224472, 32.4318161, -76.6689377, 77.9650116
3: -49.3580132, 25.0145435, -53.4202080, 27.1236706, -76.4816818, 78.4347534
4: -44.6303940, 33.3231201, -48.2235909, 36.0445251, -80.6749191, 81.5467072
5: -39.2700005, 30.7721176, -42.4984398, 33.2637596, -72.5337601, 73.2705536
6: -38.0686150, 36.6629829, -41.1556435, 39.6708069, -77.7394257, 77.8186188
7: -41.5144806, 34.8010597, -44.8568382, 37.6202049, -79.1346588, 79.6578827
8: -50.5759888, 33.5684929, -54.6988487, 36.3603058, -86.9362946, 88.2673416
9: -39.9688530, 34.9376831, -43.0846291, 37.9031677, -77.8720245, 78.0223083

Time for backsubstitution: 2.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 108

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8590501, upper bound: 73.8597151
time: 10.25 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8589152, upper bound: 73.8594789
time: 9.89 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -40.7637100, 33.9660873, -49.2598495, 40.8775940, -81.6413040, 83.2259369
1: -32.3470917, 29.6526985, -39.3455544, 35.7559700, -68.1030579, 68.9982529
2: -44.2371216, 30.0425682, -53.5642166, 36.0803413, -80.3174591, 83.6067734
3: -49.3580132, 25.0145435, -59.6769218, 30.2818813, -79.6398849, 84.6914597
4: -44.6303940, 33.3231201, -53.7468109, 40.2204132, -84.8508072, 87.0699310
5: -39.2700005, 30.7721176, -47.4318237, 37.0676842, -76.3376846, 78.2039261
6: -38.0686150, 36.6629829, -45.8418961, 44.2731819, -82.3417892, 82.5048752
7: -41.5144806, 34.8010597, -50.0206604, 41.9074860, -83.4219513, 84.8217087
8: -50.5759888, 33.5684929, -60.9874344, 40.6569252, -91.2329102, 94.5559158
9: -39.9688530, 34.9376831, -47.8645592, 42.3583069, -82.3271637, 82.8022385

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 108

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8590501, upper bound: 73.8597151
time: 10.14 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8589152, upper bound: 73.8594789
time: 8.97 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -43.9457893, 36.5499535, -36.9178352, 30.8508549, -74.7966461, 73.4677887
1: -34.9372025, 31.9285583, -29.0609856, 26.8721256, -61.8093262, 60.9895325
2: -47.7158241, 32.2807426, -39.9402695, 27.2463226, -74.9621277, 72.2210083
3: -53.2377472, 26.9531612, -44.8203583, 22.6683235, -75.9060669, 71.7734985
4: -48.0470963, 35.8881149, -40.4928818, 30.1186714, -78.1657715, 76.3809814
5: -42.3023796, 33.1072655, -35.5348396, 27.8822174, -70.1845779, 68.6421051
6: -40.9397011, 39.4940376, -34.5002975, 33.1930580, -74.1327591, 73.9943390
7: -44.6938591, 37.4258537, -37.5732918, 31.5408535, -76.2347107, 74.9991455
8: -54.4709167, 36.2473946, -45.9069710, 30.3961544, -84.8670731, 82.1543655
9: -42.9319611, 37.6428871, -36.4459114, 31.4325981, -74.3645630, 74.0887909

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 108

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8565484, upper bound: 73.8562715
time: 7.87 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8585690, upper bound: 73.8583968
time: 9.27 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -42.3093987, 35.2269096, -36.3763580, 30.4406738, -72.7500610, 71.6032639
1: -33.5595741, 30.7542896, -28.5435829, 26.4919205, -60.0514946, 59.2978630
2: -45.9030647, 31.1031494, -39.3344116, 26.8650608, -72.7681274, 70.4375610
3: -51.2833939, 25.9431591, -44.2188797, 22.2971916, -73.5805817, 70.1620407
4: -46.2959595, 34.5416031, -39.9409752, 29.6381931, -75.9341507, 74.4825668
5: -40.7221298, 31.8834953, -35.0172043, 27.4601021, -68.1822357, 66.9006958
6: -39.4311600, 38.0257187, -34.0178146, 32.7072906, -72.1384506, 72.0435257
7: -43.0427971, 36.0499649, -37.0495033, 31.0777531, -74.1205521, 73.0994720
8: -52.4770050, 34.8927269, -45.2862854, 29.9516220, -82.4286118, 80.1790085
9: -41.4300385, 36.1675110, -36.0135231, 30.8623466, -72.2923889, 72.1810226

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 108

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8561446, upper bound: 73.8560181
time: 8.43 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8581492, upper bound: 73.8581493
time: 8.71 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -44.1183624, 36.7016144, -40.7637100, 33.9660873, -78.0844498, 77.4653244
1: -35.1174622, 32.0663567, -32.3470917, 29.6526985, -64.7701569, 64.4134369
2: -47.9273491, 32.4348183, -44.2371216, 30.0425682, -77.9699173, 76.6719360
3: -53.4241257, 27.1260052, -49.3580132, 25.0145435, -78.4386597, 76.4840164
4: -48.2280540, 36.0481262, -44.6303940, 33.3231201, -81.5511780, 80.6785126
5: -42.5021019, 33.2661743, -39.2700005, 30.7721176, -73.2742157, 72.5361710
6: -41.1588783, 39.6750031, -38.0686150, 36.6629829, -77.8218536, 77.7436218
7: -44.8604622, 37.6231155, -41.5144806, 34.8010597, -79.6614990, 79.1375885
8: -54.7047386, 36.3650017, -50.5759888, 33.5684929, -88.2732315, 86.9409943
9: -43.0882721, 37.9062958, -39.9688530, 34.9376831, -78.0259552, 77.8751526

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 108

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8590264, upper bound: 73.8596921
time: 10.23 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588988, upper bound: 73.8594640
time: 11.53 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -44.1183624, 36.7016144, -45.4667549, 37.7858429, -81.9042053, 82.1683655
1: -35.1174622, 32.0663567, -36.2191391, 33.0267029, -68.1441650, 68.2854919
2: -47.9273491, 32.4348183, -49.4070053, 33.3764496, -81.3038025, 81.8418274
3: -53.4241257, 27.1260052, -55.0613365, 27.8935146, -81.3176193, 82.1873398
4: -48.2280540, 36.0481262, -49.6832771, 37.1419678, -85.3700104, 85.7313995
5: -42.5021019, 33.2661743, -43.7761650, 34.2483826, -76.7504883, 77.0423279
6: -41.1588783, 39.6750031, -42.3474121, 40.8670158, -82.0258865, 82.0224152
7: -44.8604622, 37.6231155, -46.2372017, 38.7120819, -83.5725403, 83.8603210
8: -54.7047386, 36.3650017, -56.3297806, 37.5068207, -92.2115555, 92.6947784
9: -43.0882721, 37.9062958, -44.3375549, 39.0121689, -82.1004333, 82.2438507

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 108

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8590264, upper bound: 73.8596921
time: 10.25 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588988, upper bound: 73.8594640
time: 10.82 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 13.27 + 591.85 = 605.12 seconds
