## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 9.007671511800002


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640)
1: (-6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886)
2: (-7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480)
3: (-9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012)
4: (-8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819)
5: (-6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571)
6: (-6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572)
7: (-8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652)
8: (-8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467)
9: (-6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.02 + 4.86 = 6.88 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -9.0166882, upper bound: 9.0166883

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 84

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0163774, upper bound: 9.0163448
time: 7.34 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0163390, upper bound: 9.0163390
time: 3.07 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 10.60 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 10.60
Output dim: 7, lower bound: -9.0163774, upper bound: 9.0163448
NS_A2, status: Status.UNKNOWN, split count: 1, time: 10.60
Output dim: 7, lower bound: -9.0163390, upper bound: 9.0163390

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -7.4363236, 5.7155976, -7.6105275, 5.8564177, -13.2927418, 13.3261251
1: -5.9711609, 5.1793494, -6.1264510, 5.3051434, -11.2763042, 11.3057995
2: -7.2749095, 4.3961573, -7.4654551, 4.5093651, -11.7842751, 11.8616123
3: -8.8425083, 4.2966771, -9.0613842, 4.4142790, -13.2567873, 13.3580608
4: -8.0005665, 6.2316737, -8.1995602, 6.3912582, -14.3918247, 14.4312325
5: -6.3874593, 5.3953571, -6.5538192, 5.5282950, -11.9157543, 11.9491768
6: -6.4942737, 6.8905997, -6.6539564, 7.0556540, -13.5499277, 13.5445557
7: -8.1398649, 4.0495753, -8.3390789, 4.1905942, -12.3304596, 12.3886547
8: -7.9943476, 5.8853970, -8.1884384, 6.0285306, -14.0228786, 14.0738354
9: -6.1536164, 6.6015611, -6.3338342, 6.7786651, -12.9322815, 12.9353952

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 84

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0163385, upper bound: 9.0163385
time: 3.18 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0163385, upper bound: 9.0163390
time: 5.14 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -7.6525245, 5.8821526, -7.6065598, 5.8531814, -13.5057020, 13.4887123
1: -6.1535773, 5.3257508, -6.1229701, 5.3022499, -11.4558268, 11.4487209
2: -7.4937177, 4.5179243, -7.4609838, 4.5067606, -12.0004768, 11.9789085
3: -9.1112919, 4.4179034, -9.0563316, 4.4116349, -13.5229263, 13.4742355
4: -8.2390251, 6.4093227, -8.1950226, 6.3876028, -14.6266279, 14.6043444
5: -6.5827065, 5.5467153, -6.5499520, 5.5251489, -12.1078548, 12.0966673
6: -6.6816645, 7.0888405, -6.6502771, 7.0519338, -13.7335949, 13.7391176
7: -8.3745699, 4.1748061, -8.3345652, 4.1872582, -12.5618267, 12.5093708
8: -8.2279015, 6.0499687, -8.1838818, 6.0252810, -14.2531815, 14.2338505
9: -6.3381491, 6.7948503, -6.3296652, 6.7746124, -13.1127615, 13.1245155

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 84

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0161459, upper bound: 9.0162726
time: 2.74 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0161285, upper bound: 9.0161285
time: 5.71 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 10.57 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 10.57
Output dim: 7, lower bound: -9.0163385, upper bound: 9.0163385
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 10.57
Output dim: 7, lower bound: -9.0163385, upper bound: 9.0163390
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 10.57
Output dim: 7, lower bound: -9.0161459, upper bound: 9.0162726
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 10.57
Output dim: 7, lower bound: -9.0161285, upper bound: 9.0161285

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -7.4363236, 5.7155976, -7.4363236, 5.7155976, -13.1519213, 13.1519213
1: -5.9711609, 5.1793494, -5.9711609, 5.1793494, -11.1505108, 11.1505108
2: -7.2749095, 4.3961573, -7.2749095, 4.3961573, -11.6710663, 11.6710663
3: -8.8425083, 4.2966771, -8.8425083, 4.2966771, -13.1391850, 13.1391850
4: -8.0005665, 6.2316737, -8.0005665, 6.2316737, -14.2322407, 14.2322407
5: -6.3874593, 5.3953571, -6.3874593, 5.3953571, -11.7828159, 11.7828159
6: -6.4942737, 6.8905997, -6.4942737, 6.8905997, -13.3848734, 13.3848734
7: -8.1398649, 4.0495753, -8.1398649, 4.0495753, -12.1894398, 12.1894398
8: -7.9943476, 5.8853970, -7.9943476, 5.8853970, -13.8797445, 13.8797445
9: -6.1536164, 6.6015611, -6.1536164, 6.6015611, -12.7551775, 12.7551775

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 84

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0163133, upper bound: 9.0161527
time: 3.98 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0161543, upper bound: 9.0161330
time: 3.89 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -7.4363236, 5.7155976, -7.6525245, 5.8821526, -13.3184757, 13.3681202
1: -5.9711609, 5.1793494, -6.1535773, 5.3257508, -11.2969112, 11.3329248
2: -7.2749095, 4.3961573, -7.4937177, 4.5179243, -11.7928333, 11.8898735
3: -8.8425083, 4.2966771, -9.1112919, 4.4179034, -13.2604103, 13.4079685
4: -8.0005665, 6.2316737, -8.2390251, 6.4093227, -14.4098892, 14.4706955
5: -6.3874593, 5.3953571, -6.5827065, 5.5467153, -11.9341745, 11.9780636
6: -6.4942737, 6.8905997, -6.6816645, 7.0888405, -13.5831146, 13.5722628
7: -8.1398649, 4.0495753, -8.3745699, 4.1748061, -12.3146706, 12.4241447
8: -7.9943476, 5.8853970, -8.2279015, 6.0499687, -14.0443163, 14.1132965
9: -6.1536164, 6.6015611, -6.3381491, 6.7948503, -12.9484653, 12.9397106

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 84

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0163133, upper bound: 9.0161527
time: 3.72 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0161543, upper bound: 9.0161331
time: 3.78 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -7.6525245, 5.8821526, -6.9709396, 5.3673358, -13.0198584, 12.8530922
1: -6.1535773, 5.3257508, -5.5967507, 4.8795094, -11.0330849, 10.9225016
2: -7.4937177, 4.5179243, -6.8246360, 4.1594505, -11.6531668, 11.3425598
3: -9.1112919, 4.4179034, -8.2793312, 4.0759087, -13.1872005, 12.6972332
4: -8.2390251, 6.4093227, -7.5047016, 5.8736138, -14.1126366, 13.9140244
5: -6.5827065, 5.5467153, -5.9794612, 5.0861740, -11.6688786, 11.5261765
6: -6.6816645, 7.0888405, -6.1069446, 6.4781866, -13.1598482, 13.1957855
7: -8.3745699, 4.1748061, -7.6592426, 3.8383603, -12.2129307, 11.8340473
8: -8.2279015, 6.0499687, -7.5008411, 5.5486674, -13.7765656, 13.5508099
9: -6.3381491, 6.7948503, -5.8013749, 6.2181902, -12.5563393, 12.5962238

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 84

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0161285, upper bound: 9.0161285
time: 4.84 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0161285, upper bound: 9.0161285
time: 4.95 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -7.5809908, 5.8274984, -7.0444646, 5.4247012, -13.0056915, 12.8719616
1: -6.0942497, 5.2780924, -5.6586404, 4.9290128, -11.0232620, 10.9367313
2: -7.4221478, 4.4788399, -6.9015031, 4.2025480, -11.6246958, 11.3803415
3: -9.0233860, 4.3795681, -8.3660269, 4.1113234, -13.1347084, 12.7455950
4: -8.1613350, 6.3514214, -7.5865097, 5.9345074, -14.0958424, 13.9379311
5: -6.5187430, 5.4973998, -6.0463629, 5.1379189, -11.6566620, 11.5437622
6: -6.6204214, 7.0242181, -6.1708026, 6.5473280, -13.1677485, 13.1950207
7: -8.2985458, 4.1356926, -7.7424045, 3.8807273, -12.1792727, 11.8780975
8: -8.1512489, 5.9961796, -7.5834627, 5.6048460, -13.7560921, 13.5796404
9: -6.2787037, 6.7323294, -5.8645058, 6.2853022, -12.5640059, 12.5968351

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 84

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0159099, upper bound: 9.0155430
time: 4.85 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155023, upper bound: 9.0155021
time: 3.36 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 10.20 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 10.20
Output dim: 7, lower bound: -9.0163133, upper bound: 9.0161527
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 10.20
Output dim: 7, lower bound: -9.0161543, upper bound: 9.0161330
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 10.20
Output dim: 7, lower bound: -9.0163133, upper bound: 9.0161527
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 10.20
Output dim: 7, lower bound: -9.0161543, upper bound: 9.0161331
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 10.20
Output dim: 7, lower bound: -9.0161285, upper bound: 9.0161285
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 10.20
Output dim: 7, lower bound: -9.0161285, upper bound: 9.0161285
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 10.20
Output dim: 7, lower bound: -9.0159099, upper bound: 9.0155430
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 10.20
Output dim: 7, lower bound: -9.0155023, upper bound: 9.0155021

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -6.8144217, 5.2395210, -7.4363236, 5.7155976, -12.5300198, 12.6758442
1: -5.4558735, 4.7649941, -5.9711609, 5.1793494, -10.6352234, 10.7361546
2: -6.6546135, 4.0557895, -7.2749095, 4.3961573, -11.0507708, 11.3306990
3: -8.0807743, 3.9712572, -8.8425083, 4.2966771, -12.3774509, 12.8137655
4: -7.3246727, 5.7278557, -8.0005665, 6.2316737, -13.5563469, 13.7284222
5: -5.8313422, 4.9662495, -6.3874593, 5.3953571, -11.2266998, 11.3537083
6: -5.9627452, 6.3272591, -6.4942737, 6.8905997, -12.8533449, 12.8215332
7: -7.4770341, 3.7169933, -8.1398649, 4.0495753, -11.5266094, 11.8568583
8: -7.3270836, 5.4179459, -7.9943476, 5.8853970, -13.2124805, 13.4122934
9: -5.6369500, 6.0632124, -6.1536164, 6.6015611, -12.2385111, 12.2168293

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 84

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0161659, upper bound: 9.0161660
time: 3.98 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0161659, upper bound: 9.0161660
time: 4.36 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -6.8871508, 5.2960076, -7.3650146, 5.6609464, -12.5480976, 12.6610222
1: -5.5169582, 4.8141918, -5.9120131, 5.1317196, -10.6486778, 10.7262049
2: -6.7299519, 4.0984430, -7.2037416, 4.3570995, -11.0870514, 11.3021851
3: -8.1688366, 4.0102410, -8.7546558, 4.2585626, -12.4273987, 12.7648964
4: -7.4054875, 5.7880840, -7.9231930, 6.1738281, -13.5793152, 13.7112770
5: -5.8966866, 5.0172682, -6.3235016, 5.3459830, -11.2426701, 11.3407698
6: -6.0261893, 6.3952947, -6.4331989, 6.8260756, -12.8522644, 12.8284931
7: -7.5584831, 3.7582538, -8.0639944, 4.0110030, -11.5694866, 11.8222485
8: -7.4072843, 5.4738126, -7.9180374, 5.8316965, -13.2389812, 13.3918495
9: -5.6986618, 6.1282763, -6.0942769, 6.5398240, -12.2384853, 12.2225533

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 84

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155508, upper bound: 9.0159443
time: 3.85 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155073, upper bound: 9.0155072
time: 6.64 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6.8144217, 5.2395210, -7.6525245, 5.8821526, -12.6965742, 12.8920422
1: -5.4558735, 4.7649941, -6.1535773, 5.3257508, -10.7816238, 10.9185705
2: -6.6546135, 4.0557895, -7.4937177, 4.5179243, -11.1725378, 11.5495052
3: -8.0807743, 3.9712572, -9.1112919, 4.4179034, -12.4986763, 13.0825491
4: -7.3246727, 5.7278557, -8.2390251, 6.4093227, -13.7339954, 13.9668789
5: -5.8313422, 4.9662495, -6.5827065, 5.5467153, -11.3780575, 11.5489550
6: -5.9627452, 6.3272591, -6.6816645, 7.0888405, -13.0515862, 13.0089207
7: -7.4770341, 3.7169933, -8.3745699, 4.1748061, -11.6518402, 12.0915632
8: -7.3270836, 5.4179459, -8.2279015, 6.0499687, -13.3770523, 13.6458454
9: -5.6369500, 6.0632124, -6.3381491, 6.7948503, -12.4317999, 12.4013615

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 84

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0161544, upper bound: 9.0161331
time: 3.81 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0161544, upper bound: 9.0161331
time: 11.30 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -6.8871508, 5.2960076, -7.5809908, 5.8274984, -12.7146492, 12.8769979
1: -5.5169582, 4.8141918, -6.0942497, 5.2780924, -10.7950506, 10.9084415
2: -6.7299519, 4.0984430, -7.4221478, 4.4788399, -11.2087917, 11.5205908
3: -8.1688366, 4.0102410, -9.0233860, 4.3795681, -12.5484037, 13.0336266
4: -7.4054875, 5.7880840, -8.1613350, 6.3514214, -13.7569084, 13.9494181
5: -5.8966866, 5.0172682, -6.5187430, 5.4973998, -11.3940868, 11.5360098
6: -6.0261893, 6.3952947, -6.6204214, 7.0242181, -13.0504074, 13.0157166
7: -7.5584831, 3.7582538, -8.2985458, 4.1356926, -11.6941757, 12.0567999
8: -7.4072843, 5.4738126, -8.1512489, 5.9961796, -13.4034634, 13.6250610
9: -5.6986618, 6.1282763, -6.2787037, 6.7323294, -12.4309902, 12.4069805

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 84

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155431, upper bound: 9.0159095
time: 4.12 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155027, upper bound: 9.0155011
time: 2.87 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7.0258856, 5.4027596, -6.9709396, 5.3673358, -12.3932209, 12.3736992
1: -5.6340594, 4.9086308, -5.5967507, 4.8795094, -10.5135689, 10.5053816
2: -6.8674455, 4.1751184, -6.8246360, 4.1594505, -11.0268955, 10.9997540
3: -8.3444328, 4.0879040, -8.2793312, 4.0759087, -12.4203415, 12.3672352
4: -7.5581126, 5.9016056, -7.5047016, 5.8736138, -13.4317265, 13.4063072
5: -6.0223022, 5.1147518, -5.9794612, 5.0861740, -11.1084766, 11.0942135
6: -6.1460214, 6.5216899, -6.1069446, 6.4781866, -12.6242085, 12.6286345
7: -7.7072811, 3.8358197, -7.6592426, 3.8383603, -11.5456409, 11.4950619
8: -7.5555449, 5.5787492, -7.5008411, 5.5486674, -13.1042118, 13.0795898
9: -5.8167133, 6.2491593, -5.8013749, 6.2181902, -12.0349035, 12.0505342

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 84

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0161460, upper bound: 9.0162726
time: 4.36 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0161460, upper bound: 9.0162726
time: 2.85 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.0942917, 5.4563880, -6.9709396, 5.3673358, -12.4616280, 12.4273281
1: -5.6921310, 4.9553719, -5.5967507, 4.8795094, -10.5716400, 10.5521221
2: -6.9389954, 4.2158380, -6.8246360, 4.1594505, -11.0984459, 11.0404739
3: -8.4282713, 4.1254654, -8.2793312, 4.0759087, -12.5041800, 12.4047966
4: -7.6341333, 5.9591799, -7.5047016, 5.8736138, -13.5077477, 13.4638815
5: -6.0842090, 5.1632996, -5.9794612, 5.0861740, -11.1703835, 11.1427612
6: -6.2060590, 6.5861773, -6.1069446, 6.4781866, -12.6842461, 12.6931219
7: -7.7845793, 3.8762345, -7.6592426, 3.8383603, -11.6229401, 11.5354767
8: -7.6311426, 5.6321859, -7.5008411, 5.5486674, -13.1798096, 13.1330271
9: -5.8765941, 6.3114395, -5.8013749, 6.2181902, -12.0947838, 12.1128139

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 84

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0161460, upper bound: 9.0162726
time: 3.54 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0161460, upper bound: 9.0162726
time: 8.82 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7.4877067, 5.7550998, -7.0444646, 5.4247012, -12.9124069, 12.7995644
1: -6.0154433, 5.2134838, -5.6586404, 4.9290128, -10.9444542, 10.8721237
2: -7.3253074, 4.4242496, -6.9015031, 4.2025480, -11.5278549, 11.3257523
3: -8.9040356, 4.3219671, -8.3660269, 4.1113234, -13.0153580, 12.6879930
4: -8.0588398, 6.2736535, -7.5865097, 5.9345074, -13.9933472, 13.8601618
5: -6.4347463, 5.4310966, -6.0463629, 5.1379189, -11.5726652, 11.4774590
6: -6.5377512, 6.9379692, -6.1708026, 6.5473280, -13.0850773, 13.1087704
7: -8.1945953, 4.0791507, -7.7424045, 3.8807273, -12.0753202, 11.8215551
8: -8.0504246, 5.9235573, -7.5834627, 5.6048460, -13.6552696, 13.5070190
9: -6.1981497, 6.6479797, -5.8645058, 6.2853022, -12.4834509, 12.5124855

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 84

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0159087, upper bound: 9.0155354
time: 4.35 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0159087, upper bound: 9.0155429
time: 5.87 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.5318489, 6.5518374, -7.0087934, 5.3969851, -13.9288330, 13.5606308
1: -6.8776908, 5.9018507, -5.6285548, 4.9044867, -11.7821770, 11.5304050
2: -8.3632793, 4.9853201, -6.8643994, 4.1817679, -12.5450478, 11.8497200
3: -10.1685390, 4.8517323, -8.3217382, 4.0918589, -14.2603970, 13.1734705
4: -9.1909504, 7.1131654, -7.5471516, 5.9050117, -15.0959625, 14.6603165
5: -7.3675928, 6.1442418, -6.0138474, 5.1124392, -12.4800310, 12.1580877
6: -7.4239216, 7.8792806, -6.1393962, 6.5141611, -13.9380827, 14.0186758
7: -9.2978983, 4.6302156, -7.7022748, 3.8594987, -13.1573973, 12.3324890
8: -9.1690311, 6.7029395, -7.5440145, 5.5774865, -14.7465153, 14.2469540
9: -7.0558362, 7.5525703, -5.8338122, 6.2529225, -13.3087587, 13.3863811

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 219

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155003, upper bound: 9.0154945
time: 2.73 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155003, upper bound: 9.0155023
time: 3.15 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 8.30 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 8.30
Output dim: 7, lower bound: -9.0161659, upper bound: 9.0161660
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 8.30
Output dim: 7, lower bound: -9.0161659, upper bound: 9.0161660
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 8.30
Output dim: 7, lower bound: -9.0155508, upper bound: 9.0159443
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 8.30
Output dim: 7, lower bound: -9.0155073, upper bound: 9.0155072
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 8.30
Output dim: 7, lower bound: -9.0161544, upper bound: 9.0161331
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 8.30
Output dim: 7, lower bound: -9.0161544, upper bound: 9.0161331
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 8.30
Output dim: 7, lower bound: -9.0155431, upper bound: 9.0159095
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 8.30
Output dim: 7, lower bound: -9.0155027, upper bound: 9.0155011
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 8.30
Output dim: 7, lower bound: -9.0161460, upper bound: 9.0162726
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 8.30
Output dim: 7, lower bound: -9.0161460, upper bound: 9.0162726
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 8.30
Output dim: 7, lower bound: -9.0161460, upper bound: 9.0162726
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 8.30
Output dim: 7, lower bound: -9.0161460, upper bound: 9.0162726
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 8.30
Output dim: 7, lower bound: -9.0159087, upper bound: 9.0155354
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 8.30
Output dim: 7, lower bound: -9.0159087, upper bound: 9.0155429
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 8.30
Output dim: 7, lower bound: -9.0155003, upper bound: 9.0154945
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 8.30
Output dim: 7, lower bound: -9.0155003, upper bound: 9.0155023

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -6.8144217, 5.2395210, -6.8144217, 5.2395210, -12.0539427, 12.0539427
1: -5.4558735, 4.7649941, -5.4558735, 4.7649941, -10.2208672, 10.2208672
2: -6.6546135, 4.0557895, -6.6546135, 4.0557895, -10.7104034, 10.7104034
3: -8.0807743, 3.9712572, -8.0807743, 3.9712572, -12.0520315, 12.0520315
4: -7.3246727, 5.7278557, -7.3246727, 5.7278557, -13.0525284, 13.0525284
5: -5.8313422, 4.9662495, -5.8313422, 4.9662495, -10.7975922, 10.7975922
6: -5.9627452, 6.3272591, -5.9627452, 6.3272591, -12.2900047, 12.2900047
7: -7.4770341, 3.7169933, -7.4770341, 3.7169933, -11.1940269, 11.1940269
8: -7.3270836, 5.4179459, -7.3270836, 5.4179459, -12.7450294, 12.7450294
9: -5.6369500, 6.0632124, -5.6369500, 6.0632124, -11.7001629, 11.7001629

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 84

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0160671, upper bound: 9.0155697
time: 3.41 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0157024, upper bound: 9.0155246
time: 2.84 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -6.8144217, 5.2395210, -6.8871508, 5.2960076, -12.1104298, 12.1266718
1: -5.4558735, 4.7649941, -5.5169582, 4.8141918, -10.2700653, 10.2819519
2: -6.6546135, 4.0557895, -6.7299519, 4.0984430, -10.7530565, 10.7857418
3: -8.0807743, 3.9712572, -8.1688366, 4.0102410, -12.0910149, 12.1400938
4: -7.3246727, 5.7278557, -7.4054875, 5.7880840, -13.1127567, 13.1333427
5: -5.8313422, 4.9662495, -5.8966866, 5.0172682, -10.8486099, 10.8629360
6: -5.9627452, 6.3272591, -6.0261893, 6.3952947, -12.3580399, 12.3534489
7: -7.4770341, 3.7169933, -7.5584831, 3.7582538, -11.2352877, 11.2754765
8: -7.3270836, 5.4179459, -7.4072843, 5.4738126, -12.8008957, 12.8252296
9: -5.6369500, 6.0632124, -5.6986618, 6.1282763, -11.7652264, 11.7618742

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 84

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0160671, upper bound: 9.0155697
time: 2.52 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0157024, upper bound: 9.0155245
time: 2.60 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -6.8871508, 5.2960076, -7.2734237, 5.5897326, -12.4768829, 12.5694313
1: -5.5169582, 4.8141918, -5.8346977, 5.0683093, -10.5852680, 10.6488895
2: -6.7299519, 4.0984430, -7.1090517, 4.3035045, -11.0334568, 11.2074947
3: -8.1688366, 4.0102410, -8.6376438, 4.2032795, -12.3721161, 12.6478844
4: -7.4054875, 5.7880840, -7.8225164, 6.0975461, -13.5030336, 13.6106005
5: -5.8966866, 5.0172682, -6.2410755, 5.2808762, -11.1775627, 11.2583437
6: -6.0261893, 6.3952947, -6.3521600, 6.7411156, -12.7673054, 12.7474546
7: -7.5584831, 3.7582538, -7.9615021, 3.9570065, -11.5154896, 11.7197561
8: -7.4072843, 5.4738126, -7.8189631, 5.7604666, -13.1677513, 13.2927761
9: -5.6986618, 6.1282763, -6.0154843, 6.4579711, -12.1566334, 12.1437607

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 57

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155508, upper bound: 9.0159336
time: 2.61 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155508, upper bound: 9.0159443
time: 3.29 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -6.8529739, 5.2694383, -8.2925949, 6.3681264, -13.2210999, 13.5620327
1: -5.4881549, 4.7908959, -6.6757765, 5.7414231, -11.2295780, 11.4666729
2: -6.6948562, 4.0786610, -8.1188469, 4.8518791, -11.5467358, 12.1975079
3: -8.1264791, 3.9917886, -9.8743210, 4.7173576, -12.8438368, 13.8661098
4: -7.3678555, 5.7598948, -8.9277973, 6.9175611, -14.2854166, 14.6876926
5: -5.8658705, 4.9932380, -7.1509533, 5.9783711, -11.8442421, 12.1441917
6: -5.9963527, 6.3633599, -7.2174945, 7.6609912, -13.6573439, 13.5808544
7: -7.5199099, 3.7390549, -9.0392447, 4.4845548, -12.0044651, 12.7782993
8: -7.3700428, 5.4474750, -8.9104958, 6.5211782, -13.8912210, 14.3579712
9: -5.6696563, 6.0978599, -6.8518596, 7.3315144, -13.0011711, 12.9497194

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 57

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155073, upper bound: 9.0155043
time: 2.99 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155073, upper bound: 9.0155073
time: 2.82 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -6.8144217, 5.2395210, -7.0258856, 5.4027596, -12.2171812, 12.2654066
1: -5.4558735, 4.7649941, -5.6340594, 4.9086308, -10.3645039, 10.3990536
2: -6.6546135, 4.0557895, -6.8674455, 4.1751184, -10.8297319, 10.9232349
3: -8.0807743, 3.9712572, -8.3444328, 4.0879040, -12.1686783, 12.3156900
4: -7.3246727, 5.7278557, -7.5581126, 5.9016056, -13.2262783, 13.2859688
5: -5.8313422, 4.9662495, -6.0223022, 5.1147518, -10.9460945, 10.9885521
6: -5.9627452, 6.3272591, -6.1460214, 6.5216899, -12.4844351, 12.4732800
7: -7.4770341, 3.7169933, -7.7072811, 3.8358197, -11.3128538, 11.4242744
8: -7.3270836, 5.4179459, -7.5555449, 5.5787492, -12.9058323, 12.9734907
9: -5.6369500, 6.0632124, -5.8167133, 6.2491593, -11.8861094, 11.8799257

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 84

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0160614, upper bound: 9.0155629
time: 2.94 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0157000, upper bound: 9.0155185
time: 11.52 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -6.8144217, 5.2395210, -7.0942917, 5.4563880, -12.2708092, 12.3338127
1: -5.4558735, 4.7649941, -5.6921310, 4.9553719, -10.4112453, 10.4571247
2: -6.6546135, 4.0557895, -6.9389954, 4.2158380, -10.8704510, 10.9947853
3: -8.0807743, 3.9712572, -8.4282713, 4.1254654, -12.2062397, 12.3995285
4: -7.3246727, 5.7278557, -7.6341333, 5.9591799, -13.2838526, 13.3619890
5: -5.8313422, 4.9662495, -6.0842090, 5.1632996, -10.9946423, 11.0504589
6: -5.9627452, 6.3272591, -6.2060590, 6.5861773, -12.5489225, 12.5333176
7: -7.4770341, 3.7169933, -7.7845793, 3.8762345, -11.3532686, 11.5015726
8: -7.3270836, 5.4179459, -7.6311426, 5.6321859, -12.9592695, 13.0490885
9: -5.6369500, 6.0632124, -5.8765941, 6.3114395, -11.9483891, 11.9398060

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 84

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0160614, upper bound: 9.0155629
time: 2.83 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0157000, upper bound: 9.0155185
time: 4.47 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -6.8871508, 5.2960076, -7.4877067, 5.7550998, -12.6422501, 12.7837124
1: -5.5169582, 4.8141918, -6.0154433, 5.2134838, -10.7304420, 10.8296337
2: -6.7299519, 4.0984430, -7.3253074, 4.4242496, -11.1542015, 11.4237499
3: -8.1688366, 4.0102410, -8.9040356, 4.3219671, -12.4908037, 12.9142761
4: -7.4054875, 5.7880840, -8.0588398, 6.2736535, -13.6791410, 13.8469238
5: -5.8966866, 5.0172682, -6.4347463, 5.4310966, -11.3277836, 11.4520149
6: -6.0261893, 6.3952947, -6.5377512, 6.9379692, -12.9641590, 12.9330463
7: -7.5584831, 3.7582538, -8.1945953, 4.0791507, -11.6376333, 11.9528494
8: -7.4072843, 5.4738126, -8.0504246, 5.9235573, -13.3308411, 13.5242367
9: -5.6986618, 6.1282763, -6.1981497, 6.6479797, -12.3466415, 12.3264256

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 57

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155431, upper bound: 9.0158997
time: 3.49 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155431, upper bound: 9.0159095
time: 10.57 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -6.8529739, 5.2694383, -8.5318489, 6.5518374, -13.4048119, 13.8012867
1: -5.4881549, 4.7908959, -6.8776908, 5.9018507, -11.3900051, 11.6685867
2: -6.6948562, 4.0786610, -8.3632793, 4.9853201, -11.6801758, 12.4419403
3: -8.1264791, 3.9917886, -10.1685390, 4.8517323, -12.9782114, 14.1603279
4: -7.3678555, 5.7598948, -9.1909504, 7.1131654, -14.4810209, 14.9508457
5: -5.8658705, 4.9932380, -7.3675928, 6.1442418, -12.0101128, 12.3608294
6: -5.9963527, 6.3633599, -7.4239216, 7.8792806, -13.8756332, 13.7872810
7: -7.5199099, 3.7390549, -9.2978983, 4.6302156, -12.1501255, 13.0369530
8: -7.3700428, 5.4474750, -9.1690311, 6.7029395, -14.0729828, 14.6165066
9: -5.6696563, 6.0978599, -7.0558362, 7.5525703, -13.2222252, 13.1536961

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 57

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155027, upper bound: 9.0154985
time: 2.53 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155027, upper bound: 9.0155011
time: 1.96 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -7.0258856, 5.4027596, -6.8144217, 5.2395210, -12.2654066, 12.2171812
1: -5.6340594, 4.9086308, -5.4558735, 4.7649941, -10.3990536, 10.3645039
2: -6.8674455, 4.1751184, -6.6546135, 4.0557895, -10.9232349, 10.8297319
3: -8.3444328, 4.0879040, -8.0807743, 3.9712572, -12.3156900, 12.1686783
4: -7.5581126, 5.9016056, -7.3246727, 5.7278557, -13.2859688, 13.2262783
5: -6.0223022, 5.1147518, -5.8313422, 4.9662495, -10.9885521, 10.9460945
6: -6.1460214, 6.5216899, -5.9627452, 6.3272591, -12.4732800, 12.4844351
7: -7.7072811, 3.8358197, -7.4770341, 3.7169933, -11.4242744, 11.3128538
8: -7.5555449, 5.5787492, -7.3270836, 5.4179459, -12.9734907, 12.9058323
9: -5.8167133, 6.2491593, -5.6369500, 6.0632124, -11.8799257, 11.8861094

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 57

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0161047, upper bound: 9.0157909
time: 2.80 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0157503, upper bound: 9.0157476
time: 2.98 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -7.0258856, 5.4027596, -7.0258856, 5.4027596, -12.4286451, 12.4286451
1: -5.6340594, 4.9086308, -5.6340594, 4.9086308, -10.5426903, 10.5426903
2: -6.8674455, 4.1751184, -6.8674455, 4.1751184, -11.0425644, 11.0425644
3: -8.3444328, 4.0879040, -8.3444328, 4.0879040, -12.4323368, 12.4323368
4: -7.5581126, 5.9016056, -7.5581126, 5.9016056, -13.4597187, 13.4597187
5: -6.0223022, 5.1147518, -6.0223022, 5.1147518, -11.1370544, 11.1370544
6: -6.1460214, 6.5216899, -6.1460214, 6.5216899, -12.6677113, 12.6677113
7: -7.7072811, 3.8358197, -7.7072811, 3.8358197, -11.5431004, 11.5431004
8: -7.5555449, 5.5787492, -7.5555449, 5.5787492, -13.1342945, 13.1342945
9: -5.8167133, 6.2491593, -5.8167133, 6.2491593, -12.0658722, 12.0658722

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0161047, upper bound: 9.0157941
time: 5.33 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0157503, upper bound: 9.0157505
time: 6.79 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7.0942917, 5.4563880, -6.8144217, 5.2395210, -12.3338127, 12.2708092
1: -5.6921310, 4.9553719, -5.4558735, 4.7649941, -10.4571247, 10.4112453
2: -6.9389954, 4.2158380, -6.6546135, 4.0557895, -10.9947853, 10.8704510
3: -8.4282713, 4.1254654, -8.0807743, 3.9712572, -12.3995285, 12.2062397
4: -7.6341333, 5.9591799, -7.3246727, 5.7278557, -13.3619890, 13.2838526
5: -6.0842090, 5.1632996, -5.8313422, 4.9662495, -11.0504589, 10.9946423
6: -6.2060590, 6.5861773, -5.9627452, 6.3272591, -12.5333176, 12.5489225
7: -7.7845793, 3.8762345, -7.4770341, 3.7169933, -11.5015726, 11.3532686
8: -7.6311426, 5.6321859, -7.3270836, 5.4179459, -13.0490885, 12.9592695
9: -5.8765941, 6.3114395, -5.6369500, 6.0632124, -11.9398060, 11.9483891

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 84

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0159201, upper bound: 9.0157361
time: 6.53 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155159, upper bound: 9.0156918
time: 3.30 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7.0942917, 5.4563880, -7.0258856, 5.4027596, -12.4970512, 12.4822731
1: -5.6921310, 4.9553719, -5.6340594, 4.9086308, -10.6007614, 10.5894318
2: -6.9389954, 4.2158380, -6.8674455, 4.1751184, -11.1141138, 11.0832834
3: -8.4282713, 4.1254654, -8.3444328, 4.0879040, -12.5161753, 12.4698982
4: -7.6341333, 5.9591799, -7.5581126, 5.9016056, -13.5357389, 13.5172920
5: -6.0842090, 5.1632996, -6.0223022, 5.1147518, -11.1989613, 11.1856022
6: -6.2060590, 6.5861773, -6.1460214, 6.5216899, -12.7277489, 12.7321987
7: -7.7845793, 3.8762345, -7.7072811, 3.8358197, -11.6203995, 11.5835152
8: -7.6311426, 5.6321859, -7.5555449, 5.5787492, -13.2098923, 13.1877308
9: -5.8765941, 6.3114395, -5.8167133, 6.2491593, -12.1257534, 12.1281528

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 84

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0159201, upper bound: 9.0157407
time: 4.50 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155159, upper bound: 9.0156967
time: 6.02 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -7.4877067, 5.7550998, -6.8871508, 5.2960076, -12.7837133, 12.6422501
1: -6.0154433, 5.2134838, -5.5169582, 4.8141918, -10.8296337, 10.7304420
2: -7.3253074, 4.4242496, -6.7299519, 4.0984430, -11.4237499, 11.1542015
3: -8.9040356, 4.3219671, -8.1688366, 4.0102410, -12.9142761, 12.4908028
4: -8.0588398, 6.2736535, -7.4054875, 5.7880840, -13.8469238, 13.6791410
5: -6.4347463, 5.4310966, -5.8966866, 5.0172682, -11.4520149, 11.3277836
6: -6.5377512, 6.9379692, -6.0261893, 6.3952947, -12.9330463, 12.9641590
7: -8.1945953, 4.0791507, -7.5584831, 3.7582538, -11.9528494, 11.6376324
8: -8.0504246, 5.9235573, -7.4072843, 5.4738126, -13.5242367, 13.3308411
9: -6.1981497, 6.6479797, -5.6986618, 6.1282763, -12.3264256, 12.3466415

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 84

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0159087, upper bound: 9.0155354
time: 4.30 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0159087, upper bound: 9.0155354
time: 3.60 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -7.4877067, 5.7550998, -7.0942917, 5.4563880, -12.9440937, 12.8493919
1: -6.0154433, 5.2134838, -5.6921310, 4.9553719, -10.9708147, 10.9056149
2: -7.3253074, 4.4242496, -6.9389954, 4.2158380, -11.5411434, 11.3632450
3: -8.9040356, 4.3219671, -8.4282713, 4.1254654, -13.0295010, 12.7502375
4: -8.0588398, 6.2736535, -7.6341333, 5.9591799, -14.0180187, 13.9077873
5: -6.4347463, 5.4310966, -6.0842090, 5.1632996, -11.5980444, 11.5153055
6: -6.5377512, 6.9379692, -6.2060590, 6.5861773, -13.1239271, 13.1440277
7: -8.1945953, 4.0791507, -7.7845793, 3.8762345, -12.0708294, 11.8637285
8: -8.0504246, 5.9235573, -7.6311426, 5.6321859, -13.6826105, 13.5546999
9: -6.1981497, 6.6479797, -5.8765941, 6.3114395, -12.5095892, 12.5245743

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 84

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0159087, upper bound: 9.0155429
time: 7.32 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0159087, upper bound: 9.0155429
time: 3.71 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -8.5318489, 6.5518374, -6.8529739, 5.2694383, -13.8012867, 13.4048119
1: -6.8776908, 5.9018507, -5.4881549, 4.7908959, -11.6685867, 11.3900051
2: -8.3632793, 4.9853201, -6.6948562, 4.0786610, -12.4419403, 11.6801758
3: -10.1685390, 4.8517323, -8.1264791, 3.9917886, -14.1603279, 12.9782114
4: -9.1909504, 7.1131654, -7.3678555, 5.7598948, -14.9508429, 14.4810209
5: -7.3675928, 6.1442418, -5.8658705, 4.9932380, -12.3608284, 12.0101128
6: -7.4239216, 7.8792806, -5.9963527, 6.3633599, -13.7872810, 13.8756332
7: -9.2978983, 4.6302156, -7.5199099, 3.7390549, -13.0369530, 12.1501255
8: -9.1690311, 6.7029395, -7.3700428, 5.4474750, -14.6165066, 14.0729828
9: -7.0558362, 7.5525703, -5.6696563, 6.0978599, -13.1536961, 13.2222252

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 84

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155003, upper bound: 9.0154945
time: 2.73 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155003, upper bound: 9.0154947
time: 4.02 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -8.5318489, 6.5518374, -7.0596747, 5.4294305, -13.9612789, 13.6115122
1: -6.8776908, 5.9018507, -5.6629024, 4.9317389, -11.8094292, 11.5647526
2: -8.3632793, 4.9853201, -6.9033847, 4.1957688, -12.5590477, 11.8887043
3: -10.1685390, 4.8517323, -8.3852739, 4.1067333, -14.2752724, 13.2370062
4: -9.1909504, 7.1131654, -7.5960126, 5.9305410, -15.1214914, 14.7091780
5: -7.3675928, 6.1442418, -6.0529480, 5.1389375, -12.5065298, 12.1971893
6: -7.4239216, 7.8792806, -6.1758118, 6.5537891, -13.9777107, 14.0550919
7: -9.2978983, 4.6302156, -7.7454720, 3.8566844, -13.1545830, 12.3756866
8: -9.1690311, 6.7029395, -7.5934148, 5.6054292, -14.7744598, 14.2963543
9: -7.0558362, 7.5525703, -5.8470931, 6.2805405, -13.3363762, 13.3996639

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 84

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155003, upper bound: 9.0155023
time: 3.49 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155003, upper bound: 9.0155023
time: 3.38 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 9.02 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 9.02
Output dim: 7, lower bound: -9.0160671, upper bound: 9.0155697
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 9.02
Output dim: 7, lower bound: -9.0157024, upper bound: 9.0155246
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 9.02
Output dim: 7, lower bound: -9.0160671, upper bound: 9.0155697
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 9.02
Output dim: 7, lower bound: -9.0157024, upper bound: 9.0155245
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 9.02
Output dim: 7, lower bound: -9.0155508, upper bound: 9.0159336
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 9.02
Output dim: 7, lower bound: -9.0155508, upper bound: 9.0159443
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 9.02
Output dim: 7, lower bound: -9.0155073, upper bound: 9.0155043
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 9.02
Output dim: 7, lower bound: -9.0155073, upper bound: 9.0155073
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 9.02
Output dim: 7, lower bound: -9.0160614, upper bound: 9.0155629
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 9.02
Output dim: 7, lower bound: -9.0157000, upper bound: 9.0155185
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 9.02
Output dim: 7, lower bound: -9.0160614, upper bound: 9.0155629
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 9.02
Output dim: 7, lower bound: -9.0157000, upper bound: 9.0155185
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 9.02
Output dim: 7, lower bound: -9.0155431, upper bound: 9.0158997
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 9.02
Output dim: 7, lower bound: -9.0155431, upper bound: 9.0159095
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 9.02
Output dim: 7, lower bound: -9.0155027, upper bound: 9.0154985
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 9.02
Output dim: 7, lower bound: -9.0155027, upper bound: 9.0155011
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 9.02
Output dim: 7, lower bound: -9.0161047, upper bound: 9.0157909
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 9.02
Output dim: 7, lower bound: -9.0157503, upper bound: 9.0157476
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 9.02
Output dim: 7, lower bound: -9.0161047, upper bound: 9.0157941
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 9.02
Output dim: 7, lower bound: -9.0157503, upper bound: 9.0157505
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 9.02
Output dim: 7, lower bound: -9.0159201, upper bound: 9.0157361
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 9.02
Output dim: 7, lower bound: -9.0155159, upper bound: 9.0156918
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 9.02
Output dim: 7, lower bound: -9.0159201, upper bound: 9.0157407
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 9.02
Output dim: 7, lower bound: -9.0155159, upper bound: 9.0156967
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 9.02
Output dim: 7, lower bound: -9.0159087, upper bound: 9.0155354
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 9.02
Output dim: 7, lower bound: -9.0159087, upper bound: 9.0155354
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 9.02
Output dim: 7, lower bound: -9.0159087, upper bound: 9.0155429
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 9.02
Output dim: 7, lower bound: -9.0159087, upper bound: 9.0155429
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 9.02
Output dim: 7, lower bound: -9.0155003, upper bound: 9.0154945
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 9.02
Output dim: 7, lower bound: -9.0155003, upper bound: 9.0154947
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 9.02
Output dim: 7, lower bound: -9.0155003, upper bound: 9.0155023
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 9.02
Output dim: 7, lower bound: -9.0155003, upper bound: 9.0155023

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -6.7290764, 5.1731324, -6.8144217, 5.2395210, -11.9685974, 11.9875546
1: -5.3839560, 4.7067604, -5.4558735, 4.7649941, -10.1489506, 10.1626339
2: -6.5669265, 4.0062790, -6.6546135, 4.0557895, -10.6227160, 10.6608925
3: -7.9751396, 3.9252262, -8.0807743, 3.9712572, -11.9463968, 12.0060005
4: -7.2306910, 5.6573963, -7.3246727, 5.7278557, -12.9585466, 12.9820690
5: -5.7543731, 4.9062390, -5.8313422, 4.9662495, -10.7206230, 10.7375813
6: -5.8882079, 6.2474389, -5.9627452, 6.3272591, -12.2154675, 12.2101841
7: -7.3804231, 3.6690583, -7.4770341, 3.7169933, -11.0974159, 11.1460924
8: -7.2340679, 5.3521290, -7.3270836, 5.4179459, -12.6520138, 12.6792126
9: -5.5643702, 5.9872446, -5.6369500, 6.0632124, -11.6275826, 11.6241951

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 84

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0157691, upper bound: 9.0157691
time: 4.82 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0157691, upper bound: 9.0157691
time: 4.19 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7.7431498, 5.9476161, -6.7802720, 5.2129736, -12.9561234, 12.7278881
1: -6.2209353, 5.3765755, -5.4270930, 4.7417169, -10.9626522, 10.8036690
2: -7.5715690, 4.5522113, -6.6195526, 4.0360260, -11.6075954, 11.1717644
3: -9.2054300, 4.4369597, -8.0384674, 3.9528189, -13.1582489, 12.4754276
4: -8.3307943, 6.4732018, -7.2870722, 5.6996880, -14.0304823, 13.7602739
5: -6.6597123, 5.6002378, -5.8005457, 4.9422450, -11.6019573, 11.4007835
6: -6.7493706, 7.1627626, -5.9329348, 6.2953501, -13.0447206, 13.0956974
7: -8.4528923, 4.1941271, -7.4384899, 3.6978235, -12.1507158, 11.6326170
8: -8.3203354, 6.1088390, -7.2898693, 5.3916330, -13.7119684, 13.3987083
9: -6.3964062, 6.8566337, -5.6079650, 6.0328207, -12.4292269, 12.4645987

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 57

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0157667, upper bound: 9.0157691
time: 4.77 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0157691, upper bound: 9.0157691
time: 2.56 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6.7290764, 5.1731324, -6.8871508, 5.2960076, -12.0250835, 12.0602837
1: -5.3839560, 4.7067604, -5.5169582, 4.8141918, -10.1981478, 10.2237186
2: -6.5669265, 4.0062790, -6.7299519, 4.0984430, -10.6653690, 10.7362309
3: -7.9751396, 3.9252262, -8.1688366, 4.0102410, -11.9853802, 12.0940628
4: -7.2306910, 5.6573963, -7.4054875, 5.7880840, -13.0187750, 13.0628834
5: -5.7543731, 4.9062390, -5.8966866, 5.0172682, -10.7716408, 10.8029251
6: -5.8882079, 6.2474389, -6.0261893, 6.3952947, -12.2835026, 12.2736282
7: -7.3804231, 3.6690583, -7.5584831, 3.7582538, -11.1386766, 11.2275410
8: -7.2340679, 5.3521290, -7.4072843, 5.4738126, -12.7078800, 12.7594128
9: -5.5643702, 5.9872446, -5.6986618, 6.1282763, -11.6926460, 11.6859064

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 57

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0160480, upper bound: 9.0155697
time: 3.54 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0160671, upper bound: 9.0155697
time: 2.63 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7.7431498, 5.9476161, -6.8529739, 5.2694383, -13.0125885, 12.8005905
1: -6.2209353, 5.3765755, -5.4881549, 4.7908959, -11.0118313, 10.8647308
2: -7.5715690, 4.5522113, -6.6948562, 4.0786610, -11.6502304, 11.2470675
3: -9.2054300, 4.4369597, -8.1264791, 3.9917886, -13.1972189, 12.5634384
4: -8.3307943, 6.4732018, -7.3678555, 5.7598948, -14.0906887, 13.8410568
5: -6.6597123, 5.6002378, -5.8658705, 4.9932380, -11.6529503, 11.4661083
6: -6.7493706, 7.1627626, -5.9963527, 6.3633599, -13.1127300, 13.1591148
7: -8.4528923, 4.1941271, -7.5199099, 3.7390549, -12.1919470, 11.7140369
8: -8.3203354, 6.1088390, -7.3700428, 5.4474750, -13.7678108, 13.4788818
9: -6.3964062, 6.8566337, -5.6696563, 6.0978599, -12.4942665, 12.5262899

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 57

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0156952, upper bound: 9.0155246
time: 4.58 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0157024, upper bound: 9.0155246
time: 2.95 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -6.8579116, 5.2687492, -7.2414865, 5.5645890, -12.4225006, 12.5102358
1: -5.4892516, 4.7890806, -5.8075151, 5.0460982, -10.5353498, 10.5965958
2: -6.6882114, 4.0697956, -7.0754862, 4.2840443, -10.9722557, 11.1452818
3: -8.1300840, 3.9907367, -8.5978270, 4.1856761, -12.3157597, 12.5885639
4: -7.3681412, 5.7579913, -7.7869658, 6.0706649, -13.4388065, 13.5449572
5: -5.8688226, 4.9927001, -6.2121282, 5.2580571, -11.1268797, 11.2048283
6: -5.9935799, 6.3579121, -6.3237534, 6.7105808, -12.7041607, 12.6816654
7: -7.5003362, 3.7347536, -7.9235992, 3.9382129, -11.4385490, 11.6583529
8: -7.3705292, 5.4439731, -7.7838545, 5.7353063, -13.1058350, 13.2278271
9: -5.6693549, 6.0951285, -5.9877477, 6.4289689, -12.0983238, 12.0828762

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 84

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155508, upper bound: 9.0159336
time: 4.33 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155508, upper bound: 9.0159336
time: 3.37 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -6.8227777, 5.2460175, -7.2734237, 5.5897326, -12.4125099, 12.5194416
1: -5.4628563, 4.7703457, -5.8346977, 5.0683093, -10.5311661, 10.6050434
2: -6.6641264, 4.0613899, -7.1090517, 4.3035045, -10.9676304, 11.1704416
3: -8.0891027, 3.9757609, -8.6376438, 4.2032795, -12.2923822, 12.6134052
4: -7.3346639, 5.7350268, -7.8225164, 6.0975461, -13.4322100, 13.5575428
5: -5.8386965, 4.9720621, -6.2410755, 5.2808762, -11.1195726, 11.2131376
6: -5.9701271, 6.3353448, -6.3521600, 6.7411156, -12.7112427, 12.6875048
7: -7.4861374, 3.7223163, -7.9615021, 3.9570065, -11.4431438, 11.6838188
8: -7.3373251, 5.4243231, -7.8189631, 5.7604666, -13.0977917, 13.2432861
9: -5.6440411, 6.0712538, -6.0154843, 6.4579711, -12.1020126, 12.0867386

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 84

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155508, upper bound: 9.0159443
time: 4.79 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155508, upper bound: 9.0159443
time: 5.03 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -6.8231196, 5.2416987, -8.2608318, 6.3431330, -13.1662521, 13.5025311
1: -5.4599195, 4.7653704, -6.6487417, 5.7193427, -11.1792622, 11.4141121
2: -6.6524773, 4.0496597, -8.0854797, 4.8325334, -11.4850101, 12.1351395
3: -8.0869465, 3.9719291, -9.8346815, 4.6998754, -12.7868214, 13.8066101
4: -7.3298264, 5.7292895, -8.8924408, 6.8908234, -14.2206497, 14.6217308
5: -5.8374434, 4.9682322, -7.1221657, 5.9556723, -11.7931156, 12.0903978
6: -5.9631886, 6.3253994, -7.1892509, 7.6306515, -13.5938396, 13.5146503
7: -7.4610739, 3.7152238, -9.0016146, 4.4658036, -11.9268780, 12.7168388
8: -7.3326225, 5.4171653, -8.8755856, 6.4961667, -13.8287888, 14.2927513
9: -5.6398449, 6.0641699, -6.8242855, 7.3026571, -12.9425020, 12.8884554

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 84

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155073, upper bound: 9.0155043
time: 3.70 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155073, upper bound: 9.0155043
time: 5.18 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6.7883606, 5.2192621, -8.2925949, 6.3681264, -13.1564865, 13.5118570
1: -5.4338503, 4.7468863, -6.6757765, 5.7414231, -11.1752739, 11.4226627
2: -6.6287823, 4.0414696, -8.1188469, 4.8518791, -11.4806614, 12.1603165
3: -8.0464458, 3.9571767, -9.8743210, 4.7173576, -12.7638035, 13.8314972
4: -7.2967668, 5.7066407, -8.9277973, 6.9175611, -14.2143278, 14.6344376
5: -5.8076615, 4.9478650, -7.1509533, 5.9783711, -11.7860327, 12.0988178
6: -5.9400821, 6.3031850, -7.2174945, 7.6609912, -13.6010733, 13.5206795
7: -7.4472857, 3.7029881, -9.0392447, 4.4845548, -11.9318409, 12.7422333
8: -7.2998204, 5.3978009, -8.9104958, 6.5211782, -13.8209991, 14.3082962
9: -5.6148291, 6.0406251, -6.8518596, 7.3315144, -12.9463434, 12.8924847

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 84

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155073, upper bound: 9.0155073
time: 8.88 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155073, upper bound: 9.0155073
time: 5.43 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -6.7290764, 5.1731324, -7.0258856, 5.4027596, -12.1318359, 12.1990185
1: -5.3839560, 4.7067604, -5.6340594, 4.9086308, -10.2925873, 10.3408203
2: -6.5669265, 4.0062790, -6.8674455, 4.1751184, -10.7420444, 10.8737240
3: -7.9751396, 3.9252262, -8.3444328, 4.0879040, -12.0630436, 12.2696590
4: -7.2306910, 5.6573963, -7.5581126, 5.9016056, -13.1322966, 13.2155094
5: -5.7543731, 4.9062390, -6.0223022, 5.1147518, -10.8691254, 10.9285412
6: -5.8882079, 6.2474389, -6.1460214, 6.5216899, -12.4098978, 12.3934603
7: -7.3804231, 3.6690583, -7.7072811, 3.8358197, -11.2162428, 11.3763390
8: -7.2340679, 5.3521290, -7.5555449, 5.5787492, -12.8128166, 12.9076738
9: -5.5643702, 5.9872446, -5.8167133, 6.2491593, -11.8135300, 11.8039579

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 57

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0157601, upper bound: 9.0157540
time: 9.69 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0157601, upper bound: 9.0157540
time: 7.62 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7.7431498, 5.9476161, -6.9916892, 5.3761759, -13.1193256, 12.9393053
1: -6.2209353, 5.3765755, -5.6052346, 4.8852949, -11.1062298, 10.9818096
2: -7.5715690, 4.5522113, -6.8323359, 4.1553249, -11.7268944, 11.3845472
3: -9.2054300, 4.4369597, -8.3019724, 4.0694237, -13.2748537, 12.7389317
4: -8.3307943, 6.4732018, -7.5204616, 5.8733821, -14.2041759, 13.9936638
5: -6.6597123, 5.6002378, -5.9914560, 5.0906992, -11.7504120, 11.5916939
6: -6.7493706, 7.1627626, -6.1161656, 6.4897394, -13.2391100, 13.2789288
7: -8.4528923, 4.1941271, -7.6686902, 3.8165846, -12.2694769, 11.8628178
8: -8.3203354, 6.1088390, -7.5182834, 5.5523868, -13.8727226, 13.6271229
9: -6.3964062, 6.8566337, -5.7876725, 6.2187061, -12.6151123, 12.6443062

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 57

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0157575, upper bound: 9.0157540
time: 2.57 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0157601, upper bound: 9.0157540
time: 10.06 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6.7290764, 5.1731324, -7.0942917, 5.4563880, -12.1854649, 12.2674236
1: -5.3839560, 4.7067604, -5.6921310, 4.9553719, -10.3393278, 10.3988914
2: -6.5669265, 4.0062790, -6.9389954, 4.2158380, -10.7827644, 10.9452744
3: -7.9751396, 3.9252262, -8.4282713, 4.1254654, -12.1006050, 12.3534975
4: -7.2306910, 5.6573963, -7.6341333, 5.9591799, -13.1898708, 13.2915297
5: -5.7543731, 4.9062390, -6.0842090, 5.1632996, -10.9176731, 10.9904480
6: -5.8882079, 6.2474389, -6.2060590, 6.5861773, -12.4743853, 12.4534979
7: -7.3804231, 3.6690583, -7.7845793, 3.8762345, -11.2566576, 11.4536381
8: -7.2340679, 5.3521290, -7.6311426, 5.6321859, -12.8662539, 12.9832716
9: -5.5643702, 5.9872446, -5.8765941, 6.3114395, -11.8758097, 11.8638382

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 84

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0157000, upper bound: 9.0155185
time: 2.47 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0157000, upper bound: 9.0155185
time: 3.07 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7.7431498, 5.9476161, -7.0596747, 5.4294305, -13.1725807, 13.0072908
1: -6.2209353, 5.3765755, -5.6629024, 4.9317389, -11.1526737, 11.0394783
2: -7.5715690, 4.5522113, -6.9033847, 4.1957688, -11.7673378, 11.4555960
3: -9.2054300, 4.4369597, -8.3852739, 4.1067333, -13.3121634, 12.8222332
4: -8.3307943, 6.4732018, -7.5960126, 5.9305410, -14.2613354, 14.0692139
5: -6.6597123, 5.6002378, -6.0529480, 5.1389375, -11.7986498, 11.6531858
6: -6.7493706, 7.1627626, -6.1758118, 6.5537891, -13.3031597, 13.3385744
7: -8.4528923, 4.1941271, -7.7454720, 3.8566844, -12.3095770, 11.9395990
8: -8.3203354, 6.1088390, -7.5934148, 5.6054292, -13.9257641, 13.7022533
9: -6.3964062, 6.8566337, -5.8470931, 6.2805405, -12.6769466, 12.7037268

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 84

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0156931, upper bound: 9.0155185
time: 4.27 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0157000, upper bound: 9.0155185
time: 3.17 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -6.8579116, 5.2687492, -7.4552536, 5.7296081, -12.5875196, 12.7240009
1: -5.4892516, 4.7890806, -5.9878120, 5.1909828, -10.6802349, 10.7768917
2: -6.6882114, 4.0697956, -7.2910938, 4.4045143, -11.0927258, 11.3608894
3: -8.1300840, 3.9907367, -8.8637266, 4.3039637, -12.4340477, 12.8544636
4: -7.3681412, 5.7579913, -8.0227251, 6.2463646, -13.6145058, 13.7807159
5: -5.8688226, 4.9927001, -6.4053111, 5.4079776, -11.2768002, 11.3980112
6: -5.9935799, 6.3579121, -6.5089130, 6.9070282, -12.9006081, 12.8668232
7: -7.5003362, 3.7347536, -8.1561947, 4.0595798, -11.5599155, 11.8909483
8: -7.3705292, 5.4439731, -8.0147495, 5.8980150, -13.2685442, 13.4587231
9: -5.6693549, 6.0951285, -6.1699276, 6.6181273, -12.2874823, 12.2650566

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 84

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155431, upper bound: 9.0158997
time: 4.41 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155431, upper bound: 9.0158997
time: 2.87 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -6.8227777, 5.2460175, -7.4877067, 5.7550998, -12.5778770, 12.7337236
1: -5.4628563, 4.7703457, -6.0154433, 5.2134838, -10.6763401, 10.7857876
2: -6.6641264, 4.0613899, -7.3253074, 4.4242496, -11.0883760, 11.3866968
3: -8.0891027, 3.9757609, -8.9040356, 4.3219671, -12.4110699, 12.8797970
4: -7.3346639, 5.7350268, -8.0588398, 6.2736535, -13.6083174, 13.7938671
5: -5.8386965, 4.9720621, -6.4347463, 5.4310966, -11.2697926, 11.4068079
6: -5.9701271, 6.3353448, -6.5377512, 6.9379692, -12.9080963, 12.8730946
7: -7.4861374, 3.7223163, -8.1945953, 4.0791507, -11.5652866, 11.9169121
8: -7.3373251, 5.4243231, -8.0504246, 5.9235573, -13.2608824, 13.4747477
9: -5.6440411, 6.0712538, -6.1981497, 6.6479797, -12.2920198, 12.2694035

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 84

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155431, upper bound: 9.0159095
time: 4.97 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155431, upper bound: 9.0159094
time: 2.55 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -6.8231196, 5.2416987, -8.4994192, 6.5263953, -13.3495150, 13.7411175
1: -5.4599195, 4.7653704, -6.8500957, 5.8794069, -11.3393269, 11.6154661
2: -6.6524773, 4.0496597, -8.3291416, 4.9656544, -11.6181316, 12.3787985
3: -8.0869465, 3.9719291, -10.1282711, 4.8337936, -12.9207401, 14.1002007
4: -7.3298264, 5.7292895, -9.1548710, 7.0859170, -14.4157429, 14.8841610
5: -5.8374434, 4.9682322, -7.3381906, 6.1211562, -11.9585991, 12.3064213
6: -5.9631886, 6.3253994, -7.3951387, 7.8484278, -13.8116169, 13.7205353
7: -7.4610739, 3.7152238, -9.2596912, 4.6106434, -12.0717173, 12.9749146
8: -7.3326225, 5.4171653, -9.1334038, 6.6774578, -14.0100803, 14.5505695
9: -5.6398449, 6.0641699, -7.0276756, 7.5227590, -13.1626034, 13.0918455

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 84

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155027, upper bound: 9.0154985
time: 3.13 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155027, upper bound: 9.0154985
time: 2.94 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6.7883606, 5.2192621, -8.5318489, 6.5518374, -13.3401985, 13.7511110
1: -5.4338503, 4.7468863, -6.8776908, 5.9018507, -11.3357010, 11.6245766
2: -6.6287823, 4.0414696, -8.3632793, 4.9853201, -11.6141024, 12.4047489
3: -8.0464458, 3.9571767, -10.1685390, 4.8517323, -12.8981771, 14.1257153
4: -7.2967668, 5.7066407, -9.1909504, 7.1131654, -14.4099321, 14.8975906
5: -5.8076615, 4.9478650, -7.3675928, 6.1442418, -11.9519033, 12.3154554
6: -5.9400821, 6.3031850, -7.4239216, 7.8792806, -13.8193626, 13.7271061
7: -7.4472857, 3.7029881, -9.2978983, 4.6302156, -12.0775013, 13.0008869
8: -7.2998204, 5.3978009, -9.1690311, 6.7029395, -14.0027599, 14.5668297
9: -5.6148291, 6.0406251, -7.0558362, 7.5525703, -13.1673985, 13.0964613

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 84

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155027, upper bound: 9.0155012
time: 4.54 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155027, upper bound: 9.0155012
time: 3.57 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -6.9397335, 5.3357463, -6.8144217, 5.2395210, -12.1792545, 12.1501675
1: -5.5614676, 4.8497868, -5.4558735, 4.7649941, -10.3264618, 10.3056602
2: -6.7789130, 4.1251297, -6.6546135, 4.0557895, -10.8347025, 10.7797432
3: -8.2376204, 4.0414357, -8.0807743, 3.9712572, -12.2088776, 12.1222095
4: -7.4632335, 5.8304377, -7.3246727, 5.7278557, -13.1910896, 13.1551104
5: -5.9445987, 5.0541472, -5.8313422, 4.9662495, -10.9108486, 10.8854895
6: -6.0707622, 6.4411182, -5.9627452, 6.3272591, -12.3980217, 12.4038639
7: -7.6097741, 3.7873573, -7.4770341, 3.7169933, -11.3267670, 11.2643909
8: -7.4616470, 5.5122805, -7.3270836, 5.4179459, -12.8795929, 12.8393641
9: -5.7434182, 6.1724286, -5.6369500, 6.0632124, -11.8066311, 11.8093786

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 84

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0157540, upper bound: 9.0157601
time: 2.80 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0157540, upper bound: 9.0157601
time: 2.69 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7.9770699, 6.1297364, -6.7802720, 5.2129736, -13.1900434, 12.9100084
1: -6.4192624, 5.5359287, -5.4270930, 4.7417169, -11.1609793, 10.9630222
2: -7.8092904, 4.6845865, -6.6195526, 4.0360260, -11.8453159, 11.3041391
3: -9.4979038, 4.5661583, -8.0384674, 3.9528189, -13.4507227, 12.6046257
4: -8.5884943, 6.6671433, -7.2870722, 5.6996880, -14.2881823, 13.9542160
5: -6.8718762, 5.7650266, -5.8005457, 4.9422450, -11.8141212, 11.5655727
6: -6.9524798, 7.3792539, -5.9329348, 6.2953501, -13.2478294, 13.3121891
7: -8.7092476, 4.3271031, -7.4384899, 3.6978235, -12.4070711, 11.7655926
8: -8.5733261, 6.2889290, -7.2898693, 5.3916330, -13.9649591, 13.5787983
9: -6.5982680, 7.0642533, -5.6079650, 6.0328207, -12.6310883, 12.6722183

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 57

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0157518, upper bound: 9.0157601
time: 1.91 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0157540, upper bound: 9.0157601
time: 1.89 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6.9397335, 5.3357463, -7.0258856, 5.4027596, -12.3424931, 12.3616314
1: -5.5614676, 4.8497868, -5.6340594, 4.9086308, -10.4700985, 10.4838467
2: -6.7789130, 4.1251297, -6.8674455, 4.1751184, -10.9540310, 10.9925747
3: -8.2376204, 4.0414357, -8.3444328, 4.0879040, -12.3255243, 12.3858681
4: -7.4632335, 5.8304377, -7.5581126, 5.9016056, -13.3648396, 13.3885498
5: -5.9445987, 5.0541472, -6.0223022, 5.1147518, -11.0593510, 11.0764494
6: -6.0707622, 6.4411182, -6.1460214, 6.5216899, -12.5924520, 12.5871391
7: -7.6097741, 3.7873573, -7.7072811, 3.8358197, -11.4455938, 11.4946384
8: -7.4616470, 5.5122805, -7.5555449, 5.5787492, -13.0403957, 13.0678253
9: -5.7434182, 6.1724286, -5.8167133, 6.2491593, -11.9925776, 11.9891415

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 57

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0157506, upper bound: 9.0157505
time: 3.47 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0157506, upper bound: 9.0157505
time: 4.86 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7.9770699, 6.1297364, -6.9916892, 5.3761759, -13.3532457, 13.1214256
1: -6.4192624, 5.5359287, -5.6052346, 4.8852949, -11.3045578, 11.1411629
2: -7.8092904, 4.6845865, -6.8323359, 4.1553249, -11.9646149, 11.5169220
3: -9.4979038, 4.5661583, -8.3019724, 4.0694237, -13.5673275, 12.8681307
4: -8.5884943, 6.6671433, -7.5204616, 5.8733821, -14.4618759, 14.1876049
5: -6.8718762, 5.7650266, -5.9914560, 5.0906992, -11.9625759, 11.7564831
6: -6.9524798, 7.3792539, -6.1161656, 6.4897394, -13.4422188, 13.4954195
7: -8.7092476, 4.3271031, -7.6686902, 3.8165846, -12.5258322, 11.9957933
8: -8.5733261, 6.2889290, -7.5182834, 5.5523868, -14.1257133, 13.8072128
9: -6.5982680, 7.0642533, -5.7876725, 6.2187061, -12.8169746, 12.8519258

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 57

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0157482, upper bound: 9.0157505
time: 2.69 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0157506, upper bound: 9.0157505
time: 4.21 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7.0079017, 5.3890643, -6.8144217, 5.2395210, -12.2474232, 12.2034855
1: -5.6192031, 4.8963184, -5.4558735, 4.7649941, -10.3841972, 10.3521919
2: -6.8500457, 4.1656260, -6.6546135, 4.0557895, -10.9058352, 10.8202400
3: -8.3211136, 4.0787687, -8.0807743, 3.9712572, -12.2923708, 12.1595430
4: -7.5389853, 5.8876429, -7.3246727, 5.7278557, -13.2668409, 13.2123156
5: -6.0061903, 5.1024823, -5.8313422, 4.9662495, -10.9724398, 10.9338245
6: -6.1305323, 6.5052671, -5.9627452, 6.3272591, -12.4577913, 12.4680119
7: -7.6866822, 3.8274076, -7.4770341, 3.7169933, -11.4036751, 11.3044415
8: -7.5369606, 5.5653539, -7.3270836, 5.4179459, -12.9549065, 12.8924370
9: -5.8027868, 6.2343287, -5.6369500, 6.0632124, -11.8659992, 11.8712788

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 84

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155185, upper bound: 9.0157000
time: 4.96 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155185, upper bound: 9.0157000
time: 3.00 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8.0195665, 6.1635284, -6.7802720, 5.2129736, -13.2325401, 12.9438000
1: -6.4558516, 5.5654306, -5.4270930, 4.7417169, -11.1975689, 10.9925232
2: -7.8549695, 4.7112465, -6.6195526, 4.0360260, -11.8909950, 11.3307991
3: -9.5496721, 4.5903015, -8.0384674, 3.9528189, -13.5024910, 12.6287689
4: -8.6364517, 6.7039151, -7.2870722, 5.6996880, -14.3361397, 13.9909878
5: -6.9106483, 5.7957034, -5.8005457, 4.9422450, -11.8528938, 11.5962486
6: -6.9904079, 7.4202414, -5.9329348, 6.2953501, -13.2857580, 13.3531761
7: -8.7590103, 4.3537660, -7.4384899, 3.6978235, -12.4568338, 11.7922554
8: -8.6211929, 6.3228393, -7.2898693, 5.3916330, -14.0128260, 13.6127090
9: -6.6369157, 7.1041546, -5.6079650, 6.0328207, -12.6697369, 12.7121201

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 57

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155170, upper bound: 9.0157000
time: 3.01 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155185, upper bound: 9.0157000
time: 4.21 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7.0079017, 5.3890643, -7.0258856, 5.4027596, -12.4106617, 12.4149494
1: -5.6192031, 4.8963184, -5.6340594, 4.9086308, -10.5278339, 10.5303783
2: -6.8500457, 4.1656260, -6.8674455, 4.1751184, -11.0251637, 11.0330715
3: -8.3211136, 4.0787687, -8.3444328, 4.0879040, -12.4090176, 12.4232016
4: -7.5389853, 5.8876429, -7.5581126, 5.9016056, -13.4405909, 13.4457550
5: -6.0061903, 5.1024823, -6.0223022, 5.1147518, -11.1209421, 11.1247845
6: -6.1305323, 6.5052671, -6.1460214, 6.5216899, -12.6522217, 12.6512890
7: -7.6866822, 3.8274076, -7.7072811, 3.8358197, -11.5225019, 11.5346889
8: -7.5369606, 5.5653539, -7.5555449, 5.5787492, -13.1157093, 13.1208992
9: -5.8027868, 6.2343287, -5.8167133, 6.2491593, -12.0519466, 12.0510426

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 57

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155165, upper bound: 9.0156967
time: 3.84 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155165, upper bound: 9.0156967
time: 3.99 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.0195665, 6.1635284, -6.9916892, 5.3761759, -13.3957424, 13.1552181
1: -6.4558516, 5.5654306, -5.6052346, 4.8852949, -11.3411465, 11.1706657
2: -7.8549695, 4.7112465, -6.8323359, 4.1553249, -12.0102940, 11.5435829
3: -9.5496721, 4.5903015, -8.3019724, 4.0694237, -13.6190958, 12.8922739
4: -8.6364517, 6.7039151, -7.5204616, 5.8733821, -14.5098343, 14.2243767
5: -6.9106483, 5.7957034, -5.9914560, 5.0906992, -12.0013475, 11.7871590
6: -6.9904079, 7.4202414, -6.1161656, 6.4897394, -13.4801474, 13.5364075
7: -8.7590103, 4.3537660, -7.6686902, 3.8165846, -12.5755949, 12.0224562
8: -8.6211929, 6.3228393, -7.5182834, 5.5523868, -14.1735802, 13.8411226
9: -6.6369157, 7.1041546, -5.7876725, 6.2187061, -12.8556213, 12.8918266

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 57

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155144, upper bound: 9.0156967
time: 2.77 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155165, upper bound: 9.0156967
time: 3.48 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -6.9397335, 5.3357463, -6.8871508, 5.2960076, -12.2357407, 12.2228966
1: -5.5614676, 4.8497868, -5.5169582, 4.8141918, -10.3756599, 10.3667450
2: -6.7789130, 4.1251297, -6.7299519, 4.0984430, -10.8773556, 10.8550816
3: -8.2376204, 4.0414357, -8.1688366, 4.0102410, -12.2478619, 12.2102718
4: -7.4632335, 5.8304377, -7.4054875, 5.7880840, -13.2513180, 13.2359257
5: -5.9445987, 5.0541472, -5.8966866, 5.0172682, -10.9618664, 10.9508343
6: -6.0707622, 6.4411182, -6.0261893, 6.3952947, -12.4660568, 12.4673080
7: -7.6097741, 3.7873573, -7.5584831, 3.7582538, -11.3680277, 11.3458405
8: -7.4616470, 5.5122805, -7.4072843, 5.4738126, -12.9354591, 12.9195652
9: -5.7434182, 6.1724286, -5.6986618, 6.1282763, -11.8716946, 11.8710899

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 57

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0158997, upper bound: 9.0155431
time: 5.80 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0159095, upper bound: 9.0155431
time: 4.46 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 6.88 + 601.96 = 608.84 seconds
