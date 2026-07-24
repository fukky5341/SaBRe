## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 173.89956106530002


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329)
1: (-79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183)
2: (-104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471)
3: (-110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219)
4: (-101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509)
5: (-90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867)
6: (-86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223)
7: (-95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773)
8: (-114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580)
9: (-86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.50 + 10.64 = 12.14 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -174.0736347, upper bound: 174.0736347

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0506930, upper bound: 174.0521303
time: 8.65 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0484527, upper bound: 174.0484527
time: 7.15 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 15.95 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 15.95
Output dim: 7, lower bound: -174.0506930, upper bound: 174.0521303
IS_A2, status: Status.UNKNOWN, split count: 1, time: 15.95
Output dim: 7, lower bound: -174.0484527, upper bound: 174.0484527

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -84.7945175, 67.3363419, -94.5060349, 75.0702057, -159.8647156, 161.8423767
1: -71.0174942, 59.7082291, -79.2014389, 66.5985794, -137.6160736, 138.9096680
2: -93.5649643, 61.1370964, -104.3030472, 68.0764999, -161.6414490, 165.4401398
3: -99.2057114, 52.1909523, -110.6649246, 58.1981163, -157.4037933, 162.8558350
4: -90.6216278, 69.7862701, -101.0963440, 77.7846146, -168.4062500, 170.8826141
5: -81.3096313, 63.2362137, -90.6905060, 70.5433807, -151.8530121, 153.9266968
6: -77.9796448, 75.1436157, -86.9384842, 83.7556839, -161.7353210, 162.0820618
7: -85.3573227, 72.0001450, -95.1351624, 80.1866226, -165.5439453, 167.1352692
8: -102.7011108, 69.5013962, -114.4460297, 77.5040588, -180.2051392, 183.9474030
9: -77.8513641, 75.9898987, -86.7146835, 84.7555695, -162.6069336, 162.7045898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=61, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=184, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0484529, upper bound: 174.0484529
time: 7.54 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0484529, upper bound: 174.0484527
time: 6.80 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -70.4749985, 55.9063721, -82.3258972, 65.3742981, -135.8493042, 138.2322540
1: -58.9251480, 49.5083618, -68.9416428, 57.9617653, -116.8869171, 118.4499817
2: -77.6856918, 50.9035301, -90.8320847, 59.3749657, -137.0606232, 141.7355957
3: -82.3612061, 43.3041534, -96.3086929, 50.6716766, -133.0328827, 139.6128235
4: -75.1945114, 57.9392395, -87.9613266, 67.7534180, -142.9479370, 145.9005585
5: -67.4476700, 52.3454247, -78.9354858, 61.3802490, -128.8278961, 131.2808990
6: -64.8120270, 62.4707680, -75.7116089, 72.9496994, -137.7617188, 138.1823578
7: -70.9039001, 59.9517593, -82.8681564, 69.9267349, -140.8306274, 142.8199158
8: -85.4032898, 57.6308556, -99.7242508, 67.4637909, -152.8670654, 157.3550873
9: -64.7994461, 63.0337334, -75.5943604, 73.7637863, -138.5632324, 138.6280975

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=61, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=190, inp2_unstable=184, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=248, inp2_unstable=254, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0361590, upper bound: 174.0369696
time: 7.87 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0413178, upper bound: 174.0413178
time: 8.32 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 17.72 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 17.72
Output dim: 7, lower bound: -174.0484529, upper bound: 174.0484529
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 17.72
Output dim: 7, lower bound: -174.0484529, upper bound: 174.0484527
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 17.72
Output dim: 7, lower bound: -174.0361590, upper bound: 174.0369696
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 17.72
Output dim: 7, lower bound: -174.0413178, upper bound: 174.0413178

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -84.7945175, 67.3363419, -84.7945175, 67.3363419, -152.1308594, 152.1308594
1: -71.0174942, 59.7082291, -71.0174942, 59.7082291, -130.7257233, 130.7257233
2: -93.5649643, 61.1370964, -93.5649643, 61.1370964, -154.7020264, 154.7020264
3: -99.2057114, 52.1909523, -99.2057114, 52.1909523, -151.3966217, 151.3966217
4: -90.6216278, 69.7862701, -90.6216278, 69.7862701, -160.4078827, 160.4078827
5: -81.3096313, 63.2362137, -81.3096313, 63.2362137, -144.5458374, 144.5458374
6: -77.9796448, 75.1436157, -77.9796448, 75.1436157, -153.1232605, 153.1232605
7: -85.3573227, 72.0001450, -85.3573227, 72.0001450, -157.3574371, 157.3574371
8: -102.7011108, 69.5013962, -102.7011108, 69.5013962, -172.2024841, 172.2024841
9: -77.8513641, 75.9898987, -77.8513641, 75.9898987, -153.8412628, 153.8412628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=184, inp2_unstable=184, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0396017, upper bound: 174.0400114
time: 8.80 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0435904, upper bound: 174.0450494
time: 8.06 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -84.7945175, 67.3363419, -70.4749985, 55.9063721, -140.7008820, 137.8113403
1: -71.0174942, 59.7082291, -58.9251480, 49.5083618, -120.5258408, 118.6333771
2: -93.5649643, 61.1370964, -77.6856918, 50.9035301, -144.4684448, 138.8227692
3: -99.2057114, 52.1909523, -82.3612061, 43.3041534, -142.5098114, 134.5521393
4: -90.6216278, 69.7862701, -75.1945114, 57.9392395, -148.5608673, 144.9807587
5: -81.3096313, 63.2362137, -67.4476700, 52.3454247, -133.6550446, 130.6838531
6: -77.9796448, 75.1436157, -64.8120270, 62.4707680, -140.4504089, 139.9556427
7: -85.3573227, 72.0001450, -70.9039001, 59.9517593, -145.3090820, 142.9040070
8: -102.7011108, 69.5013962, -85.4032898, 57.6308556, -160.3319397, 154.9046783
9: -77.8513641, 75.9898987, -64.7994461, 63.0337334, -140.8851013, 140.7893372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=184, inp2_unstable=190, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=248, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0396017, upper bound: 174.0400114
time: 9.81 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0435904, upper bound: 174.0450494
time: 7.57 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -63.8690872, 50.6241417, -63.8882637, 50.6799011, -114.5489883, 114.5123978
1: -53.3631134, 44.8545303, -53.4624290, 44.9592819, -98.3223877, 98.3169556
2: -70.3799133, 46.2276459, -70.4161758, 46.2691269, -116.6490402, 116.6438217
3: -74.6150742, 39.2013474, -74.7059402, 39.2611847, -113.8762589, 113.9072876
4: -68.1703796, 52.5368690, -68.3068848, 52.6657829, -120.8361664, 120.8437500
5: -61.0662804, 47.4350967, -61.1925392, 47.6695862, -108.7358704, 108.6276321
6: -58.7702980, 56.6274796, -58.8395004, 56.6103630, -115.3806458, 115.4669724
7: -64.3340454, 54.4421959, -64.4316864, 54.5194054, -118.8534393, 118.8738785
8: -77.4123764, 52.1665878, -77.4023590, 52.1951370, -129.6074982, 129.5689392
9: -58.8409538, 57.1323280, -58.9143372, 57.2949409, -116.1358871, 116.0466614

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=190, inp2_unstable=181, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=242, inp2_unstable=250, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0354981, upper bound: 174.0364388
time: 9.11 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0354100, upper bound: 174.0363557
time: 7.88 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -67.1838837, 53.2761765, -73.4136810, 58.2676506, -125.4515381, 126.6898575
1: -56.1614799, 47.1871376, -61.4638138, 51.6669540, -107.8284302, 108.6509476
2: -74.0468674, 48.5751953, -80.9568939, 53.0311623, -127.0780258, 129.5320892
3: -78.4974976, 41.2722702, -85.8421173, 45.1735153, -123.6710129, 127.1143799
4: -71.6870956, 55.2456398, -78.4418564, 60.4578362, -132.1449127, 133.6874695
5: -64.2688599, 49.9006920, -70.3494720, 54.7493248, -119.0181885, 120.2501678
6: -61.7931900, 59.5599518, -67.5267029, 65.0572891, -126.8504639, 127.0866547
7: -67.6268692, 57.2063065, -73.9457855, 62.4735260, -130.1003876, 131.1520996
8: -81.4205322, 54.9122200, -88.9241791, 60.0851822, -141.5057068, 143.8363953
9: -61.8271446, 60.0870819, -67.5262604, 65.7847443, -127.6118927, 127.6133423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=190, inp2_unstable=182, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=245, inp2_unstable=251, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0409906, upper bound: 174.0409571
time: 6.86 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0409093, upper bound: 174.0409093
time: 7.54 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 15.92 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 15.92
Output dim: 7, lower bound: -174.0396017, upper bound: 174.0400114
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 15.92
Output dim: 7, lower bound: -174.0435904, upper bound: 174.0450494
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 15.92
Output dim: 7, lower bound: -174.0396017, upper bound: 174.0400114
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 15.92
Output dim: 7, lower bound: -174.0435904, upper bound: 174.0450494
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 15.92
Output dim: 7, lower bound: -174.0354981, upper bound: 174.0364388
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 15.92
Output dim: 7, lower bound: -174.0354100, upper bound: 174.0363557
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 15.92
Output dim: 7, lower bound: -174.0409906, upper bound: 174.0409571
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 15.92
Output dim: 7, lower bound: -174.0409093, upper bound: 174.0409093

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -66.1182098, 52.4538612, -77.6929321, 61.6753387, -127.7935410, 130.1467743
1: -55.3313484, 46.5363007, -65.0468903, 54.6971397, -110.0284805, 111.5831833
2: -72.8861008, 47.8570900, -85.6983719, 56.0734863, -128.9595642, 133.5554657
3: -77.3204422, 40.6280098, -90.8787613, 47.7844734, -125.1049194, 131.5067444
4: -70.7086105, 54.5064659, -83.0511246, 63.9796982, -134.6883087, 137.5575562
5: -63.3368149, 49.3439980, -74.4757919, 57.9477921, -121.2846069, 123.8197784
6: -60.8860092, 58.5924721, -71.4769897, 68.8514099, -129.7374268, 130.0694427
7: -66.6841660, 56.3916893, -78.2573166, 66.0613174, -132.7454681, 134.6490021
8: -80.0947647, 54.0385170, -94.1026764, 63.6189194, -143.7136841, 148.1411896
9: -60.9560890, 59.3092613, -71.4288406, 69.6462021, -130.6022797, 130.7380981

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=60, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=181, inp2_unstable=184, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=250, inp2_unstable=253, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0549819, upper bound: 174.0539616
time: 8.78 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0548613, upper bound: 174.0538548
time: 9.32 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -75.9871292, 60.3190575, -81.4768143, 64.6950989, -140.6822205, 141.7958679
1: -63.6269035, 53.4884529, -68.2319031, 57.3646088, -120.9915085, 121.7203445
2: -83.8097687, 54.8659401, -89.8926239, 58.7740059, -142.5837402, 144.7585602
3: -88.8641281, 46.7527657, -95.3100128, 50.1381912, -139.0023193, 142.0627747
4: -81.2107391, 62.5805740, -87.0751801, 67.0727081, -148.2834473, 149.6557312
5: -72.8304443, 56.6832848, -78.1169815, 60.7666168, -133.5970306, 134.8002472
6: -69.8919601, 67.3421249, -74.9336472, 72.2037125, -142.0956726, 142.2757721
7: -76.5410614, 64.6338577, -82.0376434, 69.2245865, -145.7656403, 146.6715088
8: -92.0325394, 62.2128220, -98.6830292, 66.7575912, -158.7901306, 160.8958435
9: -69.8776245, 68.1081085, -74.8481064, 73.0218582, -142.8994751, 142.9562073

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=60, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=182, inp2_unstable=184, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=252, inp2_unstable=254, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0603609, upper bound: 174.0603619
time: 7.54 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0602538, upper bound: 174.0602538
time: 8.03 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -66.1182098, 52.4538612, -63.8690872, 50.6241417, -116.7423553, 116.3229446
1: -55.3313484, 46.5363007, -53.3631134, 44.8545303, -100.1858826, 99.8994141
2: -72.8861008, 47.8570900, -70.3799133, 46.2276459, -119.1137466, 118.2369995
3: -77.3204422, 40.6280098, -74.6150742, 39.2013474, -116.5217896, 115.2430725
4: -70.7086105, 54.5064659, -68.1703796, 52.5368690, -123.2454758, 122.6768341
5: -63.3368149, 49.3439980, -61.0662804, 47.4350967, -110.7719116, 110.4102783
6: -60.8860092, 58.5924721, -58.7702980, 56.6274796, -117.5134888, 117.3627701
7: -66.6841660, 56.3916893, -64.3340454, 54.4421959, -121.1263580, 120.7257385
8: -80.0947647, 54.0385170, -77.4123764, 52.1665878, -132.2613525, 131.4508972
9: -60.9560890, 59.3092613, -58.8409538, 57.1323280, -118.0884171, 118.1502075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=60, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=181, inp2_unstable=190, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=250, inp2_unstable=242, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0390840, upper bound: 174.0393982
time: 9.41 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0390087, upper bound: 174.0393549
time: 8.25 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -75.9871292, 60.3190575, -67.1838837, 53.2761765, -129.2633057, 127.5029297
1: -63.6269035, 53.4884529, -56.1614799, 47.1871376, -110.8140182, 109.6499329
2: -83.8097687, 54.8659401, -74.0468674, 48.5751953, -132.3849640, 128.9128113
3: -88.8641281, 46.7527657, -78.4974976, 41.2722702, -130.1363678, 125.2502594
4: -81.2107391, 62.5805740, -71.6870956, 55.2456398, -136.4563751, 134.2676697
5: -72.8304443, 56.6832848, -64.2688599, 49.9006920, -122.7311401, 120.9521484
6: -69.8919601, 67.3421249, -61.7931900, 59.5599518, -129.4519043, 129.1352997
7: -76.5410614, 64.6338577, -67.6268692, 57.2063065, -133.7473755, 132.2607269
8: -92.0325394, 62.2128220, -81.4205322, 54.9122200, -146.9447632, 143.6333313
9: -69.8776245, 68.1081085, -61.8271446, 60.0870819, -129.9646912, 129.9352570

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=60, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=182, inp2_unstable=190, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=252, inp2_unstable=245, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 153

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0432084, upper bound: 174.0446391
time: 8.27 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0431436, upper bound: 174.0445648
time: 8.18 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -60.7373466, 48.1488075, -62.8919525, 49.8950310, -110.6323700, 111.0407562
1: -50.7307549, 42.6457977, -52.6273613, 44.2565536, -94.9873047, 95.2731628
2: -66.9202423, 43.9982605, -69.3162994, 45.5585442, -112.4787903, 113.3145523
3: -70.9074707, 37.2666740, -73.5339737, 38.6475410, -109.5550003, 110.8006439
4: -64.8077393, 49.9618607, -67.2374954, 51.8461800, -116.6539154, 117.1993484
5: -58.0621109, 45.1179619, -60.2404289, 46.9347496, -104.9968567, 105.3583908
6: -55.8907585, 53.8572311, -57.9249496, 55.7289085, -111.6196671, 111.7821808
7: -61.1882706, 51.8165970, -63.4297028, 53.6864624, -114.8747177, 115.2462769
8: -73.6274338, 49.6010742, -76.1925125, 51.3726807, -125.0001144, 125.7935867
9: -55.9962540, 54.3316002, -58.0103035, 56.4054680, -112.4017105, 112.3419037

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=60, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=186, inp2_unstable=181, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=241, inp2_unstable=249, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0354981, upper bound: 174.0364388
time: 7.36 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0354981, upper bound: 174.0364388
time: 9.07 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -74.2587204, 58.8768921, -62.5430145, 49.6186943, -123.8774109, 121.4199066
1: -61.9761925, 52.0932503, -52.3329659, 44.0102539, -105.9864502, 104.4262085
2: -81.7725754, 53.4920883, -68.9314423, 45.3113861, -127.0839462, 122.4235306
3: -86.7705612, 45.4198380, -73.1197433, 38.4319725, -125.2025299, 118.5395737
4: -79.3501434, 60.9908371, -66.8592834, 51.5578308, -130.9079742, 127.8501205
5: -71.1028442, 55.1194649, -59.9041786, 46.6757011, -117.7785492, 115.0236435
6: -68.3276520, 65.7254868, -57.6030960, 55.4189720, -123.7466049, 123.3285828
7: -74.6282654, 63.0124817, -63.0777473, 53.3942032, -128.0224609, 126.0902252
8: -89.8551483, 60.6471863, -75.7715912, 51.0849686, -140.9401245, 136.4187775
9: -68.1762695, 66.4228439, -57.6918831, 56.0929337, -124.2692032, 124.1147232

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=60, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=190, inp2_unstable=181, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=250, inp2_unstable=249, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0346416, upper bound: 174.0356618
time: 7.55 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0354100, upper bound: 174.0363557
time: 8.04 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -64.0162277, 50.7689285, -72.3956757, 57.4640121, -121.4802399, 123.1646042
1: -53.4968071, 44.9491463, -60.6099777, 50.9484940, -104.4452972, 105.5591278
2: -70.5483627, 46.3157387, -79.8324127, 52.3034058, -122.8517685, 126.1481476
3: -74.7497864, 39.3168640, -84.6455688, 44.5465393, -119.2963181, 123.9624176
4: -68.2848892, 52.6390419, -77.3489609, 59.6180763, -127.9029617, 129.9879913
5: -61.2301178, 47.5546036, -69.3763809, 53.9968033, -115.2269211, 116.9309845
6: -58.8785934, 56.7586021, -66.5909500, 64.1565475, -123.0351410, 123.3495483
7: -64.4427109, 54.5472984, -72.9207535, 61.6201096, -126.0628204, 127.4680481
8: -77.5958252, 52.3145027, -87.6893005, 59.2438545, -136.8396759, 140.0037842
9: -58.9442024, 57.2508926, -66.5989456, 64.8736725, -123.8178711, 123.8498383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=60, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=187, inp2_unstable=182, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=243, inp2_unstable=251, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0409906, upper bound: 174.0409571
time: 6.77 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0409906, upper bound: 174.0409571
time: 6.58 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -77.5025482, 61.4840584, -72.0258102, 57.1704941, -134.6730347, 133.5098724
1: -64.7108612, 54.3833580, -60.2977409, 50.6872177, -115.3980713, 114.6810989
2: -85.3626709, 55.7953682, -79.4244080, 52.0397224, -137.4023743, 135.2197723
3: -90.5504379, 47.4417725, -84.2076721, 44.3183022, -134.8687286, 131.6494446
4: -82.7900085, 63.6362610, -76.9479446, 59.3109016, -142.1009064, 140.5841980
5: -74.2306671, 57.5329437, -69.0200958, 53.7213745, -127.9520416, 126.5530396
6: -71.2861404, 68.5977325, -66.2490158, 63.8280067, -135.1141510, 134.8467407
7: -77.8496246, 65.7227173, -72.5469513, 61.3089256, -139.1585541, 138.2696686
8: -93.7643890, 63.3391457, -87.2431030, 58.9375725, -152.7019348, 150.5822449
9: -71.1089554, 69.3108673, -66.2592468, 64.5410004, -135.6499634, 135.5701141

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=60, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=190, inp2_unstable=182, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=251, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0409093, upper bound: 174.0409092
time: 6.99 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0409093, upper bound: 174.0409093
time: 7.17 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 15.69 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.69
Output dim: 7, lower bound: -174.0549819, upper bound: 174.0539616
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.69
Output dim: 7, lower bound: -174.0548613, upper bound: 174.0538548
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.69
Output dim: 7, lower bound: -174.0603609, upper bound: 174.0603619
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.69
Output dim: 7, lower bound: -174.0602538, upper bound: 174.0602538
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.69
Output dim: 7, lower bound: -174.0390840, upper bound: 174.0393982
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.69
Output dim: 7, lower bound: -174.0390087, upper bound: 174.0393549
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.69
Output dim: 7, lower bound: -174.0432084, upper bound: 174.0446391
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.69
Output dim: 7, lower bound: -174.0431436, upper bound: 174.0445648
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.69
Output dim: 7, lower bound: -174.0354981, upper bound: 174.0364388
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.69
Output dim: 7, lower bound: -174.0354981, upper bound: 174.0364388
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.69
Output dim: 7, lower bound: -174.0346416, upper bound: 174.0356618
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.69
Output dim: 7, lower bound: -174.0354100, upper bound: 174.0363557
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.69
Output dim: 7, lower bound: -174.0409906, upper bound: 174.0409571
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.69
Output dim: 7, lower bound: -174.0409906, upper bound: 174.0409571
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.69
Output dim: 7, lower bound: -174.0409093, upper bound: 174.0409092
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.69
Output dim: 7, lower bound: -174.0409093, upper bound: 174.0409093

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -65.1229706, 51.6696320, -74.3916931, 59.0681496, -124.1911163, 126.0613174
1: -54.4973717, 45.8342667, -62.2742271, 52.3667412, -106.8641129, 108.1084900
2: -71.7875061, 47.1472435, -82.0497208, 53.7105827, -125.4980927, 129.1969604
3: -76.1499176, 40.0152283, -86.9898605, 45.7507172, -121.9006271, 127.0050888
4: -69.6404037, 53.6877441, -79.5059586, 61.2594032, -130.8997803, 133.1936646
5: -62.3855820, 48.6098785, -71.3199005, 55.5046616, -117.8902435, 119.9297791
6: -59.9721870, 57.7121162, -68.4421616, 65.9302597, -125.9024429, 126.1542816
7: -65.6833344, 55.5594788, -74.9315033, 63.2903023, -128.9736328, 130.4909821
8: -78.8867722, 53.2168732, -90.1048126, 60.8997879, -139.7865448, 143.3216858
9: -60.0526886, 58.4210129, -68.4210510, 66.6933823, -126.7460556, 126.8420486

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=181, inp2_unstable=181, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=250, inp2_unstable=252, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0539731, upper bound: 174.0529588
time: 8.49 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0549082, upper bound: 174.0539438
time: 7.92 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -64.7636185, 51.3849449, -89.0910187, 70.7405090, -135.5041046, 140.4759674
1: -54.1945610, 45.5806160, -74.5022964, 62.6604538, -116.8549805, 120.0829163
2: -71.3913269, 46.8926926, -98.2106934, 64.1049423, -135.4962769, 145.1033630
3: -75.7235184, 39.7934952, -104.2081451, 54.6107140, -130.3342285, 144.0016327
4: -69.2509232, 53.3908882, -95.3291931, 73.2875290, -142.5384521, 148.7200775
5: -62.0392380, 48.3432236, -85.4789200, 66.4230957, -128.4623413, 133.8221436
6: -59.6406517, 57.3930702, -81.9622650, 78.8448792, -138.4855042, 139.3553009
7: -65.3210602, 55.2585106, -89.5965271, 75.5234070, -140.8444519, 144.8550415
8: -78.4538116, 52.9204788, -107.7391586, 72.9557953, -151.4095917, 160.6596375
9: -59.7246017, 58.0994949, -81.7406006, 79.8334885, -139.5580902, 139.8400879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=181, inp2_unstable=185, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=250, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0538531, upper bound: 174.0528443
time: 9.18 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0547917, upper bound: 174.0538373
time: 7.95 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -74.9697418, 59.5160637, -78.1643906, 62.0787659, -137.0485077, 137.6804352
1: -62.7740974, 52.7705765, -65.4480209, 55.0257797, -117.7998810, 118.2185822
2: -82.6856613, 54.1378326, -86.2318497, 56.4022217, -139.0878754, 140.3696442
3: -87.6680679, 46.1263771, -91.4063416, 48.0964203, -135.7644501, 137.5327148
4: -80.1183929, 61.7417030, -83.5173645, 64.3429794, -144.4613647, 145.2590179
5: -71.8580475, 55.9308395, -74.9501190, 58.3146057, -130.1726227, 130.8809509
6: -68.9561081, 66.4423523, -71.8881454, 69.2717667, -138.2278748, 138.3305054
7: -75.5162811, 63.7810555, -78.7011642, 66.4429703, -141.9592133, 142.4822235
8: -90.7982788, 61.3720703, -94.6722031, 64.0293884, -154.8276672, 156.0442657
9: -68.9509125, 67.1979294, -71.8292847, 70.0578613, -139.0087738, 139.0272217

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=182, inp2_unstable=182, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=252, inp2_unstable=253, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 185

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0596035, upper bound: 174.0594784
time: 9.07 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0603402, upper bound: 174.0603422
time: 7.08 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -74.5869980, 59.2123833, -92.6620026, 73.5808029, -148.1678009, 151.8743744
1: -62.4512558, 52.5002022, -77.5079956, 65.1810074, -127.6322556, 130.0081940
2: -82.2633667, 53.8647308, -102.1590424, 66.6587067, -148.9220734, 156.0237732
3: -87.2149811, 45.8902817, -108.3916626, 56.8564339, -144.0714111, 154.2819366
4: -79.7033386, 61.4240341, -99.1338043, 76.2013016, -155.9046326, 160.5578308
5: -71.4894333, 55.6458244, -88.9047623, 69.0875931, -140.5770264, 144.5505829
6: -68.6020050, 66.1025696, -85.2241364, 82.0110245, -150.6130219, 151.3267059
7: -75.1294098, 63.4592094, -93.1605911, 78.5105972, -153.6399994, 156.6197815
8: -90.3365021, 61.0551071, -112.0557556, 75.9129715, -166.2494659, 173.1108704
9: -68.5995102, 66.8538818, -84.9676666, 83.0150833, -151.6145782, 151.8215027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=182, inp2_unstable=186, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=252, inp2_unstable=256, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 185

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0594920, upper bound: 174.0593741
time: 17.43 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0602240, upper bound: 174.0602240
time: 7.41 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -65.1229706, 51.6696320, -60.7373466, 48.1488075, -113.2717743, 112.4069672
1: -54.4973717, 45.8342667, -50.7307549, 42.6457977, -97.1431732, 96.5650177
2: -71.7875061, 47.1472435, -66.9202423, 43.9982605, -115.7857590, 114.0674744
3: -76.1499176, 40.0152283, -70.9074707, 37.2666740, -113.4165802, 110.9226990
4: -69.6404037, 53.6877441, -64.8077393, 49.9618607, -119.6022491, 118.4954834
5: -62.3855820, 48.6098785, -58.0621109, 45.1179619, -107.5035400, 106.6719818
6: -59.9721870, 57.7121162, -55.8907585, 53.8572311, -113.8294144, 113.6028748
7: -65.6833344, 55.5594788, -61.1882706, 51.8165970, -117.4999237, 116.7477341
8: -78.8867722, 53.2168732, -73.6274338, 49.6010742, -128.4878540, 126.8442993
9: -60.0526886, 58.4210129, -55.9962540, 54.3316002, -114.3842773, 114.4172440

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=181, inp2_unstable=186, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=250, inp2_unstable=241, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0382944, upper bound: 174.0385074
time: 9.82 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0390743, upper bound: 174.0393982
time: 9.45 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -64.7636185, 51.3849449, -74.2587204, 58.8768921, -123.6405029, 125.6436615
1: -54.1945610, 45.5806160, -61.9761925, 52.0932503, -106.2877884, 107.5568085
2: -71.3913269, 46.8926926, -81.7725754, 53.4920883, -124.8834152, 128.6652679
3: -75.7235184, 39.7934952, -86.7705612, 45.4198380, -121.1433563, 126.5640564
4: -69.2509232, 53.3908882, -79.3501434, 60.9908371, -130.2417603, 132.7410278
5: -62.0392380, 48.3432236, -71.1028442, 55.1194649, -117.1586990, 119.4460678
6: -59.6406517, 57.3930702, -68.3276520, 65.7254868, -125.3661346, 125.7207184
7: -65.3210602, 55.2585106, -74.6282654, 63.0124817, -128.3335266, 129.8867798
8: -78.4538116, 52.9204788, -89.8551483, 60.6471863, -139.1009827, 142.7756348
9: -59.7246017, 58.0994949, -68.1762695, 66.4228439, -126.1474457, 126.2757645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=181, inp2_unstable=190, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=250, inp2_unstable=250, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0381861, upper bound: 174.0384440
time: 9.66 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0390018, upper bound: 174.0393549
time: 7.16 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -74.9697418, 59.5160637, -64.0162277, 50.7689285, -125.7386627, 123.5322876
1: -62.7740974, 52.7705765, -53.4968071, 44.9491463, -107.7232437, 106.2673721
2: -82.6856613, 54.1378326, -70.5483627, 46.3157387, -129.0013885, 124.6861954
3: -87.6680679, 46.1263771, -74.7497864, 39.3168640, -126.9849167, 120.8761520
4: -80.1183929, 61.7417030, -68.2848892, 52.6390419, -132.7574310, 130.0265656
5: -71.8580475, 55.9308395, -61.2301178, 47.5546036, -119.4126511, 117.1609573
6: -68.9561081, 66.4423523, -58.8785934, 56.7586021, -125.7147064, 125.3209457
7: -75.5162811, 63.7810555, -64.4427109, 54.5472984, -130.0635529, 128.2237701
8: -90.7982788, 61.3720703, -77.5958252, 52.3145027, -143.1127625, 138.9678955
9: -68.9509125, 67.1979294, -58.9442024, 57.2508926, -126.2018051, 126.1421280

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=182, inp2_unstable=187, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=252, inp2_unstable=243, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 185

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0423518, upper bound: 174.0436791
time: 8.85 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0432084, upper bound: 174.0446389
time: 7.91 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -74.5869980, 59.2123833, -77.5025482, 61.4840584, -136.0710602, 136.7149353
1: -62.4512558, 52.5002022, -64.7108612, 54.3833580, -116.8346100, 117.2110596
2: -82.2633667, 53.8647308, -85.3626709, 55.7953682, -138.0587311, 139.2273865
3: -87.2149811, 45.8902817, -90.5504379, 47.4417725, -134.6567383, 136.4406891
4: -79.7033386, 61.4240341, -82.7900085, 63.6362610, -143.3395996, 144.2140503
5: -71.4894333, 55.6458244, -74.2306671, 57.5329437, -129.0223694, 129.8764954
6: -68.6020050, 66.1025696, -71.2861404, 68.5977325, -137.1997375, 137.3887024
7: -75.1294098, 63.4592094, -77.8496246, 65.7227173, -140.8521118, 141.3088379
8: -90.3365021, 61.0551071, -93.7643890, 63.3391457, -153.6756439, 154.8194885
9: -68.5995102, 66.8538818, -71.1089554, 69.3108673, -137.9103699, 137.9628143

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=182, inp2_unstable=190, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=252, inp2_unstable=251, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 185

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0422513, upper bound: 174.0436112
time: 7.80 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0431436, upper bound: 174.0445648
time: 8.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -60.7373466, 48.1488075, -65.1132584, 51.6607094, -112.3980560, 113.2620697
1: -50.7307549, 42.6457977, -54.4907265, 45.8270416, -96.5578003, 97.1365204
2: -66.9202423, 43.9982605, -71.7775421, 47.1403313, -114.0605698, 115.7757950
3: -70.9074707, 37.2666740, -76.1396027, 40.0103722, -110.9178391, 113.4062729
4: -64.8077393, 49.9618607, -69.6300812, 53.6797371, -118.4874725, 119.5919342
5: -58.0621109, 45.1179619, -62.3754730, 48.6024513, -106.6645584, 107.4934387
6: -55.8907585, 53.8572311, -59.9621124, 57.7042694, -113.5950317, 113.8193359
7: -61.1882706, 51.8165970, -65.6745377, 55.5505028, -116.7387695, 117.4911270
8: -73.6274338, 49.6010742, -78.8783035, 53.2079277, -126.8353500, 128.4793701
9: -55.9962540, 54.3316002, -60.0418434, 58.4140091, -114.4102478, 114.3734436

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=186, inp2_unstable=181, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=241, inp2_unstable=250, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0346568, upper bound: 174.0355130
time: 7.89 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0354981, upper bound: 174.0364388
time: 9.53 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -60.7373466, 48.1488075, -52.3624725, 41.4441376, -102.1814880, 100.5112762
1: -50.7307549, 42.6457977, -43.6971512, 36.7814789, -87.5122223, 86.3429489
2: -66.9202423, 43.9982605, -57.6524887, 38.1103745, -105.0306168, 101.6507492
3: -70.9074707, 37.2666740, -61.0429955, 32.0699577, -102.9774170, 98.3096695
4: -64.8077393, 49.9618607, -55.8985291, 43.1194077, -107.9271393, 105.8603897
5: -58.0621109, 45.1179619, -49.9574585, 38.9656906, -97.0277710, 95.0754089
6: -55.8907585, 53.8572311, -48.2071037, 46.4544754, -102.3452301, 102.0643311
7: -61.1882706, 51.8165970, -52.8650131, 44.8479424, -106.0362091, 104.6815948
8: -73.6274338, 49.6010742, -63.4543991, 42.6970673, -116.3245010, 113.0554733
9: -55.9962540, 54.3316002, -48.4635506, 46.8615952, -102.8578339, 102.7951431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=186, inp2_unstable=186, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=241, inp2_unstable=230, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0346568, upper bound: 174.0355130
time: 8.11 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0354981, upper bound: 174.0364388
time: 7.28 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -72.7192688, 57.6530762, -59.3399734, 47.0773239, -119.7965927, 116.9930267
1: -60.6833267, 51.0081520, -49.6421585, 41.7558441, -102.4391708, 100.6503143
2: -80.0718307, 52.4144249, -65.3938141, 43.0723228, -123.1441498, 117.8082199
3: -84.9472046, 44.4647713, -69.3382034, 36.4345093, -121.3817139, 113.8029785
4: -77.6946640, 59.7268295, -63.4232063, 48.9254761, -126.6201401, 123.1500320
5: -69.6177902, 53.9705658, -56.8211632, 44.2913017, -113.9090881, 110.7917328
6: -66.9113083, 64.3619232, -54.6682549, 52.5709496, -119.4822540, 119.0301819
7: -73.0870056, 61.7263451, -59.8688545, 50.7264061, -123.8134155, 121.5951996
8: -87.9972763, 59.3912849, -71.8997726, 48.4536514, -136.4509277, 131.2910614
9: -66.7734146, 65.0446320, -54.7697258, 53.2253876, -119.9988022, 119.8143616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=177, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=250, inp2_unstable=246, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9922458, upper bound: 173.9962000
time: 10.46 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9881364, upper bound: 173.9898888
time: 7.41 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -73.6483231, 58.3918495, -60.4882622, 47.9905663, -121.6388855, 118.8800888
1: -61.4622803, 51.6622276, -50.6068611, 42.5641365, -104.0264130, 102.2690811
2: -81.0980988, 53.0655289, -66.6613541, 43.8731804, -124.9712830, 119.7268829
3: -86.0458679, 45.0381813, -70.6892624, 37.1505280, -123.1963959, 115.7274475
4: -78.6942291, 60.4895363, -64.6550751, 49.8706436, -128.5648346, 125.1446075
5: -70.5136642, 54.6633682, -57.9265327, 45.1419182, -115.6555710, 112.5898972
6: -67.7659531, 65.1840820, -55.7172279, 53.5948563, -121.3608017, 120.9012985
7: -74.0172729, 62.5021973, -61.0219574, 51.6779938, -125.6952667, 123.5241547
8: -89.1186981, 60.1491013, -73.2917786, 49.4053307, -138.5240173, 133.4408722
9: -67.6197128, 65.8765030, -55.8182373, 54.2562561, -121.8759689, 121.6947327

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=179, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=250, inp2_unstable=246, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9932000, upper bound: 173.9970932
time: 9.55 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9888689, upper bound: 173.9905585
time: 7.85 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -64.0162277, 50.7689285, -74.9614639, 59.5100327, -123.5262604, 125.7303848
1: -53.4968071, 44.9491463, -62.7689362, 52.7651482, -106.2619553, 107.7180786
2: -70.5483627, 46.3157387, -82.6755066, 54.1293602, -124.6777191, 128.9912415
3: -74.7497864, 39.3168640, -87.6573334, 46.1218567, -120.8716354, 126.9741821
4: -68.2848892, 52.6390419, -80.1090012, 61.7360229, -130.0209045, 132.7480469
5: -61.2301178, 47.5546036, -71.8505325, 55.9239273, -117.1540451, 119.4051361
6: -58.8785934, 56.7586021, -68.9465942, 66.4364395, -125.3150330, 125.7052002
7: -64.4427109, 54.5472984, -75.5067520, 63.7745094, -128.2172241, 130.0540314
8: -77.5958252, 52.3145027, -90.7877731, 61.3655281, -138.9613495, 143.1022491
9: -58.9442024, 57.2508926, -68.9435883, 67.1918030, -126.1360016, 126.1944809

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=187, inp2_unstable=182, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=243, inp2_unstable=252, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0399334, upper bound: 174.0398325
time: 8.02 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0409906, upper bound: 174.0409571
time: 7.82 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -64.0162277, 50.7689285, -60.7812653, 48.1644058, -112.1806183, 111.5501938
1: -53.4968071, 44.9491463, -50.7720261, 42.6737213, -96.1705246, 95.7211533
2: -70.5483627, 46.3157387, -66.9617157, 44.0526810, -114.6010437, 113.2774506
3: -74.7497864, 39.3168640, -70.9559021, 37.3084831, -112.0582733, 110.2727585
4: -68.2848892, 52.6390419, -64.8540726, 49.9990082, -118.2838974, 117.4931107
5: -61.2301178, 47.5546036, -58.0770912, 45.1454353, -106.3755493, 105.6316986
6: -58.8785934, 56.7586021, -55.9126129, 53.8942032, -112.7727966, 112.6712036
7: -64.4427109, 54.5472984, -61.2460823, 51.8623390, -116.3050385, 115.7933807
8: -77.5958252, 52.3145027, -73.6572342, 49.6323204, -127.2281494, 125.9717407
9: -58.9442024, 57.2508926, -56.0465279, 54.3449440, -113.2891464, 113.2974167

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=187, inp2_unstable=187, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=243, inp2_unstable=237, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0399334, upper bound: 174.0398325
time: 7.78 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0409906, upper bound: 174.0409571
time: 6.91 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -77.5025482, 61.4840584, -74.5795212, 59.2069626, -136.7095032, 136.0635834
1: -64.7108612, 54.3833580, -62.4466019, 52.4953117, -117.2061768, 116.8299561
2: -85.3626709, 55.7953682, -82.2542191, 53.8570900, -139.2197571, 138.0495911
3: -90.5504379, 47.4417725, -87.2053299, 45.8862114, -136.4366302, 134.6470795
4: -82.7900085, 63.6362610, -79.6948547, 61.4189186, -144.2089081, 143.3311157
5: -74.2306671, 57.5329437, -71.4826584, 55.6395988, -129.8702698, 129.0155792
6: -71.2861404, 68.5977325, -68.5934219, 66.0972366, -137.3833771, 137.1911621
7: -77.8496246, 65.7227173, -75.1208191, 63.4533119, -141.3029327, 140.8435211
8: -93.7643890, 63.3391457, -90.3270340, 61.0492249, -154.8135834, 153.6661530
9: -71.1089554, 69.3108673, -68.5929031, 66.8483658, -137.9573059, 137.9037781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=190, inp2_unstable=182, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=252, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 153

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0398916, upper bound: 174.0397889
time: 7.69 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0409092, upper bound: 174.0409092
time: 7.45 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -77.5025482, 61.4840584, -60.4145851, 47.8739052, -125.3764496, 121.8986282
1: -64.7108612, 54.3833580, -50.4622040, 42.4157333, -107.1265869, 104.8455505
2: -85.3626709, 55.7953682, -66.5572968, 43.7927971, -129.1554718, 122.3526611
3: -90.5504379, 47.4417725, -70.5203400, 37.0823097, -127.6327515, 117.9621124
4: -82.7900085, 63.6362610, -64.4572296, 49.6951065, -132.4850922, 128.0934906
5: -74.2306671, 57.5329437, -57.7237358, 44.8738518, -119.1045074, 115.2566833
6: -71.2861404, 68.5977325, -55.5730629, 53.5693703, -124.8555145, 124.1707916
7: -77.8496246, 65.7227173, -60.8760948, 51.5542679, -129.4039001, 126.5988159
8: -93.7643890, 63.3391457, -73.2150040, 49.3310318, -143.0954285, 136.5541534
9: -71.1089554, 69.3108673, -55.7108307, 54.0155220, -125.1244812, 125.0216904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=190, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=237, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 153

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0398916, upper bound: 174.0397889
time: 8.65 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0409093, upper bound: 174.0409092
time: 7.65 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 17.90 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 7, lower bound: -174.0539731, upper bound: 174.0529588
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 7, lower bound: -174.0549082, upper bound: 174.0539438
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 7, lower bound: -174.0538531, upper bound: 174.0528443
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 7, lower bound: -174.0547917, upper bound: 174.0538373
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 7, lower bound: -174.0596035, upper bound: 174.0594784
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 7, lower bound: -174.0603402, upper bound: 174.0603422
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 7, lower bound: -174.0594920, upper bound: 174.0593741
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 7, lower bound: -174.0602240, upper bound: 174.0602240
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 7, lower bound: -174.0382944, upper bound: 174.0385074
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 7, lower bound: -174.0390743, upper bound: 174.0393982
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 7, lower bound: -174.0381861, upper bound: 174.0384440
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 7, lower bound: -174.0390018, upper bound: 174.0393549
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 7, lower bound: -174.0423518, upper bound: 174.0436791
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 7, lower bound: -174.0432084, upper bound: 174.0446389
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 7, lower bound: -174.0422513, upper bound: 174.0436112
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 7, lower bound: -174.0431436, upper bound: 174.0445648
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 7, lower bound: -174.0346568, upper bound: 174.0355130
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 7, lower bound: -174.0354981, upper bound: 174.0364388
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 7, lower bound: -174.0346568, upper bound: 174.0355130
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 7, lower bound: -174.0354981, upper bound: 174.0364388
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 7, lower bound: -173.9922458, upper bound: 173.9962000
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 7, lower bound: -173.9881364, upper bound: 173.9898888
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 7, lower bound: -173.9932000, upper bound: 173.9970932
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 7, lower bound: -173.9888689, upper bound: 173.9905585
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 7, lower bound: -174.0399334, upper bound: 174.0398325
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 7, lower bound: -174.0409906, upper bound: 174.0409571
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 7, lower bound: -174.0399334, upper bound: 174.0398325
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 7, lower bound: -174.0409906, upper bound: 174.0409571
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 7, lower bound: -174.0398916, upper bound: 174.0397889
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 7, lower bound: -174.0409092, upper bound: 174.0409092
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 7, lower bound: -174.0398916, upper bound: 174.0397889
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 7, lower bound: -174.0409093, upper bound: 174.0409092

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -61.9012375, 49.1103783, -72.8234177, 57.8220139, -119.7232513, 121.9337921
1: -51.7897987, 43.5632057, -60.9598579, 51.2612457, -103.0510406, 104.5230637
2: -68.2299576, 44.8933907, -80.3155746, 52.6074219, -120.8373795, 125.2089615
3: -72.3472061, 38.0076027, -85.1372223, 44.7808075, -117.1280136, 123.1448212
4: -66.1813354, 51.0393143, -77.8189163, 59.9708252, -126.1521530, 128.8582001
5: -59.2849541, 46.2114029, -69.8109283, 54.3342247, -113.6191711, 116.0223312
6: -57.0211983, 54.8478203, -67.0001755, 64.5399704, -121.5611572, 121.8479919
7: -62.4540176, 52.8746071, -73.3580246, 61.9787407, -124.4327469, 126.2326202
8: -74.9935989, 50.5703735, -88.2114410, 59.6160622, -134.6096649, 138.7817841
9: -57.1110268, 55.5367661, -66.9882736, 65.2896347, -122.4006424, 122.5250397

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=58, inp2_unstable=59, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=178, inp2_unstable=179, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=248, inp2_unstable=252, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0124441, upper bound: 174.0153627
time: 9.98 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0512643, upper bound: 174.0501892
time: 9.54 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -63.0658379, 50.0379410, -73.7612762, 58.5676193, -121.6334457, 123.7992020
1: -52.7679977, 44.3838806, -61.7446976, 51.9219360, -104.6899338, 106.1285782
2: -69.5148773, 45.7062645, -81.3522873, 53.2680130, -122.7828903, 127.0585403
3: -73.7161865, 38.7332649, -86.2437286, 45.3582344, -119.0744019, 124.9769897
4: -67.4326859, 51.9973412, -78.8285980, 60.7414207, -128.1740875, 130.8259277
5: -60.4054298, 47.0747185, -70.7132034, 55.0338097, -115.4392395, 117.7879181
6: -58.0836945, 55.8862724, -67.8626175, 65.3706436, -123.4543228, 123.7488708
7: -63.6240616, 53.8401833, -74.2995148, 62.7628899, -126.3869476, 128.1396942
8: -76.4049911, 51.5344849, -89.3438339, 60.3840866, -136.7890778, 140.8783264
9: -58.1752777, 56.5820580, -67.8451843, 66.1291962, -124.3044739, 124.4272461

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=58, inp2_unstable=59, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=179, inp2_unstable=180, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=250, inp2_unstable=252, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 153

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0125836, upper bound: 174.0154661
time: 10.93 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0523253, upper bound: 174.0512960
time: 8.22 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 20.78 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 20.78
Output dim: 7, lower bound: -174.0124441, upper bound: 174.0153627
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 20.78
Output dim: 7, lower bound: -174.0512643, upper bound: 174.0501892
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 20.78
Output dim: 7, lower bound: -174.0125836, upper bound: 174.0154661
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 20.78
Output dim: 7, lower bound: -174.0523253, upper bound: 174.0512960
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.78
Output dim: 7, lower bound: -174.0538531, upper bound: 174.0528443
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.78
Output dim: 7, lower bound: -174.0547917, upper bound: 174.0538373
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.78
Output dim: 7, lower bound: -174.0596035, upper bound: 174.0594784
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.78
Output dim: 7, lower bound: -174.0603402, upper bound: 174.0603422
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.78
Output dim: 7, lower bound: -174.0594920, upper bound: 174.0593741
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.78
Output dim: 7, lower bound: -174.0602240, upper bound: 174.0602240
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.78
Output dim: 7, lower bound: -174.0382944, upper bound: 174.0385074
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.78
Output dim: 7, lower bound: -174.0390743, upper bound: 174.0393982
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.78
Output dim: 7, lower bound: -174.0381861, upper bound: 174.0384440
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.78
Output dim: 7, lower bound: -174.0390018, upper bound: 174.0393549
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.78
Output dim: 7, lower bound: -174.0423518, upper bound: 174.0436791
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.78
Output dim: 7, lower bound: -174.0432084, upper bound: 174.0446389
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.78
Output dim: 7, lower bound: -174.0422513, upper bound: 174.0436112
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.78
Output dim: 7, lower bound: -174.0431436, upper bound: 174.0445648
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.78
Output dim: 7, lower bound: -174.0346568, upper bound: 174.0355130
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.78
Output dim: 7, lower bound: -174.0354981, upper bound: 174.0364388
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.78
Output dim: 7, lower bound: -174.0346568, upper bound: 174.0355130
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.78
Output dim: 7, lower bound: -174.0354981, upper bound: 174.0364388
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.78
Output dim: 7, lower bound: -173.9922458, upper bound: 173.9962000
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.78
Output dim: 7, lower bound: -173.9881364, upper bound: 173.9898888
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.78
Output dim: 7, lower bound: -173.9932000, upper bound: 173.9970932
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.78
Output dim: 7, lower bound: -173.9888689, upper bound: 173.9905585
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.78
Output dim: 7, lower bound: -174.0399334, upper bound: 174.0398325
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.78
Output dim: 7, lower bound: -174.0409906, upper bound: 174.0409571
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.78
Output dim: 7, lower bound: -174.0399334, upper bound: 174.0398325
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.78
Output dim: 7, lower bound: -174.0409906, upper bound: 174.0409571
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.78
Output dim: 7, lower bound: -174.0398916, upper bound: 174.0397889
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.78
Output dim: 7, lower bound: -174.0409092, upper bound: 174.0409092
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.78
Output dim: 7, lower bound: -174.0398916, upper bound: 174.0397889
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.78
Output dim: 7, lower bound: -174.0409093, upper bound: 174.0409092

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 12.14 + 601.58 = 613.72 seconds
