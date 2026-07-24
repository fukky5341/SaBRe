## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2000 seconds
Threshold: 173.89956106530002
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

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

## BASE Result
execution time: IAR + LP analysis = 1.38 + 9.71 = 11.09 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -174.0741399, upper bound: 174.0741399


# Binary Search by BASE starts (time budget: 1988.91 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=175.32177734375
rel_dist={7: [-174.07406456819004, 174.07406456764704]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=175.32177734375
rel_dist={7: [-174.07363473064066, 174.07363473064066]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=175.32177734375
rel_dist={7: [-174.07321147769613, 174.07321147870266]}

## Binary Search Result
Binary search time: 39.67 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1949.25 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0530380, upper bound: 174.0555292
time: 7.75 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0489073, upper bound: 174.0489072
time: 7.51 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 15.38 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 15.38
Output dim: 7, lower bound: -174.0530380, upper bound: 174.0555292
IS_A2, status: Status.UNKNOWN, split count: 1, time: 15.38
Output dim: 7, lower bound: -174.0489073, upper bound: 174.0489072

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

Time for backsubstitution: 1.21 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0489074, upper bound: 174.0489073
time: 6.47 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0489074, upper bound: 174.0489073
time: 6.58 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -70.4749985, 55.9063721, -89.6166077, 71.1798325, -141.6548309, 145.5229645
1: -58.9251480, 49.5083618, -75.0847397, 63.1332054, -122.0583496, 124.5930786
2: -77.6856918, 50.9035301, -98.8970261, 64.5852661, -142.2709198, 149.8005524
3: -82.3612061, 43.3041534, -104.9027405, 55.1791344, -137.5403442, 148.2068939
4: -75.1945114, 57.9392395, -95.8249893, 73.7591858, -148.9537048, 153.7642212
5: -67.4476700, 52.3454247, -85.9718246, 66.8672104, -134.3148651, 138.3172455
6: -64.8120270, 62.4707680, -82.4334412, 79.4191895, -144.2312164, 144.9042053
7: -70.9039001, 59.9517593, -90.2127457, 76.0700760, -146.9739685, 150.1645050
8: -85.4032898, 57.6308556, -108.5385208, 73.4758530, -158.8791351, 166.1693726
9: -64.7994461, 63.0337334, -82.2529068, 80.3452911, -145.1446838, 145.2866364

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 190
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
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0370024, upper bound: 174.0383716
time: 7.61 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0417222, upper bound: 174.0417222
time: 7.15 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 16.12 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 16.12
Output dim: 7, lower bound: -174.0489074, upper bound: 174.0489073
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 16.12
Output dim: 7, lower bound: -174.0489074, upper bound: 174.0489073
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 16.12
Output dim: 7, lower bound: -174.0370024, upper bound: 174.0383716
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 16.12
Output dim: 7, lower bound: -174.0417222, upper bound: 174.0417222

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

Time for backsubstitution: 1.24 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0428660, upper bound: 174.0439896
time: 8.56 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0460376, upper bound: 174.0486163
time: 7.64 seconds

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

Time for backsubstitution: 1.23 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0428660, upper bound: 174.0439897
time: 8.20 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0460376, upper bound: 174.0486163
time: 8.51 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -69.1525192, 54.8498573, -70.9380722, 56.2994804, -125.4519958, 125.7879257
1: -57.8138161, 48.5768089, -59.3851089, 49.9576721, -107.7714691, 107.9619141
2: -76.2224579, 49.9639053, -78.2193680, 51.2897835, -127.5122299, 128.1832733
3: -80.8126831, 42.4846153, -83.0093689, 43.5963745, -124.4090576, 125.4939651
4: -73.7881546, 56.8573456, -75.8997726, 58.4838295, -132.2719879, 132.7571106
5: -66.1726685, 51.3616219, -68.0064087, 52.9659538, -119.1386185, 119.3680267
6: -63.6022835, 61.3005447, -65.3319092, 62.8579597, -126.4602432, 126.6324539
7: -69.5856781, 58.8469467, -71.5375214, 60.4494019, -130.0350800, 130.3844604
8: -83.8028336, 56.5375557, -85.9356995, 58.0110741, -141.8139038, 142.4732513
9: -63.6051788, 61.8523979, -65.3502045, 63.6649780, -127.2701416, 127.2025986

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
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
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 83
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
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0354872, upper bound: 174.0354872
time: 8.62 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0354872, upper bound: 174.0383716
time: 8.65 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -70.2723083, 55.7446861, -80.6673355, 64.0536957, -134.3260040, 136.4120178
1: -58.7552223, 49.3653908, -67.5645676, 56.8123932, -115.5676117, 116.9299622
2: -77.4613647, 50.7596207, -88.9929810, 58.2102547, -135.6716156, 139.7525787
3: -82.1235962, 43.1792374, -94.3946915, 49.6388741, -131.7624664, 137.5739136
4: -74.9784470, 57.7734108, -86.2586288, 66.4405670, -141.4190063, 144.0320129
5: -67.2523651, 52.1947060, -77.3576355, 60.2052116, -127.4575577, 129.5523376
6: -64.6261520, 62.2913437, -74.2170334, 71.4858551, -136.1119843, 136.5083618
7: -70.7014771, 59.7824364, -81.2594986, 68.5804672, -139.2819519, 141.0419312
8: -85.1576920, 57.4635620, -97.6990891, 66.0721283, -151.2298279, 155.1626587
9: -64.6163559, 62.8522110, -74.1527634, 72.3383713, -136.9547272, 137.0049744

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 83
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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0383716, upper bound: 174.0370024
time: 7.54 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0383716, upper bound: 174.0417222
time: 7.32 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 16.28 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 16.28
Output dim: 7, lower bound: -174.0428660, upper bound: 174.0439896
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 16.28
Output dim: 7, lower bound: -174.0460376, upper bound: 174.0486163
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 16.28
Output dim: 7, lower bound: -174.0428660, upper bound: 174.0439897
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 16.28
Output dim: 7, lower bound: -174.0460376, upper bound: 174.0486163
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 16.28
Output dim: 7, lower bound: -174.0354872, upper bound: 174.0354872
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 16.28
Output dim: 7, lower bound: -174.0354872, upper bound: 174.0383716
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 16.28
Output dim: 7, lower bound: -174.0383716, upper bound: 174.0370024
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 16.28
Output dim: 7, lower bound: -174.0383716, upper bound: 174.0417222

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -66.1182098, 52.4538612, -83.3898010, 66.2177048, -132.3359070, 135.8436279
1: -55.3313484, 46.5363007, -69.8356705, 58.7171669, -114.0485153, 116.3719635
2: -72.8861008, 47.8570900, -92.0101395, 60.1353912, -133.0214844, 139.8672180
3: -77.3204422, 40.6280098, -97.5592041, 51.3176422, -128.6380920, 138.1872101
4: -70.7086105, 54.5064659, -89.1237030, 68.6382904, -139.3468933, 143.6301727
5: -63.3368149, 49.3439980, -79.9585342, 62.1898384, -125.5266571, 129.3025360
6: -60.8860092, 58.5924721, -76.6939163, 73.8980560, -134.7840576, 135.2863922
7: -66.6841660, 56.3916893, -83.9536285, 70.8249664, -137.5091248, 140.3452911
8: -80.0947647, 54.0385170, -101.0005188, 68.3383789, -148.4331360, 155.0390320
9: -60.9560890, 59.3092613, -76.5814514, 74.7356873, -135.6917725, 135.8907166

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
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
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 123
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0543145, upper bound: 174.0543145
time: 7.51 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0543145, upper bound: 174.0558562
time: 7.15 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -75.9871292, 60.3190575, -84.5922623, 67.1753693, -143.1625061, 144.9113159
1: -63.6269035, 53.4884529, -70.8475723, 59.5653687, -123.1922607, 124.3360214
2: -83.8097687, 54.8659401, -93.3410950, 60.9930382, -144.8028107, 148.2070312
3: -88.8641281, 46.7527657, -98.9682465, 52.0657196, -140.9298401, 145.7210083
4: -81.2107391, 62.5805740, -90.4054108, 69.6208878, -150.8316345, 152.9859924
5: -72.8304443, 56.6832848, -81.1150131, 63.0856400, -135.9160461, 137.7982635
6: -69.8919601, 67.3421249, -77.7940140, 74.9642715, -144.8562317, 145.1360931
7: -76.5410614, 64.6338577, -85.1549759, 71.8308487, -148.3719177, 149.7888336
8: -92.0325394, 62.2128220, -102.4561462, 69.3341217, -161.3666687, 164.6689758
9: -69.8776245, 68.1081085, -77.6683121, 75.8089981, -145.6866150, 145.7764130

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
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
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0558562, upper bound: 174.0573222
time: 8.52 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0558562, upper bound: 174.0613456
time: 8.10 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -66.1182098, 52.4538612, -69.1525192, 54.8498573, -120.9680634, 121.6063766
1: -55.3313484, 46.5363007, -57.8138161, 48.5768089, -103.9081573, 104.3500977
2: -72.8861008, 47.8570900, -76.2224579, 49.9639053, -122.8500061, 124.0795441
3: -77.3204422, 40.6280098, -80.8126831, 42.4846153, -119.8050461, 121.4406815
4: -70.7086105, 54.5064659, -73.7881546, 56.8573456, -127.5659561, 128.2946167
5: -63.3368149, 49.3439980, -66.1726685, 51.3616219, -114.6984329, 115.5166626
6: -60.8860092, 58.5924721, -63.6022835, 61.3005447, -122.1865540, 122.1947556
7: -66.6841660, 56.3916893, -69.5856781, 58.8469467, -125.5311127, 125.9773636
8: -80.0947647, 54.0385170, -83.8028336, 56.5375557, -136.6323242, 137.8413544
9: -60.9560890, 59.3092613, -63.6051788, 61.8523979, -122.8084869, 122.9144135

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
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
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 83
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

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0399316, upper bound: 174.0424712
time: 8.88 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0399316, upper bound: 174.0439896
time: 9.23 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -75.9871292, 60.3190575, -70.2723083, 55.7446861, -131.7318115, 130.5913696
1: -63.6269035, 53.4884529, -58.7552223, 49.3653908, -112.9922943, 112.2436752
2: -83.8097687, 54.8659401, -77.4613647, 50.7596207, -134.5693665, 132.3273010
3: -88.8641281, 46.7527657, -82.1235962, 43.1792374, -132.0433655, 128.8763428
4: -81.2107391, 62.5805740, -74.9784470, 57.7734108, -138.9841309, 137.5590210
5: -72.8304443, 56.6832848, -67.2523651, 52.1947060, -125.0251465, 123.9356537
6: -69.8919601, 67.3421249, -64.6261520, 62.2913437, -132.1832733, 131.9682770
7: -76.5410614, 64.6338577, -70.7014771, 59.7824364, -136.3235016, 135.3353271
8: -92.0325394, 62.2128220, -85.1576920, 57.4635620, -149.4960938, 147.3704987
9: -69.8776245, 68.1081085, -64.6163559, 62.8522110, -132.7298279, 132.7244568

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 83
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
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0413511, upper bound: 174.0452962
time: 8.61 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0413511, upper bound: 174.0486163
time: 9.52 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -53.4962120, 42.3403587, -70.9380722, 56.2994804, -109.7956924, 113.2784271
1: -44.6419525, 37.5796928, -59.3851089, 49.9576721, -94.5996094, 96.9647827
2: -58.9045525, 38.9118729, -78.2193680, 51.2897835, -110.1943283, 117.1312408
3: -62.3951569, 32.7622299, -83.0093689, 43.5963745, -105.9915314, 115.7715912
4: -57.1169357, 44.0458641, -75.8997726, 58.4838295, -115.6007690, 119.9456329
5: -51.0467224, 39.7939148, -68.0064087, 52.9659538, -104.0126648, 107.8003159
6: -49.2543106, 47.4556465, -65.3319092, 62.8579597, -112.1122742, 112.7875519
7: -54.0007973, 45.7917099, -71.5375214, 60.4494019, -114.4501953, 117.3292313
8: -64.8231049, 43.6149826, -85.9356995, 58.0110741, -122.8341751, 129.5506744
9: -49.4842415, 47.8767395, -65.3502045, 63.6649780, -113.1492157, 113.2269287

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 62
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
type: B, layer: 1, pos: 233
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
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0354872, upper bound: 174.0354872
time: 8.04 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0354872, upper bound: 174.0354872
time: 10.78 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -61.8228836, 48.9918251, -70.9380722, 56.2994804, -118.1223602, 119.9299011
1: -51.6522598, 43.4095039, -59.3851089, 49.9576721, -101.6099243, 102.7945938
2: -68.1197586, 44.7963371, -78.2193680, 51.2897835, -119.4095306, 123.0157013
3: -72.1893768, 37.9530525, -83.0093689, 43.5963745, -115.7857513, 120.9624176
4: -65.9723282, 50.8582993, -75.8997726, 58.4838295, -124.4561615, 126.7580719
5: -59.0778275, 45.9164162, -68.0064087, 52.9659538, -112.0437622, 113.9228210
6: -56.8723488, 54.8196983, -65.3319092, 62.8579597, -119.7303085, 120.1516037
7: -62.2940903, 52.7377510, -71.5375214, 60.4494019, -122.7434921, 124.2752609
8: -74.9276886, 50.4877396, -85.9356995, 58.0110741, -132.9387512, 136.4234314
9: -56.9941368, 55.2847824, -65.3502045, 63.6649780, -120.6591187, 120.6349869

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 62
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
type: B, layer: 1, pos: 233
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0354872, upper bound: 174.0383716
time: 7.45 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0354872, upper bound: 174.0354872
time: 8.30 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -53.4962120, 42.3403587, -80.6673355, 64.0536957, -117.5499115, 123.0076904
1: -44.6419525, 37.5796928, -67.5645676, 56.8123932, -101.4543304, 105.1442490
2: -58.9045525, 38.9118729, -88.9929810, 58.2102547, -117.1148071, 127.9048386
3: -62.3951569, 32.7622299, -94.3946915, 49.6388741, -112.0340271, 127.1569138
4: -57.1169357, 44.0458641, -86.2586288, 66.4405670, -123.5575027, 130.3044891
5: -51.0467224, 39.7939148, -77.3576355, 60.2052116, -111.2519226, 117.1515503
6: -49.2543106, 47.4556465, -74.2170334, 71.4858551, -120.7401657, 121.6726837
7: -54.0007973, 45.7917099, -81.2594986, 68.5804672, -122.5812683, 127.0512085
8: -64.8231049, 43.6149826, -97.6990891, 66.0721283, -130.8952332, 141.3140717
9: -49.4842415, 47.8767395, -74.1527634, 72.3383713, -121.8226013, 122.0294952

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0354872, upper bound: 174.0370024
time: 6.74 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0354872, upper bound: 174.0370024
time: 7.20 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -61.8228836, 48.9918251, -80.6673355, 64.0536957, -125.8765717, 129.6591644
1: -51.6522598, 43.4095039, -67.5645676, 56.8123932, -108.4646454, 110.9740677
2: -68.1197586, 44.7963371, -88.9929810, 58.2102547, -126.3300171, 133.7893219
3: -72.1893768, 37.9530525, -94.3946915, 49.6388741, -121.8282471, 132.3477325
4: -65.9723282, 50.8582993, -86.2586288, 66.4405670, -132.4128723, 137.1169128
5: -59.0778275, 45.9164162, -77.3576355, 60.2052116, -119.2830200, 123.2740479
6: -56.8723488, 54.8196983, -74.2170334, 71.4858551, -128.3582001, 129.0367279
7: -62.2940903, 52.7377510, -81.2594986, 68.5804672, -130.8745575, 133.9972534
8: -74.9276886, 50.4877396, -97.6990891, 66.0721283, -140.9998169, 148.1868286
9: -56.9941368, 55.2847824, -74.1527634, 72.3383713, -129.3325043, 129.4375153

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0354872, upper bound: 174.0354872
time: 7.16 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0354872, upper bound: 174.0417197
time: 7.04 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 15.59 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.59
Output dim: 7, lower bound: -174.0543145, upper bound: 174.0543145
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.59
Output dim: 7, lower bound: -174.0543145, upper bound: 174.0558562
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.59
Output dim: 7, lower bound: -174.0558562, upper bound: 174.0573222
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.59
Output dim: 7, lower bound: -174.0558562, upper bound: 174.0613456
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.59
Output dim: 7, lower bound: -174.0399316, upper bound: 174.0424712
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.59
Output dim: 7, lower bound: -174.0399316, upper bound: 174.0439896
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.59
Output dim: 7, lower bound: -174.0413511, upper bound: 174.0452962
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.59
Output dim: 7, lower bound: -174.0413511, upper bound: 174.0486163
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.59
Output dim: 7, lower bound: -174.0354872, upper bound: 174.0354872
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.59
Output dim: 7, lower bound: -174.0354872, upper bound: 174.0354872
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.59
Output dim: 7, lower bound: -174.0354872, upper bound: 174.0383716
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.59
Output dim: 7, lower bound: -174.0354872, upper bound: 174.0354872
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.59
Output dim: 7, lower bound: -174.0354872, upper bound: 174.0370024
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.59
Output dim: 7, lower bound: -174.0354872, upper bound: 174.0370024
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.59
Output dim: 7, lower bound: -174.0354872, upper bound: 174.0354872
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.59
Output dim: 7, lower bound: -174.0354872, upper bound: 174.0417197

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -66.1182098, 52.4538612, -66.1182098, 52.4538612, -118.5720673, 118.5720673
1: -55.3313484, 46.5363007, -55.3313484, 46.5363007, -101.8676453, 101.8676453
2: -72.8861008, 47.8570900, -72.8861008, 47.8570900, -120.7431946, 120.7431946
3: -77.3204422, 40.6280098, -77.3204422, 40.6280098, -117.9484406, 117.9484406
4: -70.7086105, 54.5064659, -70.7086105, 54.5064659, -125.2150650, 125.2150650
5: -63.3368149, 49.3439980, -63.3368149, 49.3439980, -112.6808090, 112.6808090
6: -60.8860092, 58.5924721, -60.8860092, 58.5924721, -119.4784851, 119.4784851
7: -66.6841660, 56.3916893, -66.6841660, 56.3916893, -123.0758514, 123.0758514
8: -80.0947647, 54.0385170, -80.0947647, 54.0385170, -134.1332855, 134.1332855
9: -60.9560890, 59.3092613, -60.9560890, 59.3092613, -120.2653503, 120.2653503

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 62
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
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0536404, upper bound: 174.0537722
time: 7.17 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0533886, upper bound: 174.0533886
time: 7.35 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -66.1182098, 52.4538612, -75.9871292, 60.3190575, -126.4372559, 128.4409790
1: -55.3313484, 46.5363007, -63.6269035, 53.4884529, -108.8198013, 110.1632004
2: -72.8861008, 47.8570900, -83.8097687, 54.8659401, -127.7520370, 131.6668549
3: -77.3204422, 40.6280098, -88.8641281, 46.7527657, -124.0731964, 129.4921265
4: -70.7086105, 54.5064659, -81.2107391, 62.5805740, -133.2891846, 135.7172089
5: -63.3368149, 49.3439980, -72.8304443, 56.6832848, -120.0200958, 122.1744308
6: -60.8860092, 58.5924721, -69.8919601, 67.3421249, -128.2281342, 128.4844360
7: -66.6841660, 56.3916893, -76.5410614, 64.6338577, -131.3180237, 132.9327393
8: -80.0947647, 54.0385170, -92.0325394, 62.2128220, -142.3075867, 146.0710602
9: -60.9560890, 59.3092613, -69.8776245, 68.1081085, -129.0641937, 129.1868896

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 62
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
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0536404, upper bound: 174.0553604
time: 7.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0533886, upper bound: 174.0549326
time: 7.80 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -75.9871292, 60.3190575, -66.1182098, 52.4538612, -128.4409790, 126.4372559
1: -63.6269035, 53.4884529, -55.3313484, 46.5363007, -110.1632004, 108.8198013
2: -83.8097687, 54.8659401, -72.8861008, 47.8570900, -131.6668549, 127.7520370
3: -88.8641281, 46.7527657, -77.3204422, 40.6280098, -129.4921265, 124.0731964
4: -81.2107391, 62.5805740, -70.7086105, 54.5064659, -135.7172089, 133.2891693
5: -72.8304443, 56.6832848, -63.3368149, 49.3439980, -122.1744308, 120.0200958
6: -69.8919601, 67.3421249, -60.8860092, 58.5924721, -128.4844360, 128.2281342
7: -76.5410614, 64.6338577, -66.6841660, 56.3916893, -132.9327393, 131.3180237
8: -92.0325394, 62.2128220, -80.0947647, 54.0385170, -146.0710602, 142.3075867
9: -69.8776245, 68.1081085, -60.9560890, 59.3092613, -129.1868896, 129.0641937

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0551531, upper bound: 174.0566952
time: 7.69 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0549326, upper bound: 174.0564695
time: 8.34 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -75.9871292, 60.3190575, -75.9871292, 60.3190575, -136.3061829, 136.3061829
1: -63.6269035, 53.4884529, -63.6269035, 53.4884529, -117.1153564, 117.1153564
2: -83.8097687, 54.8659401, -83.8097687, 54.8659401, -138.6757050, 138.6757050
3: -88.8641281, 46.7527657, -88.8641281, 46.7527657, -135.6168976, 135.6168976
4: -81.2107391, 62.5805740, -81.2107391, 62.5805740, -143.7913208, 143.7913055
5: -72.8304443, 56.6832848, -72.8304443, 56.6832848, -129.5137024, 129.5137024
6: -69.8919601, 67.3421249, -69.8919601, 67.3421249, -137.2340851, 137.2340851
7: -76.5410614, 64.6338577, -76.5410614, 64.6338577, -141.1749268, 141.1749268
8: -92.0325394, 62.2128220, -92.0325394, 62.2128220, -154.2453461, 154.2453461
9: -69.8776245, 68.1081085, -69.8776245, 68.1081085, -137.9857330, 137.9857330

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0551531, upper bound: 174.0566952
time: 8.77 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0549326, upper bound: 174.0607148
time: 7.73 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -66.1182098, 52.4538612, -53.4962120, 42.3403587, -108.4585724, 105.9500732
1: -55.3313484, 46.5363007, -44.6419525, 37.5796928, -92.9110336, 91.1782379
2: -72.8861008, 47.8570900, -58.9045525, 38.9118729, -111.7979736, 106.7616425
3: -77.3204422, 40.6280098, -62.3951569, 32.7622299, -110.0826645, 103.0231628
4: -70.7086105, 54.5064659, -57.1169357, 44.0458641, -114.7544708, 111.6233902
5: -63.3368149, 49.3439980, -51.0467224, 39.7939148, -103.1307220, 100.3907089
6: -60.8860092, 58.5924721, -49.2543106, 47.4556465, -108.3416595, 107.8467865
7: -66.6841660, 56.3916893, -54.0007973, 45.7917099, -112.4758759, 110.3924866
8: -80.0947647, 54.0385170, -64.8231049, 43.6149826, -123.7097473, 118.8616104
9: -60.9560890, 59.3092613, -49.4842415, 47.8767395, -108.8328247, 108.7934875

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 62
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
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0392748, upper bound: 174.0419950
time: 9.60 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0391242, upper bound: 174.0417165
time: 9.93 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -66.1182098, 52.4538612, -61.8228836, 48.9918251, -115.1100311, 114.2767334
1: -55.3313484, 46.5363007, -51.6522598, 43.4095039, -98.7408524, 98.1885529
2: -72.8861008, 47.8570900, -68.1197586, 44.7963371, -117.6824341, 115.9768524
3: -77.3204422, 40.6280098, -72.1893768, 37.9530525, -115.2734985, 112.8173828
4: -70.7086105, 54.5064659, -65.9723282, 50.8582993, -121.5669022, 120.4787750
5: -63.3368149, 49.3439980, -59.0778275, 45.9164162, -109.2532349, 108.4218063
6: -60.8860092, 58.5924721, -56.8723488, 54.8196983, -115.7057037, 115.4648209
7: -66.6841660, 56.3916893, -62.2940903, 52.7377510, -119.4218979, 118.6857681
8: -80.0947647, 54.0385170, -74.9276886, 50.4877396, -130.5825043, 128.9662018
9: -60.9560890, 59.3092613, -56.9941368, 55.2847824, -116.2408752, 116.3033905

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 62
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
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0392748, upper bound: 174.0435911
time: 8.41 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0391242, upper bound: 174.0433251
time: 8.79 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -75.9871292, 60.3190575, -53.4962120, 42.3403587, -118.3274841, 113.8152695
1: -63.6269035, 53.4884529, -44.6419525, 37.5796928, -101.2065964, 98.1303940
2: -83.8097687, 54.8659401, -58.9045525, 38.9118729, -122.7216415, 113.7704926
3: -88.8641281, 46.7527657, -62.3951569, 32.7622299, -121.6263504, 109.1479187
4: -81.2107391, 62.5805740, -57.1169357, 44.0458641, -125.2565994, 119.6975021
5: -72.8304443, 56.6832848, -51.0467224, 39.7939148, -112.6243591, 107.7300034
6: -69.8919601, 67.3421249, -49.2543106, 47.4556465, -117.3476105, 116.5964355
7: -76.5410614, 64.6338577, -54.0007973, 45.7917099, -122.3327713, 118.6346588
8: -92.0325394, 62.2128220, -64.8231049, 43.6149826, -135.6475220, 127.0359192
9: -69.8776245, 68.1081085, -49.4842415, 47.8767395, -117.7543488, 117.5923462

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0407120, upper bound: 174.0447991
time: 8.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0405561, upper bound: 174.0446123
time: 8.94 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -75.9871292, 60.3190575, -61.8228836, 48.9918251, -124.9789581, 122.1419296
1: -63.6269035, 53.4884529, -51.6522598, 43.4095039, -107.0364075, 105.1407089
2: -83.8097687, 54.8659401, -68.1197586, 44.7963371, -128.6060944, 122.9857025
3: -88.8641281, 46.7527657, -72.1893768, 37.9530525, -126.8171844, 118.9421387
4: -81.2107391, 62.5805740, -65.9723282, 50.8582993, -132.0690308, 128.5529022
5: -72.8304443, 56.6832848, -59.0778275, 45.9164162, -118.7468567, 115.7611084
6: -69.8919601, 67.3421249, -56.8723488, 54.8196983, -124.7116547, 124.2144699
7: -76.5410614, 64.6338577, -62.2940903, 52.7377510, -129.2788086, 126.9279327
8: -92.0325394, 62.2128220, -74.9276886, 50.4877396, -142.5202789, 137.1404877
9: -69.8776245, 68.1081085, -56.9941368, 55.2847824, -125.1624069, 125.1022491

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0407120, upper bound: 174.0483140
time: 8.65 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0405561, upper bound: 174.0481633
time: 8.62 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -53.4962120, 42.3403587, -66.1548691, 52.4823914, -105.9786072, 108.4952240
1: -44.6419525, 37.5796928, -55.3616371, 46.5620232, -91.2039719, 92.9413223
2: -58.9045525, 38.9118729, -72.9268265, 47.8825073, -106.7870636, 111.8386993
3: -62.3951569, 32.7622299, -77.3646622, 40.6498680, -103.0450287, 110.1268845
4: -57.1169357, 44.0458641, -70.7479553, 54.5364494, -111.6533813, 114.7938232
5: -51.0467224, 39.7939148, -63.3726463, 49.3708725, -100.4175949, 103.1665497
6: -49.2543106, 47.4556465, -60.9196625, 58.6244926, -107.8787994, 108.3753052
7: -54.0007973, 45.7917099, -66.7205276, 56.4222107, -110.4230042, 112.5122375
8: -64.8231049, 43.6149826, -80.1385117, 54.0677299, -118.8908386, 123.7534943
9: -49.4842415, 47.8767395, -60.9894028, 59.3414078, -108.8256454, 108.8661423

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0348176, upper bound: 174.0348893
time: 8.48 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0346652, upper bound: 174.0346652
time: 7.64 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -53.4962120, 42.3403587, -53.2607117, 42.1531525, -95.6493683, 95.6010742
1: -44.6419525, 37.5796928, -44.4493484, 37.4125938, -82.0545502, 82.0290298
2: -58.9045525, 38.9118729, -58.6422691, 38.7486496, -97.6531982, 97.5541382
3: -62.3951569, 32.7622299, -62.1095695, 32.6233826, -95.0185394, 94.8717957
4: -57.1169357, 44.0458641, -56.8638535, 43.8556671, -100.9725876, 100.9097137
5: -51.0467224, 39.7939148, -50.8190613, 39.6271896, -90.6738968, 90.6129608
6: -49.2543106, 47.4556465, -49.0342293, 47.2447662, -96.4990768, 96.4898758
7: -54.0007973, 45.7917099, -53.7679062, 45.6008492, -99.6016464, 99.5596161
8: -64.8231049, 43.6149826, -64.5376892, 43.4286232, -108.2517166, 108.1526718
9: -49.4842415, 47.8767395, -49.2784920, 47.6603165, -97.1445618, 97.1552200

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0348176, upper bound: 174.0348893
time: 6.78 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0346652, upper bound: 174.0346652
time: 7.96 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -61.8228836, 48.9918251, -66.1548691, 52.4823914, -114.3052597, 115.1466980
1: -51.6522598, 43.4095039, -55.3616371, 46.5620232, -98.2142792, 98.7711334
2: -68.1197586, 44.7963371, -72.9268265, 47.8825073, -116.0022659, 117.7231598
3: -72.1893768, 37.9530525, -77.3646622, 40.6498680, -112.8392487, 115.3177185
4: -65.9723282, 50.8582993, -70.7479553, 54.5364494, -120.5087738, 121.6062469
5: -59.0778275, 45.9164162, -63.3726463, 49.3708725, -108.4487000, 109.2890549
6: -56.8723488, 54.8196983, -60.9196625, 58.6244926, -115.4968414, 115.7393570
7: -62.2940903, 52.7377510, -66.7205276, 56.4222107, -118.7162933, 119.4582748
8: -74.9276886, 50.4877396, -80.1385117, 54.0677299, -128.9954224, 130.6262512
9: -56.9941368, 55.2847824, -60.9894028, 59.3414078, -116.3355408, 116.2741852

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
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
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
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

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0364087, upper bound: 174.0379234
time: 8.14 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0362601, upper bound: 174.0378164
time: 9.28 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -61.8228836, 48.9918251, -53.2607117, 42.1531525, -103.9760361, 102.2525330
1: -51.6522598, 43.4095039, -44.4493484, 37.4125938, -89.0648499, 87.8588486
2: -68.1197586, 44.7963371, -58.6422691, 38.7486496, -106.8684082, 103.4386063
3: -72.1893768, 37.9530525, -62.1095695, 32.6233826, -104.8127594, 100.0626221
4: -65.9723282, 50.8582993, -56.8638535, 43.8556671, -109.8279800, 107.7221527
5: -59.0778275, 45.9164162, -50.8190613, 39.6271896, -98.7049942, 96.7354660
6: -56.8723488, 54.8196983, -49.0342293, 47.2447662, -104.1171112, 103.8539276
7: -62.2940903, 52.7377510, -53.7679062, 45.6008492, -107.8949203, 106.5056610
8: -74.9276886, 50.4877396, -64.5376892, 43.4286232, -118.3563004, 115.0254211
9: -56.9941368, 55.2847824, -49.2784920, 47.6603165, -104.6544495, 104.5632782

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
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
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0364087, upper bound: 174.0379234
time: 7.27 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0362601, upper bound: 174.0378164
time: 7.87 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -53.4962120, 42.3403587, -75.9871292, 60.3190575, -113.8152695, 118.3274841
1: -44.6419525, 37.5796928, -63.6269035, 53.4884529, -98.1303940, 101.2065964
2: -58.9045525, 38.9118729, -83.8097687, 54.8659401, -113.7704926, 122.7216415
3: -62.3951569, 32.7622299, -88.8641281, 46.7527657, -109.1479187, 121.6263504
4: -57.1169357, 44.0458641, -81.2107391, 62.5805740, -119.6975021, 125.2565994
5: -51.0467224, 39.7939148, -72.8304443, 56.6832848, -107.7300034, 112.6243591
6: -49.2543106, 47.4556465, -69.8919601, 67.3421249, -116.5964355, 117.3476105
7: -54.0007973, 45.7917099, -76.5410614, 64.6338577, -118.6346588, 122.3327713
8: -64.8231049, 43.6149826, -92.0325394, 62.2128220, -127.0359192, 135.6475220
9: -49.4842415, 47.8767395, -69.8776245, 68.1081085, -117.5923462, 117.7543488

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0380790, upper bound: 174.0365558
time: 7.52 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0378164, upper bound: 174.0362601
time: 7.31 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -53.4962120, 42.3403587, -61.7327347, 48.9164810, -102.4126892, 104.0730896
1: -44.6419525, 37.5796928, -51.5724068, 43.3438377, -87.9857712, 89.1520996
2: -58.9045525, 38.9118729, -68.0111237, 44.7297974, -103.6343460, 106.9229889
3: -62.3951569, 32.7622299, -72.0813065, 37.8965836, -100.2917252, 104.8435287
4: -57.1169357, 44.0458641, -65.8746262, 50.7817345, -107.8986664, 109.9204865
5: -51.0467224, 39.7939148, -58.9901314, 45.8503914, -96.8971100, 98.7840271
6: -49.2543106, 47.4556465, -56.7882843, 54.7342072, -103.9885178, 104.2439194
7: -54.0007973, 45.7917099, -62.2018242, 52.6605988, -106.6613922, 107.9935303
8: -64.8231049, 43.6149826, -74.8051071, 50.4101486, -115.2332535, 118.4200897
9: -49.4842415, 47.8767395, -56.9114113, 55.1955643, -104.6798096, 104.7881317

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0380790, upper bound: 174.0365558
time: 7.92 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0378164, upper bound: 174.0362601
time: 8.08 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -61.8228836, 48.9918251, -75.9871292, 60.3190575, -122.1419296, 124.9789581
1: -51.6522598, 43.4095039, -63.6269035, 53.4884529, -105.1407089, 107.0364075
2: -68.1197586, 44.7963371, -83.8097687, 54.8659401, -122.9857025, 128.6060944
3: -72.1893768, 37.9530525, -88.8641281, 46.7527657, -118.9421387, 126.8171844
4: -65.9723282, 50.8582993, -81.2107391, 62.5805740, -128.5529022, 132.0690308
5: -59.0778275, 45.9164162, -72.8304443, 56.6832848, -115.7611008, 118.7468567
6: -56.8723488, 54.8196983, -69.8919601, 67.3421249, -124.2144699, 124.7116547
7: -62.2940903, 52.7377510, -76.5410614, 64.6338577, -126.9279327, 129.2788086
8: -74.9276886, 50.4877396, -92.0325394, 62.2128220, -137.1404877, 142.5202789
9: -56.9941368, 55.2847824, -69.8776245, 68.1081085, -125.1022491, 125.1624069

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
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
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
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

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0414978, upper bound: 174.0414454
time: 7.03 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0413627, upper bound: 174.0413678
time: 8.07 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -61.8228836, 48.9918251, -61.7327347, 48.9164810, -110.7393570, 110.7245636
1: -51.6522598, 43.4095039, -51.5724068, 43.3438377, -94.9960861, 94.9819107
2: -68.1197586, 44.7963371, -68.0111237, 44.7297974, -112.8495483, 112.8074493
3: -72.1893768, 37.9530525, -72.0813065, 37.8965836, -110.0859451, 110.0343628
4: -65.9723282, 50.8582993, -65.8746262, 50.7817345, -116.7540588, 116.7329178
5: -59.0778275, 45.9164162, -58.9901314, 45.8503914, -104.9282074, 104.9065399
6: -56.8723488, 54.8196983, -56.7882843, 54.7342072, -111.6065521, 111.6079559
7: -62.2940903, 52.7377510, -62.2018242, 52.6605988, -114.9546738, 114.9395599
8: -74.9276886, 50.4877396, -74.8051071, 50.4101486, -125.3378372, 125.2928391
9: -56.9941368, 55.2847824, -56.9114113, 55.1955643, -112.1896973, 112.1961975

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
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
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0414978, upper bound: 174.0414454
time: 6.76 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0413627, upper bound: 174.0413679
time: 6.46 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 14.67 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 7, lower bound: -174.0536404, upper bound: 174.0537722
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 7, lower bound: -174.0533886, upper bound: 174.0533886
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 7, lower bound: -174.0536404, upper bound: 174.0553604
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 7, lower bound: -174.0533886, upper bound: 174.0549326
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 7, lower bound: -174.0551531, upper bound: 174.0566952
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 7, lower bound: -174.0549326, upper bound: 174.0564695
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 7, lower bound: -174.0551531, upper bound: 174.0566952
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 7, lower bound: -174.0549326, upper bound: 174.0607148
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 7, lower bound: -174.0392748, upper bound: 174.0419950
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 7, lower bound: -174.0391242, upper bound: 174.0417165
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 7, lower bound: -174.0392748, upper bound: 174.0435911
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 7, lower bound: -174.0391242, upper bound: 174.0433251
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 7, lower bound: -174.0407120, upper bound: 174.0447991
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 7, lower bound: -174.0405561, upper bound: 174.0446123
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 7, lower bound: -174.0407120, upper bound: 174.0483140
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 7, lower bound: -174.0405561, upper bound: 174.0481633
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 7, lower bound: -174.0348176, upper bound: 174.0348893
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 7, lower bound: -174.0346652, upper bound: 174.0346652
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 7, lower bound: -174.0348176, upper bound: 174.0348893
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 7, lower bound: -174.0346652, upper bound: 174.0346652
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 7, lower bound: -174.0364087, upper bound: 174.0379234
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 7, lower bound: -174.0362601, upper bound: 174.0378164
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 7, lower bound: -174.0364087, upper bound: 174.0379234
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 7, lower bound: -174.0362601, upper bound: 174.0378164
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 7, lower bound: -174.0380790, upper bound: 174.0365558
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 7, lower bound: -174.0378164, upper bound: 174.0362601
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 7, lower bound: -174.0380790, upper bound: 174.0365558
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 7, lower bound: -174.0378164, upper bound: 174.0362601
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 7, lower bound: -174.0414978, upper bound: 174.0414454
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 7, lower bound: -174.0413627, upper bound: 174.0413678
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 7, lower bound: -174.0414978, upper bound: 174.0414454
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 7, lower bound: -174.0413627, upper bound: 174.0413679

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -62.8906555, 49.9093628, -66.1182098, 52.4538612, -115.3445129, 116.0275574
1: -52.6252098, 44.2591934, -55.3313484, 46.5363007, -99.1615143, 99.5905380
2: -69.3224258, 45.5546608, -72.8861008, 47.8570900, -117.1795197, 118.4407578
3: -73.5198212, 38.6409035, -77.3204422, 40.6280098, -114.1478195, 115.9613495
4: -67.2444305, 51.8523521, -70.7086105, 54.5064659, -121.7508850, 122.5609589
5: -60.2516022, 46.9617386, -63.3368149, 49.3439980, -109.5955963, 110.2985458
6: -57.9241104, 55.7373543, -60.8860092, 58.5924721, -116.5165863, 116.6233597
7: -63.4366455, 53.6903343, -66.6841660, 56.3916893, -119.8283310, 120.3744965
8: -76.1832428, 51.3811111, -80.0947647, 54.0385170, -130.2217560, 131.4758759
9: -58.0256653, 56.4291534, -60.9560890, 59.3092613, -117.3349152, 117.3852386

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0533886, upper bound: 174.0533886
time: 6.95 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0533886, upper bound: 174.0533886
time: 7.33 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -77.4078217, 61.4288826, -65.7090454, 52.1309853, -129.5388031, 127.1379242
1: -64.6930542, 54.4171600, -54.9879761, 46.2476387, -110.9406815, 109.4051208
2: -85.2704010, 55.8115158, -72.4346085, 47.5657768, -132.8361511, 128.2461090
3: -90.5000992, 47.3791809, -76.8380966, 40.3759651, -130.8760376, 124.2172623
4: -82.8633194, 63.7354698, -70.2683334, 54.1694870, -137.0328064, 134.0037994
5: -74.2422562, 57.7419167, -62.9448738, 49.0416946, -123.2839355, 120.6867752
6: -71.2608719, 68.4845047, -60.5098457, 58.2301521, -129.4910126, 128.9943237
7: -77.9144974, 65.7600708, -66.2724380, 56.0493851, -133.9638824, 132.0325012
8: -93.6158142, 63.2917824, -79.5991211, 53.7008209, -147.3166046, 142.8908997
9: -71.1653595, 69.4030685, -60.5840683, 58.9438744, -130.1092377, 129.9871368

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0524383, upper bound: 174.0525496
time: 8.05 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0533094, upper bound: 174.0533094
time: 7.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -62.8906555, 49.9093628, -75.9871292, 60.3190575, -123.2097092, 125.8964844
1: -52.6252098, 44.2591934, -63.6269035, 53.4884529, -106.1136627, 107.8860931
2: -69.3224258, 45.5546608, -83.8097687, 54.8659401, -124.1883698, 129.3644257
3: -73.5198212, 38.6409035, -88.8641281, 46.7527657, -120.2725754, 127.5050354
4: -67.2444305, 51.8523521, -81.2107391, 62.5805740, -129.8250122, 133.0630798
5: -60.2516022, 46.9617386, -72.8304443, 56.6832848, -116.9348831, 119.7921753
6: -57.9241104, 55.7373543, -69.8919601, 67.3421249, -125.2662201, 125.6293182
7: -63.4366455, 53.6903343, -76.5410614, 64.6338577, -128.0704956, 130.2313995
8: -76.1832428, 51.3811111, -92.0325394, 62.2128220, -138.3960571, 143.4136505
9: -58.0256653, 56.4291534, -69.8776245, 68.1081085, -126.1337738, 126.3067780

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 8
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
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 215
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
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0564695, upper bound: 174.0549326
time: 7.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0564695, upper bound: 174.0549326
time: 7.99 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -77.4078217, 61.4288826, -75.5640411, 59.9846573, -137.3924866, 136.9929047
1: -64.6930542, 54.4171600, -63.2716331, 53.1898117, -117.8828659, 117.6887817
2: -85.2704010, 55.8115158, -83.3424835, 54.5633774, -139.8337555, 139.1539917
3: -90.5000992, 47.3791809, -88.3658371, 46.4921684, -136.9922485, 135.7449799
4: -82.8633194, 63.7354698, -80.7552185, 62.2310677, -145.0943909, 144.4906921
5: -74.2422562, 57.7419167, -72.4252319, 56.3697777, -130.6120300, 130.1671448
6: -71.2608719, 68.4845047, -69.5021591, 66.9675217, -138.2283936, 137.9866333
7: -77.9144974, 65.7600708, -76.1144943, 64.2788925, -142.1933746, 141.8745728
8: -93.6158142, 63.2917824, -91.5200195, 61.8629532, -155.4787445, 154.8117981
9: -71.1653595, 69.4030685, -69.4913635, 67.7291107, -138.8944550, 138.8944244

Time for backsubstitution: 1.34 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=175.32177734375
rel_dist={7: [-174.07406456819004, 174.07406456764704]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

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

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0506930, upper bound: 174.0521303
time: 8.89 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0484527, upper bound: 174.0484527
time: 7.37 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 16.41 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 16.41
Output dim: 7, lower bound: -174.0506930, upper bound: 174.0521303
IS_A2, status: Status.UNKNOWN, split count: 1, time: 16.41
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

Time for backsubstitution: 1.22 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0484529, upper bound: 174.0484529
time: 7.76 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0484529, upper bound: 174.0484527
time: 7.04 seconds

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

Time for backsubstitution: 1.22 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0361590, upper bound: 174.0369696
time: 8.12 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0413178, upper bound: 174.0413178
time: 8.49 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 17.97 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 17.97
Output dim: 7, lower bound: -174.0484529, upper bound: 174.0484529
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 17.97
Output dim: 7, lower bound: -174.0484529, upper bound: 174.0484527
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 17.97
Output dim: 7, lower bound: -174.0361590, upper bound: 174.0369696
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 17.97
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

Time for backsubstitution: 1.23 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0396017, upper bound: 174.0400114
time: 9.08 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0435904, upper bound: 174.0450494
time: 8.37 seconds

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

Time for backsubstitution: 1.25 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0396017, upper bound: 174.0400114
time: 10.21 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0435904, upper bound: 174.0450494
time: 7.80 seconds

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

Time for backsubstitution: 1.26 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0354981, upper bound: 174.0364388
time: 9.53 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0354100, upper bound: 174.0363557
time: 8.23 seconds

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

Time for backsubstitution: 1.25 seconds

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
time: 7.08 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0409093, upper bound: 174.0409093
time: 7.80 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 16.26 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 16.26
Output dim: 7, lower bound: -174.0396017, upper bound: 174.0400114
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 16.26
Output dim: 7, lower bound: -174.0435904, upper bound: 174.0450494
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 16.26
Output dim: 7, lower bound: -174.0396017, upper bound: 174.0400114
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 16.26
Output dim: 7, lower bound: -174.0435904, upper bound: 174.0450494
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 16.26
Output dim: 7, lower bound: -174.0354981, upper bound: 174.0364388
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 16.26
Output dim: 7, lower bound: -174.0354100, upper bound: 174.0363557
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 16.26
Output dim: 7, lower bound: -174.0409906, upper bound: 174.0409571
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 16.26
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

Time for backsubstitution: 1.26 seconds

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
time: 9.08 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0548613, upper bound: 174.0538548
time: 9.53 seconds

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

Time for backsubstitution: 1.26 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0603609, upper bound: 174.0603619
time: 7.67 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0602538, upper bound: 174.0602538
time: 8.35 seconds

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

Time for backsubstitution: 1.24 seconds

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
time: 9.72 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0390087, upper bound: 174.0393549
time: 8.53 seconds

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

Time for backsubstitution: 1.25 seconds

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
time: 8.59 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0431436, upper bound: 174.0445648
time: 8.53 seconds

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

Time for backsubstitution: 1.25 seconds

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
time: 7.65 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0354981, upper bound: 174.0364388
time: 9.43 seconds

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

Time for backsubstitution: 1.29 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0346416, upper bound: 174.0356618
time: 7.70 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0354100, upper bound: 174.0363557
time: 8.39 seconds

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

Time for backsubstitution: 1.29 seconds

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
time: 7.14 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0409906, upper bound: 174.0409571
time: 7.05 seconds

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

Time for backsubstitution: 1.36 seconds

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
time: 7.38 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0409093, upper bound: 174.0409093
time: 7.56 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 16.46 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 7, lower bound: -174.0549819, upper bound: 174.0539616
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 7, lower bound: -174.0548613, upper bound: 174.0538548
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 7, lower bound: -174.0603609, upper bound: 174.0603619
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 7, lower bound: -174.0602538, upper bound: 174.0602538
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 7, lower bound: -174.0390840, upper bound: 174.0393982
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 7, lower bound: -174.0390087, upper bound: 174.0393549
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 7, lower bound: -174.0432084, upper bound: 174.0446391
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 7, lower bound: -174.0431436, upper bound: 174.0445648
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 7, lower bound: -174.0354981, upper bound: 174.0364388
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 7, lower bound: -174.0354981, upper bound: 174.0364388
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 7, lower bound: -174.0346416, upper bound: 174.0356618
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 7, lower bound: -174.0354100, upper bound: 174.0363557
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 7, lower bound: -174.0409906, upper bound: 174.0409571
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 7, lower bound: -174.0409906, upper bound: 174.0409571
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 7, lower bound: -174.0409093, upper bound: 174.0409092
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.46
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

Time for backsubstitution: 1.32 seconds

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
time: 9.01 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0549082, upper bound: 174.0539438
time: 8.36 seconds

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

Time for backsubstitution: 1.33 seconds

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

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0538531, upper bound: 174.0528443
time: 9.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0547917, upper bound: 174.0538373
time: 8.38 seconds

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

Time for backsubstitution: 1.35 seconds

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

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0596035, upper bound: 174.0594784
time: 9.64 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0603402, upper bound: 174.0603422
time: 7.39 seconds

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

Time for backsubstitution: 1.31 seconds

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
time: 18.33 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0602240, upper bound: 174.0602240
time: 7.83 seconds

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

Time for backsubstitution: 1.24 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0382944, upper bound: 174.0385074
time: 10.10 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0390743, upper bound: 174.0393982
time: 9.66 seconds

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

Time for backsubstitution: 1.22 seconds

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
time: 10.07 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0390018, upper bound: 174.0393549
time: 7.45 seconds

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

Time for backsubstitution: 1.20 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0423518, upper bound: 174.0436791
time: 9.73 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0432084, upper bound: 174.0446389
time: 8.38 seconds

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

Time for backsubstitution: 1.20 seconds

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

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0422513, upper bound: 174.0436112
time: 8.29 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0431436, upper bound: 174.0445648
time: 8.88 seconds

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

Time for backsubstitution: 1.20 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0346568, upper bound: 174.0355130
time: 8.09 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0354981, upper bound: 174.0364388
time: 9.93 seconds

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

Time for backsubstitution: 1.29 seconds

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

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0346568, upper bound: 174.0355130
time: 8.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0354981, upper bound: 174.0364388
time: 7.73 seconds

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

Time for backsubstitution: 1.29 seconds

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
time: 10.89 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9881364, upper bound: 173.9898888
time: 7.68 seconds

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

Time for backsubstitution: 1.28 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9932000, upper bound: 173.9970932
time: 9.84 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9888689, upper bound: 173.9905585
time: 8.12 seconds

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

Time for backsubstitution: 1.28 seconds

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
time: 8.28 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0409906, upper bound: 174.0409571
time: 8.06 seconds

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

Time for backsubstitution: 1.33 seconds

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
time: 8.01 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0409906, upper bound: 174.0409571
time: 7.19 seconds

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

Time for backsubstitution: 1.29 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0398916, upper bound: 174.0397889
time: 7.99 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0409092, upper bound: 174.0409092
time: 7.59 seconds

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

Time for backsubstitution: 1.37 seconds

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

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0398916, upper bound: 174.0397889
time: 9.14 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0409093, upper bound: 174.0409092
time: 7.69 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 18.35 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0539731, upper bound: 174.0529588
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0549082, upper bound: 174.0539438
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0538531, upper bound: 174.0528443
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0547917, upper bound: 174.0538373
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0596035, upper bound: 174.0594784
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0603402, upper bound: 174.0603422
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0594920, upper bound: 174.0593741
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0602240, upper bound: 174.0602240
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0382944, upper bound: 174.0385074
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0390743, upper bound: 174.0393982
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0381861, upper bound: 174.0384440
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0390018, upper bound: 174.0393549
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0423518, upper bound: 174.0436791
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0432084, upper bound: 174.0446389
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0422513, upper bound: 174.0436112
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0431436, upper bound: 174.0445648
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0346568, upper bound: 174.0355130
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0354981, upper bound: 174.0364388
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0346568, upper bound: 174.0355130
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0354981, upper bound: 174.0364388
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -173.9922458, upper bound: 173.9962000
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -173.9881364, upper bound: 173.9898888
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -173.9932000, upper bound: 173.9970932
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -173.9888689, upper bound: 173.9905585
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0399334, upper bound: 174.0398325
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0409906, upper bound: 174.0409571
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0399334, upper bound: 174.0398325
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0409906, upper bound: 174.0409571
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0398916, upper bound: 174.0397889
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0409092, upper bound: 174.0409092
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.35
Output dim: 7, lower bound: -174.0398916, upper bound: 174.0397889
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.35
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

Time for backsubstitution: 1.21 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0124441, upper bound: 174.0153627
time: 10.16 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0512643, upper bound: 174.0501892
time: 9.72 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 21.23 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 21.23
Output dim: 7, lower bound: -174.0124441, upper bound: 174.0153627
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 21.23
Output dim: 7, lower bound: -174.0512643, upper bound: 174.0501892
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -174.0549082, upper bound: 174.0539438
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -174.0538531, upper bound: 174.0528443
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -174.0547917, upper bound: 174.0538373
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -174.0596035, upper bound: 174.0594784
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -174.0603402, upper bound: 174.0603422
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -174.0594920, upper bound: 174.0593741
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -174.0602240, upper bound: 174.0602240
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -174.0382944, upper bound: 174.0385074
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -174.0390743, upper bound: 174.0393982
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -174.0381861, upper bound: 174.0384440
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -174.0390018, upper bound: 174.0393549
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -174.0423518, upper bound: 174.0436791
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -174.0432084, upper bound: 174.0446389
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -174.0422513, upper bound: 174.0436112
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -174.0431436, upper bound: 174.0445648
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -174.0346568, upper bound: 174.0355130
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -174.0354981, upper bound: 174.0364388
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -174.0346568, upper bound: 174.0355130
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -174.0354981, upper bound: 174.0364388
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -173.9922458, upper bound: 173.9962000
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -173.9881364, upper bound: 173.9898888
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -173.9932000, upper bound: 173.9970932
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -173.9888689, upper bound: 173.9905585
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -174.0399334, upper bound: 174.0398325
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -174.0409906, upper bound: 174.0409571
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -174.0399334, upper bound: 174.0398325
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -174.0409906, upper bound: 174.0409571
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -174.0398916, upper bound: 174.0397889
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -174.0409092, upper bound: 174.0409092
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -174.0398916, upper bound: 174.0397889
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.23
Output dim: 7, lower bound: -174.0409093, upper bound: 174.0409092
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=175.32177734375
rel_dist={7: [-174.07363473064066, 174.07363473064066]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0488676, upper bound: 174.0493953
time: 10.07 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0480801, upper bound: 174.0480803
time: 7.92 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 18.12 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 18.12
Output dim: 7, lower bound: -174.0488676, upper bound: 174.0493953
IS_A2, status: Status.UNKNOWN, split count: 1, time: 18.12
Output dim: 7, lower bound: -174.0480801, upper bound: 174.0480803

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -84.7945175, 67.3363419, -88.9458008, 70.6434402, -155.4379578, 156.2821350
1: -71.0174942, 59.7082291, -74.5171356, 62.6546059, -133.6721039, 134.2253723
2: -93.5649643, 61.1370964, -98.1563110, 64.1046982, -157.6696320, 159.2934113
3: -99.2057114, 52.1909523, -104.1050186, 54.7606354, -153.9663086, 156.2959442
4: -90.6216278, 69.7862701, -95.1004791, 73.2063370, -163.8279572, 164.8867340
5: -81.3096313, 63.2362137, -85.3197632, 66.3617935, -147.6714172, 148.5559387
6: -77.9796448, 75.1436157, -81.8107605, 78.8259964, -156.8056335, 156.9543762
7: -85.3573227, 72.0001450, -89.5378113, 75.5011978, -160.8585205, 161.5379028
8: -102.7011108, 69.5013962, -107.7238235, 72.9237442, -175.6248474, 177.2252197
9: -77.8513641, 75.9898987, -81.6415024, 79.7380676, -157.5894318, 157.6314087

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0361532, upper bound: 174.0369290
time: 10.76 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0417078, upper bound: 174.0422700
time: 10.30 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -70.4749985, 55.9063721, -71.5585098, 56.7855110, -127.2604904, 127.4648819
1: -58.9251480, 49.5083618, -59.8905525, 50.3235168, -109.2486649, 109.3988876
2: -77.6856918, 50.9035301, -78.9031143, 51.6880684, -129.3737335, 129.8066406
3: -82.3612061, 43.3041534, -83.6229477, 44.0402184, -126.4014282, 126.9271011
4: -75.1945114, 57.9392395, -76.3596497, 58.8697662, -134.0642700, 134.2988739
5: -67.4476700, 52.3454247, -68.5346909, 53.2829628, -120.7306290, 120.8801041
6: -64.8120270, 62.4707680, -65.7815933, 63.4061699, -128.2182007, 128.2523651
7: -70.9039001, 59.9517593, -72.0061188, 60.8630142, -131.7669067, 131.9578857
8: -85.4032898, 57.6308556, -86.6969452, 58.5733604, -143.9766388, 144.3277588
9: -64.7994461, 63.0337334, -65.7577133, 64.0333405, -128.8327789, 128.7914429

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0352793, upper bound: 174.0355802
time: 8.86 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0409058, upper bound: 174.0409057
time: 9.49 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 19.62 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 19.62
Output dim: 7, lower bound: -174.0361532, upper bound: 174.0369290
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 19.62
Output dim: 7, lower bound: -174.0417078, upper bound: 174.0422700
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 19.62
Output dim: 7, lower bound: -174.0352793, upper bound: 174.0355802
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 19.62
Output dim: 7, lower bound: -174.0409058, upper bound: 174.0409057

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -71.4930725, 56.7284546, -70.2101822, 55.7173424, -127.2104187, 126.9386368
1: -59.8387146, 50.3236008, -58.7708435, 49.4399529, -109.2786713, 109.0944443
2: -78.8289413, 51.6620789, -77.4148331, 50.7715340, -129.6004639, 129.0769043
3: -83.6210403, 43.9456596, -82.1455688, 43.1464767, -126.7675171, 126.0912323
4: -76.4495697, 58.9059868, -75.1160660, 57.8836555, -134.3332214, 134.0220184
5: -68.5071945, 53.3348007, -67.2971191, 52.4191055, -120.9263000, 120.6319199
6: -65.8062820, 63.3588524, -64.6582031, 62.2164459, -128.0227356, 128.0170593
7: -72.0586548, 60.8805885, -70.8062820, 59.8353806, -131.8940277, 131.6868744
8: -86.5934143, 58.4790802, -85.0512390, 57.4134521, -144.0068512, 143.5303192
9: -65.8230591, 64.1047668, -64.6894455, 63.0065994, -128.8296509, 128.7942200

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 215
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
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0353692, upper bound: 174.0362402
time: 11.97 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0353462, upper bound: 174.0362179
time: 12.07 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -78.2301712, 62.1069832, -80.0708237, 63.5775833, -141.8077240, 142.1778107
1: -65.5077515, 55.0719452, -67.0639801, 56.3873596, -121.8951111, 122.1359177
2: -86.2953339, 56.4627113, -88.3334503, 57.7840805, -144.0793915, 144.7961426
3: -91.4972610, 48.1355095, -93.6856384, 49.2697334, -140.7669983, 141.8211517
4: -83.6065521, 64.4160004, -85.6145859, 65.9485931, -149.5551453, 150.0305786
5: -74.9901886, 58.3515320, -76.7782288, 59.7563324, -134.7465210, 135.1297607
6: -71.9519958, 69.3281784, -73.6627579, 70.9599228, -142.9119110, 142.9909363
7: -78.7872238, 66.5093765, -80.6580505, 68.0756531, -146.8628387, 147.1674194
8: -94.7500992, 64.0698013, -96.9745407, 65.5824585, -160.3325500, 161.0443420
9: -71.9085922, 70.1157608, -73.6089401, 71.7980576, -143.7066498, 143.7246857

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0412941, upper bound: 174.0418554
time: 9.53 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0412626, upper bound: 174.0418268
time: 10.35 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -58.2543602, 46.1427155, -53.7982521, 42.6321983, -100.8865585, 99.9409637
1: -48.6365929, 40.9140930, -44.9711418, 37.8300095, -86.4665909, 85.8852158
2: -64.1710739, 42.2657547, -59.2572479, 39.1189041, -103.2899780, 101.5229950
3: -68.0162201, 35.7151184, -62.8196411, 33.0369377, -101.0531616, 98.5347519
4: -62.2032928, 47.9457245, -57.4570618, 44.3506775, -106.5539703, 105.4027786
5: -55.6461258, 43.2910843, -51.4186859, 40.1083946, -95.7545166, 94.7097702
6: -53.6306038, 51.6677780, -49.5437279, 47.6813393, -101.3119354, 101.2114792
7: -58.7507820, 49.7681389, -54.2997360, 46.0531387, -104.8039169, 104.0678711
8: -70.6029816, 47.5331841, -65.1895370, 43.8968124, -114.4997940, 112.7227173
9: -53.7871704, 52.1165390, -49.7392578, 48.1783867, -101.9655609, 101.8557816

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 126

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0345361, upper bound: 174.0349072
time: 9.98 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0345157, upper bound: 174.0348768
time: 9.63 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -63.9995537, 50.7303162, -62.7241058, 49.7397003, -113.7392426, 113.4544220
1: -53.4828835, 44.9403305, -52.4785309, 44.0839577, -97.5668335, 97.4188614
2: -70.5260468, 46.3292618, -69.1222382, 45.4205284, -115.9465790, 115.4514923
3: -74.7516785, 39.3018646, -73.2615128, 38.5942039, -113.3458710, 112.5633774
4: -68.2915802, 52.6394272, -66.9348221, 51.6392670, -119.9308472, 119.5742493
5: -61.1863174, 47.5349770, -60.0180206, 46.7166786, -107.9029999, 107.5529709
6: -58.8707466, 56.7436218, -57.6798668, 55.5812302, -114.4519806, 114.4234848
7: -64.4588165, 54.5518608, -63.1736908, 53.4827309, -117.9415436, 117.7255325
8: -77.5654526, 52.2810059, -75.9905472, 51.2616386, -128.8270721, 128.2715454
9: -58.9544754, 57.2350655, -57.7715569, 56.1206779, -115.0751495, 115.0066223

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 105
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
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 75
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
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 94
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0405210, upper bound: 174.0405128
time: 8.44 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0404924, upper bound: 174.0404924
time: 8.67 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 18.39 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 18.39
Output dim: 7, lower bound: -174.0353692, upper bound: 174.0362402
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.39
Output dim: 7, lower bound: -174.0353462, upper bound: 174.0362179
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 18.39
Output dim: 7, lower bound: -174.0412941, upper bound: 174.0418554
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 18.39
Output dim: 7, lower bound: -174.0412626, upper bound: 174.0418268
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 18.39
Output dim: 7, lower bound: -174.0345361, upper bound: 174.0349072
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.39
Output dim: 7, lower bound: -174.0345157, upper bound: 174.0348768
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 18.39
Output dim: 7, lower bound: -174.0405210, upper bound: 174.0405128
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 18.39
Output dim: 7, lower bound: -174.0404924, upper bound: 174.0404924

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -68.2532196, 54.1720772, -67.8025818, 53.8204346, -122.0736542, 121.9746552
1: -57.1211052, 48.0378685, -56.7538109, 47.7419777, -104.8630829, 104.7916794
2: -75.2500458, 49.3506012, -74.7568970, 49.0548744, -124.3049164, 124.1074982
3: -79.8065033, 41.9505119, -79.3127975, 41.6649590, -121.4714584, 121.2633057
4: -72.9727325, 56.2400208, -72.5323639, 55.9034348, -128.8761444, 128.7723694
5: -65.4102936, 50.9418030, -64.9951553, 50.6432343, -116.0535278, 115.9369507
6: -62.8323860, 60.4925652, -62.4474487, 60.0872955, -122.9196777, 122.9399948
7: -68.7975159, 58.1673241, -68.3857040, 57.8223228, -126.6198273, 126.5530243
8: -82.6696243, 55.8132591, -82.1298141, 55.4272079, -138.0968170, 137.9430695
9: -62.8786049, 61.2127190, -62.5040398, 60.8583450, -123.7369537, 123.7167587

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 94

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0344952, upper bound: 174.0353636
time: 12.35 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0353692, upper bound: 174.0362402
time: 11.54 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -82.9430237, 65.8374329, -67.7178192, 53.7509727, -136.6940002, 133.5552521
1: -69.3264389, 58.3208313, -56.6798210, 47.6818428, -117.0082779, 115.0006561
2: -91.3979034, 59.7234001, -74.6641998, 48.9977684, -140.3956757, 134.3875885
3: -96.9892578, 50.7942123, -79.2059021, 41.6118622, -138.6011200, 130.0001221
4: -88.7754974, 68.2654572, -72.4346466, 55.8314209, -144.6069031, 140.7001038
5: -79.5633621, 61.8477707, -64.9085846, 50.5778694, -130.1412354, 126.7563477
6: -76.3342361, 73.3961182, -62.3664513, 60.0104599, -136.3446655, 135.7625427
7: -83.4499664, 70.3818283, -68.2988968, 57.7505112, -141.2004700, 138.6807251
8: -100.3023376, 67.8629303, -82.0328674, 55.3579750, -155.6602783, 149.8957977
9: -76.1813431, 74.3450089, -62.4236679, 60.7813454, -136.9626923, 136.7686768

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 94

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0344460, upper bound: 174.0353166
time: 11.34 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0353462, upper bound: 174.0362179
time: 10.25 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -74.9289246, 59.4999390, -77.6554413, 61.6705475, -136.5994720, 137.1553497
1: -62.7351456, 52.7417297, -65.0355072, 54.6821861, -117.4173203, 117.7772293
2: -82.6473770, 54.0994606, -85.6651001, 56.0550766, -138.7024536, 139.7645569
3: -87.6083298, 46.1018944, -90.8416138, 47.7815514, -135.3898773, 136.9435120
4: -80.0618134, 61.6954231, -83.0204239, 63.9578247, -144.0196075, 144.7158356
5: -71.8342209, 55.9080849, -74.4690399, 57.9693642, -129.8035889, 130.3770905
6: -68.9172134, 66.4073563, -71.4412231, 68.8226776, -137.7398987, 137.8485718
7: -75.4614639, 63.7380066, -78.2261810, 66.0488968, -141.5103607, 141.9641571
8: -90.7534027, 61.3509216, -94.0478821, 63.5886192, -154.3420258, 155.3988037
9: -68.9003448, 67.1629562, -71.4078445, 69.6370087, -138.5373077, 138.5708008

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 185

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9724154, upper bound: 173.9747513
time: 11.06 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0387556, upper bound: 174.0393460
time: 11.22 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -89.3203583, 70.9274750, -77.5362320, 61.5737038, -150.8940582, 148.4637146
1: -74.7039642, 62.8193626, -64.9319153, 54.5973206, -129.3012543, 127.7512741
2: -98.4699936, 64.2759323, -85.5346298, 55.9712067, -154.4411926, 149.8105469
3: -104.4676285, 54.7751579, -90.6960983, 47.7069702, -152.1745911, 145.4712524
4: -95.5566483, 73.4726486, -82.8849106, 63.8556709, -159.4123230, 156.3575592
5: -85.6948929, 66.5977631, -74.3499374, 57.8779297, -143.5728149, 140.9476624
6: -82.1561203, 79.0522919, -71.3274384, 68.7148666, -150.8709717, 150.3796997
7: -89.8233185, 75.7160110, -78.1041794, 65.9473572, -155.7706604, 153.8201904
8: -108.0131531, 73.1533127, -93.9068985, 63.4874268, -171.5005493, 167.0602112
9: -81.9443665, 80.0283279, -71.2943115, 69.5270538, -151.4714203, 151.3226318

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 185

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9724012, upper bound: 173.9747348
time: 11.97 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0387173, upper bound: 174.0393162
time: 11.28 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -55.2155228, 43.7388573, -51.6521912, 40.9387741, -96.1542892, 95.3910370
1: -46.0788307, 38.7715759, -43.1675987, 36.3201218, -82.3989563, 81.9391632
2: -60.8126564, 40.1036758, -56.8894310, 37.5959549, -98.4086151, 96.9931030
3: -64.4096451, 33.8371544, -60.2766609, 31.7128315, -96.1224670, 94.1138153
4: -58.9387627, 45.4483795, -55.1567802, 42.5881767, -101.5269318, 100.6051636
5: -52.7285843, 41.0542641, -49.3626137, 38.5297127, -91.2583008, 90.4168625
6: -50.8335533, 48.9817429, -47.5722923, 45.7883072, -96.6218414, 96.5540314
7: -55.6939659, 47.2194023, -52.1416626, 44.2563591, -99.9503250, 99.3610687
8: -66.9286194, 45.0502090, -62.6011238, 42.1458321, -109.0744476, 107.6513290
9: -51.0285034, 49.4029350, -47.7921600, 46.2670708, -97.2955780, 97.1950989

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0336878, upper bound: 174.0340771
time: 10.17 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0345361, upper bound: 174.0349072
time: 11.21 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -68.4145050, 54.2061501, -51.5737724, 40.8751411, -109.2896423, 105.7799149
1: -57.0476685, 47.9625435, -43.0994415, 36.2655449, -93.3132172, 91.0619812
2: -75.2931900, 49.3541031, -56.8032646, 37.5439758, -112.8371658, 106.1573563
3: -79.9230423, 41.7753830, -60.1796989, 31.6633739, -111.5864182, 101.9550781
4: -73.1199341, 56.2143593, -55.0664101, 42.5208740, -115.6408005, 111.2807693
5: -65.4504013, 50.7335892, -49.2836876, 38.4705658, -103.9209595, 100.0172729
6: -62.9804955, 60.5373116, -47.4964256, 45.7171669, -108.6976624, 108.0337372
7: -68.8360367, 58.1426964, -52.0615692, 44.1909332, -113.0269623, 110.2042542
8: -82.7630844, 55.7835274, -62.5075607, 42.0801773, -124.8432617, 118.2910843
9: -62.9186592, 61.1811218, -47.7185135, 46.1955948, -109.1142578, 108.8996353

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 249

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0336476, upper bound: 174.0340419
time: 10.48 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0345157, upper bound: 174.0348768
time: 8.42 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -60.8622322, 48.2494354, -60.4662895, 47.9574318, -108.8196640, 108.7157288
1: -50.8445511, 42.7275429, -50.5834885, 42.4930115, -93.3375549, 93.3110352
2: -67.0609818, 44.0947876, -66.6307220, 43.8071404, -110.8681030, 110.7255096
3: -71.0366974, 37.3632660, -70.5999374, 37.2029953, -108.2396774, 107.9632034
4: -64.9231186, 50.0585899, -64.5137405, 49.7846336, -114.7077484, 114.5723267
5: -58.1759109, 45.2114563, -57.8576889, 45.0433426, -103.2192383, 103.0691452
6: -55.9844742, 53.9696693, -55.6049232, 53.5855370, -109.5700073, 109.5745773
7: -61.3061638, 51.9196358, -60.9031601, 51.5880585, -112.8942108, 112.8227997
8: -73.7750931, 49.7126617, -73.2646942, 49.4137573, -123.1888351, 122.9773483
9: -56.1028214, 54.4268379, -55.7191925, 54.1036072, -110.2064285, 110.1460266

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0392844, upper bound: 174.0393117
time: 9.46 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0405210, upper bound: 174.0405128
time: 11.72 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -74.1723480, 58.8086624, -60.3773613, 47.8850670, -122.0574036, 119.1860123
1: -61.9065704, 52.0148506, -50.5060730, 42.4305573, -104.3371201, 102.5209198
2: -81.6799393, 53.4365692, -66.5332870, 43.7468834, -125.4268188, 119.9698563
3: -86.6509705, 45.3874817, -70.4906235, 37.1467285, -123.7976913, 115.8781052
4: -79.2364578, 60.9166260, -64.4115677, 49.7080078, -128.9444580, 125.3281937
5: -71.0161057, 55.0602341, -57.7676659, 44.9748573, -115.9909515, 112.8278961
6: -68.2288284, 65.6528244, -55.5193443, 53.5046921, -121.7335205, 121.1721649
7: -74.5301132, 62.9434433, -60.8123741, 51.5129318, -126.0430222, 123.7557983
8: -89.7527390, 60.5806236, -73.1591644, 49.3387642, -139.0914764, 133.7397766
9: -68.0926895, 66.3293076, -55.6350403, 54.0220299, -122.1147156, 121.9643326

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0392489, upper bound: 174.0392880
time: 8.97 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0404924, upper bound: 174.0404924
time: 9.70 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 20.02 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.02
Output dim: 7, lower bound: -174.0344952, upper bound: 174.0353636
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.02
Output dim: 7, lower bound: -174.0353692, upper bound: 174.0362402
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.02
Output dim: 7, lower bound: -174.0344460, upper bound: 174.0353166
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.02
Output dim: 7, lower bound: -174.0353462, upper bound: 174.0362179
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.02
Output dim: 7, lower bound: -173.9724154, upper bound: 173.9747513
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.02
Output dim: 7, lower bound: -174.0387556, upper bound: 174.0393460
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.02
Output dim: 7, lower bound: -173.9724012, upper bound: 173.9747348
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.02
Output dim: 7, lower bound: -174.0387173, upper bound: 174.0393162
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.02
Output dim: 7, lower bound: -174.0336878, upper bound: 174.0340771
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.02
Output dim: 7, lower bound: -174.0345361, upper bound: 174.0349072
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.02
Output dim: 7, lower bound: -174.0336476, upper bound: 174.0340419
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.02
Output dim: 7, lower bound: -174.0345157, upper bound: 174.0348768
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.02
Output dim: 7, lower bound: -174.0392844, upper bound: 174.0393117
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.02
Output dim: 7, lower bound: -174.0405210, upper bound: 174.0405128
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.02
Output dim: 7, lower bound: -174.0392489, upper bound: 174.0392880
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.02
Output dim: 7, lower bound: -174.0404924, upper bound: 174.0404924

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -65.4516068, 51.9471550, -64.5868835, 51.2656860, -116.7172928, 116.5340271
1: -54.7719460, 46.0633240, -54.0508842, 45.4744911, -100.2464218, 100.1142120
2: -72.1556549, 47.3828316, -71.2063980, 46.8051033, -118.9607391, 118.5892334
3: -76.4968948, 40.2174072, -75.5146713, 39.6619072, -116.1587982, 115.7320709
4: -69.9598923, 53.9379005, -69.0790176, 53.2598343, -123.2197266, 123.0169067
5: -62.7135735, 48.8522568, -61.8999710, 48.2487335, -110.9623032, 110.7522202
6: -60.2595291, 58.0086899, -59.5016823, 57.2286339, -117.4881592, 117.5103760
7: -65.9881592, 55.8255081, -65.1614304, 55.1416245, -121.1297760, 120.9869385
8: -79.2889023, 53.5208511, -78.2459106, 52.7873802, -132.0762482, 131.7667542
9: -60.3204575, 58.7054482, -59.5671501, 57.9790268, -118.2994766, 118.2725983

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0329526, upper bound: 174.0339110
time: 11.10 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0333559, upper bound: 174.0342599
time: 10.28 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -66.7834702, 53.0052872, -65.7383270, 52.1836815, -118.9671478, 118.7436066
1: -55.8856163, 47.0003586, -55.0189743, 46.2866135, -102.1722260, 102.0193329
2: -73.6251984, 48.3199348, -72.4762268, 47.6089134, -121.2341080, 120.7961578
3: -78.0665436, 41.0347443, -76.8697052, 40.3794975, -118.4460373, 117.9044495
4: -71.3935776, 55.0320168, -70.3162842, 54.2074051, -125.6009827, 125.3482895
5: -63.9949608, 49.8443718, -63.0081558, 49.1032257, -113.0981903, 112.8525238
6: -61.4824333, 59.1873779, -60.5521584, 58.2552719, -119.7377014, 119.7395325
7: -67.3244324, 56.9376259, -66.3191376, 56.0971565, -123.4215775, 123.2567596
8: -80.8957138, 54.6113091, -79.6394272, 53.7392731, -134.6349640, 134.2507324
9: -61.5360413, 59.8969955, -60.6204491, 59.0126305, -120.5486755, 120.5174408

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9766294, upper bound: 173.9758995
time: 11.12 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0324365, upper bound: 174.0333741
time: 10.71 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -80.0933990, 63.5735664, -64.4804688, 51.1797142, -131.2731018, 128.0540314
1: -66.9362106, 56.3125305, -53.9592285, 45.3995094, -112.3357162, 110.2717438
2: -88.2485962, 57.7214622, -71.0897064, 46.7322655, -134.9808655, 128.8111572
3: -93.6247711, 49.0319481, -75.3846054, 39.5955162, -133.2202606, 124.4165421
4: -85.7120819, 65.9230804, -68.9591827, 53.1695404, -138.8816223, 134.8822632
5: -76.8215256, 59.7208328, -61.7936249, 48.1677437, -124.9892654, 121.5144577
6: -73.7158203, 70.8699570, -59.4003639, 57.1330528, -130.8488617, 130.2703247
7: -80.5919724, 67.9981155, -65.0538483, 55.0525856, -135.6445618, 133.0519409
8: -96.8627396, 65.5308533, -78.1207733, 52.6980400, -149.5607758, 143.6516113
9: -73.5776291, 71.7946472, -59.4675217, 57.8832321, -131.4608154, 131.2621460

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9755968, upper bound: 173.9748652
time: 10.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0314742, upper bound: 174.0323532
time: 10.01 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -81.4535065, 64.6552048, -65.6311417, 52.0964775, -133.5499878, 130.2863464
1: -68.0739670, 57.2699852, -54.9259872, 46.2110023, -114.2849579, 112.1959610
2: -89.7512894, 58.6789360, -72.3593750, 47.5359802, -137.2872620, 131.0383148
3: -95.2267761, 49.8656540, -76.7370453, 40.3119469, -135.5387115, 126.6026917
4: -87.1761017, 67.0409470, -70.1951065, 54.1168747, -141.2929382, 137.2360382
5: -78.1298828, 60.7348709, -62.9003792, 49.0209427, -127.1508255, 123.6352539
6: -74.9653168, 72.0738678, -60.4504128, 58.1587486, -133.1240692, 132.5242767
7: -81.9568176, 69.1354218, -66.2100143, 56.0068741, -137.9636841, 135.3453827
8: -98.5040207, 66.6445923, -79.5151215, 53.6512642, -152.1552887, 146.1597137
9: -74.8203506, 73.0124817, -60.5197258, 58.9160423, -133.7363739, 133.5321960

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9762207, upper bound: 173.9754893
time: 11.88 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0324062, upper bound: 174.0333451
time: 11.53 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -59.4050636, 47.1053619, -59.2343941, 46.9388962, -106.3439560, 106.3397446
1: -49.6485939, 41.8005867, -49.4587555, 41.7135086, -91.3620987, 91.2593384
2: -65.4816666, 43.0559998, -65.3096695, 42.9294510, -108.4111099, 108.3656693
3: -69.5245438, 36.5530548, -69.4595718, 36.4120331, -105.9365692, 106.0126190
4: -63.4893837, 48.9272499, -63.4199638, 48.8009109, -112.2902985, 112.3472137
5: -56.8539009, 44.2920227, -56.6781349, 44.1706772, -101.0245819, 100.9701538
6: -54.6596222, 52.6790771, -54.5252724, 52.5510712, -107.2106934, 107.2043457
7: -59.8929825, 50.7384872, -59.7863579, 50.6454735, -110.5384521, 110.5248337
8: -71.9020233, 48.4578056, -71.6318054, 48.2422104, -120.1442261, 120.0896149
9: -54.8113365, 53.1368446, -54.7205658, 52.9664307, -107.7777405, 107.8573990

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9715683, upper bound: 173.9738179
time: 11.92 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9724154, upper bound: 173.9747487
time: 11.59 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -73.0491562, 58.0018425, -74.1972427, 58.9146996, -131.9638519, 132.1990814
1: -61.1559410, 51.4164658, -62.1250839, 52.2428970, -113.3988266, 113.5415497
2: -80.5681458, 52.7604980, -81.8414001, 53.5910072, -134.1591492, 134.6018982
3: -85.4147949, 44.9477654, -86.8014603, 45.6561470, -131.0708923, 131.7492065
4: -78.0495834, 60.1481171, -79.3172073, 61.1119995, -139.1615906, 139.4653015
5: -70.0229492, 54.5012741, -71.1371384, 55.3810997, -125.4040527, 125.6384048
6: -67.1860428, 64.7460098, -68.2560120, 65.7647400, -132.9507751, 133.0020142
7: -73.5708160, 62.1614609, -74.7508621, 63.1464233, -136.7172241, 136.9123230
8: -88.4742966, 59.7953987, -89.8580246, 60.7260590, -149.2003479, 149.6534271
9: -67.1897659, 65.4648438, -68.2594223, 66.5131149, -133.7028656, 133.7242737

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0376217, upper bound: 174.0381878
time: 11.09 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0387555, upper bound: 174.0393460
time: 10.02 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -73.4451752, 58.2584801, -59.1165161, 46.8435593, -120.2887268, 117.3749847
1: -61.3118172, 51.6232262, -49.3569298, 41.6306267, -102.9424362, 100.9801559
2: -80.9077148, 52.9619560, -65.1803207, 42.8468933, -123.7546082, 118.1422729
3: -85.9581451, 45.0097389, -69.3170624, 36.3382034, -122.2963409, 114.3267975
4: -78.5937195, 60.4062271, -63.2866974, 48.6999893, -127.2937088, 123.6929169
5: -70.3923569, 54.7121315, -56.5607834, 44.0801620, -114.4725189, 111.2729111
6: -67.5675735, 65.0055008, -54.4126244, 52.4448204, -120.0123901, 119.4181137
7: -73.8800583, 62.4083290, -59.6659851, 50.5455704, -124.4256287, 122.0743103
8: -88.7460327, 59.9625092, -71.4911041, 48.1431351, -136.8891602, 131.4535980
9: -67.5092392, 65.6876373, -54.6097412, 52.8580666, -120.3673096, 120.2973785

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9715572, upper bound: 173.9738081
time: 10.47 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9724012, upper bound: 173.9747322
time: 10.27 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -87.4710846, 69.4536133, -74.0752640, 58.8156357, -146.2867126, 143.5288696
1: -73.1464767, 61.5139999, -62.0194893, 52.1561890, -125.3026581, 123.5334930
2: -96.4256668, 62.9580956, -81.7077484, 53.5052681, -149.9309082, 144.6658478
3: -102.3066864, 53.6375809, -86.6528854, 45.5800285, -147.8867188, 140.2904510
4: -93.5757675, 71.9517593, -79.1787262, 61.0074883, -154.5832062, 151.1304932
5: -83.9129715, 65.2133713, -71.0153732, 55.2875214, -139.2004852, 136.2287445
6: -80.4529953, 77.4167252, -68.1396561, 65.6545410, -146.1075134, 145.5563354
7: -87.9650192, 74.1634598, -74.6258698, 63.0427284, -151.0077515, 148.7893219
8: -105.7729721, 71.6223831, -89.7135391, 60.6226006, -166.3955688, 161.3359222
9: -80.2606812, 78.3580627, -68.1434250, 66.4006729, -146.6613464, 146.5014648

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9973790, upper bound: 173.9985499
time: 11.65 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9961632, upper bound: 173.9968572
time: 11.35 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -52.5393410, 41.6096573, -48.6361160, 38.5421562, -91.0814972, 90.2457733
1: -43.8367615, 36.8942146, -40.6455193, 34.2089233, -78.0456848, 77.5397263
2: -57.8562698, 38.2306709, -53.5651665, 35.4979935, -93.3542633, 91.7958298
3: -61.2318916, 32.1799965, -56.7009621, 29.8357296, -91.0676193, 88.8809433
4: -56.0586739, 43.2533112, -51.9216881, 40.1161041, -96.1747742, 95.1750031
5: -50.1453209, 39.0702667, -46.4564133, 36.2968826, -86.4421997, 85.5266724
6: -48.3654480, 46.6169281, -44.8033028, 43.1219139, -91.4873657, 91.4202271
7: -53.0146446, 44.9826431, -49.1265106, 41.7508926, -94.7655106, 94.1091537
8: -63.6910133, 42.8702202, -58.9550591, 39.6847229, -103.3757324, 101.8252792
9: -48.5901756, 47.0119095, -45.0452538, 43.5781097, -92.1682816, 92.0571594

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9885972, upper bound: 173.9900572
time: 10.96 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9871966, upper bound: 173.9879034
time: 9.61 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -53.8232307, 42.6314507, -49.7183723, 39.4053421, -93.2285538, 92.3498154
1: -44.9067650, 37.7922401, -41.5480042, 34.9644585, -79.8712234, 79.3402405
2: -59.2733612, 39.1313591, -54.7548409, 36.2500877, -95.5234528, 93.8861923
3: -62.7514877, 32.9680405, -57.9785538, 30.5085487, -93.2600250, 90.9465790
4: -57.4425545, 44.3063240, -53.0827789, 41.0044289, -98.4469757, 97.3890991
5: -51.3835487, 40.0224800, -47.4998398, 37.0992050, -88.4827576, 87.5223083
6: -49.5491867, 47.7492752, -45.7933693, 44.0791054, -93.6282959, 93.5426407
7: -54.3010635, 46.0552063, -50.2096634, 42.6450920, -96.9461517, 96.2648544
8: -65.2462769, 43.9163895, -60.2672615, 40.5741348, -105.8203964, 104.1836548
9: -49.7589722, 48.1584091, -46.0313530, 44.5436134, -94.3025818, 94.1897583

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9894980, upper bound: 173.9908945
time: 10.57 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9880167, upper bound: 173.9886213
time: 9.32 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -65.7059937, 52.0583534, -48.5604095, 38.4810715, -104.1870651, 100.6187515
1: -54.7793159, 46.0623245, -40.5800819, 34.1565514, -88.9358597, 86.6424103
2: -72.3051300, 47.4633293, -53.4812584, 35.4476433, -107.7527771, 100.9445877
3: -76.7137680, 40.1012840, -56.6084137, 29.7887287, -106.5024872, 96.7097015
4: -70.2109604, 53.9935570, -51.8348427, 40.0510712, -110.2620316, 105.8283997
5: -62.8419533, 48.7278938, -46.3810043, 36.2399979, -99.0819397, 95.1088867
6: -60.4886322, 58.1440163, -44.7301559, 43.0535126, -103.5421448, 102.8741760
7: -66.1236496, 55.8842392, -49.0492096, 41.6879234, -107.8115616, 104.9334488
8: -79.4946594, 53.5810242, -58.8646851, 39.6212730, -119.1159210, 112.4457092
9: -60.4522285, 58.7636642, -44.9739838, 43.5094872, -103.9617081, 103.7376480

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 126

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9885735, upper bound: 173.9900383
time: 8.99 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9871965, upper bound: 173.9878928
time: 10.10 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -67.0146942, 53.0967102, -49.6178131, 39.3243027, -106.3389893, 102.7145233
1: -55.8724098, 46.9786797, -41.4614601, 34.8943291, -90.7667389, 88.4401398
2: -73.7484436, 48.3786888, -54.6444931, 36.1825409, -109.9309845, 103.0231781
3: -78.2589417, 40.9024811, -57.8552513, 30.4453335, -108.7042770, 98.7577362
4: -71.6176834, 55.0661049, -52.9684792, 40.9189720, -112.5366516, 108.0345612
5: -64.1018066, 49.6963463, -47.3998184, 37.0235596, -101.1253662, 97.0961456
6: -61.6919746, 59.2984734, -45.6969604, 43.9887123, -105.6806870, 104.9954376
7: -67.4345169, 56.9747353, -50.1070137, 42.5613556, -109.9958725, 107.0817261
8: -81.0739899, 54.6448364, -60.1467514, 40.4902420, -121.5642319, 114.7915802
9: -61.6429062, 59.9318275, -45.9374924, 44.4523811, -106.0952911, 105.8693237

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9894810, upper bound: 173.9908859
time: 9.76 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9880167, upper bound: 173.9886206
time: 8.50 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 19.61 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.61
Output dim: 7, lower bound: -174.0329526, upper bound: 174.0339110
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.61
Output dim: 7, lower bound: -174.0333559, upper bound: 174.0342599
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.61
Output dim: 7, lower bound: -173.9766294, upper bound: 173.9758995
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.61
Output dim: 7, lower bound: -174.0324365, upper bound: 174.0333741
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.61
Output dim: 7, lower bound: -173.9755968, upper bound: 173.9748652
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.61
Output dim: 7, lower bound: -174.0314742, upper bound: 174.0323532
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.61
Output dim: 7, lower bound: -173.9762207, upper bound: 173.9754893
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.61
Output dim: 7, lower bound: -174.0324062, upper bound: 174.0333451
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.61
Output dim: 7, lower bound: -173.9715683, upper bound: 173.9738179
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.61
Output dim: 7, lower bound: -173.9724154, upper bound: 173.9747487
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.61
Output dim: 7, lower bound: -174.0376217, upper bound: 174.0381878
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.61
Output dim: 7, lower bound: -174.0387555, upper bound: 174.0393460
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.61
Output dim: 7, lower bound: -173.9715572, upper bound: 173.9738081
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.61
Output dim: 7, lower bound: -173.9724012, upper bound: 173.9747322
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.61
Output dim: 7, lower bound: -173.9973790, upper bound: 173.9985499
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.61
Output dim: 7, lower bound: -173.9961632, upper bound: 173.9968572
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.61
Output dim: 7, lower bound: -173.9885972, upper bound: 173.9900572
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.61
Output dim: 7, lower bound: -173.9871966, upper bound: 173.9879034
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.61
Output dim: 7, lower bound: -173.9894980, upper bound: 173.9908945
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.61
Output dim: 7, lower bound: -173.9880167, upper bound: 173.9886213
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.61
Output dim: 7, lower bound: -173.9885735, upper bound: 173.9900383
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.61
Output dim: 7, lower bound: -173.9871965, upper bound: 173.9878928
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.61
Output dim: 7, lower bound: -173.9894810, upper bound: 173.9908859
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.61
Output dim: 7, lower bound: -173.9880167, upper bound: 173.9886206
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 19.61
Output dim: 7, lower bound: -174.0392844, upper bound: 174.0393117
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.61
Output dim: 7, lower bound: -174.0405210, upper bound: 174.0405128
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.61
Output dim: 7, lower bound: -174.0392489, upper bound: 174.0392880
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.61
Output dim: 7, lower bound: -174.0404924, upper bound: 174.0404924
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=175.32177734375
rel_dist={7: [-174.07321147769613, 174.07321147870266]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1823.17 seconds
