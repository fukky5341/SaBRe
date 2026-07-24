## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 157.221417204


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168)
1: (-73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138)
2: (-96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084)
3: (-102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802)
4: (-93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124)
5: (-84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665)
6: (-80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456)
7: (-87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022)
8: (-105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349)
9: (-79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.96 + 10.50 = 12.45 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -157.3787960, upper bound: 157.3787960

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 57

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3787960, upper bound: 157.3787960
time: 8.20 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3787960, upper bound: 157.3787960
time: 7.24 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 15.71 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 15.71
Output dim: 9, lower bound: -157.3787960, upper bound: 157.3787960
NS_A2, status: Status.UNKNOWN, split count: 1, time: 15.71
Output dim: 9, lower bound: -157.3787960, upper bound: 157.3787960

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -85.3586121, 68.4641953, -86.3537369, 69.2596283, -154.6182098, 154.8179321
1: -71.9158249, 61.0958023, -72.7442474, 61.7905540, -133.7063751, 133.8400574
2: -94.4184952, 61.8785095, -95.5122528, 62.5884781, -157.0069733, 157.3907623
3: -100.2560425, 53.5495110, -101.4244766, 54.1568565, -154.4128876, 154.9739838
4: -91.9746552, 70.6984863, -93.0460434, 71.5199814, -163.4946289, 163.7445374
5: -82.4562836, 64.8553467, -83.4220200, 65.6002197, -148.0565033, 148.2773743
6: -78.9177246, 76.1577988, -79.8363266, 77.0355530, -155.9532776, 155.9941254
7: -85.9947052, 72.4834442, -86.9862900, 73.3159485, -159.3106537, 159.4697266
8: -103.6159668, 71.0141830, -104.8210983, 71.8594208, -175.4753571, 175.8352814
9: -78.2394333, 77.3749542, -79.1457443, 78.2647781, -156.5041962, 156.5206604

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3740829, upper bound: 157.3740557
time: 8.19 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3731230, upper bound: 157.3731405
time: 5.49 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -86.2967606, 69.2144089, -86.7743912, 69.5940704, -155.8908081, 155.9888000
1: -72.6958008, 61.7509117, -73.0952072, 62.0868301, -134.7826233, 134.8460693
2: -95.4493637, 62.5489082, -95.9734802, 62.8871078, -158.3364716, 158.5223694
3: -101.3572159, 54.1220665, -101.9210510, 54.4194794, -155.7766418, 156.0430908
4: -92.9855728, 71.4727936, -93.4968872, 71.8641052, -164.8496704, 164.9696655
5: -83.3670044, 65.5575943, -83.8263550, 65.9150467, -149.2820282, 149.3839264
6: -79.7840958, 76.9849854, -80.2248535, 77.4077377, -157.1918335, 157.2098389
7: -86.9299469, 73.2681122, -87.4054108, 73.6713028, -160.6012573, 160.6735229
8: -104.7521591, 71.8130646, -105.3280182, 72.2077026, -176.9598541, 177.1410370
9: -79.0951996, 78.2140656, -79.5321426, 78.6445160, -157.7396393, 157.7461700

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3742219, upper bound: 157.3742163
time: 7.16 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3734474, upper bound: 157.3734474
time: 6.54 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 15.89 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 15.89
Output dim: 9, lower bound: -157.3740829, upper bound: 157.3740557
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 15.89
Output dim: 9, lower bound: -157.3731230, upper bound: 157.3731405
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 15.89
Output dim: 9, lower bound: -157.3742219, upper bound: 157.3742163
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 15.89
Output dim: 9, lower bound: -157.3734474, upper bound: 157.3734474

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -85.3470459, 68.4550400, -85.7155762, 68.7535629, -154.1005707, 154.1706238
1: -71.9061966, 61.0877075, -72.2115860, 61.3432617, -133.2494507, 133.2992859
2: -94.4057922, 61.8703270, -94.8107605, 62.1353416, -156.5411377, 156.6810913
3: -100.2425690, 53.5424232, -100.6795425, 53.7644768, -154.0070190, 154.2219696
4: -91.9622116, 70.6890182, -92.3581314, 70.9972763, -162.9594421, 163.0471497
5: -82.4452209, 64.8466949, -82.8112564, 65.1226425, -147.5678558, 147.6579590
6: -78.9071274, 76.1475906, -79.2506409, 76.4717331, -155.3788605, 155.3982239
7: -85.9831314, 72.4737244, -86.3465042, 72.7785950, -158.7617188, 158.8202209
8: -103.6020432, 71.0047226, -104.0524826, 71.3362122, -174.9382629, 175.0572052
9: -78.2288208, 77.3645859, -78.5593185, 77.6915588, -155.9203491, 155.9239044

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3674316, upper bound: 157.3673676
time: 8.08 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3663581, upper bound: 157.3660877
time: 8.53 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -84.6739578, 67.9209747, -92.9273300, 74.4575958, -159.1315613, 160.8482819
1: -71.3436890, 60.6160583, -78.2775955, 66.4237671, -137.7674255, 138.8936462
2: -93.6653137, 61.3926926, -102.7682953, 67.2520218, -160.9172974, 164.1609802
3: -99.4575348, 53.1285706, -109.1646957, 58.2007370, -157.6582336, 162.2932434
4: -91.2362976, 70.1377640, -100.0803680, 76.9103241, -168.1466217, 170.2181244
5: -81.8003845, 64.3435516, -89.7662964, 70.4954910, -152.2958679, 154.1098328
6: -78.2892227, 75.5525513, -85.8271637, 82.8563232, -161.1455383, 161.3797150
7: -85.3082123, 71.9068604, -93.5778275, 78.7672958, -164.0755005, 165.4846802
8: -102.7912598, 70.4532089, -112.7828445, 77.3461838, -180.1374512, 183.2360535
9: -77.6104736, 76.7599564, -85.0546722, 84.0451202, -161.6555786, 161.8145905

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 57

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3659901, upper bound: 157.3662978
time: 6.46 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3649151, upper bound: 157.3649648
time: 7.30 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -85.8647537, 68.8691864, -82.0705261, 65.8356705, -151.7004242, 150.9396973
1: -72.3288956, 61.4409256, -69.1006699, 58.7122536, -131.0411530, 130.5415955
2: -94.9680557, 62.2378540, -90.7338791, 59.5007095, -154.4687653, 152.9716949
3: -100.8450928, 53.8501892, -96.3456345, 51.4596481, -152.3047485, 150.1958160
4: -92.5197678, 71.1159821, -88.4249649, 67.9802856, -160.5000610, 159.5409546
5: -82.9493484, 65.2323456, -79.2795486, 62.3751183, -145.3244629, 144.5118866
6: -79.3827438, 76.5983429, -75.8563538, 73.1984711, -152.5812073, 152.4546661
7: -86.4933548, 72.9030685, -82.6532135, 69.6974106, -156.1907501, 155.5562592
8: -104.2281647, 71.4552231, -99.6230774, 68.3116913, -172.5398407, 171.0782623
9: -78.7014084, 77.8229141, -75.2458496, 74.3870850, -153.0885010, 153.0687561

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 57

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 175

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3675427, upper bound: 157.3676790
time: 7.87 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3664675, upper bound: 157.3662331
time: 7.61 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -82.1321793, 65.8926849, -78.5809555, 63.0561562, -145.1883240, 144.4736328
1: -69.1658401, 58.7736206, -66.1381073, 56.2079201, -125.3737488, 124.9117279
2: -90.8205414, 59.5611267, -86.8480682, 56.9993172, -147.8198090, 146.4091949
3: -96.4283142, 51.5104523, -92.2226028, 49.2569695, -145.6852875, 143.7330322
4: -88.5006332, 68.0414124, -84.6669312, 65.0997391, -153.6003418, 152.7082977
5: -79.3465195, 62.4352913, -75.9130783, 59.7599945, -139.1065063, 138.3483734
6: -75.9244156, 73.2658081, -72.6321182, 70.0746994, -145.9991150, 145.8979187
7: -82.7311554, 69.7603760, -79.1390915, 66.7680740, -149.4992371, 148.8994751
8: -99.7107239, 68.3715668, -95.3999939, 65.4262009, -165.1369324, 163.7715454
9: -75.3083572, 74.4542236, -72.0973434, 71.2434616, -146.5518188, 146.5515442

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 57

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 175

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3665443, upper bound: 157.3669166
time: 6.99 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3653830, upper bound: 157.3653830
time: 6.28 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 15.70 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 15.70
Output dim: 9, lower bound: -157.3674316, upper bound: 157.3673676
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 15.70
Output dim: 9, lower bound: -157.3663581, upper bound: 157.3660877
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 15.70
Output dim: 9, lower bound: -157.3659901, upper bound: 157.3662978
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 15.70
Output dim: 9, lower bound: -157.3649151, upper bound: 157.3649648
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 15.70
Output dim: 9, lower bound: -157.3675427, upper bound: 157.3676790
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 15.70
Output dim: 9, lower bound: -157.3664675, upper bound: 157.3662331
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 15.70
Output dim: 9, lower bound: -157.3665443, upper bound: 157.3669166
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 15.70
Output dim: 9, lower bound: -157.3653830, upper bound: 157.3653830

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -80.6133041, 64.6715851, -85.2843933, 68.4089737, -149.0222778, 149.9559479
1: -67.8866196, 57.6912422, -71.8453522, 61.0338478, -128.9204712, 129.5365753
2: -89.1319427, 58.4622154, -94.3303299, 61.8248520, -150.9567871, 152.7925415
3: -94.6335907, 50.5658226, -100.1683884, 53.4930878, -148.1266632, 150.7342072
4: -86.8576050, 66.7791595, -91.8931503, 70.6411362, -157.4987183, 158.6723022
5: -77.8686676, 61.2837219, -82.3943939, 64.7980347, -142.6666870, 143.6781158
6: -74.5111237, 71.9117661, -78.8500214, 76.0858002, -150.5968628, 150.7617798
7: -81.1997299, 68.4739609, -85.9107056, 72.4142380, -153.6139679, 154.3846283
8: -97.8596497, 67.0836639, -103.5294495, 70.9790802, -168.8386993, 170.6130981
9: -73.9145889, 73.0794907, -78.1663132, 77.3011169, -151.2156830, 151.2458038

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 57

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3628259, upper bound: 157.3629373
time: 7.23 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3624244, upper bound: 157.3625384
time: 9.12 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -77.3334503, 62.0619125, -81.5484390, 65.4297409, -142.7631836, 143.6103516
1: -65.1010818, 55.3383102, -68.6793594, 58.3640442, -123.4651184, 124.0176697
2: -85.4808350, 56.1133766, -90.1791153, 59.1457977, -144.6266174, 146.2924957
3: -90.7566605, 48.4941826, -95.7469406, 51.1505814, -141.9072266, 144.2411041
4: -83.3257217, 64.0754547, -87.8706055, 67.5636292, -150.8893433, 151.9460602
5: -74.7079086, 58.8292122, -78.7882996, 61.9983025, -136.7062073, 137.6175079
6: -71.4811401, 68.9751358, -75.3882828, 72.7502823, -144.2314148, 144.3634186
7: -77.8975449, 65.7235184, -82.1453018, 69.2685623, -147.1661072, 147.8688202
8: -93.8900528, 64.3762054, -99.0079727, 67.8925095, -161.7825317, 163.3841858
9: -70.9574127, 70.1266785, -74.7701340, 73.9291458, -144.8865356, 144.8968201

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 57

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3611726, upper bound: 157.3606714
time: 7.32 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3609990, upper bound: 157.3605371
time: 7.72 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -79.9381027, 64.1358032, -92.5023499, 74.1177902, -154.0558929, 156.6381531
1: -67.3223267, 57.2181015, -77.9166794, 66.1188354, -133.4411621, 135.1347809
2: -88.3890762, 57.9831352, -102.2947998, 66.9460907, -155.3351593, 160.2779388
3: -93.8461990, 50.1506577, -108.6611023, 57.9333267, -151.7794952, 158.8117371
4: -86.1294250, 66.2261276, -99.6220474, 76.5592194, -162.6886444, 165.8481750
5: -77.2217789, 60.7790070, -89.3552933, 70.1754532, -147.3972168, 150.1343079
6: -73.8912659, 71.3147812, -85.4323273, 82.4759445, -156.3672180, 156.7471008
7: -80.5226440, 67.9053345, -93.1483536, 78.4081879, -158.9308319, 161.0536804
8: -97.0462494, 66.5305786, -112.2671661, 76.9940567, -174.0402832, 178.7977448
9: -73.2943268, 72.4729004, -84.6673279, 83.6602478, -156.9545746, 157.1402130

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3613055, upper bound: 157.3616283
time: 7.80 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3606732, upper bound: 157.3610754
time: 7.30 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -76.6631393, 61.5302124, -88.7084656, 71.0921478, -147.7552795, 150.2386780
1: -64.5409393, 54.8686600, -74.6991577, 63.4073067, -127.9482422, 129.5678101
2: -84.7435455, 55.6377869, -98.0770645, 64.2233200, -148.9668579, 153.7148438
3: -89.9749527, 48.0821877, -104.1716614, 55.5526009, -145.5275574, 152.2538452
4: -82.6030807, 63.5265160, -95.5365524, 73.4322662, -156.0353394, 159.0630646
5: -74.0657883, 58.3282013, -85.6927719, 67.3310013, -141.3967590, 144.0209656
6: -70.8661118, 68.3827286, -81.9149551, 79.0880356, -149.9541168, 150.2976685
7: -77.2253418, 65.1591568, -89.3226700, 75.2116852, -152.4370270, 154.4818268
8: -93.0829010, 63.8272133, -107.6731186, 73.8587799, -166.9416504, 171.5003357
9: -70.3417740, 69.5245132, -81.2170258, 80.2359695, -150.5777435, 150.7415466

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3596716, upper bound: 157.3594632
time: 6.22 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3592269, upper bound: 157.3591196
time: 6.84 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -85.2292099, 68.3650742, -82.0590210, 65.8265686, -151.0557861, 150.4240875
1: -71.7982712, 60.9953728, -69.0910645, 58.7041931, -130.5024719, 130.0864410
2: -94.2693024, 61.7864037, -90.7212372, 59.4925575, -153.7618561, 152.5076294
3: -100.1030350, 53.4592438, -96.3322144, 51.4525795, -151.5556183, 149.7914581
4: -91.8344955, 70.5953140, -88.4125824, 67.9708633, -159.8053589, 159.0079041
5: -82.3409729, 64.7567215, -79.2685242, 62.3665123, -144.7074432, 144.0252380
6: -78.7992859, 76.0366898, -75.8458099, 73.1883163, -151.9876099, 151.8824921
7: -85.8560181, 72.3677444, -82.6416702, 69.6877365, -155.5437622, 155.0093994
8: -103.4625473, 70.9339905, -99.6092453, 68.3022842, -171.7648010, 170.5432129
9: -78.1172256, 77.2517700, -75.2353058, 74.3767700, -152.4939880, 152.4870758

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 57

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3637767, upper bound: 157.3637229
time: 8.55 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3635269, upper bound: 157.3635141
time: 8.17 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -92.4269943, 74.0580215, -81.3715134, 65.2813416, -157.7083435, 155.4295349
1: -77.8530045, 66.0666885, -68.5165787, 58.2226181, -136.0756226, 134.5832672
2: -102.2121429, 66.8941956, -89.9650421, 59.0051613, -161.2172852, 156.8592377
3: -108.5720367, 57.8880157, -95.5304718, 51.0300980, -159.6021423, 153.4184875
4: -99.5422363, 76.4975357, -87.6712341, 67.4080429, -166.9502869, 164.1687622
5: -89.2831955, 70.1197205, -78.6100693, 61.8528709, -151.1360626, 148.7297974
6: -85.3631973, 82.4094162, -75.2149124, 72.5806656, -157.9438629, 157.6242828
7: -93.0741348, 78.3457260, -81.9524002, 69.1091995, -162.1833344, 160.2981262
8: -112.1766891, 76.9330368, -98.7814102, 67.7395401, -179.9162292, 175.7144470
9: -84.6005554, 83.5936432, -74.6041641, 73.7594604, -158.3600159, 158.1977997

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 57

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3607546, upper bound: 157.3619112
time: 6.97 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3620887, upper bound: 157.3616766
time: 7.42 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -81.4989166, 65.3903046, -78.5699997, 63.0474854, -144.5464020, 143.9602966
1: -68.6370544, 58.3297081, -66.1289749, 56.2002563, -124.8373108, 124.4586792
2: -90.1242218, 59.1113129, -86.8360519, 56.9915619, -147.1157837, 145.9473419
3: -95.6889267, 51.1208687, -92.2098312, 49.2502480, -144.9391785, 143.3306885
4: -87.8177795, 67.5225830, -84.6551437, 65.0907974, -152.9085541, 152.1777344
5: -78.7402649, 61.9614105, -75.9026108, 59.7518158, -138.4920502, 137.8640137
6: -75.3430405, 72.7061539, -72.6220856, 70.0650482, -145.4080505, 145.3282471
7: -82.0960541, 69.2269058, -79.1281433, 66.7588654, -148.8548889, 148.3550415
8: -98.9478302, 67.8521729, -95.3868256, 65.4172287, -164.3650513, 163.2389679
9: -74.7261581, 73.8850327, -72.0872879, 71.2336426, -145.9598083, 145.9722900

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 104

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3653830, upper bound: 157.3653830
time: 5.96 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3653830, upper bound: 157.3653830
time: 6.03 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -88.6031342, 71.0086288, -77.8943253, 62.5116653, -151.1148071, 148.9029388
1: -74.6099930, 63.3331413, -65.5641785, 55.7268600, -130.3368530, 128.8972931
2: -97.9618454, 64.1504822, -86.0928802, 56.5122910, -154.4741364, 150.2433624
3: -104.0425949, 55.4841919, -91.4215546, 48.8351364, -152.8777008, 146.9057465
4: -95.4261246, 73.3450394, -83.9265060, 64.5376053, -159.9637146, 157.2715454
5: -85.5920486, 67.2514343, -75.2553940, 59.2469139, -144.8389435, 142.5068359
6: -81.8166504, 78.9951935, -72.0021820, 69.4678345, -151.2844849, 150.9973602
7: -89.2194443, 75.1236649, -78.4505005, 66.1901703, -155.4096069, 153.5741577
8: -107.5466843, 73.7729187, -94.5734100, 64.8639908, -172.4106750, 168.3463287
9: -81.1233368, 80.1412659, -71.4668503, 70.6267166, -151.7500458, 151.6081238

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3596716, upper bound: 157.3610250
time: 6.75 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3592269, upper bound: 157.3606457
time: 7.17 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 15.95 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.95
Output dim: 9, lower bound: -157.3628259, upper bound: 157.3629373
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.95
Output dim: 9, lower bound: -157.3624244, upper bound: 157.3625384
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.95
Output dim: 9, lower bound: -157.3611726, upper bound: 157.3606714
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.95
Output dim: 9, lower bound: -157.3609990, upper bound: 157.3605371
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.95
Output dim: 9, lower bound: -157.3613055, upper bound: 157.3616283
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.95
Output dim: 9, lower bound: -157.3606732, upper bound: 157.3610754
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.95
Output dim: 9, lower bound: -157.3596716, upper bound: 157.3594632
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.95
Output dim: 9, lower bound: -157.3592269, upper bound: 157.3591196
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.95
Output dim: 9, lower bound: -157.3637767, upper bound: 157.3637229
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.95
Output dim: 9, lower bound: -157.3635269, upper bound: 157.3635141
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.95
Output dim: 9, lower bound: -157.3607546, upper bound: 157.3619112
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.95
Output dim: 9, lower bound: -157.3620887, upper bound: 157.3616766
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.95
Output dim: 9, lower bound: -157.3653830, upper bound: 157.3653830
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.95
Output dim: 9, lower bound: -157.3653830, upper bound: 157.3653830
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.95
Output dim: 9, lower bound: -157.3596716, upper bound: 157.3610250
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.95
Output dim: 9, lower bound: -157.3592269, upper bound: 157.3606457

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -78.8123779, 63.2409439, -80.1510391, 64.3329926, -143.1453705, 143.3919678
1: -66.3471298, 56.3971443, -67.4578247, 57.3465271, -123.6936569, 123.8549576
2: -87.1195297, 57.1691933, -88.5964737, 58.1435127, -145.2630463, 145.7656708
3: -92.4965210, 49.4350510, -94.0748749, 50.2665024, -142.7629852, 143.5099030
4: -84.9133072, 65.2891388, -86.3560410, 66.3957825, -151.3090820, 151.6451569
5: -76.1346512, 59.9344406, -77.4537201, 60.9524422, -137.0870819, 137.3881531
6: -72.8362350, 70.2996063, -74.0779037, 71.4932632, -144.3294983, 144.3775024
7: -79.3668976, 66.9502640, -80.6901169, 68.0725937, -147.4394836, 147.6403656
8: -95.6703415, 65.6067429, -97.2930679, 66.7726593, -162.4429932, 162.8998108
9: -72.2737961, 71.4541855, -73.4906540, 72.6686783, -144.9424744, 144.9448242

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 175

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3628259, upper bound: 157.3629373
time: 7.37 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3628259, upper bound: 157.3629373
time: 7.92 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -78.1244888, 62.6922722, -81.4627151, 65.3600311, -143.4844971, 144.1549835
1: -65.7637711, 55.9065704, -68.5339432, 58.2464867, -124.0102539, 124.4405060
2: -86.3557587, 56.6803055, -90.0130081, 59.0676460, -145.4234009, 146.6932983
3: -91.6819992, 49.0041962, -95.5886383, 51.0491829, -142.7311554, 144.5928345
4: -84.1727524, 64.7230377, -87.7661743, 67.4564056, -151.6291504, 152.4892120
5: -75.4705887, 59.4218636, -78.6972275, 61.9310074, -137.4015961, 138.1190948
6: -72.2019348, 69.6863861, -75.2753906, 72.6380844, -144.8400269, 144.9617767
7: -78.6741791, 66.3736801, -81.9940567, 69.1761017, -147.8502502, 148.3676605
8: -94.8356323, 65.0404816, -98.8496399, 67.8398285, -162.6754608, 163.8901215
9: -71.6509018, 70.8370285, -74.6957550, 73.8270569, -145.4779663, 145.5327759

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 175

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3624244, upper bound: 157.3625384
time: 7.55 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3624244, upper bound: 157.3625384
time: 6.76 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -75.5740738, 60.6651917, -76.4868851, 61.4110489, -136.9851074, 137.1520538
1: -63.5963745, 54.0741386, -64.3521194, 54.7279129, -118.3242874, 118.4262543
2: -83.5142517, 54.8501701, -84.5237579, 55.5148697, -139.0291138, 139.3739319
3: -88.6699753, 47.3892670, -89.7424164, 47.9706993, -136.6406403, 137.1316833
4: -81.4258728, 62.6202888, -82.4090500, 63.3779335, -144.8038025, 145.0292969
5: -73.0139847, 57.5119476, -73.9169464, 58.2075500, -131.2215118, 131.4288940
6: -69.8455276, 67.4002686, -70.6835251, 68.2218094, -138.0673370, 138.0838013
7: -76.1060257, 64.2352676, -76.9953156, 64.9870453, -141.0930634, 141.2305908
8: -91.7513733, 62.9347649, -92.8576279, 63.7456284, -155.4970093, 155.7923889
9: -69.3548279, 68.5396042, -70.1585922, 69.3634338, -138.7182159, 138.6981964

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 175

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3611726, upper bound: 157.3606714
time: 8.26 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3611726, upper bound: 157.3606714
time: 7.39 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -74.8862152, 60.1164055, -77.7795639, 62.4227371, -137.3089600, 137.8959503
1: -63.0134392, 53.5835419, -65.4120560, 55.6155052, -118.6289444, 118.9955978
2: -82.7508545, 54.3607254, -85.9202652, 56.4255180, -139.1763763, 140.2809601
3: -87.8558426, 46.9584007, -91.2363358, 48.7425232, -136.5983582, 138.1947021
4: -80.6849899, 62.0542030, -83.7996750, 64.4244385, -145.1094360, 145.8538666
5: -72.3503113, 56.9977837, -75.1424255, 59.1720047, -131.5222931, 132.1402130
6: -69.2113190, 66.7864380, -71.8636856, 69.3499222, -138.5612488, 138.6501160
7: -75.4113541, 63.6583290, -78.2807007, 66.0750046, -141.4863586, 141.9390106
8: -90.9154053, 62.3680649, -94.3911438, 64.7966537, -155.7120361, 156.7592163
9: -68.7306671, 67.9209595, -71.3466415, 70.5049438, -139.2356110, 139.2675934

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 57

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 175

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3609990, upper bound: 157.3605371
time: 6.83 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3609990, upper bound: 157.3605371
time: 7.73 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -78.1367493, 62.7048111, -87.2851639, 69.9732132, -148.1099548, 149.9899750
1: -65.7824249, 55.9237175, -73.4571152, 62.3710403, -128.1534729, 129.3807983
2: -86.3761826, 56.6898041, -96.4661560, 63.2036781, -149.5798492, 153.1559296
3: -91.7086487, 49.0196266, -102.4657440, 54.6517754, -146.3604126, 151.4853668
4: -84.1846085, 64.7357941, -93.9948273, 72.2425308, -156.4271393, 158.7306213
5: -75.4873199, 59.4293785, -84.3326416, 66.2659454, -141.7532654, 143.7620239
6: -72.2159882, 69.7022247, -80.5807114, 77.8069077, -150.0228882, 150.2829285
7: -78.6893463, 66.3812561, -87.8425293, 73.9956741, -152.6850281, 154.2237396
8: -94.8564224, 65.0533371, -105.9275970, 72.7171249, -167.5735474, 170.9809265
9: -71.6531601, 70.8472214, -79.9167709, 78.9506760, -150.6037903, 150.7639771

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 57

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3280289, upper bound: 157.3285179
time: 8.34 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3612996, upper bound: 157.3616251
time: 7.01 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3613055, upper bound: 157.3616283
time: 7.23 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -77.4481659, 62.1555786, -88.5754929, 70.9830017, -148.4311676, 150.7310638
1: -65.1985092, 55.4326706, -74.5144958, 63.2575493, -128.4560394, 129.9471283
2: -85.6115875, 56.2004967, -97.8615036, 64.1130219, -149.7246094, 154.0619965
3: -90.8932343, 48.5883751, -103.9575119, 55.4196548, -146.3128815, 152.5458679
4: -83.4432983, 64.1690979, -95.3859558, 73.2851028, -156.7283936, 159.5550385
5: -74.8225403, 58.9162979, -85.5575562, 67.2267075, -142.0492249, 144.4738464
6: -71.5810623, 69.0883408, -81.7609711, 78.9323807, -150.5134430, 150.8493042
7: -77.9959641, 65.8041306, -89.1281891, 75.0778961, -153.0738373, 154.9323120
8: -94.0208817, 64.4865112, -107.4592514, 73.7690506, -167.7899170, 171.9457397
9: -71.0296783, 70.2294159, -81.0994492, 80.0873337, -151.1170044, 151.3288574

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 57

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3606689, upper bound: 157.3610751
time: 7.72 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3606732, upper bound: 157.3610754
time: 7.42 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -74.9167786, 60.1436768, -83.5648651, 67.0070724, -141.9238434, 143.7085266
1: -63.0465584, 53.6134644, -70.3000565, 59.7110138, -122.7575684, 123.9135056
2: -82.7912369, 54.3833961, -92.3302689, 60.5324249, -143.3236694, 146.7136688
3: -87.9027863, 46.9855461, -98.0629578, 52.3176041, -140.2203674, 145.0484772
4: -80.7169800, 62.0817871, -89.9877396, 69.1768494, -149.8938141, 152.0695190
5: -72.3839798, 57.0204201, -80.7407608, 63.4766312, -135.8606110, 137.7611847
6: -69.2422333, 66.8189926, -77.1324158, 74.4841537, -143.7263794, 143.9514160
7: -75.4465714, 63.6815681, -84.0896606, 70.8603363, -146.3069153, 147.7712250
8: -90.9600296, 62.3961792, -101.4227448, 69.6434402, -160.6034698, 163.8189087
9: -68.7508087, 67.9488754, -76.5325241, 75.5931168, -144.3439331, 144.4813843

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 44

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3206829, upper bound: 157.3210445
time: 7.62 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3595907, upper bound: 157.3593987
time: 6.67 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3596716, upper bound: 157.3594632
time: 6.27 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -74.2340393, 59.5988388, -84.8433228, 68.0054626, -142.2395020, 144.4421539
1: -62.4676743, 53.1263962, -71.3484268, 60.5904961, -123.0581589, 124.4748154
2: -82.0334167, 53.8974304, -93.7117081, 61.4341469, -143.4675293, 147.6091309
3: -87.0942917, 46.5579567, -99.5415802, 53.0819054, -140.1761932, 146.0995331
4: -79.9813995, 61.5197105, -91.3642273, 70.2109299, -150.1923065, 152.8839417
5: -71.7250443, 56.5099297, -81.9529724, 64.4286194, -136.1536560, 138.4628754
6: -68.6126251, 66.2094498, -78.3013077, 75.5979614, -144.2105865, 144.5107574
7: -74.7568588, 63.1087685, -85.3636703, 71.9337387, -146.6905975, 148.4724426
8: -90.1302338, 61.8334694, -102.9397430, 70.6837234, -160.8139191, 164.7732086
9: -68.1311569, 67.3346939, -77.7046738, 76.7193298, -144.8504944, 145.0393677

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3591462, upper bound: 157.3590706
time: 5.67 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3592269, upper bound: 157.3591196
time: 6.23 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -83.4639587, 66.9632874, -76.9108582, 61.7384872, -145.2024078, 143.8741455
1: -70.2895432, 59.7270470, -64.6914062, 55.0054245, -125.2949677, 124.4184341
2: -92.2975464, 60.5203476, -84.9703217, 55.7997322, -148.0972595, 145.4906616
3: -98.0061646, 52.3490334, -90.2236862, 48.2186966, -146.2248383, 142.5727081
4: -89.9307404, 69.1345901, -82.8577652, 63.7137985, -153.6445312, 151.9923553
5: -80.6422043, 63.4335556, -74.3139725, 58.5103531, -139.1525574, 137.7475128
6: -77.1575546, 74.4573975, -71.0599289, 68.5815048, -145.7390442, 145.5173035
7: -84.0604858, 70.8744659, -77.4044342, 65.3328857, -149.3933411, 148.2789001
8: -101.3179855, 69.4870148, -93.3544312, 64.0831528, -165.4011383, 162.8414459
9: -76.5093536, 75.6580963, -70.5449753, 69.7310181, -146.2403564, 146.2030640

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 57

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3406687, upper bound: 157.3408997
time: 7.30 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3358888, upper bound: 157.3359194
time: 6.89 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -82.6318130, 66.2994614, -78.3000412, 62.8293762, -145.4611359, 144.5995026
1: -69.5831985, 59.1321831, -65.8320236, 55.9611816, -125.5443802, 124.9642029
2: -91.3712997, 59.9263306, -86.4731827, 56.7787476, -148.1500549, 146.3994904
3: -97.0181351, 51.8272705, -91.8293991, 49.0483055, -146.0664215, 143.6566620
4: -89.0315704, 68.4488297, -84.3527298, 64.8383713, -153.8699341, 152.8015594
5: -79.8379822, 62.8124733, -75.6332016, 59.5484200, -139.3863831, 138.4456787
6: -76.3879852, 73.7134628, -72.3296890, 69.7968521, -146.1848450, 146.0431366
7: -83.2199097, 70.1757202, -78.7869492, 66.5019608, -149.7218628, 148.9626770
8: -100.3060608, 68.8002853, -95.0043793, 65.2156296, -165.5216980, 163.8046570
9: -75.7553482, 74.9092636, -71.8202362, 70.9613953, -146.7167358, 146.7294769

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 57

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3381463, upper bound: 157.3386322
time: 7.56 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3341345, upper bound: 157.3343468
time: 7.86 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -90.6449738, 72.6420517, -76.2218323, 61.1921692, -151.8371429, 148.8638611
1: -76.3298950, 64.7865372, -64.1156158, 54.5228577, -130.8527222, 128.9021606
2: -100.2215424, 65.6158218, -84.2124252, 55.3113823, -155.5329132, 149.8282318
3: -106.4554367, 56.7675743, -89.4203415, 47.7952957, -154.2507324, 146.1879120
4: -97.6203613, 75.0226212, -82.1147385, 63.1498833, -160.7702484, 157.1373596
5: -87.5676804, 68.7840500, -73.6540756, 57.9956322, -145.5633087, 142.4381256
6: -83.7058945, 80.8148499, -70.4277344, 67.9724426, -151.6783447, 151.2425537
7: -91.2617264, 76.8384857, -76.7136765, 64.7531662, -156.0148773, 153.5521393
8: -110.0111847, 75.4718323, -92.5248260, 63.5195656, -173.5307312, 167.9966583
9: -82.9779892, 81.9846497, -69.9126053, 69.1124420, -152.0903931, 151.8972473

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 57

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3387424, upper bound: 157.3388166
time: 8.32 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3267095, upper bound: 157.3256416
time: 6.15 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -89.8423386, 72.0026245, -77.6134033, 62.2849159, -152.1272583, 149.6160278
1: -75.6477585, 64.2124557, -65.2580872, 55.4802742, -131.1280365, 129.4705505
2: -99.3284988, 65.0421677, -85.7178726, 56.2918816, -155.6203766, 150.7600250
3: -105.5034561, 56.2623100, -91.0282288, 48.6271210, -154.1305542, 147.2905426
4: -96.7543640, 74.3603745, -83.6120682, 64.2760696, -161.0304260, 157.9724121
5: -86.7929916, 68.1840820, -74.9753571, 59.0351334, -145.8280945, 143.1594238
6: -82.9636993, 80.0974579, -71.6994781, 69.1897507, -152.1534424, 151.7969360
7: -90.4514465, 76.1636658, -78.0984879, 65.9241714, -156.3756104, 154.2621307
8: -109.0353165, 74.8103180, -94.1776733, 64.6536636, -173.6889801, 168.9879913
9: -82.2496948, 81.2626877, -71.1897659, 70.3449631, -152.5946350, 152.4524536

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3362909, upper bound: 157.3365707
time: 7.21 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3259219, upper bound: 157.3250217
time: 5.30 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -81.4989166, 65.3903046, -77.9519424, 62.5573044, -144.0562134, 143.3422241
1: -68.6370544, 58.3297081, -65.6128922, 55.7670288, -124.4040833, 123.9425964
2: -90.1242218, 59.1113129, -86.1564331, 56.5527115, -146.6769409, 145.2677460
3: -95.6889267, 51.1208687, -91.4881744, 48.8701553, -144.5590668, 142.6090393
4: -87.8177795, 67.5225830, -83.9887924, 64.5845108, -152.4022675, 151.5113831
5: -78.7402649, 61.9614105, -75.3109055, 59.2895050, -138.0297546, 137.2723083
6: -75.3430405, 72.7061539, -72.0547867, 69.5189133, -144.8619232, 144.7609100
7: -82.0960541, 69.2269058, -78.5082474, 66.2383804, -148.3344269, 147.7351227
8: -98.9478302, 67.8521729, -94.6424255, 64.9105072, -163.8583374, 162.4945374
9: -74.7261581, 73.8850327, -71.5192108, 70.6781769, -145.4043274, 145.4042358

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3622799, upper bound: 157.3627377
time: 7.95 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3614947, upper bound: 157.3625048
time: 8.16 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -81.4989166, 65.3903046, -84.9111862, 68.0622482, -149.5611572, 150.3014832
1: -68.6370544, 58.3297081, -71.4624939, 60.6680908, -129.3050995, 129.7922058
2: -90.1242218, 59.1113129, -93.8345184, 61.4847908, -151.6090088, 152.9458008
3: -95.6889267, 51.1208687, -99.6689758, 53.1416321, -148.8305206, 150.7898102
4: -87.8177795, 67.5225830, -91.4434814, 70.2827377, -158.1005249, 158.9660492
5: -78.7402649, 61.9614105, -82.0189133, 64.4682388, -143.2084808, 143.9803162
6: -75.3430405, 72.7061539, -78.3913116, 75.6800461, -151.0230560, 151.0974731
7: -82.0960541, 69.2269058, -85.4832993, 72.0116653, -154.1076965, 154.7101898
8: -98.9478302, 67.8521729, -103.0644073, 70.7056503, -169.6534729, 170.9165497
9: -74.7261581, 73.8850327, -77.7827072, 76.8029633, -151.5291138, 151.6677246

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 57

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3622799, upper bound: 157.3627377
time: 8.61 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3614947, upper bound: 157.3625048
time: 7.95 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -86.8444672, 69.6116791, -72.9137039, 58.5577698, -145.4022369, 142.5253754
1: -73.1062775, 62.0696602, -61.3023949, 52.1454277, -125.2517014, 123.3720322
2: -95.9966583, 62.8885498, -80.5261459, 52.9353218, -148.9319458, 143.4146729
3: -101.9556503, 54.3802643, -85.5094604, 45.7043343, -147.6599884, 139.8897247
4: -93.5280228, 71.8901672, -78.5505142, 60.4170036, -153.9450226, 150.4406738
5: -83.8990173, 65.9340897, -70.4619141, 55.5162811, -139.4152832, 136.3959808
6: -80.1816635, 77.4210739, -67.3701477, 65.0099945, -145.1916351, 144.7912140
7: -87.4293060, 73.6360626, -73.3792419, 61.9743156, -149.4035950, 147.0153046
8: -105.4091644, 72.3316803, -88.5193710, 60.7823143, -166.1914825, 160.8510437
9: -79.5214996, 78.5542374, -66.9261322, 66.1320953, -145.6535950, 145.4803772

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3349381, upper bound: 157.3354343
time: 7.90 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3251320, upper bound: 157.3244744
time: 5.95 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -86.0615997, 68.9867630, -74.0541077, 59.4494972, -145.5110931, 143.0408630
1: -72.4420242, 61.5107040, -62.2351074, 52.9267502, -125.3687668, 123.7458115
2: -95.1259308, 62.3295097, -81.7573624, 53.7380791, -148.8640137, 144.0868683
3: -101.0281219, 53.8891869, -86.8227310, 46.3839455, -147.4120636, 140.7119141
4: -92.6837921, 71.2449951, -79.7792358, 61.3375587, -154.0213470, 151.0242310
5: -83.1427383, 65.3487854, -71.5397797, 56.3680191, -139.5107574, 136.8885498
6: -79.4591370, 76.7211609, -68.4082260, 66.0042267, -145.4633484, 145.1293640
7: -86.6396484, 72.9788742, -74.5142593, 62.9348373, -149.5744934, 147.4931183
8: -104.4579544, 71.6844254, -89.8715057, 61.7098198, -166.1677704, 161.5559387
9: -78.8119736, 77.8502884, -67.9747543, 67.1369400, -145.9488831, 145.8250427

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 57

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3271296, upper bound: 157.3284561
time: 7.46 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3227384, upper bound: 157.3227384
time: 5.19 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 14.70 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 9, lower bound: -157.3628259, upper bound: 157.3629373
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 9, lower bound: -157.3628259, upper bound: 157.3629373
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 9, lower bound: -157.3624244, upper bound: 157.3625384
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 9, lower bound: -157.3624244, upper bound: 157.3625384
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 9, lower bound: -157.3611726, upper bound: 157.3606714
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 9, lower bound: -157.3611726, upper bound: 157.3606714
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 9, lower bound: -157.3609990, upper bound: 157.3605371
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 9, lower bound: -157.3609990, upper bound: 157.3605371
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 9, lower bound: -157.3612996, upper bound: 157.3616251
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 9, lower bound: -157.3613055, upper bound: 157.3616283
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 9, lower bound: -157.3606689, upper bound: 157.3610751
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 9, lower bound: -157.3606732, upper bound: 157.3610754
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 9, lower bound: -157.3595907, upper bound: 157.3593987
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 9, lower bound: -157.3596716, upper bound: 157.3594632
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 9, lower bound: -157.3591462, upper bound: 157.3590706
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 9, lower bound: -157.3592269, upper bound: 157.3591196
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 9, lower bound: -157.3406687, upper bound: 157.3408997
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 9, lower bound: -157.3358888, upper bound: 157.3359194
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 9, lower bound: -157.3381463, upper bound: 157.3386322
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 9, lower bound: -157.3341345, upper bound: 157.3343468
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 9, lower bound: -157.3387424, upper bound: 157.3388166
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 9, lower bound: -157.3267095, upper bound: 157.3256416
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 9, lower bound: -157.3362909, upper bound: 157.3365707
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 9, lower bound: -157.3259219, upper bound: 157.3250217
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 9, lower bound: -157.3622799, upper bound: 157.3627377
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 9, lower bound: -157.3614947, upper bound: 157.3625048
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 9, lower bound: -157.3622799, upper bound: 157.3627377
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 9, lower bound: -157.3614947, upper bound: 157.3625048
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 9, lower bound: -157.3349381, upper bound: 157.3354343
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 9, lower bound: -157.3251320, upper bound: 157.3244744
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 9, lower bound: -157.3271296, upper bound: 157.3284561
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 9, lower bound: -157.3227384, upper bound: 157.3227384

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -78.1908264, 62.7477951, -80.1510391, 64.3329926, -142.5238190, 142.8988342
1: -65.8279877, 55.9613228, -67.4578247, 57.3465271, -123.1745148, 123.4191360
2: -86.4359741, 56.7276726, -88.5964737, 58.1435127, -144.5794678, 145.3241425
3: -91.7708664, 49.0526047, -94.0748749, 50.2665024, -142.0373688, 143.1274567
4: -84.2430344, 64.7798462, -86.3560410, 66.3957825, -150.6388092, 151.1358643
5: -75.5395584, 59.4692421, -77.4537201, 60.9524422, -136.4920044, 136.9229584
6: -72.2655334, 69.7502213, -74.0779037, 71.4932632, -143.7587891, 143.8281250
7: -78.7434006, 66.4265900, -80.6901169, 68.0725937, -146.8159943, 147.1166687
8: -94.9214172, 65.0970001, -97.2930679, 66.7726593, -161.6940308, 162.3900452
9: -71.7024231, 70.8954544, -73.4906540, 72.6686783, -144.3710938, 144.3861084

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 57

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3628259, upper bound: 157.3629373
time: 7.59 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3628259, upper bound: 157.3629373
time: 7.27 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -85.2652283, 68.3450165, -80.1510391, 64.3329926, -149.5982208, 148.4960480
1: -71.7790146, 60.9489250, -67.4578247, 57.3465271, -129.1255493, 128.4067078
2: -94.2446594, 61.7475395, -88.5964737, 58.1435127, -152.3881683, 150.3440094
3: -100.0973892, 53.4001045, -94.0748749, 50.2665024, -150.3638916, 147.4749756
4: -91.8241272, 70.5779572, -86.3560410, 66.3957825, -158.2199097, 156.9339905
5: -82.3629761, 64.7383270, -77.4537201, 60.9524422, -143.3154144, 142.1920471
6: -78.7139282, 76.0165253, -74.0779037, 71.4932632, -150.2071838, 150.0944214
7: -85.8418884, 72.3003693, -80.6901169, 68.0725937, -153.9144592, 152.9904785
8: -103.4857483, 70.9892578, -97.2930679, 66.7726593, -170.2584076, 168.2823181
9: -78.0742035, 77.1273499, -73.4906540, 72.6686783, -150.7428894, 150.6179810

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 57

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3628259, upper bound: 157.3629373
time: 8.68 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3628259, upper bound: 157.3629373
time: 7.77 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -77.5028229, 62.1990318, -81.4627151, 65.3600311, -142.8628387, 143.6617126
1: -65.2445526, 55.4706764, -68.5339432, 58.2464867, -123.4910355, 124.0046005
2: -85.6720657, 56.2387276, -90.0130081, 59.0676460, -144.7397156, 146.2517242
3: -90.9561844, 48.6216583, -95.5886383, 51.0491829, -142.0053558, 144.2102966
4: -83.5023651, 64.2136536, -87.7661743, 67.4564056, -150.9587708, 151.9798279
5: -74.8753738, 58.9565926, -78.6972275, 61.9310074, -136.8063812, 137.6538239
6: -71.6311188, 69.1368866, -75.2753906, 72.6380844, -144.2691956, 144.4122772
7: -78.0506058, 65.8499451, -81.9940567, 69.1761017, -147.2266998, 147.8439636
8: -94.0865555, 64.5306320, -98.8496399, 67.8398285, -161.9263611, 163.3802643
9: -71.0794830, 70.2781982, -74.6957550, 73.8270569, -144.9065094, 144.9739532

Time for backsubstitution: 1.90 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 12.45 + 588.10 = 600.55 seconds
