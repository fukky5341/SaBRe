## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 143.61867486269998


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766)
1: (-62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901)
2: (-83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913)
3: (-88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352)
4: (-81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725)
5: (-73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507)
6: (-70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321)
7: (-76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043)
8: (-92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513)
9: (-69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.96 + 10.35 = 11.31 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -143.7624373, upper bound: 143.7624373

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 204

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7309054, upper bound: 143.7321927
time: 6.84 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7601942, upper bound: 143.7601942
time: 6.09 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 13.02 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 13.02
Output dim: 4, lower bound: -143.7309054, upper bound: 143.7321927
NS_A2, status: Status.UNKNOWN, split count: 1, time: 13.02
Output dim: 4, lower bound: -143.7601942, upper bound: 143.7601942

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -55.0390320, 43.7957993, -66.5466003, 53.0615616, -108.1005936, 110.3423996
1: -45.1367798, 38.8839874, -54.9268608, 47.0518150, -92.1885986, 93.8108368
2: -60.0479012, 39.4024048, -72.8657990, 47.6361923, -107.6840973, 112.2681885
3: -64.2611771, 33.9397697, -77.6831360, 41.0598679, -105.3210297, 111.6228943
4: -59.2569656, 45.5209084, -71.7174225, 55.1389694, -114.3959274, 117.2383270
5: -52.9686508, 41.2229843, -64.0385284, 49.8615379, -102.8301849, 105.2615051
6: -50.8919487, 48.3378906, -61.5194969, 58.5896835, -109.4816208, 109.8573914
7: -54.9219017, 46.2777328, -66.4840088, 56.0085411, -110.9304199, 112.7617416
8: -66.3510666, 45.8323746, -80.4472275, 55.7119560, -122.0630188, 126.2795944
9: -49.9999199, 48.8834915, -60.6247635, 59.4725571, -109.4724731, 109.5082550

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 62

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7298914, upper bound: 143.7314227
time: 6.55 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7302794, upper bound: 143.7318708
time: 5.97 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -71.1890564, 56.7948189, -75.1077347, 59.9236641, -131.1127167, 131.9025574
1: -58.8822021, 50.3534431, -62.2158775, 53.1305351, -112.0127258, 112.5693207
2: -78.0392151, 50.9567719, -82.3876419, 53.7504921, -131.7897034, 133.3444214
3: -83.1072540, 43.9345856, -87.6848450, 46.3555756, -129.4627991, 131.6194153
4: -76.7635193, 59.0121994, -81.0013504, 62.2556610, -139.0191650, 140.0135498
5: -68.5076675, 53.3455086, -72.2776184, 56.2717133, -124.7793732, 125.6231232
6: -65.8120346, 62.7355423, -69.4315338, 66.2258224, -132.0378571, 132.1670380
7: -71.1526871, 59.9258461, -75.0897903, 63.2329407, -134.3856201, 135.0156250
8: -86.1432419, 59.6953735, -90.9099884, 62.9922180, -149.1354370, 150.6053619
9: -64.9082565, 63.7381058, -68.5168076, 67.3122177, -132.2204437, 132.2549133

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 62

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7321927, upper bound: 143.7309054
time: 7.13 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7321927, upper bound: 143.7601942
time: 7.03 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 15.11 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 15.11
Output dim: 4, lower bound: -143.7298914, upper bound: 143.7314227
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 15.11
Output dim: 4, lower bound: -143.7302794, upper bound: 143.7318708
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 15.11
Output dim: 4, lower bound: -143.7321927, upper bound: 143.7309054
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 15.11
Output dim: 4, lower bound: -143.7321927, upper bound: 143.7601942

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -52.4630661, 41.7479744, -59.7987213, 47.7020187, -100.1650848, 101.5466919
1: -42.9811211, 37.0672035, -49.2404518, 42.2870560, -85.2681732, 86.3076477
2: -57.1989441, 37.5742035, -65.4041901, 42.8292656, -100.0281982, 102.9783936
3: -61.2084389, 32.3592262, -69.6936951, 36.9207802, -98.1292038, 102.0529175
4: -56.4685020, 43.4005089, -64.4042282, 49.5774651, -106.0459671, 107.8047333
5: -50.4856834, 39.2858810, -57.5255852, 44.7901154, -95.2758026, 96.8114548
6: -48.5134201, 46.0617714, -55.2779999, 52.6047096, -101.1181335, 101.3397675
7: -52.3183937, 44.0906792, -59.6681061, 50.2745705, -102.5929642, 103.7587891
8: -63.2537537, 43.7324257, -72.3300247, 50.2121468, -113.4658890, 116.0624542
9: -47.6405449, 46.5662498, -54.4366570, 53.3768425, -101.0173798, 101.0029068

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 170

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7151543, upper bound: 143.7163229
time: 6.95 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7075591, upper bound: 143.7094009
time: 7.05 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -53.2226562, 42.3497086, -64.4436798, 51.3783150, -104.6009674, 106.7933884
1: -43.6156731, 37.6029320, -53.1331902, 45.5598488, -89.1755219, 90.7361221
2: -58.0376892, 38.1094818, -70.5356979, 46.1153831, -104.1530609, 108.6451797
3: -62.1111984, 32.8252373, -75.2054062, 39.7753792, -101.8865662, 108.0306396
4: -57.2914085, 44.0249977, -69.4493866, 53.3966331, -110.6880417, 113.4743576
5: -51.2179146, 39.8543930, -62.0048523, 48.2427597, -99.4606781, 101.8592453
6: -49.2144394, 46.7319641, -59.6036148, 56.7200623, -105.9345016, 106.3355789
7: -53.0871620, 44.7347298, -64.3515396, 54.1869621, -107.2741241, 109.0862732
8: -64.1682358, 44.3508186, -77.9461441, 54.0108109, -118.1790466, 122.2969589
9: -48.3364792, 47.2486191, -58.6798477, 57.5549507, -105.8914337, 105.9284592

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 124

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6999193, upper bound: 143.7020883
time: 6.63 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7275078, upper bound: 143.7289782
time: 7.70 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -71.1890564, 56.7948189, -55.0390320, 43.7957993, -114.9848557, 111.8338470
1: -58.8822021, 50.3534431, -45.1367798, 38.8839874, -97.7661743, 95.4902191
2: -78.0392151, 50.9567719, -60.0479012, 39.4024048, -117.4416199, 111.0046692
3: -83.1072540, 43.9345856, -64.2611771, 33.9397697, -117.0470047, 108.1957626
4: -76.7635193, 59.0121994, -59.2569656, 45.5209084, -122.2844238, 118.2691650
5: -68.5076675, 53.3455086, -52.9686508, 41.2229843, -109.7306366, 106.3141556
6: -65.8120346, 62.7355423, -50.8919487, 48.3378906, -114.1499252, 113.6274796
7: -71.1526871, 59.9258461, -54.9219017, 46.2777328, -117.4304199, 114.8477478
8: -86.1432419, 59.6953735, -66.3510666, 45.8323746, -131.9756165, 126.0464172
9: -64.9082565, 63.7381058, -49.9999199, 48.8834915, -113.7917480, 113.7380142

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 62

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7314227, upper bound: 143.7298914
time: 6.99 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7318708, upper bound: 143.7302794
time: 7.93 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -71.1890564, 56.7948189, -71.1890564, 56.7948189, -127.9838715, 127.9838715
1: -58.8822021, 50.3534431, -58.8822021, 50.3534431, -109.2356415, 109.2356415
2: -78.0392151, 50.9567719, -78.0392151, 50.9567719, -128.9959717, 128.9959717
3: -83.1072540, 43.9345856, -83.1072540, 43.9345856, -127.0418396, 127.0418396
4: -76.7635193, 59.0121994, -76.7635193, 59.0121994, -135.7757111, 135.7757111
5: -68.5076675, 53.3455086, -68.5076675, 53.3455086, -121.8531647, 121.8531647
6: -65.8120346, 62.7355423, -65.8120346, 62.7355423, -128.5475616, 128.5475769
7: -71.1526871, 59.9258461, -71.1526871, 59.9258461, -131.0785065, 131.0785065
8: -86.1432419, 59.6953735, -86.1432419, 59.6953735, -145.8386078, 145.8386078
9: -64.9082565, 63.7381058, -64.9082565, 63.7381058, -128.6463165, 128.6463318

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 62

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7314227, upper bound: 143.7593562
time: 8.48 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7318708, upper bound: 143.7302794
time: 7.20 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 16.50 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 16.50
Output dim: 4, lower bound: -143.7151543, upper bound: 143.7163229
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 16.50
Output dim: 4, lower bound: -143.7075591, upper bound: 143.7094009
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 16.50
Output dim: 4, lower bound: -143.6999193, upper bound: 143.7020883
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 16.50
Output dim: 4, lower bound: -143.7275078, upper bound: 143.7289782
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 16.50
Output dim: 4, lower bound: -143.7314227, upper bound: 143.7298914
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 16.50
Output dim: 4, lower bound: -143.7318708, upper bound: 143.7302794
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 16.50
Output dim: 4, lower bound: -143.7314227, upper bound: 143.7593562
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 16.50
Output dim: 4, lower bound: -143.7318708, upper bound: 143.7302794

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -45.9935989, 36.5555992, -58.4043312, 46.5854378, -92.5790329, 94.9599228
1: -37.6308098, 32.5312347, -48.0841179, 41.3114624, -78.9422684, 80.6153488
2: -50.0427094, 32.9890213, -63.8654938, 41.8405190, -91.8832245, 96.8545151
3: -53.6136589, 28.3991184, -68.0591812, 36.0650711, -89.6787262, 96.4582901
4: -49.5008774, 38.0955772, -62.8986588, 48.4329147, -97.9337769, 100.9942322
5: -44.2320595, 34.4070702, -56.1753159, 43.7409706, -87.9730301, 90.5823746
6: -42.5677986, 40.3746834, -53.9939766, 51.3776779, -93.9454803, 94.3686600
7: -45.8449287, 38.6427727, -58.2737122, 49.1030998, -94.9480286, 96.9164886
8: -55.4129486, 38.2841339, -70.6462631, 49.0360031, -104.4489517, 108.9303970
9: -41.7236671, 40.7408638, -53.1621552, 52.1186600, -93.8423233, 93.9030151

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 170

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7054899, upper bound: 143.7065604
time: 6.58 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7046661, upper bound: 143.7056128
time: 7.49 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -43.4460564, 34.4741364, -54.4698868, 43.4343376, -86.8803940, 88.9440155
1: -35.5910645, 30.8079872, -44.8184471, 38.5607834, -74.1518478, 75.6264343
2: -47.2726555, 31.2106819, -59.5253143, 39.0560341, -86.3286896, 90.7359924
3: -50.6639938, 26.8489437, -63.4468231, 33.6460381, -84.3100281, 90.2957687
4: -46.7762146, 36.0429611, -58.6549301, 45.2048416, -91.9810562, 94.6978836
5: -41.7731209, 32.4716110, -52.3661575, 40.7828751, -82.5559921, 84.8377533
6: -40.2856483, 38.1675186, -50.3732910, 47.9145889, -88.2002411, 88.5408096
7: -43.3887253, 36.5078278, -54.3480453, 45.8007393, -89.1894684, 90.8558731
8: -52.3644066, 36.0811234, -65.8906174, 45.7184563, -98.0828629, 101.9717407
9: -39.3942757, 38.4688148, -49.5720329, 48.5775070, -87.9717789, 88.0408478

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 170

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6971930, upper bound: 143.6991671
time: 6.67 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6964895, upper bound: 143.6983620
time: 6.39 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -41.4439125, 32.8915558, -57.8386116, 46.1111145, -87.5550232, 90.7301636
1: -33.6844254, 29.2529831, -47.4816513, 40.8744125, -74.5588379, 76.7346344
2: -44.9570274, 29.8077965, -63.2339287, 41.4430504, -86.4000626, 93.0417023
3: -48.4337997, 25.5509300, -67.5257263, 35.6746063, -84.1084061, 93.0766525
4: -44.6591225, 34.1274338, -62.3276253, 47.8982506, -92.5573654, 96.4550476
5: -39.9627800, 31.0465927, -55.6648254, 43.3229408, -83.2856903, 86.7114182
6: -38.3584099, 36.2198524, -53.4899063, 50.8388443, -89.1972504, 89.7097626
7: -41.2908134, 34.7623444, -57.7251053, 48.6326561, -89.9234619, 92.4874496
8: -49.6284103, 34.1885834, -69.8791122, 48.3699913, -97.9983749, 104.0676956
9: -37.5218468, 36.5535126, -52.6237984, 51.5349693, -89.0568161, 89.1772995

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 170

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6913326, upper bound: 143.6935048
time: 7.08 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6910732, upper bound: 143.6930939
time: 6.76 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -49.4361382, 39.3283501, -63.5739861, 50.6862526, -100.1223907, 102.9023285
1: -40.3988457, 34.9173279, -52.3888474, 44.9433517, -85.3421631, 87.3061752
2: -53.8451576, 35.4441833, -69.5738449, 45.5004997, -99.3456573, 105.0180283
3: -57.7029266, 30.4809589, -74.1940994, 39.2354317, -96.9383545, 104.6750488
4: -53.2253876, 40.8817902, -68.5109482, 52.6750145, -105.9004059, 109.3927383
5: -47.5984459, 37.0325584, -61.1705742, 47.5958748, -95.1943054, 98.2031326
6: -45.7271576, 43.3682671, -58.7982903, 55.9444008, -101.6715469, 102.1665573
7: -49.2919807, 41.5468140, -63.4800262, 53.4560776, -102.7480545, 105.0268402
8: -59.5382233, 41.1365356, -76.8845673, 53.2723045, -112.8105316, 118.0211029
9: -44.8688011, 43.8189430, -57.8819809, 56.7633095, -101.6321106, 101.7009277

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 170

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7213973, upper bound: 143.7223491
time: 6.53 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7211889, upper bound: 143.7220855
time: 6.53 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -64.3955994, 51.3998566, -52.4630661, 41.7479744, -106.1435699, 103.8629227
1: -53.1505241, 45.5554619, -42.9811211, 37.0672035, -90.2177048, 88.5365829
2: -70.5260239, 46.1158981, -57.1989441, 37.5742035, -108.1002274, 103.3148422
3: -75.0708618, 39.7683411, -61.2084389, 32.3592262, -107.4300842, 100.9767761
4: -69.4003906, 53.4169083, -56.4685020, 43.4005089, -112.8009033, 109.8854065
5: -61.9543114, 48.2407532, -50.4856834, 39.2858810, -101.2401733, 98.7264404
6: -59.5268135, 56.7059937, -48.5134201, 46.0617714, -105.5885773, 105.2193985
7: -64.2931061, 54.1554527, -52.3183937, 44.0906792, -108.3837891, 106.4738388
8: -77.9704895, 54.1610222, -63.2537537, 43.7324257, -121.7029114, 117.4147644
9: -58.6758537, 57.6022644, -47.6405449, 46.5662498, -105.2421036, 105.2428131

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 170

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7163229, upper bound: 143.7151543
time: 6.87 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7094009, upper bound: 143.7075591
time: 6.98 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -69.0568771, 55.0875969, -53.2226562, 42.3497086, -111.4065857, 108.3102570
1: -57.0652199, 48.8404388, -43.6156731, 37.6029320, -94.6681519, 92.4561157
2: -75.6762848, 49.4146347, -58.0376892, 38.1094818, -113.7857590, 107.4523087
3: -80.5944138, 42.6306839, -62.1111984, 32.8252373, -113.4196472, 104.7418671
4: -74.4635086, 57.2453079, -57.2914085, 44.0249977, -118.4884872, 114.5367126
5: -66.4463730, 51.7040939, -51.2179146, 39.8543930, -106.3007660, 102.9219971
6: -63.8677101, 60.8395119, -49.2144394, 46.7319641, -110.5996704, 110.0539551
7: -68.9898224, 58.0798111, -53.0871620, 44.7347298, -113.7245483, 111.1669617
8: -83.6032639, 57.9682922, -64.1682358, 44.3508186, -127.9540863, 122.1365280
9: -62.9355888, 61.7938652, -48.3364792, 47.2486191, -110.1842041, 110.1303406

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 124

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7020883, upper bound: 143.6999192
time: 7.84 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7289782, upper bound: 143.7275078
time: 7.17 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -64.3955994, 51.3998566, -68.5622406, 54.7090263, -119.1046143, 119.9620895
1: -53.1505241, 45.5554619, -56.6673698, 48.4988251, -101.6493454, 102.2228012
2: -70.5260239, 46.1158981, -75.1357193, 49.0862579, -119.6122818, 121.2516174
3: -75.0708618, 39.7683411, -80.0010529, 42.3231354, -117.3939972, 119.7693787
4: -69.4003906, 53.4169083, -73.9177475, 56.8490906, -126.2494812, 127.3346558
5: -61.9543114, 48.2407532, -65.9739075, 51.3721352, -113.3264389, 114.2146530
6: -59.5268135, 56.7059937, -63.3822403, 60.4051819, -119.9319916, 120.0882111
7: -64.2931061, 54.1554527, -68.5009384, 57.6956787, -121.9887848, 122.6563873
8: -77.9704895, 54.1610222, -82.9842987, 57.5550499, -135.5255280, 137.1453094
9: -58.6758537, 57.6022644, -62.4976196, 61.3661995, -120.0420532, 120.0998840

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 62

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7336745, upper bound: 143.7322686
time: 8.00 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7575539, upper bound: 143.7575603
time: 6.99 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -69.0568771, 55.0875969, -69.3172607, 55.3071480, -124.3640213, 124.4048538
1: -57.0652199, 48.8404388, -57.3038521, 49.0317345, -106.0969543, 106.1442871
2: -75.6762848, 49.4146347, -75.9700775, 49.6213531, -125.2976379, 125.3847046
3: -80.5944138, 42.6306839, -80.8971863, 42.7850494, -123.3794632, 123.5278702
4: -74.4635086, 57.2453079, -74.7374725, 57.4715385, -131.9350433, 131.9827576
5: -66.4463730, 51.7040939, -66.7030487, 51.9362106, -118.3825836, 118.4071426
6: -63.8677101, 60.8395119, -64.0827332, 61.0748749, -124.9425812, 124.9222412
7: -68.9898224, 58.0798111, -69.2643433, 58.3360596, -127.3258820, 127.3441467
8: -83.6032639, 57.9682922, -83.8940277, 58.1700325, -141.7732849, 141.8623199
9: -62.9355888, 61.7938652, -63.1913757, 62.0476265, -124.9832077, 124.9852448

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 62

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7338165, upper bound: 143.7324253
time: 6.81 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7581921, upper bound: 143.7581921
time: 6.12 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 13.76 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 13.76
Output dim: 4, lower bound: -143.7054899, upper bound: 143.7065604
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.76
Output dim: 4, lower bound: -143.7046661, upper bound: 143.7056128
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 13.76
Output dim: 4, lower bound: -143.6971930, upper bound: 143.6991671
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 13.76
Output dim: 4, lower bound: -143.6964895, upper bound: 143.6983620
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 13.76
Output dim: 4, lower bound: -143.6913326, upper bound: 143.6935048
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.76
Output dim: 4, lower bound: -143.6910732, upper bound: 143.6930939
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 13.76
Output dim: 4, lower bound: -143.7213973, upper bound: 143.7223491
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 13.76
Output dim: 4, lower bound: -143.7211889, upper bound: 143.7220855
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 13.76
Output dim: 4, lower bound: -143.7163229, upper bound: 143.7151543
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.76
Output dim: 4, lower bound: -143.7094009, upper bound: 143.7075591
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 13.76
Output dim: 4, lower bound: -143.7020883, upper bound: 143.6999192
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 13.76
Output dim: 4, lower bound: -143.7289782, upper bound: 143.7275078
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 13.76
Output dim: 4, lower bound: -143.7336745, upper bound: 143.7322686
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.76
Output dim: 4, lower bound: -143.7575539, upper bound: 143.7575603
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 13.76
Output dim: 4, lower bound: -143.7338165, upper bound: 143.7324253
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 13.76
Output dim: 4, lower bound: -143.7581921, upper bound: 143.7581921

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -44.3410263, 35.2415619, -51.7553368, 41.3197021, -85.6607132, 86.9968872
1: -36.2642288, 31.3714161, -42.5714417, 36.6612930, -72.9255219, 73.9428558
2: -48.2143402, 31.8263130, -56.5398445, 37.1597214, -85.3740616, 88.3661499
3: -51.6727028, 27.3828793, -60.2824478, 31.9713326, -83.6440277, 87.6653061
4: -47.7253265, 36.7502480, -55.7478676, 43.0169220, -90.7422409, 92.4981079
5: -42.6449127, 33.1760445, -49.7815514, 38.8112831, -81.4561920, 82.9575958
6: -41.0629387, 38.9201469, -47.9297371, 45.5195045, -86.5824432, 86.8498840
7: -44.1826591, 37.2543831, -51.6040916, 43.5387154, -87.7213745, 88.8584747
8: -53.4243584, 36.9314384, -62.6933403, 43.5734482, -96.9978027, 99.6247787
9: -40.2177315, 39.2462502, -47.1069794, 46.1018524, -86.3195801, 86.3532257

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 124

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7042295, upper bound: 143.7054575
time: 7.46 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7054899, upper bound: 143.7065604
time: 8.20 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -42.4066162, 33.7027130, -52.9135094, 42.2301407, -84.6367569, 86.6162262
1: -34.6638947, 30.0208244, -43.5264130, 37.4753304, -72.1392212, 73.5472412
2: -46.0813789, 30.4722042, -57.8280106, 37.9819565, -84.0633316, 88.3002014
3: -49.4036217, 26.1940823, -61.6477661, 32.6487389, -82.0523605, 87.8418503
4: -45.6596451, 35.1765976, -57.0095673, 43.9741402, -89.6337891, 92.1861649
5: -40.7842636, 31.7400169, -50.8906136, 39.6631279, -80.4473801, 82.6306305
6: -39.3082504, 37.2230415, -49.0366669, 46.5461578, -85.8544083, 86.2597046
7: -42.2585945, 35.6354637, -52.7877464, 44.5004005, -86.7589951, 88.4232101
8: -51.1012573, 35.3423767, -64.1317596, 44.5276680, -95.6289215, 99.4741287
9: -38.4600868, 37.5059052, -48.1405182, 47.1125565, -85.5726318, 85.6464233

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 124

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7034740, upper bound: 143.7045417
time: 7.23 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7046661, upper bound: 143.7056128
time: 7.32 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -41.8223763, 33.1845131, -47.8583946, 38.1968689, -80.0192413, 81.0428925
1: -34.2527237, 29.6719284, -39.3450203, 33.9389496, -68.1916733, 69.0169449
2: -45.4852104, 30.0740948, -52.2370605, 34.4022903, -79.8874969, 82.3111572
3: -48.7570190, 25.8538132, -55.7115517, 29.5747929, -78.3318100, 81.5653687
4: -45.0410728, 34.7212372, -51.5395470, 39.8197823, -84.8608551, 86.2607880
5: -40.2120743, 31.2674789, -46.0124435, 35.8766632, -76.0887375, 77.2799072
6: -38.8123817, 36.7437172, -44.3416214, 42.0950470, -80.9074249, 81.0853348
7: -41.7669067, 35.1480827, -47.7132301, 40.2730637, -82.0399551, 82.8613052
8: -50.4178314, 34.7515755, -57.9735985, 40.2836189, -90.7014465, 92.7251740
9: -37.9179611, 37.0046692, -43.5547104, 42.5985832, -80.5165405, 80.5593796

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 247

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6956781, upper bound: 143.6977922
time: 7.99 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6971930, upper bound: 143.6991671
time: 6.37 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -39.9080925, 31.6650162, -49.0448761, 39.1296158, -79.0376740, 80.7098923
1: -32.6758652, 28.3385372, -40.3242607, 34.7723389, -67.4482040, 68.6627960
2: -43.3804474, 28.7359829, -53.5563087, 35.2460403, -78.6264877, 82.2922897
3: -46.5109787, 24.6804733, -57.1091232, 30.2713127, -76.7822800, 81.7895889
4: -43.0035400, 33.1633415, -52.8304825, 40.8012886, -83.8048248, 85.9938049
5: -38.3742905, 29.8509998, -47.1486893, 36.7502251, -75.1245041, 76.9996872
6: -37.0784264, 35.0670509, -45.4766464, 43.1466484, -80.2250748, 80.5437012
7: -39.8638191, 33.5487366, -48.9237633, 41.2571869, -81.1209946, 82.4725037
8: -48.1195145, 33.1815720, -59.4465904, 41.2609787, -89.3804932, 92.6281586
9: -36.1808662, 35.2840576, -44.6131630, 43.6336517, -79.8145142, 79.8972168

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 247

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6948376, upper bound: 143.6968761
time: 6.95 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6964895, upper bound: 143.6983620
time: 6.23 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -39.8978348, 31.6644955, -51.0559540, 40.7383080, -80.6361389, 82.7204361
1: -32.4169693, 28.1772938, -41.8666382, 36.1302986, -68.5472717, 70.0439301
2: -43.2516022, 28.7290726, -55.7571907, 36.6686287, -79.9202271, 84.4862518
3: -46.6128273, 24.6042709, -59.5895424, 31.4987411, -78.1115723, 84.1937943
4: -43.0040932, 32.8697929, -55.0251007, 42.3754234, -85.3794937, 87.8948822
5: -38.4751892, 29.8950233, -49.1425056, 38.2964859, -76.7716751, 79.0375290
6: -36.9613533, 34.8591309, -47.2978554, 44.8657951, -81.8271484, 82.1569672
7: -39.7422409, 33.4610977, -50.9169312, 42.9593544, -82.7015991, 84.3780289
8: -47.7735710, 32.9267731, -61.7652130, 42.7953568, -90.5689240, 94.6919785
9: -36.1159935, 35.1643639, -46.4494553, 45.4092560, -81.5252457, 81.6138153

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 247

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6868254, upper bound: 143.6895882
time: 6.40 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6912868, upper bound: 143.6935044
time: 6.70 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -38.1361618, 30.2650127, -52.9844398, 42.2617874, -80.3979492, 83.2494507
1: -30.9648266, 26.9533005, -43.4646988, 37.4899788, -68.4547882, 70.4179993
2: -41.3066978, 27.5008259, -57.8994522, 38.0453987, -79.3520813, 85.4002762
3: -44.5372467, 23.5165596, -61.8678551, 32.6458549, -77.1830978, 85.3844070
4: -41.1234818, 31.4359550, -57.1230812, 43.9693336, -85.0928116, 88.5590210
5: -36.7819595, 28.5819283, -50.9955215, 39.7234497, -76.5054016, 79.5774536
6: -35.3704910, 33.3068161, -49.1226845, 46.5742836, -81.9447708, 82.4294968
7: -37.9841423, 31.9807224, -52.8861008, 44.5748100, -82.5589447, 84.8668213
8: -45.6521187, 31.4824505, -64.1416779, 44.3880959, -90.0402145, 95.6241150
9: -34.5173416, 33.5850677, -48.1901169, 47.1143799, -81.6317215, 81.7751770

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6817123, upper bound: 143.6840613
time: 6.43 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6786103, upper bound: 143.6815764
time: 6.52 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -47.7249107, 37.9698792, -56.7590370, 45.2903671, -93.0152740, 94.7289124
1: -38.9887657, 33.7183952, -46.7303619, 40.1732330, -79.1619797, 80.4487610
2: -51.9560585, 34.2417297, -62.0631981, 40.6971855, -92.6532211, 96.3049088
3: -55.6978493, 29.4293423, -66.2212601, 35.0413551, -90.7392044, 95.6505890
4: -51.3886604, 39.4905548, -61.1822090, 47.1216545, -98.5103149, 100.6727600
5: -45.9558220, 35.7598686, -54.6143303, 42.5419769, -88.4978027, 90.3741989
6: -44.1714096, 41.8622818, -52.5792923, 49.9347649, -94.1061707, 94.4415741
7: -47.5722847, 40.1094551, -56.6409531, 47.7495689, -95.3218460, 96.7504044
8: -57.4853592, 39.7367706, -68.7277756, 47.6743469, -105.1596909, 108.4645462
9: -43.3092270, 42.2753258, -51.6735573, 50.5926247, -93.9018478, 93.9488754

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 170

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7043326, upper bound: 143.7055757
time: 7.82 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6958496, upper bound: 143.6979731
time: 6.72 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -45.8621826, 36.4915314, -58.7090836, 46.8305817, -92.6927643, 95.2006073
1: -37.4643440, 32.4228134, -48.3473511, 41.5484619, -79.0127945, 80.7701416
2: -49.9011803, 32.9342079, -64.2292175, 42.0882950, -91.9894714, 97.1634216
3: -53.5188866, 28.2891350, -68.5236816, 36.2032127, -89.7220840, 96.8128052
4: -49.3991470, 37.9761314, -63.3030663, 48.7333183, -98.1324615, 101.2791901
5: -44.1697884, 34.3745918, -56.4879837, 43.9850769, -88.1548615, 90.8625565
6: -42.4814110, 40.2267570, -54.4202309, 51.6639633, -94.1453705, 94.6469879
7: -45.7138443, 38.5473862, -58.6299324, 49.3831520, -95.0970001, 97.1773224
8: -55.2453346, 38.2104988, -71.1294403, 49.2857590, -104.5310974, 109.3399353
9: -41.6152229, 40.6047783, -53.4330978, 52.3209381, -93.9361572, 94.0378723

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 170

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7040525, upper bound: 143.7051641
time: 7.63 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6956051, upper bound: 143.6976446
time: 6.66 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -62.9806671, 50.2675591, -45.9935989, 36.5555992, -99.5362701, 96.2611542
1: -51.9744339, 44.5647278, -37.6308098, 32.5312347, -84.5056534, 82.1955414
2: -68.9632339, 45.1115265, -50.0427094, 32.9890213, -101.9522552, 95.1542282
3: -73.4121475, 38.8999405, -53.6136589, 28.3991184, -101.8112564, 92.5135956
4: -67.8710938, 52.2557068, -49.5008774, 38.0955772, -105.9666672, 101.7565842
5: -60.5834465, 47.1772270, -44.2320595, 34.4070702, -94.9905090, 91.4092636
6: -58.2222786, 55.4578094, -42.5677986, 40.3746834, -98.5969620, 98.0256042
7: -62.8778915, 52.9667587, -45.8449287, 38.6427727, -101.5206528, 98.8116837
8: -76.2604752, 52.9681320, -55.4129486, 38.2841339, -114.5446091, 108.3810806
9: -57.3816986, 56.3243294, -41.7236671, 40.7408638, -98.1225586, 98.0479965

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 62

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6872171, upper bound: 143.6873788
time: 6.36 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7131086, upper bound: 143.7119664
time: 8.25 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -59.0787354, 47.1448097, -43.4460564, 34.4741364, -93.5528641, 90.5908661
1: -48.7322807, 41.8348923, -35.5910645, 30.8079872, -79.5402679, 77.4259567
2: -64.6597443, 42.3477058, -47.2726555, 31.2106819, -95.8704224, 89.6203613
3: -68.8400192, 36.4999161, -50.6639938, 26.8489437, -95.6889572, 87.1639099
4: -63.6620407, 49.0566978, -46.7762146, 36.0429611, -99.7049942, 95.8329086
5: -56.8044777, 44.2448769, -41.7731209, 32.4716110, -89.2760773, 86.0179825
6: -54.6297646, 52.0200996, -40.2856483, 38.1675186, -92.7972870, 92.3057480
7: -58.9847374, 49.6922684, -43.3887253, 36.5078278, -95.4925690, 93.0809937
8: -71.5432358, 49.6797180, -52.3644066, 36.0811234, -107.6243591, 102.0441284
9: -53.8210335, 52.8126030, -39.3942757, 38.4688148, -92.2898331, 92.2068787

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 170

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6991671, upper bound: 143.6971930
time: 6.90 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6983620, upper bound: 143.6964895
time: 6.18 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -62.3716736, 49.7590904, -41.4439125, 32.8915558, -95.2632294, 91.2030029
1: -51.3290482, 44.0926628, -33.6844254, 29.2529831, -80.5820312, 77.7770844
2: -68.2802505, 44.6789513, -44.9570274, 29.8077965, -98.0880356, 89.6359711
3: -72.8252945, 38.4780159, -48.4337997, 25.5509300, -98.3762207, 86.9118195
4: -67.2463150, 51.6865616, -44.6591225, 34.1274338, -101.3737411, 96.3456879
5: -60.0289383, 46.7278671, -39.9627800, 31.0465927, -91.0755310, 86.6906204
6: -57.6723633, 54.8730583, -38.3584099, 36.2198524, -93.8922119, 93.2314606
7: -62.2830467, 52.4579201, -41.2908134, 34.7623444, -97.0453949, 93.7487259
8: -75.4326096, 52.2640800, -49.6284103, 34.1885834, -109.6211929, 101.8924789
9: -56.7998047, 55.6977882, -37.5218468, 36.5535126, -93.3533096, 93.2196274

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 170

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6935048, upper bound: 143.6913326
time: 7.17 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6930939, upper bound: 143.6910732
time: 8.34 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -68.1862793, 54.3956223, -49.4361382, 39.3283501, -107.5146179, 103.8317566
1: -56.3199158, 48.2227631, -40.3988457, 34.9173279, -91.2372437, 88.6215897
2: -74.7147903, 48.7997246, -53.8451576, 35.4441833, -110.1589661, 102.6448669
3: -79.5834122, 42.0898857, -57.7029266, 30.4809589, -110.0643616, 99.7928162
4: -73.5249405, 56.5236320, -53.2253876, 40.8817902, -114.4067307, 109.7490158
5: -65.6119308, 51.0567551, -47.5984459, 37.0325584, -102.6444855, 98.6551895
6: -63.0618706, 60.0638313, -45.7271576, 43.3682671, -106.4301376, 105.7909851
7: -68.1181641, 57.3485680, -49.2919807, 41.5468140, -109.6649704, 106.6405487
8: -82.5419693, 57.2294044, -59.5382233, 41.1365356, -123.6785049, 116.7676239
9: -62.1374741, 61.0027657, -44.8688011, 43.8189430, -105.9564133, 105.8715591

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7223491, upper bound: 143.7213973
time: 7.16 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7220855, upper bound: 143.7211889
time: 34.69 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -57.7215080, 46.0768929, -55.1150360, 43.9669151, -101.6884155, 101.1919250
1: -47.4371490, 40.8194923, -45.1079407, 38.9326859, -86.3698349, 85.9274216
2: -63.1458321, 41.3928337, -60.2516556, 39.5712624, -102.7170944, 101.6444778
3: -67.3073807, 35.6242065, -64.4097290, 33.9412956, -101.2486725, 100.0339203
4: -62.2006531, 47.8608475, -59.4118652, 45.5947151, -107.7953491, 107.2726974
5: -55.5467491, 43.2699318, -53.0825272, 41.3522758, -96.8990250, 96.3524628
6: -53.3463478, 50.7599449, -50.9231491, 48.3975601, -101.7439041, 101.6830902
7: -57.5958443, 48.5415573, -55.0121689, 46.3785362, -103.9743805, 103.5537262
8: -69.8157120, 48.4600143, -66.5221863, 45.9676018, -115.7833099, 114.9821930
9: -52.5546875, 51.5163269, -50.1445236, 49.0451851, -101.5998688, 101.6608505

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 62

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7250803, upper bound: 143.7231856
time: 8.02 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7238674, upper bound: 143.7222807
time: 6.95 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -63.5316582, 50.7129707, -64.5644302, 51.5306282, -115.0622864, 115.2774048
1: -52.4111938, 44.9430389, -53.2480621, 45.6660500, -98.0772400, 98.1911011
2: -69.5709610, 45.5055199, -70.7161026, 46.2622032, -115.8331604, 116.2216110
3: -74.0664520, 39.2317696, -75.3549194, 39.8409958, -113.9074402, 114.5866852
4: -68.4684677, 52.7001076, -69.6061935, 53.5335922, -122.0020599, 122.3063049
5: -61.1255074, 47.5981598, -62.1403542, 48.3985062, -109.5240097, 109.7385101
6: -58.7269707, 55.9354630, -59.6823845, 56.8402748, -115.5672379, 115.6178360
7: -63.4272728, 53.4294548, -64.4950256, 54.3373032, -117.7645721, 117.9244766
8: -76.9167938, 53.4277992, -78.1087723, 54.1636581, -131.0804291, 131.5365601
9: -57.8835793, 56.8161392, -58.8309975, 57.7282181, -115.6117859, 115.6471405

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 62

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7512545, upper bound: 143.7510000
time: 6.40 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7501007, upper bound: 143.7501204
time: 6.69 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -62.3716736, 49.7590904, -55.8824615, 44.5754662, -106.9471436, 105.6415482
1: -51.3290482, 44.0926628, -45.7509460, 39.4737282, -90.8027725, 89.8436050
2: -68.2802505, 44.6789513, -61.1009407, 40.1138954, -108.3941422, 105.7798691
3: -72.8252945, 38.4780159, -65.3230133, 34.4110146, -107.2362976, 103.8010254
4: -67.2463150, 51.6865616, -60.2467461, 46.2275887, -113.4738922, 111.9333038
5: -60.0289383, 46.7278671, -53.8240128, 41.9245911, -101.9535217, 100.5518646
6: -57.6723633, 54.8730583, -51.6348038, 49.0769081, -106.7492676, 106.5078506
7: -62.2830467, 52.4579201, -55.7895279, 47.0294380, -109.3124847, 108.2474518
8: -75.4326096, 52.2640800, -67.4465485, 46.5938759, -122.0264893, 119.7106247
9: -56.7998047, 55.6977882, -50.8478699, 49.7357826, -106.5355835, 106.5456467

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 170

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7271462, upper bound: 143.7255411
time: 8.22 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7266781, upper bound: 143.7252039
time: 5.83 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -68.1862793, 54.3956223, -65.3229828, 52.1313400, -120.3176117, 119.7186050
1: -56.3199158, 48.2227631, -53.8876114, 46.2015877, -102.5214996, 102.1103592
2: -74.7147903, 48.7997246, -71.5543289, 46.8001060, -121.5148926, 120.3540421
3: -79.5834122, 42.0898857, -76.2551193, 40.3055382, -119.8889389, 118.3450012
4: -73.5249405, 56.5236320, -70.4300842, 54.1591568, -127.6840973, 126.9537125
5: -65.6119308, 51.0567551, -62.8730583, 48.9654503, -114.5773773, 113.9298096
6: -63.0618706, 60.0638313, -60.3863640, 57.5136032, -120.5754623, 120.4501953
7: -68.1181641, 57.3485680, -65.2623596, 54.9811287, -123.0992813, 122.6109314
8: -82.5419693, 57.2294044, -79.0232239, 54.7815666, -137.3235168, 136.2526093
9: -62.1374741, 61.0027657, -59.5280952, 58.4131775, -120.5506516, 120.5308456

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7547938, upper bound: 143.7545846
time: 7.72 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7542063, upper bound: 143.7542069
time: 6.92 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 15.46 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 4, lower bound: -143.7042295, upper bound: 143.7054575
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 4, lower bound: -143.7054899, upper bound: 143.7065604
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 4, lower bound: -143.7034740, upper bound: 143.7045417
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 4, lower bound: -143.7046661, upper bound: 143.7056128
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 4, lower bound: -143.6956781, upper bound: 143.6977922
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 4, lower bound: -143.6971930, upper bound: 143.6991671
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 4, lower bound: -143.6948376, upper bound: 143.6968761
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 4, lower bound: -143.6964895, upper bound: 143.6983620
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 4, lower bound: -143.6868254, upper bound: 143.6895882
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 4, lower bound: -143.6912868, upper bound: 143.6935044
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 4, lower bound: -143.6817123, upper bound: 143.6840613
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 4, lower bound: -143.6786103, upper bound: 143.6815764
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 4, lower bound: -143.7043326, upper bound: 143.7055757
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 4, lower bound: -143.6958496, upper bound: 143.6979731
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 4, lower bound: -143.7040525, upper bound: 143.7051641
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 4, lower bound: -143.6956051, upper bound: 143.6976446
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 4, lower bound: -143.6872171, upper bound: 143.6873788
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 4, lower bound: -143.7131086, upper bound: 143.7119664
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 4, lower bound: -143.6991671, upper bound: 143.6971930
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 4, lower bound: -143.6983620, upper bound: 143.6964895
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 4, lower bound: -143.6935048, upper bound: 143.6913326
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 4, lower bound: -143.6930939, upper bound: 143.6910732
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 4, lower bound: -143.7223491, upper bound: 143.7213973
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 4, lower bound: -143.7220855, upper bound: 143.7211889
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 4, lower bound: -143.7250803, upper bound: 143.7231856
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 4, lower bound: -143.7238674, upper bound: 143.7222807
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 4, lower bound: -143.7512545, upper bound: 143.7510000
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 4, lower bound: -143.7501007, upper bound: 143.7501204
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 4, lower bound: -143.7271462, upper bound: 143.7255411
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 4, lower bound: -143.7266781, upper bound: 143.7252039
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 4, lower bound: -143.7547938, upper bound: 143.7545846
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 4, lower bound: -143.7542063, upper bound: 143.7542069

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -41.8875351, 33.3002663, -50.1701927, 40.0666237, -81.9541550, 83.4704437
1: -34.2488403, 29.6674309, -41.2635574, 35.5574799, -69.8063202, 70.9309845
2: -45.5156670, 30.0907593, -54.7957726, 36.0386658, -81.5543213, 84.8865280
3: -48.7905579, 25.8635311, -58.4173622, 30.9937611, -79.7843170, 84.2808914
4: -45.0902672, 34.7455521, -54.0378075, 41.7224998, -86.8127670, 88.7833557
5: -40.3080788, 31.3870602, -48.2668762, 37.6439972, -77.9520645, 79.6539230
6: -38.8237991, 36.7705040, -46.4763832, 44.1281776, -82.9519806, 83.2468872
7: -41.7339439, 35.2070999, -50.0137215, 42.2157326, -83.9496613, 85.2208252
8: -50.4615440, 34.9501915, -60.7826691, 42.2879906, -92.7495346, 95.7328491
9: -38.0016861, 37.0638008, -45.6726761, 44.6883888, -82.6900558, 82.7364655

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 170

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6931275, upper bound: 143.6925179
time: 6.07 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6931275, upper bound: 143.7054575
time: 6.72 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -42.7862396, 34.0148964, -51.3115807, 40.9701042, -83.7563477, 85.3264618
1: -34.9876938, 30.2929020, -42.2069588, 36.3534470, -71.3411407, 72.4998474
2: -46.5100594, 30.7300167, -56.0535736, 36.8459244, -83.3559723, 86.7835922
3: -49.8429184, 26.4277401, -59.7606430, 31.6977081, -81.5406265, 86.1883850
4: -46.0564499, 35.4822617, -55.2699966, 42.6547966, -88.7112350, 90.7522354
5: -41.1596794, 32.0425835, -49.3582001, 38.4875908, -79.6472702, 81.4007645
6: -39.6396332, 37.5641479, -47.5226250, 45.1317520, -84.7713699, 85.0867691
7: -42.6332130, 35.9616127, -51.1604271, 43.1699562, -85.8031693, 87.1220398
8: -51.5479736, 35.6708145, -62.1583595, 43.2135658, -94.7615356, 97.8291779
9: -38.8150787, 37.8666344, -46.7065163, 45.7074242, -84.5225067, 84.5731430

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 170

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6943729, upper bound: 143.6936815
time: 5.35 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6943729, upper bound: 143.7065604
time: 5.79 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -39.8945160, 31.7177944, -51.3106728, 40.9635849, -80.8581009, 83.0284576
1: -32.6033859, 28.2772751, -42.2046242, 36.3595238, -68.9629059, 70.4819031
2: -43.3246841, 28.6966743, -56.0646477, 36.8491364, -80.1738205, 84.7613220
3: -46.4543457, 24.6416664, -59.7636681, 31.6601715, -78.1145172, 84.4053345
4: -42.9677849, 33.1273384, -55.2812157, 42.6664352, -85.6342010, 88.4085541
5: -38.3941498, 29.9094925, -49.3590736, 38.4834442, -76.8775940, 79.2685623
6: -37.0189323, 35.0235252, -47.5672264, 45.1389122, -82.1578445, 82.5907516
7: -39.7540970, 33.5408859, -51.1805496, 43.1627197, -82.9167938, 84.7214279
8: -48.0717621, 33.3138466, -62.2012672, 43.2280960, -91.2998352, 95.5151138
9: -36.1930161, 35.2752876, -46.6900291, 45.6841049, -81.8771210, 81.9653168

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6949782, upper bound: 143.6962090
time: 7.51 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6936490, upper bound: 143.6946365
time: 7.71 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -40.8542290, 32.4789124, -52.4644890, 41.8764305, -82.7306595, 84.9433975
1: -33.3906288, 28.9447765, -43.1573830, 37.1636848, -70.5543137, 72.1021576
2: -44.3811684, 29.3775177, -57.3360023, 37.6644707, -82.0456390, 86.7135162
3: -47.5770760, 25.2429638, -61.1198654, 32.3716049, -79.9486771, 86.3628235
4: -43.9968414, 33.9110260, -56.5261421, 43.6076546, -87.6044846, 90.4371643
5: -39.3018456, 30.6093254, -50.4615974, 39.3358574, -78.6376724, 81.0709229
6: -37.8888855, 35.8698120, -48.6245270, 46.1534424, -84.0423050, 84.4943314
7: -40.7137642, 34.3444633, -52.3389626, 44.1270180, -84.8407822, 86.6834183
8: -49.2297897, 34.0827827, -63.5912590, 44.1638565, -93.3936386, 97.6740417
9: -37.0607986, 36.1301041, -47.7352219, 46.7136726, -83.7744522, 83.8653259

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6926256, upper bound: 143.6916318
time: 7.19 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6926256, upper bound: 143.6916318
time: 6.21 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -39.4106293, 31.2801876, -46.2879257, 36.9561310, -76.3667603, 77.5681076
1: -32.2762794, 27.9997826, -38.0515099, 32.8467979, -65.1230621, 66.0512924
2: -42.8374176, 28.3708916, -50.5091400, 33.2921066, -76.1295242, 78.8800278
3: -45.9283409, 24.3646317, -53.8652000, 28.6082668, -74.5366058, 78.2298279
4: -42.4562378, 32.7500267, -49.8487892, 38.5391541, -80.9953842, 82.5988083
5: -37.9187851, 29.5116043, -44.5124741, 34.7209282, -72.6396942, 74.0240784
6: -36.6100998, 34.6313591, -42.9044724, 40.7181168, -77.3281937, 77.5358276
7: -39.3638573, 33.1376495, -46.1375580, 38.9630928, -78.3269501, 79.2752075
8: -47.5048027, 32.8013115, -56.0795670, 39.0117798, -86.5165863, 88.8808746
9: -35.7394905, 34.8633041, -42.1352730, 41.1999588, -76.9394531, 76.9985733

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 242

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6877518, upper bound: 143.6881902
time: 6.33 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6877518, upper bound: 143.6881902
time: 6.37 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -40.2364883, 31.9359531, -47.4149475, 37.8475800, -78.0840530, 79.3508987
1: -32.9541588, 28.5732136, -38.9811325, 33.6315994, -66.5857391, 67.5543365
2: -43.7505722, 28.9574776, -51.7509003, 34.0887527, -77.8393173, 80.7083740
3: -46.8939285, 24.8820744, -55.1902924, 29.3018360, -76.1957550, 80.0723572
4: -43.3444977, 33.4270248, -51.0627327, 39.4578514, -82.8023148, 84.4897461
5: -38.6987534, 30.1126575, -45.5893707, 35.5531693, -74.2519226, 75.7020187
6: -37.3626938, 35.3620338, -43.9353485, 41.7077713, -79.0704651, 79.2973709
7: -40.1896439, 33.8299866, -47.2699280, 39.9046669, -80.0942917, 81.0999146
8: -48.5052299, 33.4665794, -57.4383698, 39.9242516, -88.4294662, 90.9049530
9: -36.4889183, 35.6002350, -43.1549225, 42.2044754, -78.6933899, 78.7551575

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 20

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6892199, upper bound: 143.6895809
time: 6.08 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6892199, upper bound: 143.6895809
time: 6.10 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -37.4584351, 29.7308159, -47.4666100, 37.8825302, -75.3409576, 77.1974258
1: -30.6748371, 26.6440258, -39.0244637, 33.6752510, -64.3500748, 65.6684875
2: -40.6949883, 27.0116844, -51.8195381, 34.1307983, -74.8257904, 78.8312149
3: -43.6383095, 23.1693878, -55.2542305, 29.2991524, -72.9374466, 78.4236145
4: -40.3803596, 31.1647835, -51.1313171, 39.5144844, -79.8948288, 82.2960968
5: -36.0465927, 28.0669212, -45.6416130, 35.5877533, -71.6343155, 73.7085342
6: -34.8434639, 32.9213676, -44.0313759, 41.7633705, -76.6068344, 76.9527435
7: -37.4243050, 31.5100555, -47.3397636, 39.9411812, -77.3654633, 78.8498230
8: -45.1599655, 31.2062302, -57.5417747, 39.9823647, -85.1423264, 88.7480011
9: -33.9724236, 33.1110535, -43.1862183, 42.2283401, -76.2007599, 76.2972717

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6852982, upper bound: 143.6878623
time: 7.67 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6846053, upper bound: 143.6868516
time: 6.86 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -38.3332481, 30.4253616, -48.5989761, 38.7784538, -77.1117020, 79.0243301
1: -31.3871117, 27.2488556, -39.9582024, 34.4632378, -65.8503494, 67.2070618
2: -41.6599808, 27.6312389, -53.0675125, 34.9309464, -76.5909042, 80.6987457
3: -44.6610603, 23.7149563, -56.5849190, 29.9966545, -74.6577072, 80.2998734
4: -41.3197517, 31.8790455, -52.3510666, 40.4371910, -81.7569427, 84.2301102
5: -36.8730698, 28.7043247, -46.7230644, 36.4247894, -73.2978592, 75.4273911
6: -35.6397476, 33.6949997, -45.0681190, 42.7572327, -78.3969803, 78.7631149
7: -38.2979774, 32.2419815, -48.4779663, 40.8866119, -79.1845856, 80.7199249
8: -46.2196198, 31.9077015, -58.9086914, 40.8997002, -87.1193237, 90.8163910
9: -34.7629700, 33.8896255, -44.2111092, 43.2374039, -78.0003738, 78.1007233

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6883394, upper bound: 143.6906000
time: 7.30 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6875495, upper bound: 143.6894309
time: 6.76 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 11.31 + 602.37 = 613.68 seconds
