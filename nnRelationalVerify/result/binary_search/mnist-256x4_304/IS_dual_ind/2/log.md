## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2000 seconds
Threshold: 143.61867486269998
Search space: {k/256 | k = 1, 2, ..., 12}


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

## BASE Result
execution time: IAR + LP analysis = 1.31 + 8.27 = 9.57 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -143.7624614, upper bound: 143.7624614


# Binary Search by BASE starts (time budget: 1990.43 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=145.0042724609375
rel_dist={4: [-143.76245312009848, 143.76245312009854]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=145.0042724609375
rel_dist={4: [-143.76243730833275, 143.76243730833278]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=145.0042724609375
rel_dist={4: [-143.7624133928287, 143.76241339486478]}

## Binary Search Result
Binary search time: 33.35 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1957.07 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7350495, upper bound: 143.7377536
time: 6.08 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7602417, upper bound: 143.7602417
time: 4.85 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 11.08 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 11.08
Output dim: 4, lower bound: -143.7350495, upper bound: 143.7377536
IS_A2, status: Status.UNKNOWN, split count: 1, time: 11.08
Output dim: 4, lower bound: -143.7602417, upper bound: 143.7602417

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -55.0390320, 43.7957993, -72.4897690, 57.8241692, -112.8632050, 116.2855682
1: -45.1367798, 38.8839874, -59.9855843, 51.2697868, -96.4065704, 98.8695526
2: -60.0479012, 39.4024048, -79.4760437, 51.8804016, -111.9282990, 118.8784409
3: -64.2611771, 33.9397697, -84.6278458, 44.7349052, -108.9960709, 118.5675964
4: -59.2569656, 45.5209084, -78.1617203, 60.0776825, -119.3346481, 123.6826248
5: -52.9686508, 41.2229843, -69.7588577, 54.3106346, -107.2792740, 110.9818420
6: -50.8919487, 48.3378906, -67.0118179, 63.8906174, -114.7825546, 115.3497086
7: -54.9219017, 46.2777328, -72.4588318, 61.0238991, -115.9458008, 118.7365646
8: -66.3510666, 45.8323746, -87.7077255, 60.7607002, -127.1117554, 133.5401001
9: -49.9999199, 48.8834915, -66.1034775, 64.9136963, -114.9136047, 114.9869614

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
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
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 62

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7251565, upper bound: 143.7251565
time: 5.90 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7251565, upper bound: 143.7377536
time: 5.68 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -71.1890564, 56.7948189, -76.0228043, 60.6544914, -131.8435516, 132.8176117
1: -58.8822021, 50.3534431, -62.9943390, 53.7792549, -112.6614532, 113.3477783
2: -78.0392151, 50.9567719, -83.4031906, 54.4029045, -132.4421234, 134.3599548
3: -83.1072540, 43.9345856, -88.7539597, 46.9208794, -130.0281067, 132.6885376
4: -76.7635193, 59.0121994, -81.9910431, 63.0132256, -139.7767181, 141.0032349
5: -68.5076675, 53.3455086, -73.1580811, 56.9551659, -125.4628067, 126.5035858
6: -65.8120346, 62.7355423, -70.2768860, 67.0410385, -132.8530731, 133.0124207
7: -71.1526871, 59.9258461, -76.0090866, 64.0053253, -135.1579895, 135.9349213
8: -86.1432419, 59.6953735, -92.0233536, 63.7623940, -149.9056396, 151.7187195
9: -64.9082565, 63.7381058, -69.3594666, 68.1468353, -133.0550690, 133.0975342

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 204

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7377536, upper bound: 143.7350495
time: 6.35 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7377536, upper bound: 143.7602417
time: 6.16 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 13.87 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 13.87
Output dim: 4, lower bound: -143.7251565, upper bound: 143.7251565
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 13.87
Output dim: 4, lower bound: -143.7251565, upper bound: 143.7377536
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 13.87
Output dim: 4, lower bound: -143.7377536, upper bound: 143.7350495
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 13.87
Output dim: 4, lower bound: -143.7377536, upper bound: 143.7602417

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -55.0390320, 43.7957993, -55.0390320, 43.7957993, -98.8348312, 98.8348312
1: -45.1367798, 38.8839874, -45.1367798, 38.8839874, -84.0207596, 84.0207596
2: -60.0479012, 39.4024048, -60.0479012, 39.4024048, -99.4503021, 99.4503021
3: -64.2611771, 33.9397697, -64.2611771, 33.9397697, -98.2009277, 98.2009277
4: -59.2569656, 45.5209084, -59.2569656, 45.5209084, -104.7778778, 104.7778778
5: -52.9686508, 41.2229843, -52.9686508, 41.2229843, -94.1916275, 94.1916275
6: -50.8919487, 48.3378906, -50.8919487, 48.3378906, -99.2298279, 99.2298279
7: -54.9219017, 46.2777328, -54.9219017, 46.2777328, -101.1996307, 101.1996307
8: -66.3510666, 45.8323746, -66.3510666, 45.8323746, -112.1834335, 112.1834335
9: -49.9999199, 48.8834915, -49.9999199, 48.8834915, -98.8834000, 98.8834000

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 124

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7245227, upper bound: 143.7245338
time: 5.96 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7248432, upper bound: 143.7248432
time: 6.45 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -55.0390320, 43.7957993, -71.1890564, 56.7948189, -111.8338470, 114.9848557
1: -45.1367798, 38.8839874, -58.8822021, 50.3534431, -95.4902191, 97.7661743
2: -60.0479012, 39.4024048, -78.0392151, 50.9567719, -111.0046692, 117.4416199
3: -64.2611771, 33.9397697, -83.1072540, 43.9345856, -108.1957626, 117.0470047
4: -59.2569656, 45.5209084, -76.7635193, 59.0121994, -118.2691650, 122.2844238
5: -52.9686508, 41.2229843, -68.5076675, 53.3455086, -106.3141556, 109.7306366
6: -50.8919487, 48.3378906, -65.8120346, 62.7355423, -113.6274796, 114.1499252
7: -54.9219017, 46.2777328, -71.1526871, 59.9258461, -114.8477478, 117.4304199
8: -66.3510666, 45.8323746, -86.1432419, 59.6953735, -126.0464172, 131.9756165
9: -49.9999199, 48.8834915, -64.9082565, 63.7381058, -113.7380142, 113.7917480

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 124

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7245227, upper bound: 143.7370922
time: 5.95 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7248432, upper bound: 143.7374636
time: 6.19 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7369791, upper bound: 143.7340695
time: 7.02 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7374636, upper bound: 143.7345243
time: 6.99 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7369791, upper bound: 143.7595127
time: 8.00 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7374636, upper bound: 143.7600751
time: 7.32 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 16.83 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 16.83
Output dim: 4, lower bound: -143.7245227, upper bound: 143.7245338
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 16.83
Output dim: 4, lower bound: -143.7248432, upper bound: 143.7248432
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 16.83
Output dim: 4, lower bound: -143.7245227, upper bound: 143.7370922
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 16.83
Output dim: 4, lower bound: -143.7248432, upper bound: 143.7374636
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 16.83
Output dim: 4, lower bound: -143.7369791, upper bound: 143.7340695
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 16.83
Output dim: 4, lower bound: -143.7374636, upper bound: 143.7345243
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 16.83
Output dim: 4, lower bound: -143.7369791, upper bound: 143.7595127
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 16.83
Output dim: 4, lower bound: -143.7374636, upper bound: 143.7600751

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -48.4783058, 38.5816154, -54.7951622, 43.6019554, -92.0802612, 93.3767776
1: -39.6424866, 34.2516251, -44.9326439, 38.7121544, -78.3546295, 79.1842651
2: -52.7891083, 34.7452850, -59.7782974, 39.2291832, -92.0182800, 94.5235825
3: -56.4817047, 29.9176483, -63.9723816, 33.7898521, -90.2715530, 93.8900299
4: -52.1529922, 40.1202431, -58.9926682, 45.3199463, -97.4729385, 99.1129150
5: -46.6482468, 36.2875671, -52.7332382, 41.0397720, -87.6880035, 89.0208054
6: -44.8350182, 42.5409431, -50.6664124, 48.1222534, -92.9572754, 93.2073364
7: -48.2896652, 40.7025223, -54.6754837, 46.0709229, -94.3605881, 95.3780060
8: -58.4571152, 40.4870491, -66.0579987, 45.6334190, -104.0905304, 106.5450439
9: -43.9927902, 42.9840393, -49.7766724, 48.6639519, -92.6567383, 92.7607117

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 124

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7243533, upper bound: 143.7243533
time: 5.93 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7243533, upper bound: 143.7245338
time: 5.74 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -52.9735260, 42.1415138, -54.7213440, 43.5429459, -96.5164719, 96.8628464
1: -43.3925552, 37.4190674, -44.8706741, 38.6601257, -82.0526810, 82.2897415
2: -57.7568817, 37.9149361, -59.6964569, 39.1762199, -96.9331055, 97.6113892
3: -61.8209190, 32.6746101, -63.8853302, 33.7445755, -95.5654907, 96.5599365
4: -57.0251656, 43.8118134, -58.9127960, 45.2590599, -102.2842178, 102.7246094
5: -50.9711494, 39.6323318, -52.6621742, 40.9837685, -91.9549179, 92.2945099
6: -49.0095940, 46.5107346, -50.5983086, 48.0569229, -97.0665131, 97.1090393
7: -52.8241425, 44.4897537, -54.6010284, 46.0080948, -98.8322372, 99.0907822
8: -63.8972740, 44.1607933, -65.9694366, 45.5730782, -109.4703522, 110.1302261
9: -48.0968819, 47.0131721, -49.7090836, 48.5973701, -96.6942520, 96.7222519

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 29
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
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 124

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7245338, upper bound: 143.7245227
time: 5.40 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7245338, upper bound: 143.7248432
time: 5.48 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -48.4783058, 38.5816154, -70.9399414, 56.5969810, -105.0752792, 109.5215607
1: -39.6424866, 34.2516251, -58.6722984, 50.1775894, -89.8200760, 92.9239197
2: -52.7891083, 34.7452850, -77.7639313, 50.7794991, -103.5685959, 112.5092163
3: -56.4817047, 29.9176483, -82.8127670, 43.7816582, -100.2633514, 112.7304153
4: -52.1529922, 40.1202431, -76.4937820, 58.8071480, -110.9601364, 116.6140213
5: -46.6482468, 36.2875671, -68.2673035, 53.1583176, -99.8065643, 104.5548553
6: -44.8350182, 42.5409431, -65.5817032, 62.5146103, -107.3496246, 108.1226349
7: -48.2896652, 40.7025223, -70.9012375, 59.7144318, -108.0040970, 111.6037598
8: -58.4571152, 40.4870491, -85.8437729, 59.4924126, -117.9495239, 126.3308258
9: -43.9927902, 42.9840393, -64.6796722, 63.5132065, -107.5059967, 107.6636963

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 62

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7339763, upper bound: 143.7368332
time: 6.69 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7339763, upper bound: 143.7370921
time: 6.83 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -52.9735260, 42.1415138, -70.8629608, 56.5356255, -109.5091476, 113.0044708
1: -43.3925552, 37.4190674, -58.6072578, 50.1231880, -93.5157471, 96.0263214
2: -57.7568817, 37.9149361, -77.6787338, 50.7241745, -108.4810486, 115.5936737
3: -61.8209190, 32.6746101, -82.7221451, 43.7343102, -105.5552216, 115.3967438
4: -57.0251656, 43.8118134, -76.4104996, 58.7438164, -115.7689819, 120.2223053
5: -50.9711494, 39.6323318, -68.1932526, 53.1000557, -104.0712051, 107.8255844
6: -49.0095940, 46.5107346, -65.5107422, 62.4462395, -111.4558334, 112.0214767
7: -52.8241425, 44.4897537, -70.8236465, 59.6489296, -112.4730682, 115.3134003
8: -63.8972740, 44.1607933, -85.7513199, 59.4296494, -123.3269196, 129.9121094
9: -48.0968819, 47.0131721, -64.6091309, 63.4435692, -111.5404510, 111.6222992

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 62

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7108369, upper bound: 143.7109940
time: 7.45 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7318715, upper bound: 143.7347527
time: 6.60 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -64.3955994, 51.3998566, -54.7951622, 43.6019554, -107.9975586, 106.1950226
1: -53.1505241, 45.5554619, -44.9326439, 38.7121544, -91.8626556, 90.4880981
2: -70.5260239, 46.1158981, -59.7782974, 39.2291832, -109.7551880, 105.8941956
3: -75.0708618, 39.7683411, -63.9723816, 33.7898521, -108.8607178, 103.7407074
4: -69.4003906, 53.4169083, -58.9926682, 45.3199463, -114.7203369, 112.4095764
5: -61.9543114, 48.2407532, -52.7332382, 41.0397720, -102.9940720, 100.9739914
6: -59.5268135, 56.7059937, -50.6664124, 48.1222534, -107.6490555, 107.3723755
7: -64.2931061, 54.1554527, -54.6754837, 46.0709229, -110.3640289, 108.8309326
8: -77.9704895, 54.1610222, -66.0579987, 45.6334190, -123.6039124, 120.2190247
9: -58.6758537, 57.6022644, -49.7766724, 48.6639519, -107.3398056, 107.3789368

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 124

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7368332, upper bound: 143.7339763
time: 7.46 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7368332, upper bound: 143.7340695
time: 7.65 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -69.0568771, 55.0875969, -54.7213440, 43.5429459, -112.5998230, 109.8089294
1: -57.0652199, 48.8404388, -44.8706741, 38.6601257, -95.7253418, 93.7111130
2: -75.6762848, 49.4146347, -59.6964569, 39.1762199, -114.8525085, 109.1110764
3: -80.5944138, 42.6306839, -63.8853302, 33.7445755, -114.3389893, 106.5160141
4: -74.4635086, 57.2453079, -58.9127960, 45.2590599, -119.7225647, 116.1580963
5: -66.4463730, 51.7040939, -52.6621742, 40.9837685, -107.4301376, 104.3662643
6: -63.8677101, 60.8395119, -50.5983086, 48.0569229, -111.9246292, 111.4378204
7: -68.9898224, 58.0798111, -54.6010284, 46.0080948, -114.9979172, 112.6808395
8: -83.6032639, 57.9682922, -65.9694366, 45.5730782, -129.1763153, 123.9377289
9: -62.9355888, 61.7938652, -49.7090836, 48.5973701, -111.5329437, 111.5029449

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 29
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
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 124

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7370921, upper bound: 143.7342611
time: 7.91 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7370921, upper bound: 143.7345243
time: 6.01 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -64.3955994, 51.3998566, -70.9399414, 56.5969810, -120.9925766, 122.3397903
1: -53.1505241, 45.5554619, -58.6722984, 50.1775894, -103.3281021, 104.2277451
2: -70.5260239, 46.1158981, -77.7639313, 50.7794991, -121.3055038, 123.8798294
3: -75.0708618, 39.7683411, -82.8127670, 43.7816582, -118.8525238, 122.5811005
4: -69.4003906, 53.4169083, -76.4937820, 58.8071480, -128.2075195, 129.9106445
5: -61.9543114, 48.2407532, -68.2673035, 53.1583176, -115.1126251, 116.5080566
6: -59.5268135, 56.7059937, -65.5817032, 62.5146103, -122.0414200, 122.2876740
7: -64.2931061, 54.1554527, -70.9012375, 59.7144318, -124.0075378, 125.0566864
8: -77.9704895, 54.1610222, -85.8437729, 59.4924126, -137.4629059, 140.0047760
9: -58.6758537, 57.6022644, -64.6796722, 63.5132065, -122.1890564, 122.2819366

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 62

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7591705, upper bound: 143.7591726
time: 5.71 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7591705, upper bound: 143.7595127
time: 5.71 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -69.0568771, 55.0875969, -70.8629608, 56.5356255, -125.5924988, 125.9505463
1: -57.0652199, 48.8404388, -58.6072578, 50.1231880, -107.1884079, 107.4476929
2: -75.6762848, 49.4146347, -77.6787338, 50.7241745, -126.4004517, 127.0933609
3: -80.5944138, 42.6306839, -82.7221451, 43.7343102, -124.3287201, 125.3528214
4: -74.4635086, 57.2453079, -76.4104996, 58.7438164, -133.2073212, 133.6558075
5: -66.4463730, 51.7040939, -68.1932526, 53.1000557, -119.5464325, 119.8973389
6: -63.8677101, 60.8395119, -65.5107422, 62.4462395, -126.3139496, 126.3502502
7: -68.9898224, 58.0798111, -70.8236465, 59.6489296, -128.6387482, 128.9034576
8: -83.6032639, 57.9682922, -85.7513199, 59.4296494, -143.0329132, 143.7196045
9: -62.9355888, 61.7938652, -64.6091309, 63.4435692, -126.3791504, 126.4029999

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 62

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7380331, upper bound: 143.7357299
time: 7.24 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7582291, upper bound: 143.7582291
time: 6.01 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 14.66 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.66
Output dim: 4, lower bound: -143.7243533, upper bound: 143.7243533
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.66
Output dim: 4, lower bound: -143.7243533, upper bound: 143.7245338
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.66
Output dim: 4, lower bound: -143.7245338, upper bound: 143.7245227
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.66
Output dim: 4, lower bound: -143.7245338, upper bound: 143.7248432
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.66
Output dim: 4, lower bound: -143.7339763, upper bound: 143.7368332
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.66
Output dim: 4, lower bound: -143.7339763, upper bound: 143.7370921
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.66
Output dim: 4, lower bound: -143.7108369, upper bound: 143.7109940
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.66
Output dim: 4, lower bound: -143.7318715, upper bound: 143.7347527
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.66
Output dim: 4, lower bound: -143.7368332, upper bound: 143.7339763
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.66
Output dim: 4, lower bound: -143.7368332, upper bound: 143.7340695
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.66
Output dim: 4, lower bound: -143.7370921, upper bound: 143.7342611
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.66
Output dim: 4, lower bound: -143.7370921, upper bound: 143.7345243
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.66
Output dim: 4, lower bound: -143.7591705, upper bound: 143.7591726
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.66
Output dim: 4, lower bound: -143.7591705, upper bound: 143.7595127
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.66
Output dim: 4, lower bound: -143.7380331, upper bound: 143.7357299
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.66
Output dim: 4, lower bound: -143.7582291, upper bound: 143.7582291

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -48.4783058, 38.5816154, -48.4783058, 38.5816154, -87.0599213, 87.0599213
1: -39.6424866, 34.2516251, -39.6424866, 34.2516251, -73.8941116, 73.8941116
2: -52.7891083, 34.7452850, -52.7891083, 34.7452850, -87.5343933, 87.5343933
3: -56.4817047, 29.9176483, -56.4817047, 29.9176483, -86.3993530, 86.3993530
4: -52.1529922, 40.1202431, -52.1529922, 40.1202431, -92.2732391, 92.2732391
5: -46.6482468, 36.2875671, -46.6482468, 36.2875671, -82.9357986, 82.9357986
6: -44.8350182, 42.5409431, -44.8350182, 42.5409431, -87.3759613, 87.3759613
7: -48.2896652, 40.7025223, -48.2896652, 40.7025223, -88.9921875, 88.9921875
8: -58.4571152, 40.4870491, -58.4571152, 40.4870491, -98.9441681, 98.9441681
9: -43.9927902, 42.9840393, -43.9927902, 42.9840393, -86.9768143, 86.9768143

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7064389, upper bound: 143.7048697
time: 6.45 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6988989, upper bound: 143.6988989
time: 6.49 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -48.4783058, 38.5816154, -52.9735260, 42.1415138, -90.6198196, 91.5551453
1: -39.6424866, 34.2516251, -43.3925552, 37.4190674, -77.0615540, 77.6441803
2: -52.7891083, 34.7452850, -57.7568817, 37.9149361, -90.7040405, 92.5021667
3: -56.4817047, 29.9176483, -61.8209190, 32.6746101, -89.1563110, 91.7385635
4: -52.1529922, 40.1202431, -57.0251656, 43.8118134, -95.9648056, 97.1454086
5: -46.6482468, 36.2875671, -50.9711494, 39.6323318, -86.2805786, 87.2587128
6: -44.8350182, 42.5409431, -49.0095940, 46.5107346, -91.3457489, 91.5505371
7: -48.2896652, 40.7025223, -52.8241425, 44.4897537, -92.7794189, 93.5266647
8: -58.4571152, 40.4870491, -63.8972740, 44.1607933, -102.6179047, 104.3843231
9: -43.9927902, 42.9840393, -48.0968819, 47.0131721, -91.0059662, 91.0809174

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7064389, upper bound: 143.7051432
time: 7.58 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6988989, upper bound: 143.6991636
time: 5.67 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -52.9735260, 42.1415138, -48.4783058, 38.5816154, -91.5551453, 90.6198196
1: -43.3925552, 37.4190674, -39.6424866, 34.2516251, -77.6441803, 77.0615540
2: -57.7568817, 37.9149361, -52.7891083, 34.7452850, -92.5021667, 90.7040405
3: -61.8209190, 32.6746101, -56.4817047, 29.9176483, -91.7385635, 89.1563110
4: -57.0251656, 43.8118134, -52.1529922, 40.1202431, -97.1454086, 95.9648056
5: -50.9711494, 39.6323318, -46.6482468, 36.2875671, -87.2587128, 86.2805786
6: -49.0095940, 46.5107346, -44.8350182, 42.5409431, -91.5505371, 91.3457489
7: -52.8241425, 44.4897537, -48.2896652, 40.7025223, -93.5266647, 92.7794189
8: -63.8972740, 44.1607933, -58.4571152, 40.4870491, -104.3843231, 102.6179047
9: -48.0968819, 47.0131721, -43.9927902, 42.9840393, -91.0809174, 91.0059662

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7172287, upper bound: 143.7170442
time: 5.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7163049, upper bound: 143.7162699
time: 5.71 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -52.9735260, 42.1415138, -52.9735260, 42.1415138, -95.1150360, 95.1150360
1: -43.3925552, 37.4190674, -43.3925552, 37.4190674, -80.8116226, 80.8116226
2: -57.7568817, 37.9149361, -57.7568817, 37.9149361, -95.6718140, 95.6718140
3: -61.8209190, 32.6746101, -61.8209190, 32.6746101, -94.4955215, 94.4955215
4: -57.0251656, 43.8118134, -57.0251656, 43.8118134, -100.8369751, 100.8369751
5: -50.9711494, 39.6323318, -50.9711494, 39.6323318, -90.6034851, 90.6034851
6: -49.0095940, 46.5107346, -49.0095940, 46.5107346, -95.5203247, 95.5203247
7: -52.8241425, 44.4897537, -52.8241425, 44.4897537, -97.3138885, 97.3138885
8: -63.8972740, 44.1607933, -63.8972740, 44.1607933, -108.0580673, 108.0580673
9: -48.0968819, 47.0131721, -48.0968819, 47.0131721, -95.1100540, 95.1100540

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 170

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7172287, upper bound: 143.7186718
time: 6.15 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7163049, upper bound: 143.7178323
time: 6.03 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -48.4783058, 38.5816154, -64.3955994, 51.3998566, -99.8781586, 102.9772186
1: -39.6424866, 34.2516251, -53.1505241, 45.5554619, -85.1979446, 87.4021378
2: -52.7891083, 34.7452850, -70.5260239, 46.1158981, -98.9050064, 105.2713089
3: -56.4817047, 29.9176483, -75.0708618, 39.7683411, -96.2500305, 104.9885101
4: -52.1529922, 40.1202431, -69.4003906, 53.4169083, -105.5699005, 109.5206299
5: -46.6482468, 36.2875671, -61.9543114, 48.2407532, -94.8889999, 98.2418671
6: -44.8350182, 42.5409431, -59.5268135, 56.7059937, -101.5410080, 102.0677490
7: -48.2896652, 40.7025223, -64.2931061, 54.1554527, -102.4451141, 104.9956284
8: -58.4571152, 40.4870491, -77.9704895, 54.1610222, -112.6181335, 118.4575348
9: -43.9927902, 42.9840393, -58.6758537, 57.6022644, -101.5950546, 101.6598740

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7241555, upper bound: 143.7261226
time: 7.78 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7138437, upper bound: 143.7172206
time: 6.20 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -48.4783058, 38.5816154, -69.0568771, 55.0875969, -103.5658951, 107.6384888
1: -39.6424866, 34.2516251, -57.0652199, 48.8404388, -88.4829254, 91.3168488
2: -52.7891083, 34.7452850, -75.6762848, 49.4146347, -102.2037430, 110.4215698
3: -56.4817047, 29.9176483, -80.5944138, 42.6306839, -99.1123886, 110.5120621
4: -52.1529922, 40.1202431, -74.4635086, 57.2453079, -109.3983002, 114.5837479
5: -46.6482468, 36.2875671, -66.4463730, 51.7040939, -98.3523407, 102.7339249
6: -44.8350182, 42.5409431, -63.8677101, 60.8395119, -105.6745300, 106.4086533
7: -48.2896652, 40.7025223, -68.9898224, 58.0798111, -106.3694763, 109.6923447
8: -58.4571152, 40.4870491, -83.6032639, 57.9682922, -116.4254074, 124.0903091
9: -43.9927902, 42.9840393, -62.9355888, 61.7938652, -105.7866516, 105.9196091

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 170

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7241555, upper bound: 143.7262687
time: 7.99 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7138437, upper bound: 143.7175635
time: 6.43 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -50.6475716, 40.2811584, -57.4137764, 45.7946777, -96.4422379, 97.6949310
1: -41.4115295, 35.7656975, -47.0316315, 40.5512962, -81.9628143, 82.7973328
2: -55.1789169, 36.2729797, -62.7920990, 41.2011757, -96.3800964, 99.0650787
3: -59.1135330, 31.2343159, -67.1271515, 35.3531265, -94.4666519, 98.3614655
4: -54.5250549, 41.8773041, -61.9041138, 47.4881096, -102.0131607, 103.7814178
5: -48.7452278, 37.8972588, -55.3032990, 43.0808372, -91.8260651, 93.2005539
6: -46.8648643, 44.4413795, -53.0437584, 50.4315033, -97.2963638, 97.4851379
7: -50.4892960, 42.5294609, -57.3309593, 48.3288689, -98.8181534, 99.8604050
8: -61.0479774, 42.1769295, -69.2797470, 47.8412476, -108.8892212, 111.4566727
9: -45.9670029, 44.9022560, -52.2432785, 51.1104164, -97.0774231, 97.1455307

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7045776, upper bound: 143.7038244
time: 7.33 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7029966, upper bound: 143.7025595
time: 7.25 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -52.9735260, 42.1415138, -66.8644791, 53.3567657, -106.3302841, 109.0059814
1: -43.3925552, 37.4190674, -55.1874237, 47.2894173, -90.6819763, 92.6064911
2: -57.7568817, 37.9149361, -73.2593765, 47.8999405, -105.6568222, 111.1743164
3: -61.8209190, 32.6746101, -78.0761948, 41.2514801, -103.0723801, 110.7508011
4: -57.0251656, 43.8118134, -72.0990448, 55.4282532, -112.4534149, 115.9108582
5: -50.9711494, 39.6323318, -64.3598709, 50.1261024, -101.0972519, 103.9922028
6: -49.0095940, 46.5107346, -61.8103142, 58.8815994, -107.8911896, 108.3210449
7: -52.8241425, 44.4897537, -66.8178177, 56.2905464, -109.1146851, 111.3075714
8: -63.8972740, 44.1607933, -80.8755951, 56.0375328, -119.9348068, 125.0363922
9: -48.0968819, 47.0131721, -60.9421577, 59.8057861, -107.9026642, 107.9553299

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 170

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7274201, upper bound: 143.7293802
time: 7.73 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7258300, upper bound: 143.7278539
time: 7.56 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -64.3955994, 51.3998566, -48.4783058, 38.5816154, -102.9772186, 99.8781586
1: -53.1505241, 45.5554619, -39.6424866, 34.2516251, -87.4021378, 85.1979446
2: -70.5260239, 46.1158981, -52.7891083, 34.7452850, -105.2713089, 98.9050064
3: -75.0708618, 39.7683411, -56.4817047, 29.9176483, -104.9885101, 96.2500305
4: -69.4003906, 53.4169083, -52.1529922, 40.1202431, -109.5206299, 105.5699005
5: -61.9543114, 48.2407532, -46.6482468, 36.2875671, -98.2418671, 94.8889999
6: -59.5268135, 56.7059937, -44.8350182, 42.5409431, -102.0677490, 101.5410080
7: -64.2931061, 54.1554527, -48.2896652, 40.7025223, -104.9956284, 102.4451141
8: -77.9704895, 54.1610222, -58.4571152, 40.4870491, -118.4575348, 112.6181335
9: -58.6758537, 57.6022644, -43.9927902, 42.9840393, -101.6598740, 101.5950546

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 114
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
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 62

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7109085, upper bound: 143.7107938
time: 6.68 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7342278, upper bound: 143.7314512
time: 6.81 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -64.3955994, 51.3998566, -52.9735260, 42.1415138, -106.5371094, 104.3733826
1: -53.1505241, 45.5554619, -43.3925552, 37.4190674, -90.5695801, 88.9480133
2: -70.5260239, 46.1158981, -57.7568817, 37.9149361, -108.4409637, 103.8727798
3: -75.0708618, 39.7683411, -61.8209190, 32.6746101, -107.7454681, 101.5892487
4: -69.4003906, 53.4169083, -57.0251656, 43.8118134, -113.2121964, 110.4420776
5: -61.9543114, 48.2407532, -50.9711494, 39.6323318, -101.5866394, 99.2118988
6: -59.5268135, 56.7059937, -49.0095940, 46.5107346, -106.0375443, 105.7155762
7: -64.2931061, 54.1554527, -52.8241425, 44.4897537, -108.7828598, 106.9795837
8: -77.9704895, 54.1610222, -63.8972740, 44.1607933, -122.1312866, 118.0582962
9: -58.6758537, 57.6022644, -48.0968819, 47.0131721, -105.6890259, 105.6991425

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 114
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
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 62

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7109085, upper bound: 143.7108177
time: 6.83 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7342278, upper bound: 143.7314976
time: 6.28 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -69.0568771, 55.0875969, -48.4783058, 38.5816154, -107.6384888, 103.5658951
1: -57.0652199, 48.8404388, -39.6424866, 34.2516251, -91.3168488, 88.4829254
2: -75.6762848, 49.4146347, -52.7891083, 34.7452850, -110.4215698, 102.2037430
3: -80.5944138, 42.6306839, -56.4817047, 29.9176483, -110.5120621, 99.1123886
4: -74.4635086, 57.2453079, -52.1529922, 40.1202431, -114.5837479, 109.3983002
5: -66.4463730, 51.7040939, -46.6482468, 36.2875671, -102.7339249, 98.3523407
6: -63.8677101, 60.8395119, -44.8350182, 42.5409431, -106.4086533, 105.6745300
7: -68.9898224, 58.0798111, -48.2896652, 40.7025223, -109.6923447, 106.3694763
8: -83.6032639, 57.9682922, -58.4571152, 40.4870491, -124.0903091, 116.4254074
9: -62.9355888, 61.7938652, -43.9927902, 42.9840393, -105.9196091, 105.7866516

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 188
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
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 36
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
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 170

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7108536, upper bound: 143.7107753
time: 9.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7344715, upper bound: 143.7317312
time: 6.98 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -69.0568771, 55.0875969, -52.9735260, 42.1415138, -111.1983871, 108.0611115
1: -57.0652199, 48.8404388, -43.3925552, 37.4190674, -94.4842834, 92.2329941
2: -75.6762848, 49.4146347, -57.7568817, 37.9149361, -113.5912170, 107.1715164
3: -80.5944138, 42.6306839, -61.8209190, 32.6746101, -113.2690201, 104.4515991
4: -74.4635086, 57.2453079, -57.0251656, 43.8118134, -118.2753220, 114.2704620
5: -66.4463730, 51.7040939, -50.9711494, 39.6323318, -106.0787048, 102.6752472
6: -63.8677101, 60.8395119, -49.0095940, 46.5107346, -110.3784485, 109.8491058
7: -68.9898224, 58.0798111, -52.8241425, 44.4897537, -113.4795761, 110.9039383
8: -83.6032639, 57.9682922, -63.8972740, 44.1607933, -127.7640381, 121.8655701
9: -62.9355888, 61.7938652, -48.0968819, 47.0131721, -109.9487534, 109.8907471

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 188
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
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 36
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
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 170

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7108536, upper bound: 143.7108305
time: 7.34 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7344715, upper bound: 143.7318715
time: 6.36 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -64.3955994, 51.3998566, -64.3955994, 51.3998566, -115.7954559, 115.7954559
1: -53.1505241, 45.5554619, -53.1505241, 45.5554619, -98.7059708, 98.7059708
2: -70.5260239, 46.1158981, -70.5260239, 46.1158981, -116.6419220, 116.6419220
3: -75.0708618, 39.7683411, -75.0708618, 39.7683411, -114.8392029, 114.8392029
4: -69.4003906, 53.4169083, -69.4003906, 53.4169083, -122.8172989, 122.8172989
5: -61.9543114, 48.2407532, -61.9543114, 48.2407532, -110.1950684, 110.1950684
6: -59.5268135, 56.7059937, -59.5268135, 56.7059937, -116.2327881, 116.2327881
7: -64.2931061, 54.1554527, -64.2931061, 54.1554527, -118.4485626, 118.4485626
8: -77.9704895, 54.1610222, -77.9704895, 54.1610222, -132.1315155, 132.1315155
9: -58.6758537, 57.6022644, -58.6758537, 57.6022644, -116.2781143, 116.2781143

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 114
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
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 62

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7353917, upper bound: 143.7377915
time: 6.74 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7574151, upper bound: 143.7574164
time: 6.05 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -64.3955994, 51.3998566, -69.0568771, 55.0875969, -119.4831848, 120.4567337
1: -53.1505241, 45.5554619, -57.0652199, 48.8404388, -101.9909668, 102.6206741
2: -70.5260239, 46.1158981, -75.6762848, 49.4146347, -119.9406509, 121.7921829
3: -75.0708618, 39.7683411, -80.5944138, 42.6306839, -117.7015457, 120.3627472
4: -69.4003906, 53.4169083, -74.4635086, 57.2453079, -126.6456833, 127.8803940
5: -61.9543114, 48.2407532, -66.4463730, 51.7040939, -113.6584015, 114.6871262
6: -59.5268135, 56.7059937, -63.8677101, 60.8395119, -120.3663177, 120.5736923
7: -64.2931061, 54.1554527, -68.9898224, 58.0798111, -122.3729172, 123.1452713
8: -77.9704895, 54.1610222, -83.6032639, 57.9682922, -135.9387817, 137.7642822
9: -58.6758537, 57.6022644, -62.9355888, 61.7938652, -120.4697189, 120.5378494

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 114
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
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 62

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7353917, upper bound: 143.7379822
time: 6.74 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7574151, upper bound: 143.7577062
time: 5.70 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -66.5789719, 53.1133270, -57.4137764, 45.7946777, -112.3736420, 110.5270996
1: -54.9380913, 47.0797577, -47.0316315, 40.5512962, -95.4893875, 94.1113892
2: -72.9356003, 47.6592598, -62.7920990, 41.2011757, -114.1367798, 110.4513550
3: -77.7156372, 41.0908051, -67.1271515, 35.3531265, -113.0687561, 108.2179565
4: -71.7882538, 55.1856346, -61.9041138, 47.4881096, -119.2763519, 117.0897446
5: -64.0681152, 49.8598022, -55.3032990, 43.0808372, -107.1489563, 105.1631012
6: -61.5709114, 58.6279182, -53.0437584, 50.4315033, -112.0024109, 111.6716766
7: -66.5043488, 55.9958649, -57.3309593, 48.3288689, -114.8332138, 113.3268127
8: -80.5748749, 55.8542900, -69.2797470, 47.8412476, -128.4161224, 125.1340332
9: -60.6612587, 59.5350800, -52.2432785, 51.1104164, -111.7716751, 111.7783508

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 161
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
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7323969, upper bound: 143.7295143
time: 7.38 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7313993, upper bound: 143.7288703
time: 7.46 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -69.0568771, 55.0875969, -66.8644791, 53.3567657, -122.4136353, 121.9520569
1: -57.0652199, 48.8404388, -55.1874237, 47.2894173, -104.3546371, 104.0278625
2: -75.6762848, 49.4146347, -73.2593765, 47.8999405, -123.5762253, 122.6740036
3: -80.5944138, 42.6306839, -78.0761948, 41.2514801, -121.8458786, 120.7068787
4: -74.4635086, 57.2453079, -72.0990448, 55.4282532, -129.8917542, 129.3443604
5: -66.4463730, 51.7040939, -64.3598709, 50.1261024, -116.5724792, 116.0639648
6: -63.8677101, 60.8395119, -61.8103142, 58.8815994, -122.7493134, 122.6498260
7: -68.9898224, 58.0798111, -66.8178177, 56.2905464, -125.2803650, 124.8976212
8: -83.6032639, 57.9682922, -80.8755951, 56.0375328, -139.6407928, 138.8438873
9: -62.9355888, 61.7938652, -60.9421577, 59.8057861, -122.7413635, 122.7360229

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 188
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
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 36
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
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7356208, upper bound: 143.7379994
time: 7.26 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7356208, upper bound: 143.7582291
time: 7.54 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 16.35 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.35
Output dim: 4, lower bound: -143.7064389, upper bound: 143.7048697
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.35
Output dim: 4, lower bound: -143.6988989, upper bound: 143.6988989
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.35
Output dim: 4, lower bound: -143.7064389, upper bound: 143.7051432
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.35
Output dim: 4, lower bound: -143.6988989, upper bound: 143.6991636
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.35
Output dim: 4, lower bound: -143.7172287, upper bound: 143.7170442
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.35
Output dim: 4, lower bound: -143.7163049, upper bound: 143.7162699
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.35
Output dim: 4, lower bound: -143.7172287, upper bound: 143.7186718
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.35
Output dim: 4, lower bound: -143.7163049, upper bound: 143.7178323
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.35
Output dim: 4, lower bound: -143.7241555, upper bound: 143.7261226
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.35
Output dim: 4, lower bound: -143.7138437, upper bound: 143.7172206
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.35
Output dim: 4, lower bound: -143.7241555, upper bound: 143.7262687
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.35
Output dim: 4, lower bound: -143.7138437, upper bound: 143.7175635
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.35
Output dim: 4, lower bound: -143.7045776, upper bound: 143.7038244
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.35
Output dim: 4, lower bound: -143.7029966, upper bound: 143.7025595
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.35
Output dim: 4, lower bound: -143.7274201, upper bound: 143.7293802
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.35
Output dim: 4, lower bound: -143.7258300, upper bound: 143.7278539
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.35
Output dim: 4, lower bound: -143.7109085, upper bound: 143.7107938
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.35
Output dim: 4, lower bound: -143.7342278, upper bound: 143.7314512
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.35
Output dim: 4, lower bound: -143.7109085, upper bound: 143.7108177
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.35
Output dim: 4, lower bound: -143.7342278, upper bound: 143.7314976
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.35
Output dim: 4, lower bound: -143.7108536, upper bound: 143.7107753
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.35
Output dim: 4, lower bound: -143.7344715, upper bound: 143.7317312
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.35
Output dim: 4, lower bound: -143.7108536, upper bound: 143.7108305
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.35
Output dim: 4, lower bound: -143.7344715, upper bound: 143.7318715
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.35
Output dim: 4, lower bound: -143.7353917, upper bound: 143.7377915
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.35
Output dim: 4, lower bound: -143.7574151, upper bound: 143.7574164
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.35
Output dim: 4, lower bound: -143.7353917, upper bound: 143.7379822
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.35
Output dim: 4, lower bound: -143.7574151, upper bound: 143.7577062
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.35
Output dim: 4, lower bound: -143.7323969, upper bound: 143.7295143
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.35
Output dim: 4, lower bound: -143.7313993, upper bound: 143.7288703
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.35
Output dim: 4, lower bound: -143.7356208, upper bound: 143.7379994
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.35
Output dim: 4, lower bound: -143.7356208, upper bound: 143.7582291

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -42.1556892, 33.4943504, -48.4783058, 38.5816154, -80.7372971, 81.9726562
1: -34.4150848, 29.8211346, -39.6424866, 34.2516251, -68.6667099, 69.4636230
2: -45.7912292, 30.2686481, -52.7891083, 34.7452850, -80.5365143, 83.0577545
3: -49.0511742, 26.0531349, -56.4817047, 29.9176483, -78.9688263, 82.5348358
4: -45.3482513, 34.9343224, -52.1529922, 40.1202431, -85.4684906, 87.0873108
5: -40.5261497, 31.5134678, -46.6482468, 36.2875671, -76.8136902, 78.1617126
6: -39.0270195, 36.9872093, -44.8350182, 42.5409431, -81.5679474, 81.8222275
7: -41.9700394, 35.3782997, -48.2896652, 40.7025223, -82.6725616, 83.6679688
8: -50.7836723, 35.1575356, -58.4571152, 40.4870491, -91.2707214, 93.6146545
9: -38.2130852, 37.2932281, -43.9927902, 42.9840393, -81.1971130, 81.2860184

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6988989, upper bound: 143.6988989
time: 5.91 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6988989, upper bound: 143.6988989
time: 5.73 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -39.6646805, 31.4591141, -46.5949898, 37.0696487, -76.7343292, 78.0541000
1: -32.4338303, 28.1434269, -38.0874939, 32.9327240, -65.3665390, 66.2309189
2: -43.0962601, 28.5413208, -50.7067833, 33.4114761, -76.5077362, 79.2481079
3: -46.1713371, 24.5400734, -54.2714539, 28.7651367, -74.9364624, 78.8115234
4: -42.7000504, 32.9237480, -50.1278305, 38.5766335, -81.2766876, 83.0515747
5: -38.1230812, 29.6256714, -44.8286514, 34.8663177, -72.9893951, 74.4543228
6: -36.8022308, 34.8356285, -43.1066132, 40.8865547, -77.6887817, 77.9422455
7: -39.5815811, 33.2981911, -46.4076424, 39.1191330, -78.7007141, 79.7058334
8: -47.8046150, 33.0035744, -56.1696548, 38.9010124, -86.7056198, 89.1732254
9: -35.9397125, 35.0744324, -42.2747574, 41.2927933, -77.2325058, 77.3491898

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6891583, upper bound: 143.6897337
time: 8.62 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6877022, upper bound: 143.6877022
time: 5.19 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -42.1556892, 33.4943504, -52.9735260, 42.1415138, -84.2971954, 86.4678802
1: -34.4150848, 29.8211346, -43.3925552, 37.4190674, -71.8341522, 73.2136917
2: -45.7912292, 30.2686481, -57.7568817, 37.9149361, -83.7061615, 88.0255280
3: -49.0511742, 26.0531349, -61.8209190, 32.6746101, -81.7257843, 87.8740540
4: -45.3482513, 34.9343224, -57.0251656, 43.8118134, -89.1600647, 91.9594879
5: -40.5261497, 31.5134678, -50.9711494, 39.6323318, -80.1584778, 82.4846191
6: -39.0270195, 36.9872093, -49.0095940, 46.5107346, -85.5377426, 85.9968033
7: -41.9700394, 35.3782997, -52.8241425, 44.4897537, -86.4597931, 88.2024384
8: -50.7836723, 35.1575356, -63.8972740, 44.1607933, -94.9444656, 99.0548096
9: -38.2130852, 37.2932281, -48.0968819, 47.0131721, -85.2262573, 85.3901062

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6979579, upper bound: 143.6965017
time: 6.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6963506, upper bound: 143.6945999
time: 6.42 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -39.6646805, 31.4591141, -51.0561333, 40.6064224, -80.2710953, 82.5152359
1: -32.4338303, 28.1434269, -41.8076591, 36.0779190, -68.5117493, 69.9510880
2: -43.0962601, 28.5413208, -55.6390343, 36.5596695, -79.6559296, 84.1803589
3: -46.1713371, 24.5400734, -59.5727844, 31.5007915, -77.6721115, 84.1128540
4: -42.7000504, 32.9237480, -54.9629517, 42.2420425, -84.9420929, 87.8866806
5: -38.1230812, 29.6256714, -49.1199913, 38.1902046, -76.3132858, 78.7456665
6: -36.8022308, 34.8356285, -47.2497787, 44.8270721, -81.6292877, 82.0854034
7: -39.5815811, 33.2981911, -50.9107018, 42.8792686, -82.4608383, 84.2088852
8: -47.8046150, 33.0035744, -61.5750237, 42.5461349, -90.3507538, 94.5785980
9: -35.9397125, 35.0744324, -46.3464088, 45.2917557, -81.2314682, 81.4208298

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 170

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6893849, upper bound: 143.6898737
time: 7.24 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6884285, upper bound: 143.6884337
time: 5.90 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -46.4414024, 36.9571381, -48.4783058, 38.5816154, -85.0230179, 85.4354401
1: -37.9970360, 32.8419724, -39.6424866, 34.2516251, -72.2486572, 72.4844589
2: -50.5496445, 33.3262253, -52.7891083, 34.7452850, -85.2949295, 86.1153336
3: -54.1662292, 28.6689415, -56.4817047, 29.9176483, -84.0838776, 85.1506348
4: -50.0214157, 38.5012627, -52.1529922, 40.1202431, -90.1416550, 90.6542511
5: -44.6981812, 34.7783966, -46.6482468, 36.2875671, -80.9857407, 81.4266434
6: -43.0724220, 40.7694969, -44.8350182, 42.5409431, -85.6133423, 85.6045151
7: -46.2701073, 39.0112839, -48.2896652, 40.7025223, -86.9726181, 87.3009491
8: -56.0627136, 38.8086853, -58.4571152, 40.4870491, -96.5497513, 97.2658005
9: -42.1472549, 41.1205864, -43.9927902, 42.9840393, -85.1312790, 85.1133728

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 170

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6965017, upper bound: 143.6979579
time: 7.57 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6898737, upper bound: 143.6893849
time: 6.06 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -48.4003105, 38.5058975, -47.2265282, 37.5882568, -85.9885635, 85.7324219
1: -39.6220589, 34.2235947, -38.6102676, 33.3754196, -72.9974823, 72.8338470
2: -52.7269745, 34.7253151, -51.4077950, 33.8651161, -86.5920792, 86.1331024
3: -56.4769630, 29.8367443, -55.0147362, 29.1500053, -85.6269684, 84.8514786
4: -52.1557732, 40.1186638, -50.8132744, 39.1032829, -91.2590485, 90.9319382
5: -46.5803413, 36.2305679, -45.4473457, 35.3567848, -81.9371185, 81.6779099
6: -44.9221573, 42.5052147, -43.6978722, 41.4411125, -86.3632660, 86.2030869
7: -48.2734718, 40.6512413, -47.0339127, 39.6536026, -87.9270630, 87.6851501
8: -58.4746933, 40.4312592, -56.9526443, 39.4616280, -97.9363251, 97.3839035
9: -43.9149704, 42.8577919, -42.8540115, 41.8565216, -85.7714920, 85.7118073

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 170

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6945999, upper bound: 143.6963506
time: 6.05 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6884337, upper bound: 143.6884285
time: 5.78 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -46.4414024, 36.9571381, -52.9735260, 42.1415138, -88.5829163, 89.9306641
1: -37.9970360, 32.8419724, -43.3925552, 37.4190674, -75.4160995, 76.2345276
2: -50.5496445, 33.3262253, -57.7568817, 37.9149361, -88.4645844, 91.0831070
3: -54.1662292, 28.6689415, -61.8209190, 32.6746101, -86.8408356, 90.4898529
4: -50.0214157, 38.5012627, -57.0251656, 43.8118134, -93.8332291, 95.5264206
5: -44.6981812, 34.7783966, -50.9711494, 39.6323318, -84.3305130, 85.7495422
6: -43.0724220, 40.7694969, -49.0095940, 46.5107346, -89.5831375, 89.7790909
7: -46.2701073, 39.0112839, -52.8241425, 44.4897537, -90.7598419, 91.8354187
8: -56.0627136, 38.8086853, -63.8972740, 44.1607933, -100.2234879, 102.7059631
9: -42.1472549, 41.1205864, -48.0968819, 47.0131721, -89.1604233, 89.2174683

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7178306, upper bound: 143.7178323
time: 4.86 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7178306, upper bound: 143.7178322
time: 4.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -48.4003105, 38.5058975, -51.7336502, 41.1600838, -89.5603943, 90.2395477
1: -39.6220589, 34.2235947, -42.3673363, 36.5522919, -76.1743469, 76.5909271
2: -52.7269745, 34.7253151, -56.3900299, 37.0448875, -89.7718582, 91.1153412
3: -56.4769630, 29.8367443, -60.3693810, 31.9125557, -88.3895111, 90.2061234
4: -52.1557732, 40.1186638, -55.6972198, 42.8045769, -94.9603348, 95.8158798
5: -46.5803413, 36.2305679, -49.7825851, 38.7130661, -85.2934036, 86.0131378
6: -44.9221573, 42.5052147, -47.8827667, 45.4211960, -90.3433533, 90.3879776
7: -48.2734718, 40.6512413, -51.5834732, 43.4518127, -91.7252808, 92.2347107
8: -58.4746933, 40.4312592, -62.4108124, 43.1441460, -101.6188354, 102.8420715
9: -43.9149704, 42.8577919, -46.9682350, 45.8965645, -89.8115387, 89.8260193

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6964342, upper bound: 143.6981414
time: 7.04 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6899123, upper bound: 143.6899149
time: 6.08 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -42.1556892, 33.4943504, -64.3955994, 51.3998566, -93.5555344, 97.8899536
1: -34.4150848, 29.8211346, -53.1505241, 45.5554619, -79.9705276, 82.9716568
2: -45.7912292, 30.2686481, -70.5260239, 46.1158981, -91.9071274, 100.7946701
3: -49.0511742, 26.0531349, -75.0708618, 39.7683411, -88.8195190, 101.1239929
4: -45.3482513, 34.9343224, -69.4003906, 53.4169083, -98.7651596, 104.3347168
5: -40.5261497, 31.5134678, -61.9543114, 48.2407532, -88.7668915, 93.4677811
6: -39.0270195, 36.9872093, -59.5268135, 56.7059937, -95.7329865, 96.5140228
7: -41.9700394, 35.3782997, -64.2931061, 54.1554527, -96.1254883, 99.6714020
8: -50.7836723, 35.1575356, -77.9704895, 54.1610222, -104.9446869, 113.1280212
9: -38.2130852, 37.2932281, -58.6758537, 57.6022644, -95.8153534, 95.9690857

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 62

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7004361, upper bound: 143.6999949
time: 7.89 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7214088, upper bound: 143.7233510
time: 6.41 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -39.6646805, 31.4591141, -62.4392815, 49.8345718, -89.4992447, 93.8983917
1: -32.4338303, 28.1434269, -51.5244102, 44.1865616, -76.6203842, 79.6678238
2: -43.0962601, 28.5413208, -68.3673630, 44.7291069, -87.8253632, 96.9086838
3: -46.1713371, 24.5400734, -72.7782135, 38.5659180, -84.7372513, 97.3182831
4: -42.7000504, 32.9237480, -67.2888412, 51.8124008, -94.5124512, 100.2125854
5: -38.1230812, 29.6256714, -60.0591698, 46.7713928, -84.8944702, 89.6848450
6: -36.8022308, 34.8356285, -57.7247696, 54.9810295, -91.7832565, 92.5603943
7: -39.5815811, 33.2981911, -62.3403473, 52.5137901, -92.0953674, 95.6385345
8: -47.8046150, 33.0035744, -75.6051559, 52.5124359, -100.3170471, 108.6087265
9: -35.9397125, 35.0744324, -56.8893967, 55.8395576, -91.7792664, 91.9638290

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 62

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6909004, upper bound: 143.6917310
time: 5.97 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7110304, upper bound: 143.7144454
time: 6.20 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 13.48 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.48
Output dim: 4, lower bound: -143.6988989, upper bound: 143.6988989
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.48
Output dim: 4, lower bound: -143.6988989, upper bound: 143.6988989
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.48
Output dim: 4, lower bound: -143.6891583, upper bound: 143.6897337
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.48
Output dim: 4, lower bound: -143.6877022, upper bound: 143.6877022
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.48
Output dim: 4, lower bound: -143.6979579, upper bound: 143.6965017
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.48
Output dim: 4, lower bound: -143.6963506, upper bound: 143.6945999
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.48
Output dim: 4, lower bound: -143.6893849, upper bound: 143.6898737
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.48
Output dim: 4, lower bound: -143.6884285, upper bound: 143.6884337
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.48
Output dim: 4, lower bound: -143.6965017, upper bound: 143.6979579
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.48
Output dim: 4, lower bound: -143.6898737, upper bound: 143.6893849
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.48
Output dim: 4, lower bound: -143.6945999, upper bound: 143.6963506
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.48
Output dim: 4, lower bound: -143.6884337, upper bound: 143.6884285
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.48
Output dim: 4, lower bound: -143.7178306, upper bound: 143.7178323
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.48
Output dim: 4, lower bound: -143.7178306, upper bound: 143.7178322
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.48
Output dim: 4, lower bound: -143.6964342, upper bound: 143.6981414
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.48
Output dim: 4, lower bound: -143.6899123, upper bound: 143.6899149
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.48
Output dim: 4, lower bound: -143.7004361, upper bound: 143.6999949
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.48
Output dim: 4, lower bound: -143.7214088, upper bound: 143.7233510
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.48
Output dim: 4, lower bound: -143.6909004, upper bound: 143.6917310
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.48
Output dim: 4, lower bound: -143.7110304, upper bound: 143.7144454
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 4, lower bound: -143.7241555, upper bound: 143.7262687
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 4, lower bound: -143.7138437, upper bound: 143.7175635
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 4, lower bound: -143.7045776, upper bound: 143.7038244
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 4, lower bound: -143.7029966, upper bound: 143.7025595
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 4, lower bound: -143.7274201, upper bound: 143.7293802
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 4, lower bound: -143.7258300, upper bound: 143.7278539
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 4, lower bound: -143.7109085, upper bound: 143.7107938
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 4, lower bound: -143.7342278, upper bound: 143.7314512
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 4, lower bound: -143.7109085, upper bound: 143.7108177
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 4, lower bound: -143.7342278, upper bound: 143.7314976
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 4, lower bound: -143.7108536, upper bound: 143.7107753
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 4, lower bound: -143.7344715, upper bound: 143.7317312
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 4, lower bound: -143.7108536, upper bound: 143.7108305
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 4, lower bound: -143.7344715, upper bound: 143.7318715
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 4, lower bound: -143.7353917, upper bound: 143.7377915
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 4, lower bound: -143.7574151, upper bound: 143.7574164
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 4, lower bound: -143.7353917, upper bound: 143.7379822
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 4, lower bound: -143.7574151, upper bound: 143.7577062
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 4, lower bound: -143.7323969, upper bound: 143.7295143
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 4, lower bound: -143.7313993, upper bound: 143.7288703
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 4, lower bound: -143.7356208, upper bound: 143.7379994
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 4, lower bound: -143.7356208, upper bound: 143.7582291
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=145.0042724609375
rel_dist={4: [-143.76245312009848, 143.76245312009854]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7309054, upper bound: 143.7321927
time: 6.83 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7601942, upper bound: 143.7601942
time: 6.18 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 13.14 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 13.14
Output dim: 4, lower bound: -143.7309054, upper bound: 143.7321927
IS_A2, status: Status.UNKNOWN, split count: 1, time: 13.14
Output dim: 4, lower bound: -143.7601942, upper bound: 143.7601942

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7298914, upper bound: 143.7314227
time: 6.70 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7302794, upper bound: 143.7318708
time: 6.20 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7321927, upper bound: 143.7309054
time: 7.40 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7321927, upper bound: 143.7601942
time: 7.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 16.11 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 16.11
Output dim: 4, lower bound: -143.7298914, upper bound: 143.7314227
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 16.11
Output dim: 4, lower bound: -143.7302794, upper bound: 143.7318708
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 16.11
Output dim: 4, lower bound: -143.7321927, upper bound: 143.7309054
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 16.11
Output dim: 4, lower bound: -143.7321927, upper bound: 143.7601942

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7151543, upper bound: 143.7163229
time: 7.33 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7075591, upper bound: 143.7094009
time: 7.29 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6999193, upper bound: 143.7020883
time: 6.82 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7275078, upper bound: 143.7289782
time: 7.98 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7314227, upper bound: 143.7298914
time: 7.32 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7318708, upper bound: 143.7302794
time: 8.45 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7314227, upper bound: 143.7593562
time: 9.37 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7318708, upper bound: 143.7600328
time: 8.39 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 19.07 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 19.07
Output dim: 4, lower bound: -143.7151543, upper bound: 143.7163229
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 19.07
Output dim: 4, lower bound: -143.7075591, upper bound: 143.7094009
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 19.07
Output dim: 4, lower bound: -143.6999193, upper bound: 143.7020883
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 19.07
Output dim: 4, lower bound: -143.7275078, upper bound: 143.7289782
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 19.07
Output dim: 4, lower bound: -143.7314227, upper bound: 143.7298914
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 19.07
Output dim: 4, lower bound: -143.7318708, upper bound: 143.7302794
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 19.07
Output dim: 4, lower bound: -143.7314227, upper bound: 143.7593562
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 19.07
Output dim: 4, lower bound: -143.7318708, upper bound: 143.7600328

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7054899, upper bound: 143.7065604
time: 7.25 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7046661, upper bound: 143.7056128
time: 8.44 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6971930, upper bound: 143.6991671
time: 7.15 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6964895, upper bound: 143.6983620
time: 6.85 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6913326, upper bound: 143.6935048
time: 7.76 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6910732, upper bound: 143.6930939
time: 7.53 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7213973, upper bound: 143.7223491
time: 6.90 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7211889, upper bound: 143.7220855
time: 6.81 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7163229, upper bound: 143.7151543
time: 7.27 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7094009, upper bound: 143.7075591
time: 7.57 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7020883, upper bound: 143.6999192
time: 8.31 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7289782, upper bound: 143.7275078
time: 7.51 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7336745, upper bound: 143.7322686
time: 8.51 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7575539, upper bound: 143.7575603
time: 7.53 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7338165, upper bound: 143.7324253
time: 7.52 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7581921, upper bound: 143.7581921
time: 6.99 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 15.93 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.93
Output dim: 4, lower bound: -143.7054899, upper bound: 143.7065604
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.93
Output dim: 4, lower bound: -143.7046661, upper bound: 143.7056128
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.93
Output dim: 4, lower bound: -143.6971930, upper bound: 143.6991671
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.93
Output dim: 4, lower bound: -143.6964895, upper bound: 143.6983620
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.93
Output dim: 4, lower bound: -143.6913326, upper bound: 143.6935048
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.93
Output dim: 4, lower bound: -143.6910732, upper bound: 143.6930939
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.93
Output dim: 4, lower bound: -143.7213973, upper bound: 143.7223491
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.93
Output dim: 4, lower bound: -143.7211889, upper bound: 143.7220855
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.93
Output dim: 4, lower bound: -143.7163229, upper bound: 143.7151543
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.93
Output dim: 4, lower bound: -143.7094009, upper bound: 143.7075591
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.93
Output dim: 4, lower bound: -143.7020883, upper bound: 143.6999192
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.93
Output dim: 4, lower bound: -143.7289782, upper bound: 143.7275078
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.93
Output dim: 4, lower bound: -143.7336745, upper bound: 143.7322686
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.93
Output dim: 4, lower bound: -143.7575539, upper bound: 143.7575603
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.93
Output dim: 4, lower bound: -143.7338165, upper bound: 143.7324253
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.93
Output dim: 4, lower bound: -143.7581921, upper bound: 143.7581921

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7042295, upper bound: 143.7054575
time: 8.18 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7054899, upper bound: 143.7065604
time: 8.96 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7034740, upper bound: 143.7045417
time: 8.00 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7046661, upper bound: 143.7056128
time: 8.04 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6956781, upper bound: 143.6977922
time: 7.79 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6971930, upper bound: 143.6991671
time: 7.03 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6948376, upper bound: 143.6968761
time: 7.59 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6964895, upper bound: 143.6983620
time: 6.72 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6868254, upper bound: 143.6895882
time: 6.90 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6912868, upper bound: 143.6935044
time: 7.34 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6817123, upper bound: 143.6840613
time: 7.03 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6786103, upper bound: 143.6815764
time: 7.10 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7043326, upper bound: 143.7055757
time: 8.44 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6958496, upper bound: 143.6979731
time: 7.41 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7040525, upper bound: 143.7051641
time: 8.29 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6956051, upper bound: 143.6976446
time: 7.29 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6872171, upper bound: 143.6873788
time: 6.92 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7131086, upper bound: 143.7119664
time: 8.95 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6991671, upper bound: 143.6971930
time: 7.49 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6983620, upper bound: 143.6964895
time: 6.72 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6935048, upper bound: 143.6913326
time: 7.72 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6930939, upper bound: 143.6910732
time: 8.99 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7223491, upper bound: 143.7213973
time: 7.74 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7220855, upper bound: 143.7211889
time: 37.13 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7250803, upper bound: 143.7231856
time: 8.60 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7238674, upper bound: 143.7222807
time: 7.88 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7512545, upper bound: 143.7510000
time: 6.96 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7501007, upper bound: 143.7501204
time: 7.32 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7271462, upper bound: 143.7255411
time: 9.04 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7266781, upper bound: 143.7252039
time: 6.40 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7547938, upper bound: 143.7545846
time: 8.42 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7542063, upper bound: 143.7542069
time: 7.46 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 17.24 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 4, lower bound: -143.7042295, upper bound: 143.7054575
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 4, lower bound: -143.7054899, upper bound: 143.7065604
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 4, lower bound: -143.7034740, upper bound: 143.7045417
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 4, lower bound: -143.7046661, upper bound: 143.7056128
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 4, lower bound: -143.6956781, upper bound: 143.6977922
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 4, lower bound: -143.6971930, upper bound: 143.6991671
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 4, lower bound: -143.6948376, upper bound: 143.6968761
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 4, lower bound: -143.6964895, upper bound: 143.6983620
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 4, lower bound: -143.6868254, upper bound: 143.6895882
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 4, lower bound: -143.6912868, upper bound: 143.6935044
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 4, lower bound: -143.6817123, upper bound: 143.6840613
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 4, lower bound: -143.6786103, upper bound: 143.6815764
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 4, lower bound: -143.7043326, upper bound: 143.7055757
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 4, lower bound: -143.6958496, upper bound: 143.6979731
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 4, lower bound: -143.7040525, upper bound: 143.7051641
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 4, lower bound: -143.6956051, upper bound: 143.6976446
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 4, lower bound: -143.6872171, upper bound: 143.6873788
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 4, lower bound: -143.7131086, upper bound: 143.7119664
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 4, lower bound: -143.6991671, upper bound: 143.6971930
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 4, lower bound: -143.6983620, upper bound: 143.6964895
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 4, lower bound: -143.6935048, upper bound: 143.6913326
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 4, lower bound: -143.6930939, upper bound: 143.6910732
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 4, lower bound: -143.7223491, upper bound: 143.7213973
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 4, lower bound: -143.7220855, upper bound: 143.7211889
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 4, lower bound: -143.7250803, upper bound: 143.7231856
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 4, lower bound: -143.7238674, upper bound: 143.7222807
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 4, lower bound: -143.7512545, upper bound: 143.7510000
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 4, lower bound: -143.7501007, upper bound: 143.7501204
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 4, lower bound: -143.7271462, upper bound: 143.7255411
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 4, lower bound: -143.7266781, upper bound: 143.7252039
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 4, lower bound: -143.7547938, upper bound: 143.7545846
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 4, lower bound: -143.7542063, upper bound: 143.7542069

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6931275, upper bound: 143.6925179
time: 6.53 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6931275, upper bound: 143.7054575
time: 7.19 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6943729, upper bound: 143.6936815
time: 5.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6943729, upper bound: 143.7065604
time: 6.19 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6949782, upper bound: 143.6962090
time: 8.03 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6936490, upper bound: 143.6946365
time: 8.20 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6926256, upper bound: 143.6916318
time: 7.75 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6926256, upper bound: 143.6916318
time: 6.72 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 15.82 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 15.82
Output dim: 4, lower bound: -143.6931275, upper bound: 143.6925179
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 15.82
Output dim: 4, lower bound: -143.6931275, upper bound: 143.7054575
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 15.82
Output dim: 4, lower bound: -143.6943729, upper bound: 143.6936815
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 15.82
Output dim: 4, lower bound: -143.6943729, upper bound: 143.7065604
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 15.82
Output dim: 4, lower bound: -143.6949782, upper bound: 143.6962090
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 15.82
Output dim: 4, lower bound: -143.6936490, upper bound: 143.6946365
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 15.82
Output dim: 4, lower bound: -143.6926256, upper bound: 143.6916318
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 15.82
Output dim: 4, lower bound: -143.6926256, upper bound: 143.6916318
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.82
Output dim: 4, lower bound: -143.6956781, upper bound: 143.6977922
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.82
Output dim: 4, lower bound: -143.6971930, upper bound: 143.6991671
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.82
Output dim: 4, lower bound: -143.6948376, upper bound: 143.6968761
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.82
Output dim: 4, lower bound: -143.6964895, upper bound: 143.6983620
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.82
Output dim: 4, lower bound: -143.6868254, upper bound: 143.6895882
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.82
Output dim: 4, lower bound: -143.6912868, upper bound: 143.6935044
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.82
Output dim: 4, lower bound: -143.6817123, upper bound: 143.6840613
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.82
Output dim: 4, lower bound: -143.6786103, upper bound: 143.6815764
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.82
Output dim: 4, lower bound: -143.7043326, upper bound: 143.7055757
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.82
Output dim: 4, lower bound: -143.6958496, upper bound: 143.6979731
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.82
Output dim: 4, lower bound: -143.7040525, upper bound: 143.7051641
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.82
Output dim: 4, lower bound: -143.6956051, upper bound: 143.6976446
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.82
Output dim: 4, lower bound: -143.6872171, upper bound: 143.6873788
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.82
Output dim: 4, lower bound: -143.7131086, upper bound: 143.7119664
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.82
Output dim: 4, lower bound: -143.6991671, upper bound: 143.6971930
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.82
Output dim: 4, lower bound: -143.6983620, upper bound: 143.6964895
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.82
Output dim: 4, lower bound: -143.6935048, upper bound: 143.6913326
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.82
Output dim: 4, lower bound: -143.6930939, upper bound: 143.6910732
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.82
Output dim: 4, lower bound: -143.7223491, upper bound: 143.7213973
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.82
Output dim: 4, lower bound: -143.7220855, upper bound: 143.7211889
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.82
Output dim: 4, lower bound: -143.7250803, upper bound: 143.7231856
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.82
Output dim: 4, lower bound: -143.7238674, upper bound: 143.7222807
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.82
Output dim: 4, lower bound: -143.7512545, upper bound: 143.7510000
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.82
Output dim: 4, lower bound: -143.7501007, upper bound: 143.7501204
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.82
Output dim: 4, lower bound: -143.7271462, upper bound: 143.7255411
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.82
Output dim: 4, lower bound: -143.7266781, upper bound: 143.7252039
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.82
Output dim: 4, lower bound: -143.7547938, upper bound: 143.7545846
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.82
Output dim: 4, lower bound: -143.7542063, upper bound: 143.7542069
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=145.0042724609375
rel_dist={4: [-143.76243730833275, 143.76243730833278]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7270413, upper bound: 143.7274371
time: 9.22 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7601239, upper bound: 143.7601239
time: 10.08 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 19.43 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 19.43
Output dim: 4, lower bound: -143.7270413, upper bound: 143.7274371
IS_A2, status: Status.UNKNOWN, split count: 1, time: 19.43
Output dim: 4, lower bound: -143.7601239, upper bound: 143.7601239

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -55.0390320, 43.7957993, -58.8561897, 46.8925056, -101.9315338, 102.6519928
1: -45.1367798, 38.8839874, -48.3958817, 41.6008301, -86.7376099, 87.2798538
2: -60.0479012, 39.4024048, -64.3148727, 42.1498413, -102.1977386, 103.7172775
3: -64.2611771, 33.9397697, -68.6915512, 36.3064651, -100.5676193, 102.6313019
4: -59.2569656, 45.5209084, -63.3841362, 48.7413292, -107.9982910, 108.9050446
5: -52.9686508, 41.2229843, -56.6363220, 44.0995102, -97.0681610, 97.8593063
6: -50.8919487, 48.3378906, -54.4214973, 51.7424088, -102.6343384, 102.7593842
7: -54.9219017, 46.2777328, -58.7512779, 49.5167351, -104.4386292, 105.0290070
8: -66.3510666, 45.8323746, -71.0570145, 49.1746140, -115.5256805, 116.8893814
9: -49.9999199, 48.8834915, -53.5385208, 52.4332848, -102.4331970, 102.4220047

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 62

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7262246, upper bound: 143.7267546
time: 10.31 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7267494, upper bound: 143.7272782
time: 9.14 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -71.1890564, 56.7948189, -72.7248001, 58.0206757, -129.2097321, 129.5195923
1: -58.8822021, 50.3534431, -60.1884041, 51.4415245, -110.3237152, 110.5418472
2: -78.0392151, 50.9567719, -79.7431717, 52.0515213, -130.0907288, 130.6999512
3: -83.1072540, 43.9345856, -84.9009781, 44.8833237, -127.9905472, 128.8355560
4: -76.7635193, 59.0121994, -78.4242935, 60.2830429, -137.0465546, 137.4364929
5: -68.5076675, 53.3455086, -69.9849548, 54.4921265, -122.9997787, 123.3304596
6: -65.8120346, 62.7355423, -67.2304001, 64.1031113, -129.9151459, 129.9659424
7: -71.1526871, 59.9258461, -72.6955566, 61.2218094, -132.3744659, 132.6213684
8: -86.1432419, 59.6953735, -88.0108566, 60.9870377, -147.1302795, 147.7062225
9: -64.9082565, 63.7381058, -66.3224640, 65.1386261, -130.0468445, 130.0605774

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 62

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7591937, upper bound: 143.7591807
time: 7.62 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7599786, upper bound: 143.7599786
time: 7.23 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 16.19 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 16.19
Output dim: 4, lower bound: -143.7262246, upper bound: 143.7267546
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 16.19
Output dim: 4, lower bound: -143.7267494, upper bound: 143.7272782
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 16.19
Output dim: 4, lower bound: -143.7591937, upper bound: 143.7591807
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 16.19
Output dim: 4, lower bound: -143.7599786, upper bound: 143.7599786

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -50.0973358, 39.8689079, -52.1848869, 41.5930634, -91.6903839, 92.0537949
1: -40.9989281, 35.3965530, -42.7926521, 36.8939095, -77.8928299, 78.1892090
2: -54.5810699, 35.8954124, -56.9406242, 37.4041367, -91.9851837, 92.8360367
3: -58.4019165, 30.9100285, -60.7875595, 32.2114410, -90.6133499, 91.6975861
4: -53.9064713, 41.4533043, -56.1547050, 43.2389565, -97.1454315, 97.6080093
5: -48.2079086, 37.5069275, -50.1970062, 39.0861969, -87.2941055, 87.7039337
6: -46.3298531, 43.9716911, -48.2520294, 45.8380013, -92.1678543, 92.2237244
7: -49.9265900, 42.0802422, -52.0114937, 43.8502960, -93.7768860, 94.0917282
8: -60.4071960, 41.8048630, -63.0359154, 43.7343903, -104.1415863, 104.8407669
9: -45.4749107, 44.4392281, -47.4304047, 46.4168701, -91.8917770, 91.8696213

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 226
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
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7049218, upper bound: 143.7052726
time: 9.41 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7019100, upper bound: 143.7025644
time: 9.63 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -51.6528015, 41.1012726, -56.8154488, 45.2600250, -96.9128113, 97.9167175
1: -42.3012199, 36.4952545, -46.6586151, 40.1529121, -82.4541321, 83.1538696
2: -56.3006096, 36.9940262, -62.0550880, 40.6739426, -96.9745483, 99.0491180
3: -60.2535667, 31.8628902, -66.2833710, 35.0597382, -95.3133087, 98.1462555
4: -55.5939789, 42.7334175, -61.1852112, 47.0473862, -102.6413574, 103.9185944
5: -49.7072525, 38.6699142, -54.6599007, 42.5267563, -92.2340088, 93.3298035
6: -47.7682152, 45.3457489, -52.5636520, 49.9320526, -97.7002716, 97.9094009
7: -51.5020866, 43.4005737, -56.6810188, 47.7452812, -99.2473679, 100.0815735
8: -62.2823372, 43.0714722, -68.6319351, 47.5239754, -109.8062973, 111.7034073
9: -46.9010963, 45.8373985, -51.6552010, 50.5706177, -97.4717102, 97.4925842

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 254
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
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 124

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7057199, upper bound: 143.7060297
time: 10.34 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7026299, upper bound: 143.7032725
time: 8.50 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -66.0978241, 52.7519226, -65.9274902, 52.6233559, -118.7211761, 118.6794128
1: -54.5876884, 46.7585335, -54.4534416, 46.6400452, -101.2277298, 101.2119446
2: -72.4095001, 47.3294945, -72.2276306, 47.2083511, -119.6178513, 119.5571289
3: -77.0853119, 40.8123703, -76.8623123, 40.7138786, -117.7991867, 117.6746826
4: -71.2460098, 54.8190346, -71.0581207, 54.6853867, -125.9313965, 125.8771439
5: -63.5966759, 49.5203285, -63.4291801, 49.3848419, -112.9815216, 112.9495087
6: -61.1018677, 58.2173767, -60.9418182, 58.0711975, -119.1730652, 119.1591949
7: -66.0123138, 55.6019859, -65.8335419, 55.4489250, -121.4612427, 121.4355316
8: -80.0189209, 55.5473595, -79.8343735, 55.4491196, -135.4680176, 135.3817291
9: -60.2368393, 59.1397324, -60.0867958, 59.0005569, -119.2373962, 119.2265091

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 62

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7293195, upper bound: 143.7296853
time: 10.20 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7574174, upper bound: 143.7574067
time: 8.16 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -67.7003021, 54.0219154, -70.5901489, 56.3116493, -124.0119476, 124.6120605
1: -55.9393768, 47.8901634, -58.3694763, 49.9270973, -105.8664703, 106.2596436
2: -74.1813507, 48.4664536, -77.3776245, 50.5076637, -124.6890030, 125.8440704
3: -78.9873962, 41.7933273, -82.3852463, 43.5781670, -122.5655518, 124.1785660
4: -72.9862137, 56.1395378, -76.1217880, 58.5140610, -131.5002747, 132.2612915
5: -65.1443558, 50.7182922, -67.9212570, 52.8490295, -117.9933777, 118.6395416
6: -62.5884476, 59.6393623, -65.2838669, 62.2048111, -124.7932434, 124.9232330
7: -67.6327057, 56.9617271, -70.5300522, 59.3738403, -127.0065460, 127.4917755
8: -81.9501724, 56.8520508, -85.4682236, 59.2582855, -141.2084045, 142.3202820
9: -61.7083397, 60.5863152, -64.3471603, 63.1918335, -124.9001770, 124.9334717

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 62

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7293913, upper bound: 143.7297469
time: 10.19 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7581402, upper bound: 143.7581402
time: 8.27 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 19.92 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 19.92
Output dim: 4, lower bound: -143.7049218, upper bound: 143.7052726
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 19.92
Output dim: 4, lower bound: -143.7019100, upper bound: 143.7025644
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 19.92
Output dim: 4, lower bound: -143.7057199, upper bound: 143.7060297
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 19.92
Output dim: 4, lower bound: -143.7026299, upper bound: 143.7032725
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 19.92
Output dim: 4, lower bound: -143.7293195, upper bound: 143.7296853
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 19.92
Output dim: 4, lower bound: -143.7574174, upper bound: 143.7574067
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 19.92
Output dim: 4, lower bound: -143.7293913, upper bound: 143.7297469
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 19.92
Output dim: 4, lower bound: -143.7581402, upper bound: 143.7581402

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -43.6917305, 34.7216606, -47.6277695, 37.9435577, -81.6352844, 82.3494110
1: -35.7030983, 30.9046955, -39.0227051, 33.7076263, -69.4107208, 69.9273834
2: -47.4921265, 31.3559322, -51.9060860, 34.1769829, -81.6690979, 83.2620163
3: -50.8781700, 26.9898720, -55.4447899, 29.4169273, -80.2950974, 82.4346619
4: -47.0087891, 36.1988792, -51.2339935, 39.5005417, -86.5093307, 87.4328766
5: -42.0114899, 32.6712685, -45.7897415, 35.6554070, -77.6669006, 78.4610062
6: -40.4430161, 38.3417015, -44.0589409, 41.8333931, -82.2764130, 82.4006424
7: -43.5183067, 36.6843414, -47.4533806, 40.0248146, -83.5431137, 84.1377258
8: -52.6352158, 36.4106369, -57.5252647, 39.8891792, -92.5243912, 93.9358978
9: -39.6184807, 38.6708641, -43.2683220, 42.3121223, -81.9305954, 81.9391861

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6942982, upper bound: 143.6946722
time: 7.95 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6940661, upper bound: 143.6944191
time: 9.51 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -41.1513634, 32.6454163, -42.7965927, 34.0654945, -75.2168579, 75.4420090
1: -33.6758041, 29.1902103, -35.0298920, 30.3279762, -64.0037842, 64.2201004
2: -44.7377853, 29.5901775, -46.5656052, 30.7634773, -75.5012589, 76.1557693
3: -47.9354973, 25.4466496, -49.7778587, 26.4569244, -74.3924026, 75.2245102
4: -44.3009300, 34.1520157, -46.0459518, 35.5416946, -79.8426208, 80.1979675
5: -39.5571327, 30.7455750, -41.1254425, 32.0124016, -71.5695190, 71.8710175
6: -38.1721611, 36.1462326, -39.6245308, 37.5933647, -75.7655182, 75.7707596
7: -41.0777626, 34.5597191, -42.6349373, 35.9675255, -77.0452881, 77.1946487
8: -49.6002846, 34.2131424, -51.6485825, 35.8199234, -85.4202118, 85.8617096
9: -37.2957458, 36.4092827, -38.8631859, 37.9818649, -75.2775955, 75.2724686

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6909964, upper bound: 143.6916966
time: 9.33 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6907456, upper bound: 143.6914472
time: 9.53 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -45.1933441, 35.9153519, -52.2245178, 41.5841064, -86.7774506, 88.1398621
1: -36.9597321, 31.9656830, -42.8593750, 36.9430542, -73.9027786, 74.8250427
2: -49.1546402, 32.4155388, -56.9861794, 37.4204865, -86.5751038, 89.4017181
3: -52.6699295, 27.9081650, -60.9054298, 32.2400742, -84.9099808, 88.8135910
4: -48.6371994, 37.4362144, -56.2257576, 43.2804604, -91.9176636, 93.6619644
5: -43.4626884, 33.7958336, -50.2161484, 39.0745735, -82.5372620, 84.0119781
6: -41.8322754, 39.6675301, -48.3345833, 45.8943787, -87.7266541, 88.0021133
7: -45.0386238, 37.9600525, -52.0904121, 43.8928070, -88.9314270, 90.0504608
8: -54.4509735, 37.6331825, -63.0882187, 43.6523590, -98.1033096, 100.7214050
9: -40.9949951, 40.0203819, -47.4626198, 46.4359093, -87.4308853, 87.4829941

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 170

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6964704, upper bound: 143.6968797
time: 9.40 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6963455, upper bound: 143.6967306
time: 10.72 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -42.6782227, 33.8588676, -47.2822800, 37.6230583, -80.3012848, 81.1411362
1: -34.9481659, 30.2663689, -38.7663918, 33.4892235, -68.4373932, 69.0327606
2: -46.4226227, 30.6631851, -51.5285454, 33.9271126, -80.3497314, 82.1917267
3: -49.7585106, 26.3791981, -55.1087952, 29.2042656, -78.9627762, 81.4879761
4: -45.9507370, 35.4095039, -50.9054184, 39.2286072, -85.1793365, 86.3149261
5: -41.0320969, 31.8881226, -45.4402847, 35.3578415, -76.3899384, 77.3284073
6: -39.5815620, 37.4901581, -43.7926598, 41.5512505, -81.1328049, 81.2828140
7: -42.6190834, 35.8536453, -47.1630592, 39.7481537, -82.3672333, 83.0167007
8: -51.4421082, 35.4565468, -57.0935936, 39.4870453, -90.9291534, 92.5501404
9: -38.6950417, 37.7792053, -42.9575844, 42.0023842, -80.6974258, 80.7367859

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 242

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6930345, upper bound: 143.6937560
time: 10.97 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6929986, upper bound: 143.6937079
time: 9.39 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -52.6846390, 42.0366020, -53.9786644, 43.0918884, -95.7765045, 96.0152588
1: -43.0815468, 37.2236023, -44.2285614, 38.1622162, -81.2437592, 81.4521637
2: -57.5609055, 37.8458824, -59.0118675, 38.7550850, -96.3159866, 96.8577499
3: -61.5337181, 32.4584885, -62.9651260, 33.2937775, -94.8274994, 95.4236069
4: -56.7752151, 43.5863533, -58.1665916, 44.7392464, -101.5144501, 101.7529373
5: -50.7385330, 39.5246811, -51.9595375, 40.4857750, -91.2242889, 91.4842224
6: -48.6767921, 46.2444115, -49.8732224, 47.4283485, -96.1051254, 96.1176300
7: -52.5583115, 44.3126602, -53.8403091, 45.4016113, -97.9599228, 98.1529694
8: -63.5999222, 43.9793854, -65.2348404, 45.2380791, -108.8379974, 109.2142258
9: -47.9183044, 46.8608818, -49.1270447, 48.1074753, -96.0257568, 95.9879227

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 62

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7193711, upper bound: 143.7200702
time: 8.64 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7190193, upper bound: 143.7196280
time: 9.67 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -62.1046066, 49.5767899, -63.1607285, 50.4237480, -112.5283508, 112.7375183
1: -51.1724052, 43.9286270, -52.0860214, 44.6790123, -95.8514099, 96.0146484
2: -67.9954453, 44.5092125, -69.1691513, 45.2540512, -113.2494965, 113.6783600
3: -72.4436569, 38.3326035, -73.6461487, 38.9955139, -111.4391708, 111.9787521
4: -66.9392929, 51.5065804, -68.0735016, 52.3900871, -119.3293610, 119.5800705
5: -59.7673569, 46.5496864, -60.7755737, 47.3267670, -107.0941238, 107.3252563
6: -57.4065323, 54.6573677, -58.3808708, 55.6037750, -113.0103073, 113.0382385
7: -62.0106926, 52.2471809, -63.0607719, 53.1243553, -115.1350403, 115.3079529
8: -75.1494217, 52.1594162, -76.4601669, 53.1015472, -128.2509460, 128.6195679
9: -56.5748138, 55.5066795, -57.5492592, 56.4832687, -113.0580826, 113.0559387

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 62

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7503081, upper bound: 143.7503974
time: 8.01 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7499994, upper bound: 143.7499878
time: 5.88 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -54.3317261, 43.3419800, -58.5488892, 46.7118187, -101.0435181, 101.8908615
1: -44.4504547, 38.3796272, -48.0491409, 41.3793869, -85.8298416, 86.4287643
2: -59.3842468, 39.0097618, -64.0622864, 41.9838333, -101.3680801, 103.0720291
3: -63.4886818, 33.4595413, -68.3929443, 36.1001587, -99.5888367, 101.8524780
4: -58.5651054, 44.9474068, -63.1318779, 48.4978943, -107.0629883, 108.0792770
5: -52.3285446, 40.7559662, -56.3650360, 43.8830795, -96.2116241, 97.1210022
6: -50.2033310, 47.7022972, -54.1316109, 51.4700203, -101.6733551, 101.8339005
7: -54.2240829, 45.7114220, -58.4518394, 49.2497597, -103.4738464, 104.1632614
8: -65.5847855, 45.3278160, -70.7584457, 48.9803772, -114.5651627, 116.0862579
9: -49.4312477, 48.3410339, -53.2999916, 52.2146683, -101.6459198, 101.6410217

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 62

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7221547, upper bound: 143.7226617
time: 8.17 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7220276, upper bound: 143.7224899
time: 7.98 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -63.7056007, 50.8456383, -67.7994232, 54.0936203, -117.7992249, 118.6450653
1: -52.5223312, 45.0595360, -55.9809227, 47.9475174, -100.4698486, 101.0404434
2: -69.7647934, 45.6449051, -74.2953796, 48.5368004, -118.3015900, 119.9402771
3: -74.3441010, 39.3135376, -79.1446381, 41.8443909, -116.1884842, 118.4581604
4: -68.6780014, 52.8268204, -73.1131744, 56.2007484, -124.8787537, 125.9399872
5: -61.3136559, 47.7473373, -65.2467499, 50.7740250, -112.0876770, 112.9940872
6: -58.8915291, 56.0776825, -62.7011642, 59.7184792, -118.6100082, 118.7788467
7: -63.6299057, 53.6063309, -67.7361526, 57.0303383, -120.6602478, 121.3424835
8: -77.0788498, 53.4635353, -82.0664368, 56.8901329, -133.9689789, 135.5299683
9: -58.0448570, 56.9515305, -61.7890701, 60.6557884, -118.7006454, 118.7406006

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7542361, upper bound: 143.7543292
time: 6.95 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7541134, upper bound: 143.7541134
time: 8.39 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 16.68 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.68
Output dim: 4, lower bound: -143.6942982, upper bound: 143.6946722
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.68
Output dim: 4, lower bound: -143.6940661, upper bound: 143.6944191
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.68
Output dim: 4, lower bound: -143.6909964, upper bound: 143.6916966
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.68
Output dim: 4, lower bound: -143.6907456, upper bound: 143.6914472
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.68
Output dim: 4, lower bound: -143.6964704, upper bound: 143.6968797
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.68
Output dim: 4, lower bound: -143.6963455, upper bound: 143.6967306
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.68
Output dim: 4, lower bound: -143.6930345, upper bound: 143.6937560
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.68
Output dim: 4, lower bound: -143.6929986, upper bound: 143.6937079
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.68
Output dim: 4, lower bound: -143.7193711, upper bound: 143.7200702
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.68
Output dim: 4, lower bound: -143.7190193, upper bound: 143.7196280
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.68
Output dim: 4, lower bound: -143.7503081, upper bound: 143.7503974
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.68
Output dim: 4, lower bound: -143.7499994, upper bound: 143.7499878
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.68
Output dim: 4, lower bound: -143.7221547, upper bound: 143.7226617
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.68
Output dim: 4, lower bound: -143.7220276, upper bound: 143.7224899
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.68
Output dim: 4, lower bound: -143.7542361, upper bound: 143.7543292
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.68
Output dim: 4, lower bound: -143.7541134, upper bound: 143.7541134

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -39.3846436, 31.2956524, -41.3637123, 32.9646034, -72.3492432, 72.6593475
1: -32.1577606, 27.8972969, -33.8518944, 29.3202133, -61.4779739, 61.7491913
2: -42.7469292, 28.3455544, -44.9879684, 29.7765064, -72.5234375, 73.3335190
3: -45.8222122, 24.3572445, -48.1043854, 25.5842457, -71.4064560, 72.4616318
4: -42.4085426, 32.6998901, -44.5251503, 34.4126282, -76.8211670, 77.2250366
5: -37.8682480, 29.4748306, -39.7749786, 30.9957161, -68.8639679, 69.2498016
6: -36.5396271, 34.5643883, -38.3665619, 36.3338890, -72.8735199, 72.9309464
7: -39.2154160, 33.0769844, -41.1644630, 34.7736397, -73.9890518, 74.2414474
8: -47.4695435, 32.8854065, -49.9976044, 34.7495308, -82.2190704, 82.8830109
9: -35.7054863, 34.7920418, -37.5608025, 36.6638374, -72.3693237, 72.3528442

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 247

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6925983, upper bound: 143.6929924
time: 34.01 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6942982, upper bound: 143.6946722
time: 10.10 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -37.4765511, 29.7809296, -42.6011734, 33.9345169, -71.4110565, 72.3821030
1: -30.5976009, 26.5784721, -34.8718109, 30.1898708, -60.7874718, 61.4502831
2: -40.6546249, 27.0174370, -46.3677139, 30.6593838, -71.3140106, 73.3851471
3: -43.5879631, 23.1898766, -49.5580864, 26.3091202, -69.8970795, 72.7479630
4: -40.3885384, 31.1514721, -45.8753357, 35.4352264, -75.8237610, 77.0268021
5: -36.0385017, 28.0599194, -40.9582291, 31.9063911, -67.9448776, 69.0181427
6: -34.8147888, 32.8958893, -39.5494766, 37.4319077, -72.2466888, 72.4453430
7: -37.3327866, 31.4864502, -42.4342422, 35.8017845, -73.1345673, 73.9206772
8: -45.1781464, 31.3229675, -51.5338898, 35.7719841, -80.9501343, 82.8568573
9: -33.9820518, 33.0864487, -38.6669006, 37.7466660, -71.7287140, 71.7533264

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 247

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6853591, upper bound: 143.6857154
time: 8.02 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6846800, upper bound: 143.6850513
time: 10.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -36.9560051, 29.3121204, -36.6907654, 29.2067509, -66.1627502, 66.0028839
1: -30.2362633, 26.2761364, -29.9930096, 26.0591812, -56.2954445, 56.2691460
2: -40.1284485, 26.6639366, -39.8362808, 26.4940224, -66.6224670, 66.5002136
3: -43.0265656, 22.9001656, -42.6180077, 22.7229595, -65.7495270, 65.5181732
4: -39.8329620, 30.7434120, -39.5282173, 30.5826321, -70.4155960, 70.2716293
5: -35.5294952, 27.6395226, -35.2552261, 27.4752808, -63.0047684, 62.8947334
6: -34.3798294, 32.4724655, -34.0927467, 32.2364693, -66.6162949, 66.5652084
7: -36.8994293, 31.0559349, -36.5282059, 30.8522530, -67.7516785, 67.5841370
8: -44.5737343, 30.7897778, -44.3255196, 30.8227882, -75.3965225, 75.1152954
9: -33.4976845, 32.6363602, -33.3153458, 32.4740067, -65.9716873, 65.9516983

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 124

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6890825, upper bound: 143.6897840
time: 8.16 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6909964, upper bound: 143.6916966
time: 8.62 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -35.1919022, 27.9123554, -37.9533844, 30.1979084, -65.3898087, 65.8657379
1: -28.7837830, 25.0601902, -31.0336800, 26.9525757, -55.7363548, 56.0938683
2: -38.1954956, 25.4396191, -41.2431908, 27.3995094, -65.5950012, 66.6828079
3: -40.9638977, 21.8270664, -44.1086845, 23.4733295, -64.4372253, 65.9357529
4: -37.9570312, 29.3137035, -40.8977737, 31.6276379, -69.5846710, 70.2114716
5: -33.8325005, 26.3361950, -36.4647942, 28.4083328, -62.2408333, 62.8009834
6: -32.7891235, 30.9263306, -35.3008804, 33.3576317, -66.1467514, 66.2271881
7: -35.1520882, 29.5843792, -37.8238297, 31.9069767, -67.0590668, 67.4082031
8: -42.4511719, 29.3568020, -45.8902435, 31.8707657, -74.3219299, 75.2470322
9: -31.9063797, 31.0641804, -34.4466476, 33.5793571, -65.4857330, 65.5108261

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 124

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6887626, upper bound: 143.6894612
time: 9.43 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6907456, upper bound: 143.6914472
time: 9.26 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -40.7033691, 32.3445168, -45.6306419, 36.3542519, -77.0576172, 77.9751587
1: -33.2545509, 28.8237324, -37.4096031, 32.3239288, -65.5784683, 66.2333374
2: -44.2004852, 29.2691269, -49.7079849, 32.7812614, -76.9817352, 78.9771042
3: -47.3983002, 25.1612377, -53.1811638, 28.1931534, -75.5914536, 78.3423996
4: -43.8314247, 33.7856903, -49.1433144, 37.9166985, -81.7481232, 82.9290009
5: -39.1433144, 30.4612999, -43.8856544, 34.1738853, -73.3171921, 74.3469543
6: -37.7567978, 35.7261925, -42.3294144, 40.0981522, -77.8549271, 78.0556030
7: -40.5459251, 34.1960144, -45.4658661, 38.3677597, -78.9136810, 79.6618805
8: -49.0633507, 33.9544067, -55.1728706, 38.2413330, -87.3046875, 89.1272736
9: -36.9117279, 35.9717331, -41.4564896, 40.4819221, -77.3936462, 77.4282227

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 124

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6953846, upper bound: 143.6957650
time: 10.10 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6964704, upper bound: 143.6968797
time: 7.68 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -39.0564537, 31.0382824, -47.6156006, 37.9211082, -76.9775467, 78.6538773
1: -31.9077282, 27.6858940, -39.0555267, 33.7230034, -65.6307297, 66.7414246
2: -42.3942986, 28.1265774, -51.9124336, 34.1968498, -76.5911484, 80.0389938
3: -45.4743996, 24.1484547, -55.5244789, 29.3772125, -74.8515930, 79.6729202
4: -42.0930328, 32.4492455, -51.3062134, 39.5573845, -81.6504135, 83.7554626
5: -37.5651436, 29.2425804, -45.7913094, 35.6439133, -73.2090607, 75.0338898
6: -36.2727966, 34.2855186, -44.2046890, 41.8565788, -78.1293793, 78.4901962
7: -38.9243851, 32.8244743, -47.4941635, 40.0280457, -78.9524307, 80.3186340
8: -47.0830231, 32.6048851, -57.6198959, 39.8829460, -86.9659653, 90.2247543
9: -35.4263878, 34.5005913, -43.2478065, 42.2420425, -77.6684265, 77.7483978

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 124

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6952812, upper bound: 143.6956218
time: 8.16 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6963455, upper bound: 143.6967306
time: 10.31 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -38.2442894, 30.3375912, -40.8393517, 32.5009651, -70.7452393, 71.1769409
1: -31.3062325, 27.1762543, -33.4465752, 28.9741974, -60.2804298, 60.6228294
2: -41.5467911, 27.5663433, -44.4140282, 29.4021111, -70.9488983, 71.9803696
3: -44.5591011, 23.6764793, -47.5553055, 25.2571468, -69.8162460, 71.2317810
4: -41.2237740, 31.8022842, -44.0012207, 33.9876022, -75.2113800, 75.8035049
5: -36.7741051, 28.6031399, -39.2511101, 30.5634136, -67.3375168, 67.8542328
6: -35.5659485, 33.6057472, -37.9355049, 35.8934326, -71.4593811, 71.5412521
7: -38.1963806, 32.1467056, -40.6973915, 34.3428802, -72.5392609, 72.8441010
8: -46.1262779, 31.8301659, -49.3489609, 34.2053833, -80.3316650, 81.1791229
9: -34.6725655, 33.7854538, -37.0883369, 36.1896210, -70.8621826, 70.8737946

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 161

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6918904, upper bound: 143.6926220
time: 10.38 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6930345, upper bound: 143.6937560
time: 9.24 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -36.7008591, 29.1138306, -42.8633232, 34.0980225, -70.7988739, 71.9771500
1: -30.0413380, 26.1152248, -35.1212044, 30.4000340, -60.4413719, 61.2364273
2: -39.8562164, 26.4969521, -46.6623917, 30.8439770, -70.7001953, 73.1593246
3: -42.7590714, 22.7345676, -49.9386978, 26.4622650, -69.2213364, 72.6732483
4: -39.5912476, 30.5525417, -46.1977272, 35.6600609, -75.2512970, 76.7502518
5: -35.2940903, 27.4654541, -41.1934128, 32.0622635, -67.3563538, 68.6588593
6: -34.1792564, 32.2551994, -39.8423538, 37.6841354, -71.8633881, 72.0975494
7: -36.6753120, 30.8630085, -42.7614059, 36.0370064, -72.7123184, 73.6244049
8: -44.2721710, 30.5753860, -51.8426285, 35.8786125, -80.1507797, 82.4180069
9: -33.2831993, 32.4109459, -38.9130669, 37.9813499, -71.2645416, 71.3240128

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 119

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6918222, upper bound: 143.6925180
time: 7.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6929986, upper bound: 143.6937079
time: 8.55 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -48.1352005, 38.3900185, -47.2350540, 37.6931458, -85.8283386, 85.6250763
1: -39.3331070, 34.0441017, -38.6557579, 33.4497185, -72.7828064, 72.6998596
2: -52.5300407, 34.6231918, -51.5578766, 33.9817581, -86.5117950, 86.1810684
3: -56.2067108, 29.6743622, -55.0712051, 29.1607933, -85.3675079, 84.7455597
4: -51.8684998, 39.8493652, -50.8834610, 39.2004509, -91.0689468, 90.7328186
5: -46.3373413, 36.0974503, -45.4316673, 35.4134636, -81.7508087, 81.5291138
6: -44.4927406, 42.2395859, -43.6686134, 41.4907608, -85.9834976, 85.9081802
7: -48.0097656, 40.4875603, -47.0985184, 39.7414703, -87.7512360, 87.5860748
8: -58.0940933, 40.1447029, -57.0894623, 39.5481796, -97.6422729, 97.2341614
9: -43.7637138, 42.7699509, -42.9740753, 42.0361214, -85.7998352, 85.7440033

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 124

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7124410, upper bound: 143.7129925
time: 10.13 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7119636, upper bound: 143.7126460
time: 8.04 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -43.4472351, 34.6247520, -44.9664116, 35.8374710, -79.2847061, 79.5911560
1: -35.4686661, 30.7706261, -36.8549309, 31.9228630, -67.3915253, 67.6255569
2: -47.3441696, 31.3098736, -49.1000481, 32.3980484, -79.7422180, 80.4099197
3: -50.7156296, 26.8063450, -52.4584389, 27.7838821, -78.4995117, 79.2647781
4: -46.8303871, 36.0071564, -48.4597054, 37.3813934, -84.2117691, 84.4668579
5: -41.8104134, 32.5596504, -43.2400169, 33.6856384, -75.4960480, 75.7996597
6: -40.1964645, 38.1127319, -41.6443939, 39.5254517, -79.7219162, 79.7571259
7: -43.3357658, 36.5401611, -44.9125061, 37.8415527, -81.1773148, 81.4526596
8: -52.4036903, 36.1953430, -54.3932457, 37.5760460, -89.9797287, 90.5885849
9: -39.4866447, 38.5727882, -40.8979111, 40.0163155, -79.5029526, 79.4707031

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 124

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7120713, upper bound: 143.7125607
time: 8.69 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7116105, upper bound: 143.7122253
time: 9.56 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -57.4121513, 45.8190117, -56.3182182, 44.9467545, -102.3589020, 102.1372223
1: -47.2808952, 40.6460152, -46.4068260, 39.8901711, -87.1710663, 87.0528412
2: -62.8163528, 41.1820107, -61.6150703, 40.4004059, -103.2167587, 102.7970810
3: -66.9423981, 35.4535255, -65.6243591, 34.7964058, -101.7388000, 101.0778809
4: -61.8723793, 47.6541100, -60.6820335, 46.7738113, -108.6461792, 108.3361435
5: -55.2236786, 43.0188866, -54.1487694, 42.1800308, -97.4037094, 97.1676559
6: -53.0852127, 50.5277596, -52.0764122, 49.5773582, -102.6625595, 102.6041718
7: -57.3175468, 48.3045616, -56.2166176, 47.3744621, -104.6920013, 104.5211792
8: -69.4833755, 48.2012138, -68.1956100, 47.3313866, -116.8147583, 116.3968201
9: -52.2854652, 51.2719154, -51.2929344, 50.3067780, -102.5922394, 102.5648346

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
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
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 62

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7422444, upper bound: 143.7422052
time: 7.62 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7419673, upper bound: 143.7419966
time: 7.65 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -52.5233612, 41.9029121, -53.8610229, 42.9438591, -95.4672165, 95.7639313
1: -43.2232590, 37.2287788, -44.4353561, 38.2277794, -81.4510269, 81.6641388
2: -57.4262199, 37.7238579, -58.9527092, 38.6811180, -96.1073303, 96.6765671
3: -61.2141342, 32.4433441, -62.7821465, 33.2955933, -94.5097198, 95.2254868
4: -56.6031075, 43.6448059, -58.0410652, 44.7994728, -101.4025650, 101.6858673
5: -50.4913177, 39.3437958, -51.7696762, 40.3175011, -90.8088226, 91.1134720
6: -48.5877113, 46.2256660, -49.8628235, 47.4436226, -96.0313187, 96.0884857
7: -52.4449387, 44.2042503, -53.8371048, 45.3219566, -97.7668762, 98.0413513
8: -63.5708885, 44.0778656, -65.2801437, 45.1891975, -108.7600861, 109.3580093
9: -47.8268929, 46.8784409, -49.0362167, 48.1131516, -95.9400482, 95.9146576

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7418844, upper bound: 143.7417026
time: 8.97 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7416405, upper bound: 143.7415330
time: 8.43 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -49.6438293, 39.6276474, -51.7898941, 41.3577690, -91.0016022, 91.4175262
1: -40.6128044, 35.1152344, -42.4571571, 36.6535301, -77.2663269, 77.5723877
2: -54.2107468, 35.7138557, -56.6088676, 37.2277184, -91.4384613, 92.3227234
3: -58.0099831, 30.5898647, -60.4867325, 31.9425640, -89.9525452, 91.0765991
4: -53.5240021, 41.1276169, -55.8575859, 42.9911995, -96.5151901, 96.9851990
5: -47.8242874, 37.2761536, -49.8648186, 38.8728333, -86.6971207, 87.1409760
6: -45.9297028, 43.5736313, -47.9660835, 45.5144043, -91.4441071, 91.5397110
7: -49.5204659, 41.7843399, -51.6690788, 43.5949478, -93.1154175, 93.4534073
8: -59.9650536, 41.4817123, -62.6728935, 43.4251328, -103.3901825, 104.1546021
9: -45.1653175, 44.1230011, -47.1505241, 46.1098557, -91.2751770, 91.2735214

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 124

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7130311, upper bound: 143.7135499
time: 9.61 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7124702, upper bound: 143.7130786
time: 8.63 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -47.7407608, 38.1179886, -53.3984375, 42.6266975, -90.3674469, 91.5164261
1: -39.0529518, 33.7941780, -43.7880325, 37.7868347, -76.8397827, 77.5822144
2: -52.1128120, 34.3819237, -58.3997803, 38.3785172, -90.4913101, 92.7817078
3: -55.7886620, 29.4216442, -62.3895988, 32.8923416, -88.6809998, 91.8112411
4: -51.4945450, 39.5800705, -57.6043587, 44.3221245, -95.8166656, 97.1844330
5: -46.0005798, 35.8658104, -51.4104042, 40.0655937, -86.0661621, 87.2762070
6: -44.2089767, 41.8988152, -49.5031929, 46.9410248, -91.1500015, 91.4020081
7: -47.6297493, 40.1914749, -53.3179665, 44.9450188, -92.5747452, 93.5094299
8: -57.6775398, 39.9148216, -64.6681366, 44.7526398, -102.4301682, 104.5829468
9: -43.4373741, 42.4183083, -48.6041260, 47.5348091, -90.9721832, 91.0224304

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 124

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7129013, upper bound: 143.7133540
time: 9.15 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7123264, upper bound: 143.7128872
time: 10.15 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -58.8703842, 47.0162392, -60.9480934, 48.6715775, -107.5419617, 107.9643250
1: -48.5081329, 41.6768456, -50.2783394, 43.1490593, -91.6571884, 91.9551849
2: -64.4369049, 42.2375679, -66.7395325, 43.7023392, -108.1392441, 108.9770966
3: -68.6876831, 36.3375282, -71.1330338, 37.6270981, -106.3147812, 107.4705582
4: -63.4783936, 48.8871994, -65.7383957, 50.6230698, -114.1014557, 114.6255951
5: -56.6637726, 44.1618309, -58.6556778, 45.6967163, -102.3604736, 102.8175049
6: -54.4794273, 51.8134041, -56.4427071, 53.6637001, -108.1431274, 108.2561111
7: -58.7788544, 49.5593681, -60.8607330, 51.2947464, -110.0736008, 110.4200897
8: -71.2913437, 49.4930191, -73.8600006, 51.2663116, -122.5576553, 123.3530197
9: -53.6389275, 52.5748978, -55.5410576, 54.4502220, -108.0891495, 108.1159515

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 170

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7457743, upper bound: 143.7457376
time: 7.76 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7451802, upper bound: 143.7452860
time: 8.92 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 18.04 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.04
Output dim: 4, lower bound: -143.6925983, upper bound: 143.6929924
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.04
Output dim: 4, lower bound: -143.6942982, upper bound: 143.6946722
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.04
Output dim: 4, lower bound: -143.6853591, upper bound: 143.6857154
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.04
Output dim: 4, lower bound: -143.6846800, upper bound: 143.6850513
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.04
Output dim: 4, lower bound: -143.6890825, upper bound: 143.6897840
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.04
Output dim: 4, lower bound: -143.6909964, upper bound: 143.6916966
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.04
Output dim: 4, lower bound: -143.6887626, upper bound: 143.6894612
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.04
Output dim: 4, lower bound: -143.6907456, upper bound: 143.6914472
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.04
Output dim: 4, lower bound: -143.6953846, upper bound: 143.6957650
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.04
Output dim: 4, lower bound: -143.6964704, upper bound: 143.6968797
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.04
Output dim: 4, lower bound: -143.6952812, upper bound: 143.6956218
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.04
Output dim: 4, lower bound: -143.6963455, upper bound: 143.6967306
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.04
Output dim: 4, lower bound: -143.6918904, upper bound: 143.6926220
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.04
Output dim: 4, lower bound: -143.6930345, upper bound: 143.6937560
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.04
Output dim: 4, lower bound: -143.6918222, upper bound: 143.6925180
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.04
Output dim: 4, lower bound: -143.6929986, upper bound: 143.6937079
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.04
Output dim: 4, lower bound: -143.7124410, upper bound: 143.7129925
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.04
Output dim: 4, lower bound: -143.7119636, upper bound: 143.7126460
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.04
Output dim: 4, lower bound: -143.7120713, upper bound: 143.7125607
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.04
Output dim: 4, lower bound: -143.7116105, upper bound: 143.7122253
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.04
Output dim: 4, lower bound: -143.7422444, upper bound: 143.7422052
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.04
Output dim: 4, lower bound: -143.7419673, upper bound: 143.7419966
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.04
Output dim: 4, lower bound: -143.7418844, upper bound: 143.7417026
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.04
Output dim: 4, lower bound: -143.7416405, upper bound: 143.7415330
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.04
Output dim: 4, lower bound: -143.7130311, upper bound: 143.7135499
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.04
Output dim: 4, lower bound: -143.7124702, upper bound: 143.7130786
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.04
Output dim: 4, lower bound: -143.7129013, upper bound: 143.7133540
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.04
Output dim: 4, lower bound: -143.7123264, upper bound: 143.7128872
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.04
Output dim: 4, lower bound: -143.7457743, upper bound: 143.7457376
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.04
Output dim: 4, lower bound: -143.7451802, upper bound: 143.7452860
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.04
Output dim: 4, lower bound: -143.7541134, upper bound: 143.7541134
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=145.0042724609375
rel_dist={4: [-143.7624133928287, 143.76241339486478]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1832.75 seconds
