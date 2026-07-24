## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2000 seconds
Threshold: 106.6602947382
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797)
1: (-48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317)
2: (-65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747)
3: (-68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471)
4: (-64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390)
5: (-57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115)
6: (-54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235)
7: (-58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829)
8: (-71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617)
9: (-54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888)

## BASE Result
execution time: IAR + LP analysis = 1.34 + 9.28 = 10.62 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -106.7671438, upper bound: 106.7671437


# Binary Search by BASE starts (time budget: 1989.38 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=108.02017974853516
rel_dist={0: [-106.7671031285499, 106.7671031285499]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=108.02017974853516
rel_dist={0: [-106.76706178863964, 106.76706178863964]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=108.02017974853516
rel_dist={0: [-106.76692036843552, 106.76692036843548]}

## Binary Search Result
Binary search time: 38.33 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1951.04 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7451674, upper bound: 106.7345582
time: 9.49 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7301109, upper bound: 106.7301109
time: 6.69 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 16.32 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 16.32
Output dim: 0, lower bound: -106.7451674, upper bound: 106.7345582
IS_A2, status: Status.UNKNOWN, split count: 1, time: 16.32
Output dim: 0, lower bound: -106.7301109, upper bound: 106.7301109

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -52.2137947, 41.5897751, -60.1304131, 47.8897781, -100.1035690, 101.7201843
1: -42.4293823, 36.0394936, -48.9951973, 41.5569344, -83.9863129, 85.0346909
2: -56.4136429, 36.4521370, -65.1370010, 42.1076736, -98.5213165, 101.5891342
3: -59.6665001, 31.4038601, -68.8342667, 36.2327805, -95.8992767, 100.2381134
4: -55.6501541, 42.0648537, -64.1016769, 48.5745659, -104.2247162, 106.1665268
5: -50.2768250, 39.4043541, -57.9381943, 45.3561211, -95.6329498, 97.3425293
6: -47.6397820, 45.6848717, -54.9252052, 52.6139221, -100.2537079, 100.6100769
7: -51.0145836, 43.7962303, -58.9155884, 50.4363098, -101.4508896, 102.7117920
8: -61.8245430, 41.1875153, -71.5229416, 47.6925201, -109.5170593, 112.7104416
9: -47.0728683, 45.8725777, -54.2685204, 52.9324722, -100.0053253, 100.1410980

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7278006, upper bound: 106.7196358
time: 8.48 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7266400, upper bound: 106.7161532
time: 9.10 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -34.0230293, 27.0736713, -54.6940079, 43.5616722, -77.5847015, 81.7676773
1: -27.2583656, 23.3865738, -44.4843025, 37.7657433, -65.0241089, 67.8708725
2: -36.4209671, 23.6102924, -59.1399689, 38.2167244, -74.6376877, 82.7502594
3: -38.5580063, 20.3228893, -62.5407028, 32.9206886, -71.4786987, 82.8635941
4: -36.2169151, 27.1975498, -58.2839813, 44.0975723, -80.3144684, 85.4815292
5: -32.4356346, 25.6262245, -52.6798210, 41.2654266, -73.7010574, 78.3060455
6: -31.0156384, 29.8003063, -49.9137001, 47.8540077, -78.8696365, 79.7139893
7: -33.0768013, 28.5532303, -53.4844551, 45.8778458, -78.9546509, 82.0376740
8: -39.6946602, 26.2763023, -64.8331375, 43.2070389, -82.9016876, 91.1094360
9: -30.5448456, 29.6413536, -49.3185081, 48.0785294, -78.6233749, 78.9598389

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 255

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7128966, upper bound: 106.7155416
time: 8.41 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7119263, upper bound: 106.7119264
time: 7.16 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 17.00 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 17.00
Output dim: 0, lower bound: -106.7278006, upper bound: 106.7196358
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 17.00
Output dim: 0, lower bound: -106.7266400, upper bound: 106.7161532
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 17.00
Output dim: 0, lower bound: -106.7128966, upper bound: 106.7155416
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 17.00
Output dim: 0, lower bound: -106.7119263, upper bound: 106.7119264

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -52.2137947, 41.5897751, -43.2367897, 34.4949570, -86.7087402, 84.8265610
1: -42.4293823, 36.0394936, -34.9878044, 29.8007889, -72.2301636, 71.0272980
2: -56.4136429, 36.4521370, -46.5671349, 30.1880875, -86.6017303, 83.0192642
3: -59.6665001, 31.4038601, -49.2782364, 25.9414082, -85.6079102, 80.6820908
4: -55.6501541, 42.0648537, -46.0794792, 34.7557182, -90.4058685, 88.1443329
5: -50.2768250, 39.4043541, -41.5383759, 32.7077026, -82.9845276, 80.9427109
6: -47.6397820, 45.6848717, -39.4550514, 37.8932304, -85.5330124, 85.1399231
7: -51.0145836, 43.7962303, -42.1981354, 36.3016472, -87.3162308, 85.9943542
8: -61.8245430, 41.1875153, -51.0103607, 33.9446335, -95.7691727, 92.1978683
9: -47.0728683, 45.8725777, -38.9905930, 38.0316658, -85.1045227, 84.8631744

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7146186, upper bound: 106.7044425
time: 6.97 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7136022, upper bound: 106.7040081
time: 9.84 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -46.6634941, 37.1883812, -35.7654915, 28.4981251, -75.1616211, 72.9538727
1: -37.8110580, 32.1684380, -28.6854954, 24.5347595, -62.3458099, 60.8539352
2: -50.3005028, 32.5214157, -38.3050308, 24.8694439, -75.1699371, 70.8264465
3: -53.2132874, 28.0238228, -40.5462570, 21.3767605, -74.5900497, 68.5700836
4: -49.7006989, 37.5097046, -37.9453735, 28.6490612, -78.3497543, 75.4550781
5: -44.8910255, 35.2493820, -34.1815300, 27.0604343, -71.9514542, 69.4309082
6: -42.5346069, 40.8410873, -32.5468407, 31.3406620, -73.8752594, 73.3879166
7: -45.4901161, 39.1443634, -34.7955933, 30.0198460, -75.5099640, 73.9399567
8: -55.0627136, 36.6517448, -41.8391533, 27.7593937, -82.8221054, 78.4908981
9: -42.0287323, 40.9401283, -32.1872482, 31.2693481, -73.2980728, 73.1273651

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7137378, upper bound: 106.7014147
time: 10.32 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7127625, upper bound: 106.7011213
time: 8.56 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -34.0230293, 27.0736713, -38.6010666, 30.7999268, -64.8229523, 65.6747360
1: -27.2583656, 23.3865738, -31.1437950, 26.5828953, -53.8412628, 54.5303688
2: -36.4209671, 23.6102924, -41.4810982, 26.9238396, -63.3448067, 65.0913925
3: -38.5580063, 20.3228893, -43.8916588, 23.1327190, -61.6907196, 64.2145462
4: -36.2169151, 27.1975498, -41.1125374, 30.9873962, -67.2043076, 68.3100815
5: -32.4356346, 25.6262245, -37.0021935, 29.2093334, -61.6449585, 62.6284180
6: -31.0156384, 29.8003063, -35.2188034, 33.8418846, -64.8575211, 65.0190964
7: -33.0768013, 28.5532303, -37.6246567, 32.4340439, -65.5108490, 66.1778793
8: -39.6946602, 26.2763023, -45.3681717, 30.1390648, -69.8337250, 71.6444702
9: -30.5448456, 29.6413536, -34.7997742, 33.8853073, -64.4301529, 64.4411163

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6983490, upper bound: 106.7031246
time: 7.31 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6974108, upper bound: 106.6998994
time: 7.36 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -30.1956596, 24.0095978, -31.9751625, 25.4665604, -55.6622200, 55.9847603
1: -24.0566959, 20.7206974, -25.5246277, 21.8966866, -45.9533844, 46.2453232
2: -32.2357750, 20.9598064, -34.1499252, 22.2311687, -54.4669380, 55.1097298
3: -34.1073608, 18.0154457, -36.1275520, 19.0653515, -53.1727142, 54.1429977
4: -32.1012688, 24.1101227, -33.8890877, 25.5779781, -57.6792450, 57.9992104
5: -28.6618786, 22.7439957, -30.4565353, 24.1821327, -52.8440094, 53.2005196
6: -27.5005836, 26.4525566, -29.0773792, 28.0064411, -55.5070152, 55.5299301
7: -29.3305283, 25.3464603, -31.0879078, 26.8262424, -56.1567612, 56.4343681
8: -35.1606255, 23.2337818, -37.2918587, 24.7114429, -59.8720703, 60.5256424
9: -27.0784683, 26.2972260, -28.7440796, 27.9055328, -54.9840012, 55.0413055

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6972580, upper bound: 106.6995194
time: 7.78 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6964472, upper bound: 106.6964472
time: 6.28 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 15.39 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 15.39
Output dim: 0, lower bound: -106.7146186, upper bound: 106.7044425
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 15.39
Output dim: 0, lower bound: -106.7136022, upper bound: 106.7040081
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 15.39
Output dim: 0, lower bound: -106.7137378, upper bound: 106.7014147
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 15.39
Output dim: 0, lower bound: -106.7127625, upper bound: 106.7011213
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 15.39
Output dim: 0, lower bound: -106.6983490, upper bound: 106.7031246
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 15.39
Output dim: 0, lower bound: -106.6974108, upper bound: 106.6998994
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 15.39
Output dim: 0, lower bound: -106.6972580, upper bound: 106.6995194
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 15.39
Output dim: 0, lower bound: -106.6964472, upper bound: 106.6964472

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -37.5701714, 29.9448643, -43.2367897, 34.4949570, -72.0651245, 73.1816483
1: -30.2940674, 25.8668327, -34.9878044, 29.8007889, -60.0948563, 60.8546333
2: -40.3736229, 26.1199093, -46.5671349, 30.1880875, -70.5617065, 72.6870422
3: -42.6850548, 22.4790840, -49.2782364, 25.9414082, -68.6264648, 71.7573242
4: -39.9820671, 30.1261158, -46.0794792, 34.7557182, -74.7377853, 76.2055969
5: -36.0434074, 28.4576359, -41.5383759, 32.7077026, -68.7511139, 69.9960022
6: -34.2014885, 32.9139633, -39.4550514, 37.8932304, -72.0947189, 72.3690109
7: -36.5678253, 31.5788727, -42.1981354, 36.3016472, -72.8694763, 73.7770004
8: -44.0254898, 29.1850643, -51.0103607, 33.9446335, -77.9701233, 80.1954269
9: -33.8262520, 32.8544693, -38.9905930, 38.0316658, -71.8579025, 71.8450623

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7136021, upper bound: 106.7040081
time: 9.38 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7136021, upper bound: 106.7040081
time: 9.74 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -37.0212746, 29.4880219, -40.2501755, 32.1177864, -69.1390533, 69.7381897
1: -29.7897205, 25.4620075, -32.5073280, 27.7270374, -57.5167542, 57.9693336
2: -39.7516670, 25.6717911, -43.2931671, 28.0825653, -67.8342285, 68.9649582
3: -42.0551643, 22.1224766, -45.8067818, 24.1240654, -66.1792297, 67.9292603
4: -39.3350525, 29.6845913, -42.8624306, 32.3313179, -71.6663589, 72.5470200
5: -35.4961929, 28.0030594, -38.6254120, 30.4744263, -65.9706192, 66.6284714
6: -33.6691475, 32.4309502, -36.7154617, 35.2843704, -68.9535141, 69.1464005
7: -35.9998589, 31.1071415, -39.2522202, 33.8140678, -69.8139267, 70.3593597
8: -43.3077049, 28.6815662, -47.3772049, 31.4958057, -74.8035049, 76.0587692
9: -33.3171692, 32.3196297, -36.2949562, 35.3633347, -68.6804886, 68.6145859

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 170

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6647852, upper bound: 106.6675276
time: 8.22 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6522658, upper bound: 106.6474906
time: 7.70 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -32.8774757, 26.2186432, -35.7654915, 28.4981251, -61.3755951, 61.9841309
1: -26.3918095, 22.6089802, -28.6854954, 24.5347595, -50.9265633, 51.2944756
2: -35.2322769, 22.8474655, -38.3050308, 24.8694439, -60.1017227, 61.1524963
3: -37.2304306, 19.6385841, -40.5462570, 21.3767605, -58.6071930, 60.1848373
4: -34.9476242, 26.3464012, -37.9453735, 28.6490612, -63.5966835, 64.2917786
5: -31.4581985, 24.9443455, -34.1815300, 27.0604343, -58.5186272, 59.1258736
6: -29.9276180, 28.8172970, -32.5468407, 31.3406620, -61.2682762, 61.3641319
7: -31.9725075, 27.6545639, -34.7955933, 30.0198460, -61.9923477, 62.4501572
8: -38.4067764, 25.4240398, -41.8391533, 27.7593937, -66.1661682, 67.2631912
9: -29.6136398, 28.7075615, -32.1872482, 31.2693481, -60.8829880, 60.8948021

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7127624, upper bound: 106.7011213
time: 9.85 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7127624, upper bound: 106.7011213
time: 9.68 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -32.6110268, 25.9806099, -33.2940712, 26.5248623, -59.1358871, 59.2746811
1: -26.1408653, 22.3962708, -26.6329575, 22.8147793, -48.9556389, 49.0292282
2: -34.9156151, 22.6095829, -35.5977745, 23.1461582, -58.0617752, 58.2073555
3: -36.9297981, 19.4472523, -37.6700745, 19.8662682, -56.7960587, 57.1173248
4: -34.6188774, 26.1408558, -35.2895889, 26.6556187, -61.2744904, 61.4304428
5: -31.1838245, 24.6931324, -31.7707233, 25.2029209, -56.3867455, 56.4638557
6: -29.6508484, 28.5870132, -30.2842426, 29.1726952, -58.8235435, 58.8712502
7: -31.7017441, 27.4079628, -32.3850975, 27.9480686, -59.6498108, 59.7930603
8: -38.0334129, 25.1617603, -38.8687820, 25.7666721, -63.8000870, 64.0305405
9: -29.3565483, 28.4354706, -29.9563026, 29.0720577, -58.4286041, 58.3917732

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6636612, upper bound: 106.6664394
time: 8.92 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6316044, upper bound: 106.6183436
time: 8.63 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -34.0230293, 27.0736713, -26.6290054, 21.2308712, -55.2538986, 53.7026749
1: -27.2583656, 23.3865738, -21.2019005, 18.2851276, -45.5434952, 44.5884743
2: -36.4209671, 23.6102924, -28.4235287, 18.5550442, -54.9760132, 52.0338135
3: -38.5580063, 20.3228893, -29.9837646, 15.8815470, -54.4395523, 50.3066521
4: -36.2169151, 27.1975498, -28.2837868, 21.3350639, -57.5519753, 55.4813347
5: -32.4356346, 25.6262245, -25.3154335, 20.2082825, -52.6439095, 50.9416580
6: -31.0156384, 29.8003063, -24.2701454, 23.3541603, -54.3697968, 54.0704498
7: -33.0768013, 28.5532303, -25.9543686, 22.4460430, -55.5228386, 54.5075951
8: -39.6946602, 26.2763023, -31.0319023, 20.4986591, -60.1933212, 57.3082047
9: -30.5448456, 29.6413536, -24.0159264, 23.2758884, -53.8207321, 53.6572800

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6974108, upper bound: 106.6998994
time: 7.02 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6974108, upper bound: 106.6998994
time: 7.29 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -31.5845890, 25.1233368, -25.9018421, 20.6304722, -52.2150612, 51.0251732
1: -25.2271957, 21.6937160, -20.5571480, 17.7552071, -42.9824028, 42.2508583
2: -33.7589035, 21.9095573, -27.5955887, 18.0045681, -51.7634735, 49.5051346
3: -35.7245560, 18.8483810, -29.1334953, 15.4222889, -51.1468391, 47.9818764
4: -33.5940628, 25.2401924, -27.4512711, 20.7513561, -54.3454094, 52.6914635
5: -30.0475655, 23.8028316, -24.5899868, 19.6187592, -49.6663246, 48.3928185
6: -28.7724457, 27.6685200, -23.5719986, 22.7136803, -51.4861221, 51.2405167
7: -30.6867752, 26.5173798, -25.2260113, 21.8141098, -52.5008850, 51.7433891
8: -36.7976837, 24.3269081, -30.1241817, 19.8924675, -56.6901474, 54.4510880
9: -28.3358860, 27.4993267, -23.3350525, 22.6075916, -50.9434776, 50.8343811

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6974108, upper bound: 106.6998994
time: 8.42 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6974108, upper bound: 106.6998994
time: 7.24 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -30.1956596, 24.0095978, -21.5376453, 17.1259022, -47.3215637, 45.5472412
1: -24.0566959, 20.7206974, -16.8352699, 14.7077999, -38.7644920, 37.5559692
2: -32.2357750, 20.9598064, -22.7732487, 14.9395657, -47.1753349, 43.7330551
3: -34.1073608, 18.0154457, -24.0428696, 12.7894211, -46.8967819, 42.0583115
4: -32.1012688, 24.1101227, -22.6918888, 17.1695461, -49.2708092, 46.8020096
5: -28.6618786, 22.7439957, -20.3004856, 16.2986336, -44.9605103, 43.0444794
6: -27.5005836, 26.4525566, -19.5620518, 18.8452930, -46.3458786, 46.0146103
7: -29.3305283, 25.3464603, -20.9131985, 18.0972691, -47.4277878, 46.2596588
8: -35.1606255, 23.2337818, -24.8853321, 16.4900188, -51.6506386, 48.1191139
9: -27.0784683, 26.2972260, -19.3216476, 18.6816330, -45.7601013, 45.6188736

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6964472, upper bound: 106.6964472
time: 6.35 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6964472, upper bound: 106.6964472
time: 5.87 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -27.9948692, 22.2522068, -21.8850250, 17.3860912, -45.3809586, 44.1372299
1: -22.2332096, 19.2028294, -17.0893326, 14.9271088, -37.1603088, 36.2921600
2: -29.8374386, 19.4380131, -23.1287403, 15.1468229, -44.9842567, 42.5667496
3: -31.5540638, 16.7042294, -24.4390430, 12.9802227, -44.5342865, 41.1432724
4: -29.7490616, 22.3357601, -23.0271053, 17.4460297, -47.1950912, 45.3628616
5: -26.5160789, 21.0924168, -20.6185017, 16.5155678, -43.0316429, 41.7109184
6: -25.4942150, 24.5303974, -19.8623257, 19.1536388, -44.6478539, 44.3927231
7: -27.1923428, 23.5082932, -21.2467098, 18.3604412, -45.5527840, 44.7550011
8: -32.5523796, 21.5105152, -25.2807846, 16.7342682, -49.2866478, 46.7912979
9: -25.1024437, 24.3611603, -19.6171761, 18.9736423, -44.0760803, 43.9783325

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 255

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6628049, upper bound: 106.6491087
time: 8.30 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6427317, upper bound: 106.6427317
time: 6.66 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 16.40 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.40
Output dim: 0, lower bound: -106.7136021, upper bound: 106.7040081
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.40
Output dim: 0, lower bound: -106.7136021, upper bound: 106.7040081
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.40
Output dim: 0, lower bound: -106.6647852, upper bound: 106.6675276
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 16.40
Output dim: 0, lower bound: -106.6522658, upper bound: 106.6474906
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.40
Output dim: 0, lower bound: -106.7127624, upper bound: 106.7011213
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.40
Output dim: 0, lower bound: -106.7127624, upper bound: 106.7011213
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.40
Output dim: 0, lower bound: -106.6636612, upper bound: 106.6664394
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 16.40
Output dim: 0, lower bound: -106.6316044, upper bound: 106.6183436
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 16.40
Output dim: 0, lower bound: -106.6974108, upper bound: 106.6998994
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 16.40
Output dim: 0, lower bound: -106.6974108, upper bound: 106.6998994
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 16.40
Output dim: 0, lower bound: -106.6974108, upper bound: 106.6998994
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 16.40
Output dim: 0, lower bound: -106.6974108, upper bound: 106.6998994
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 16.40
Output dim: 0, lower bound: -106.6964472, upper bound: 106.6964472
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 16.40
Output dim: 0, lower bound: -106.6964472, upper bound: 106.6964472
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 16.40
Output dim: 0, lower bound: -106.6628049, upper bound: 106.6491087
IS_A2_B2_B2_A2, status: Status.VERIFIED, split count: 4, time: 16.40
Output dim: 0, lower bound: -106.6427317, upper bound: 106.6427317

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -37.5701714, 29.9448643, -30.2983894, 24.1736546, -61.7438278, 60.2432480
1: -30.2940674, 25.8668327, -24.2702217, 20.8249187, -51.1189880, 50.1370506
2: -40.3736229, 26.1199093, -32.4354630, 21.1161346, -61.4897575, 58.5553703
3: -42.6850548, 22.4790840, -34.2638359, 18.0821152, -60.7671700, 56.7429199
4: -39.9820671, 30.1261158, -32.2010345, 24.3079014, -64.2899704, 62.3271484
5: -36.0434074, 28.4576359, -28.9254189, 23.0072803, -59.0506859, 57.3830566
6: -34.2014885, 32.9139633, -27.6293278, 26.5868244, -60.7883072, 60.5432892
7: -36.5678253, 31.5788727, -29.5311012, 25.5281830, -62.0960083, 61.1099739
8: -44.0254898, 29.1850643, -35.3996544, 23.4165115, -67.4420013, 64.5847168
9: -33.8262520, 32.8544693, -27.3463688, 26.5218925, -60.3481369, 60.2008362

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7146186, upper bound: 106.7044425
time: 9.11 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7146186, upper bound: 106.7044425
time: 10.38 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -37.5701714, 29.9448643, -29.4221592, 23.4520073, -61.0221786, 59.3670197
1: -30.2940674, 25.8668327, -23.5109882, 20.1929646, -50.4870300, 49.3778152
2: -40.3736229, 26.1199093, -31.4479980, 20.4644489, -60.8380737, 57.5679054
3: -42.6850548, 22.4790840, -33.2386818, 17.5285931, -60.2136459, 55.7177620
4: -39.9820671, 30.1261158, -31.2157784, 23.6078911, -63.5899582, 61.3418961
5: -36.0434074, 28.4576359, -28.0523510, 22.3039017, -58.3472939, 56.5099869
6: -34.2014885, 32.9139633, -26.7943306, 25.8205051, -60.0219917, 59.7082939
7: -36.5678253, 31.5788727, -28.6619797, 24.7738514, -61.3416710, 60.2408524
8: -44.0254898, 29.1850643, -34.3139229, 22.6856480, -66.7111359, 63.4989853
9: -33.8262520, 32.8544693, -26.5366707, 25.7191715, -59.5454254, 59.3911400

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7146186, upper bound: 106.7044425
time: 8.52 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7146186, upper bound: 106.7044425
time: 7.74 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -37.0212746, 29.4880219, -37.5859909, 30.0014229, -67.0226898, 67.0740128
1: -29.7897205, 25.4620075, -30.3132477, 25.8907452, -55.6804657, 55.7752495
2: -39.7516670, 25.6717911, -40.3934669, 26.2194138, -65.9710846, 66.0652542
3: -42.0551643, 22.1224766, -42.7111588, 22.5090866, -64.5642471, 64.8336258
4: -39.3350525, 29.6845913, -40.0268440, 30.1807079, -69.5157623, 69.7114258
5: -35.4961929, 28.0030594, -36.0240974, 28.4842091, -63.9804001, 64.0271606
6: -33.6691475, 32.4309502, -34.2887917, 32.9655418, -66.6346664, 66.7197266
7: -35.9998589, 31.1071415, -36.6481438, 31.6043682, -67.6042252, 67.7552872
8: -43.3077049, 28.6815662, -44.1691666, 29.3343849, -72.6420898, 72.8507309
9: -33.3171692, 32.3196297, -33.9036522, 32.9992256, -66.3163910, 66.2232819

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6426385, upper bound: 106.6392442
time: 11.16 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6593432, upper bound: 106.6621943
time: 8.01 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -32.8774757, 26.2186432, -24.2926273, 19.3323441, -52.2098198, 50.5112648
1: -26.3918095, 22.6089802, -19.1628304, 16.6132050, -43.0050087, 41.7718010
2: -35.2322769, 22.8474655, -25.7982388, 16.8710804, -52.1033554, 48.6457062
3: -37.2304306, 19.6385841, -27.2355919, 14.4507828, -51.6812134, 46.8741684
4: -34.9476242, 26.3464012, -25.6685677, 19.3991814, -54.3468056, 52.0149689
5: -31.4581985, 24.9443455, -22.9982738, 18.4141979, -49.8723907, 47.9426155
6: -29.9276180, 28.8172970, -22.0769939, 21.2813282, -51.2089424, 50.8942909
7: -31.9725075, 27.6545639, -23.6358089, 20.4335804, -52.4060860, 51.2903748
8: -38.4067764, 25.4240398, -28.1675053, 18.6277008, -57.0344772, 53.5915375
9: -29.6136398, 28.7075615, -21.8402634, 21.1401691, -50.7538071, 50.5478210

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6754659, upper bound: 106.6535736
time: 8.26 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6648545, upper bound: 106.6497386
time: 9.19 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -32.8774757, 26.2186432, -24.6423244, 19.5853596, -52.4628334, 50.8609619
1: -26.3918095, 22.6089802, -19.4170876, 16.8321686, -43.2239761, 42.0260658
2: -35.2322769, 22.8474655, -26.1509151, 17.0733318, -52.3056107, 48.9983788
3: -37.2304306, 19.6385841, -27.6330128, 14.6436996, -51.8741302, 47.2715950
4: -34.9476242, 26.3464012, -26.0013371, 19.6789017, -54.6265259, 52.3477364
5: -31.4581985, 24.9443455, -23.3125057, 18.6287575, -50.0869522, 48.2568512
6: -29.9276180, 28.8172970, -22.3738232, 21.5895176, -51.5171318, 51.1911201
7: -31.9725075, 27.6545639, -23.9705963, 20.6975803, -52.6700821, 51.6251602
8: -38.4067764, 25.4240398, -28.5579834, 18.8749676, -57.2817459, 53.9820213
9: -29.6136398, 28.7075615, -22.1394291, 21.4284668, -51.0420952, 50.8469849

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6754659, upper bound: 106.6535736
time: 9.27 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6648545, upper bound: 106.6497388
time: 8.16 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -32.6110268, 25.9806099, -31.0238037, 24.7205334, -57.3315582, 57.0044136
1: -26.1408653, 22.3962708, -24.7574711, 21.2503986, -47.3912621, 47.1537399
2: -34.9156151, 22.6095829, -33.1371841, 21.5752811, -56.4908981, 55.7467651
3: -36.9297981, 19.4472523, -35.0300560, 18.4854641, -55.4152603, 54.4773064
4: -34.6188774, 26.1408558, -32.8768158, 24.8311424, -59.4500160, 59.0176697
5: -31.1838245, 24.6931324, -29.5483894, 23.4974728, -54.6812973, 54.2415237
6: -29.6508484, 28.5870132, -28.2141914, 27.1858692, -56.8367157, 56.8012009
7: -31.7017441, 27.4079628, -30.1863461, 26.0577869, -57.7595253, 57.5943069
8: -38.0334129, 25.1617603, -36.1768036, 23.9680061, -62.0014191, 61.3385620
9: -29.3565483, 28.4354706, -27.9117107, 27.0767155, -56.4332657, 56.3471794

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6430868, upper bound: 106.6443193
time: 10.12 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6377325, upper bound: 106.6404370
time: 8.96 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -23.5083866, 18.6810322, -26.6290054, 21.2308712, -44.7392578, 45.3100357
1: -18.5061665, 16.1164627, -21.2019005, 18.2851276, -36.7912941, 37.3183632
2: -24.9529324, 16.3100491, -28.4235287, 18.5550442, -43.5079727, 44.7335739
3: -26.3706341, 13.9870529, -29.9837646, 15.8815470, -42.2521820, 43.9708176
4: -24.9241734, 18.7177143, -28.2837868, 21.3350639, -46.2592354, 47.0014954
5: -22.1852741, 17.7204208, -25.3154335, 20.2082825, -42.3935509, 43.0358543
6: -21.4274635, 20.5832615, -24.2701454, 23.3541603, -44.7816200, 44.8534088
7: -22.8259888, 19.7639828, -25.9543686, 22.4460430, -45.2720299, 45.7183533
8: -27.2345123, 17.9850044, -31.0319023, 20.4986591, -47.7331696, 49.0169067
9: -21.0847645, 20.3869476, -24.0159264, 23.2758884, -44.3606529, 44.4028740

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6506319, upper bound: 106.6668864
time: 10.25 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6457721, upper bound: 106.6528549
time: 6.79 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -23.2752666, 18.4808750, -26.6290054, 21.2308712, -44.5061340, 45.1098785
1: -18.2684498, 15.9317112, -21.2019005, 18.2851276, -36.5535774, 37.1336136
2: -24.6654663, 16.1136551, -28.4235287, 18.5550442, -43.2205048, 44.5371819
3: -26.0781326, 13.8163795, -29.9837646, 15.8815470, -41.9596786, 43.8001442
4: -24.6186600, 18.5342636, -28.2837868, 21.3350639, -45.9537239, 46.8180466
5: -21.9377804, 17.5106812, -25.3154335, 20.2082825, -42.1460648, 42.8261147
6: -21.1806221, 20.3771305, -24.2701454, 23.3541603, -44.5347824, 44.6472778
7: -22.5711422, 19.5337181, -25.9543686, 22.4460430, -45.0171814, 45.4880867
8: -26.9134712, 17.7980118, -31.0319023, 20.4986591, -47.4121323, 48.8299141
9: -20.8599720, 20.1556530, -24.0159264, 23.2758884, -44.1358604, 44.1715775

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6506319, upper bound: 106.6668864
time: 7.38 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6457721, upper bound: 106.6528549
time: 7.46 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -21.1116505, 16.7757492, -25.9018421, 20.6304722, -41.7421227, 42.6775742
1: -16.4420033, 14.4446974, -20.5571480, 17.7552071, -34.1972046, 35.0018425
2: -22.3062134, 14.6858902, -27.5955887, 18.0045681, -40.3107834, 42.2814789
3: -23.5796833, 12.5430098, -29.1334953, 15.4222889, -39.0019684, 41.6765060
4: -22.3132744, 16.7820320, -27.4512711, 20.7513561, -43.0646286, 44.2333031
5: -19.7656803, 15.8127747, -24.5899868, 19.6187592, -39.3844376, 40.4027634
6: -19.2425556, 18.4691181, -23.5719986, 22.7136803, -41.9562340, 42.0411110
7: -20.4832191, 17.7040157, -25.2260113, 21.8141098, -42.2973289, 42.9300270
8: -24.4019814, 16.1869106, -30.1241817, 19.8924675, -44.2944489, 46.3110924
9: -18.9094143, 18.2936459, -23.3350525, 22.6075916, -41.5170059, 41.6287003

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6498782, upper bound: 106.6646994
time: 8.33 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6443814, upper bound: 106.6486824
time: 7.02 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -18.1899948, 14.4475346, -25.9018421, 20.6304722, -38.8204651, 40.3493690
1: -13.8884544, 12.3730555, -20.5571480, 17.7552071, -31.6436615, 32.9301949
2: -19.0909615, 12.6232376, -27.5955887, 18.0045681, -37.0955276, 40.2188263
3: -20.1832809, 10.7341928, -29.1334953, 15.4222889, -35.6055679, 39.8676872
4: -19.0752907, 14.4188442, -27.4512711, 20.7513561, -39.8266449, 41.8701172
5: -16.9003220, 13.5609379, -24.5899868, 19.6187592, -36.5190735, 38.1509247
6: -16.5671673, 15.9256372, -23.5719986, 22.7136803, -39.2808456, 39.4976273
7: -17.5479889, 15.1932068, -25.2260113, 21.8141098, -39.3620949, 40.4192200
8: -20.8992119, 13.9342861, -30.1241817, 19.8924675, -40.7916794, 44.0584679
9: -16.1974926, 15.6673307, -23.3350525, 22.6075916, -38.8050842, 39.0023842

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6789672, upper bound: 106.6830238
time: 7.71 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6784864, upper bound: 106.6815360
time: 7.90 seconds

## BFS IS instance: IS_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -20.7082787, 16.4562359, -21.5376453, 17.1259022, -37.8341827, 37.9938774
1: -16.1347885, 14.1615553, -16.8352699, 14.7077999, -30.8425884, 30.9968262
2: -21.8793640, 14.3650713, -22.7732487, 14.9395657, -36.8189240, 37.1383133
3: -23.1035957, 12.2831068, -24.0428696, 12.7894211, -35.8930092, 36.3259773
4: -21.8629742, 16.4701366, -22.6918888, 17.1695461, -39.0325127, 39.1620255
5: -19.4419498, 15.5670757, -20.3004856, 16.2986336, -35.7405815, 35.8675613
6: -18.8687305, 18.1240082, -19.5620518, 18.8452930, -37.7140236, 37.6860580
7: -20.0594883, 17.3818264, -20.9131985, 18.0972691, -38.1567574, 38.2950249
8: -23.8906860, 15.8398094, -24.8853321, 16.4900188, -40.3807030, 40.7251396
9: -18.5478458, 17.9061031, -19.3216476, 18.6816330, -37.2294769, 37.2277451

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_B1_A1_B1

### Relational analysis result of IS_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6498064, upper bound: 106.6648149
time: 7.75 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2

### Relational analysis result of IS_A2_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6438760, upper bound: 106.6470349
time: 6.51 seconds

## BFS IS instance: IS_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -20.6352768, 16.3703289, -21.5376453, 17.1259022, -37.7611771, 37.9079666
1: -16.0287285, 14.0876799, -16.8352699, 14.7077999, -30.7365284, 30.9229507
2: -21.7656612, 14.2775612, -22.7732487, 14.9395657, -36.7052155, 37.0508041
3: -23.0060177, 12.2095718, -24.0428696, 12.7894211, -35.7954407, 36.2524338
4: -21.7329578, 16.4147987, -22.6918888, 17.1695461, -38.9024963, 39.1066818
5: -19.3459110, 15.4800186, -20.3004856, 16.2986336, -35.6445465, 35.7805023
6: -18.7779484, 18.0588951, -19.5620518, 18.8452930, -37.6232414, 37.6209488
7: -19.9606476, 17.2836533, -20.9131985, 18.0972691, -38.0579109, 38.1968536
8: -23.7642612, 15.7690001, -24.8853321, 16.4900188, -40.2542763, 40.6543312
9: -18.4601784, 17.8125362, -19.3216476, 18.6816330, -37.1418114, 37.1341858

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_B1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6498064, upper bound: 106.6648149
time: 7.75 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6438760, upper bound: 106.6470349
time: 7.15 seconds

## BFS IS instance: IS_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -23.6908436, 18.7961655, -21.8850250, 17.3860912, -41.0769348, 40.6811867
1: -18.6362152, 16.2337360, -17.0893326, 14.9271088, -33.5633125, 33.3230667
2: -25.1109467, 16.4272137, -23.1287403, 15.1468229, -40.2577705, 39.5559540
3: -26.6091728, 14.1050959, -24.4390430, 12.9802227, -39.5893898, 38.5441399
4: -25.1122932, 18.8732452, -23.0271053, 17.4460297, -42.5583229, 41.9003525
5: -22.3228741, 17.7906971, -20.6185017, 16.5155678, -38.8384399, 38.4091949
6: -21.6060753, 20.7462311, -19.8623257, 19.1536388, -40.7597122, 40.6085587
7: -23.0035610, 19.8754101, -21.2467098, 18.3604412, -41.3639908, 41.1221123
8: -27.4310341, 18.1381950, -25.2807846, 16.7342682, -44.1653023, 43.4189796
9: -21.2406311, 20.5618153, -19.6171761, 18.9736423, -40.2142639, 40.1789932

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_B2_A1_A1

### Relational analysis result of IS_A2_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6427657, upper bound: 106.6353772
time: 8.44 seconds

## Relational analysis of IS_A2_B2_B2_A1_A2

### Relational analysis result of IS_A2_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6568632, upper bound: 106.6428519
time: 8.32 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 20.75 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.75
Output dim: 0, lower bound: -106.7146186, upper bound: 106.7044425
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.75
Output dim: 0, lower bound: -106.7146186, upper bound: 106.7044425
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.75
Output dim: 0, lower bound: -106.7146186, upper bound: 106.7044425
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.75
Output dim: 0, lower bound: -106.7146186, upper bound: 106.7044425
IS_A1_B1_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 20.75
Output dim: 0, lower bound: -106.6426385, upper bound: 106.6392442
IS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 20.75
Output dim: 0, lower bound: -106.6593432, upper bound: 106.6621943
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.75
Output dim: 0, lower bound: -106.6754659, upper bound: 106.6535736
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.75
Output dim: 0, lower bound: -106.6648545, upper bound: 106.6497386
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.75
Output dim: 0, lower bound: -106.6754659, upper bound: 106.6535736
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.75
Output dim: 0, lower bound: -106.6648545, upper bound: 106.6497388
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 20.75
Output dim: 0, lower bound: -106.6430868, upper bound: 106.6443193
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 20.75
Output dim: 0, lower bound: -106.6377325, upper bound: 106.6404370
IS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 20.75
Output dim: 0, lower bound: -106.6506319, upper bound: 106.6668864
IS_A2_B1_B1_A1_B2, status: Status.VERIFIED, split count: 5, time: 20.75
Output dim: 0, lower bound: -106.6457721, upper bound: 106.6528549
IS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 20.75
Output dim: 0, lower bound: -106.6506319, upper bound: 106.6668864
IS_A2_B1_B1_A2_B2, status: Status.VERIFIED, split count: 5, time: 20.75
Output dim: 0, lower bound: -106.6457721, upper bound: 106.6528549
IS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 20.75
Output dim: 0, lower bound: -106.6498782, upper bound: 106.6646994
IS_A2_B1_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 20.75
Output dim: 0, lower bound: -106.6443814, upper bound: 106.6486824
IS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 20.75
Output dim: 0, lower bound: -106.6789672, upper bound: 106.6830238
IS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 20.75
Output dim: 0, lower bound: -106.6784864, upper bound: 106.6815360
IS_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 20.75
Output dim: 0, lower bound: -106.6498064, upper bound: 106.6648149
IS_A2_B2_B1_A1_B2, status: Status.VERIFIED, split count: 5, time: 20.75
Output dim: 0, lower bound: -106.6438760, upper bound: 106.6470349
IS_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 20.75
Output dim: 0, lower bound: -106.6498064, upper bound: 106.6648149
IS_A2_B2_B1_A2_B2, status: Status.VERIFIED, split count: 5, time: 20.75
Output dim: 0, lower bound: -106.6438760, upper bound: 106.6470349
IS_A2_B2_B2_A1_A1, status: Status.VERIFIED, split count: 5, time: 20.75
Output dim: 0, lower bound: -106.6427657, upper bound: 106.6353772
IS_A2_B2_B2_A1_A2, status: Status.VERIFIED, split count: 5, time: 20.75
Output dim: 0, lower bound: -106.6568632, upper bound: 106.6428519

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -25.1917973, 20.0812168, -30.2983894, 24.1736546, -49.3654518, 50.3795929
1: -20.0092201, 17.2965965, -24.2702217, 20.8249187, -40.8341370, 41.5668182
2: -26.8536167, 17.5546837, -32.4354630, 21.1161346, -47.9697495, 49.9901428
3: -28.3202724, 15.0234823, -34.2638359, 18.0821152, -46.4023857, 49.2873192
4: -26.7474060, 20.1773663, -32.2010345, 24.3079014, -51.0553055, 52.3784027
5: -23.9145222, 19.1142712, -28.9254189, 23.0072803, -46.9217987, 48.0396881
6: -22.9567261, 22.0853806, -27.6293278, 26.5868244, -49.5435410, 49.7147064
7: -24.5475559, 21.2434349, -29.5311012, 25.5281830, -50.0757370, 50.7745323
8: -29.3334694, 19.3709183, -35.3996544, 23.4165115, -52.7499771, 54.7705727
9: -22.7194691, 22.0096493, -27.3463688, 26.5218925, -49.2413483, 49.3560143

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 210

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6849009, upper bound: 106.6866888
time: 9.31 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6747160, upper bound: 106.6658929
time: 9.70 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -20.0306988, 15.9320297, -30.2983894, 24.1736546, -44.2043495, 46.2304153
1: -15.5773478, 13.6665001, -24.2702217, 20.8249187, -36.4022675, 37.9367218
2: -21.1398811, 13.8952723, -32.4354630, 21.1161346, -42.2560158, 46.3307343
3: -22.3031349, 11.8781166, -34.2638359, 18.0821152, -40.3852501, 46.1419525
4: -21.0795269, 15.9595718, -32.2010345, 24.3079014, -45.3874245, 48.1606026
5: -18.8295135, 15.1386166, -28.9254189, 23.0072803, -41.8367920, 44.0640335
6: -18.1858978, 17.5234032, -27.6293278, 26.5868244, -44.7727203, 45.1527252
7: -19.4210453, 16.8238525, -29.5311012, 25.5281830, -44.9492264, 46.3549461
8: -23.1121693, 15.3367167, -35.3996544, 23.4165115, -46.5286751, 50.7363663
9: -17.9588184, 17.3444347, -27.3463688, 26.5218925, -44.4807053, 44.6907921

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7025913, upper bound: 106.6947992
time: 7.86 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7016964, upper bound: 106.6936849
time: 9.08 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -25.1917973, 20.0812168, -29.4221592, 23.4520073, -48.6438026, 49.5033684
1: -20.0092201, 17.2965965, -23.5109882, 20.1929646, -40.2021866, 40.8075829
2: -26.8536167, 17.5546837, -31.4479980, 20.4644489, -47.3180656, 49.0026779
3: -28.3202724, 15.0234823, -33.2386818, 17.5285931, -45.8488655, 48.2621651
4: -26.7474060, 20.1773663, -31.2157784, 23.6078911, -50.3552971, 51.3931427
5: -23.9145222, 19.1142712, -28.0523510, 22.3039017, -46.2184067, 47.1666222
6: -22.9567261, 22.0853806, -26.7943306, 25.8205051, -48.7772293, 48.8797112
7: -24.5475559, 21.2434349, -28.6619797, 24.7738514, -49.3214035, 49.9054146
8: -29.3334694, 19.3709183, -34.3139229, 22.6856480, -52.0191154, 53.6848412
9: -22.7194691, 22.0096493, -26.5366707, 25.7191715, -48.4386368, 48.5463181

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6799288, upper bound: 106.6809109
time: 9.24 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6678700, upper bound: 106.6576508
time: 8.02 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -20.0306988, 15.9320297, -29.4221592, 23.4520073, -43.4827003, 45.3541832
1: -15.5773478, 13.6665001, -23.5109882, 20.1929646, -35.7703133, 37.1774864
2: -21.1398811, 13.8952723, -31.4479980, 20.4644489, -41.6043320, 45.3432693
3: -22.3031349, 11.8781166, -33.2386818, 17.5285931, -39.8317261, 45.1167984
4: -21.0795269, 15.9595718, -31.2157784, 23.6078911, -44.6874123, 47.1753502
5: -18.8295135, 15.1386166, -28.0523510, 22.3039017, -41.1334000, 43.1909599
6: -18.1858978, 17.5234032, -26.7943306, 25.8205051, -44.0064011, 44.3177338
7: -19.4210453, 16.8238525, -28.6619797, 24.7738514, -44.1948853, 45.4858284
8: -23.1121693, 15.3367167, -34.3139229, 22.6856480, -45.7978134, 49.6506310
9: -17.9588184, 17.3444347, -26.5366707, 25.7191715, -43.6779900, 43.8810997

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6983162, upper bound: 106.6890026
time: 8.32 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6974311, upper bound: 106.6872229
time: 9.08 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 18.78 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 18.78
Output dim: 0, lower bound: -106.6849009, upper bound: 106.6866888
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 18.78
Output dim: 0, lower bound: -106.6747160, upper bound: 106.6658929
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 18.78
Output dim: 0, lower bound: -106.7025913, upper bound: 106.6947992
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 18.78
Output dim: 0, lower bound: -106.7016964, upper bound: 106.6936849
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 18.78
Output dim: 0, lower bound: -106.6799288, upper bound: 106.6809109
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 18.78
Output dim: 0, lower bound: -106.6678700, upper bound: 106.6576508
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 18.78
Output dim: 0, lower bound: -106.6983162, upper bound: 106.6890026
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 18.78
Output dim: 0, lower bound: -106.6974311, upper bound: 106.6872229
IS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 18.78
Output dim: 0, lower bound: -106.6593432, upper bound: 106.6621943
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.78
Output dim: 0, lower bound: -106.6754659, upper bound: 106.6535736
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.78
Output dim: 0, lower bound: -106.6648545, upper bound: 106.6497386
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.78
Output dim: 0, lower bound: -106.6754659, upper bound: 106.6535736
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.78
Output dim: 0, lower bound: -106.6648545, upper bound: 106.6497388
IS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 18.78
Output dim: 0, lower bound: -106.6506319, upper bound: 106.6668864
IS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 18.78
Output dim: 0, lower bound: -106.6506319, upper bound: 106.6668864
IS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 18.78
Output dim: 0, lower bound: -106.6498782, upper bound: 106.6646994
IS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 18.78
Output dim: 0, lower bound: -106.6789672, upper bound: 106.6830238
IS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 18.78
Output dim: 0, lower bound: -106.6784864, upper bound: 106.6815360
IS_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 18.78
Output dim: 0, lower bound: -106.6498064, upper bound: 106.6648149
IS_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 18.78
Output dim: 0, lower bound: -106.6498064, upper bound: 106.6648149
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=108.02017974853516
rel_dist={0: [-106.7671031285499, 106.7671031285499]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7554331, upper bound: 106.7556257
time: 9.29 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7563527, upper bound: 106.7563527
time: 7.81 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 17.25 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 17.25
Output dim: 0, lower bound: -106.7554331, upper bound: 106.7556257
IS_A2, status: Status.UNKNOWN, split count: 1, time: 17.25
Output dim: 0, lower bound: -106.7563527, upper bound: 106.7563527

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -40.4215355, 32.1656380, -52.4963608, 41.7913055, -82.2128220, 84.6620026
1: -32.5613441, 27.7784863, -42.6171036, 36.2078896, -68.7692337, 70.3955917
2: -43.4267426, 28.1031914, -56.7059441, 36.6510315, -80.0777435, 84.8091278
3: -45.9748688, 24.2641106, -59.9826431, 31.5825367, -77.5574036, 84.2467499
4: -42.8716888, 32.4032478, -55.8638458, 42.2842445, -85.1559296, 88.2670898
5: -38.8010483, 30.6035709, -50.5371666, 39.6374054, -78.4384537, 81.1407394
6: -36.7541809, 35.3119507, -47.8690758, 45.8901176, -82.6442871, 83.1810303
7: -39.3098488, 33.9188652, -51.2901459, 44.0228157, -83.3326645, 85.2089920
8: -47.4530792, 31.4979420, -62.1591644, 41.3959351, -88.8490143, 93.6570969
9: -36.3031998, 35.3297119, -47.2926941, 46.1051826, -82.4083710, 82.6224060

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 168

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7550576, upper bound: 106.7550576
time: 8.66 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7550576, upper bound: 106.7556257
time: 8.82 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -43.0184174, 34.2086563, -50.9955406, 40.5937691, -83.6121826, 85.2041931
1: -34.7182274, 29.5691071, -41.3734207, 35.1553078, -69.8735275, 70.9425201
2: -46.2603874, 29.9053726, -55.0451431, 35.5759583, -81.8363342, 84.9505081
3: -48.9885712, 25.8325176, -58.2324982, 30.6709747, -79.6595459, 84.0649948
4: -45.6809273, 34.5013237, -54.2551613, 41.0516281, -86.7325516, 88.7564850
5: -41.2985191, 32.5255585, -49.0815697, 38.5158157, -79.8143311, 81.6071167
6: -39.1464157, 37.5944214, -46.4886284, 44.5845070, -83.7309113, 84.0830383
7: -41.8991547, 36.0853615, -49.7956772, 42.7667351, -84.6658936, 85.8810425
8: -50.5748444, 33.5495453, -60.3187370, 40.1566734, -90.7315140, 93.8682861
9: -38.6610641, 37.6220551, -45.9355354, 44.7589874, -83.4200516, 83.5575867

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7556258, upper bound: 106.7551749
time: 8.49 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7556258, upper bound: 106.7563527
time: 10.53 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 20.41 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 20.41
Output dim: 0, lower bound: -106.7550576, upper bound: 106.7550576
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 20.41
Output dim: 0, lower bound: -106.7550576, upper bound: 106.7556257
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 20.41
Output dim: 0, lower bound: -106.7556258, upper bound: 106.7551749
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 20.41
Output dim: 0, lower bound: -106.7556258, upper bound: 106.7563527

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -40.4215355, 32.1656380, -40.4215355, 32.1656380, -72.5871735, 72.5871735
1: -32.5613441, 27.7784863, -32.5613441, 27.7784863, -60.3398285, 60.3398285
2: -43.4267426, 28.1031914, -43.4267426, 28.1031914, -71.5298920, 71.5298920
3: -45.9748688, 24.2641106, -45.9748688, 24.2641106, -70.2389832, 70.2389832
4: -42.8716888, 32.4032478, -42.8716888, 32.4032478, -75.2749329, 75.2749329
5: -38.8010483, 30.6035709, -38.8010483, 30.6035709, -69.4046173, 69.4046173
6: -36.7541809, 35.3119507, -36.7541809, 35.3119507, -72.0661316, 72.0661316
7: -39.3098488, 33.9188652, -39.3098488, 33.9188652, -73.2287140, 73.2287140
8: -47.4530792, 31.4979420, -47.4530792, 31.4979420, -78.9510193, 78.9510193
9: -36.3031998, 35.3297119, -36.3031998, 35.3297119, -71.6329117, 71.6329117

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7373110, upper bound: 106.7343085
time: 9.60 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7345154, upper bound: 106.7336190
time: 8.37 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -40.4215355, 32.1656380, -43.0184174, 34.2086563, -74.6301880, 75.1840515
1: -32.5613441, 27.7784863, -34.7182274, 29.5691071, -62.1304512, 62.4967117
2: -43.4267426, 28.1031914, -46.2603874, 29.9053726, -73.3320923, 74.3635559
3: -45.9748688, 24.2641106, -48.9885712, 25.8325176, -71.8073883, 73.2526703
4: -42.8716888, 32.4032478, -45.6809273, 34.5013237, -77.3730164, 78.0841751
5: -38.8010483, 30.6035709, -41.2985191, 32.5255585, -71.3266068, 71.9020920
6: -36.7541809, 35.3119507, -39.1464157, 37.5944214, -74.3485794, 74.4583664
7: -39.3098488, 33.9188652, -41.8991547, 36.0853615, -75.3952103, 75.8180237
8: -47.4530792, 31.4979420, -50.5748444, 33.5495453, -81.0026245, 82.0727768
9: -36.3031998, 35.3297119, -38.6610641, 37.6220551, -73.9252548, 73.9907761

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7358105, upper bound: 106.7382072
time: 8.67 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7345154, upper bound: 106.7336190
time: 9.23 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -43.0184174, 34.2086563, -40.4215355, 32.1656380, -75.1840515, 74.6301880
1: -34.7182274, 29.5691071, -32.5613441, 27.7784863, -62.4967117, 62.1304512
2: -46.2603874, 29.9053726, -43.4267426, 28.1031914, -74.3635559, 73.3320923
3: -48.9885712, 25.8325176, -45.9748688, 24.2641106, -73.2526703, 71.8073883
4: -45.6809273, 34.5013237, -42.8716888, 32.4032478, -78.0841751, 77.3730164
5: -41.2985191, 32.5255585, -38.8010483, 30.6035709, -71.9020920, 71.3266068
6: -39.1464157, 37.5944214, -36.7541809, 35.3119507, -74.4583664, 74.3485794
7: -41.8991547, 36.0853615, -39.3098488, 33.9188652, -75.8180237, 75.3952103
8: -50.5748444, 33.5495453, -47.4530792, 31.4979420, -82.0727768, 81.0026245
9: -38.6610641, 37.6220551, -36.3031998, 35.3297119, -73.9907761, 73.9252548

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7377862, upper bound: 106.7343919
time: 10.28 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7333969, upper bound: 106.7334017
time: 6.89 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -43.0184174, 34.2086563, -43.0184174, 34.2086563, -77.2270737, 77.2270737
1: -34.7182274, 29.5691071, -34.7182274, 29.5691071, -64.2873230, 64.2873230
2: -46.2603874, 29.9053726, -46.2603874, 29.9053726, -76.1657486, 76.1657486
3: -48.9885712, 25.8325176, -48.9885712, 25.8325176, -74.8210754, 74.8210754
4: -45.6809273, 34.5013237, -45.6809273, 34.5013237, -80.1822510, 80.1822510
5: -41.2985191, 32.5255585, -41.2985191, 32.5255585, -73.8240662, 73.8240662
6: -39.1464157, 37.5944214, -39.1464157, 37.5944214, -76.7408218, 76.7408218
7: -41.8991547, 36.0853615, -41.8991547, 36.0853615, -77.9845123, 77.9845123
8: -50.5748444, 33.5495453, -50.5748444, 33.5495453, -84.1243820, 84.1243820
9: -38.6610641, 37.6220551, -38.6610641, 37.6220551, -76.2831192, 76.2831192

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7377862, upper bound: 106.7344097
time: 8.67 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7333969, upper bound: 106.7334073
time: 8.39 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 18.42 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 18.42
Output dim: 0, lower bound: -106.7373110, upper bound: 106.7343085
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.42
Output dim: 0, lower bound: -106.7345154, upper bound: 106.7336190
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 18.42
Output dim: 0, lower bound: -106.7358105, upper bound: 106.7382072
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 18.42
Output dim: 0, lower bound: -106.7345154, upper bound: 106.7336190
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 18.42
Output dim: 0, lower bound: -106.7377862, upper bound: 106.7343919
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.42
Output dim: 0, lower bound: -106.7333969, upper bound: 106.7334017
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 18.42
Output dim: 0, lower bound: -106.7377862, upper bound: 106.7344097
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 18.42
Output dim: 0, lower bound: -106.7333969, upper bound: 106.7334073

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -26.7422485, 21.2718410, -37.8471146, 30.1299877, -56.8722305, 59.1189537
1: -21.2280235, 18.2902088, -30.4349766, 25.9966240, -47.2246323, 48.7251816
2: -28.4548073, 18.6351051, -40.6121483, 26.3152657, -54.7700691, 59.2472534
3: -30.0922546, 15.9876270, -42.9920540, 22.7105942, -52.8028488, 58.9796829
4: -28.2785721, 21.3875751, -40.1298027, 30.3276749, -58.6062469, 61.5173721
5: -25.3948956, 20.3002529, -36.2888680, 28.6771679, -54.0720634, 56.5891190
6: -24.3348961, 23.3814526, -34.4116020, 33.0819054, -57.4168015, 57.7930489
7: -26.0041809, 22.4731750, -36.7924232, 31.7752075, -57.7793846, 59.2655907
8: -31.1208191, 20.5765305, -44.3611984, 29.4308472, -60.5516586, 64.9377289
9: -24.0275688, 23.3268414, -33.9962540, 33.0737152, -57.1012840, 57.3230934

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 168

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7364786, upper bound: 106.7364786
time: 8.98 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7364786, upper bound: 106.7364786
time: 8.68 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -22.5513592, 17.8861790, -30.0552025, 23.9169464, -46.4683075, 47.9413795
1: -17.6214981, 15.3477020, -23.9647007, 20.5707531, -38.1922531, 39.3124008
2: -23.7973385, 15.6399269, -32.0421410, 20.8775845, -44.6749191, 47.6820679
3: -25.1999550, 13.4318771, -33.9282684, 17.9947720, -43.1947250, 47.3601456
4: -23.6879063, 17.9321709, -31.7673378, 24.0477104, -47.7356186, 49.6995087
5: -21.2457676, 17.0805855, -28.6590309, 22.8299713, -44.0757256, 45.7396049
6: -20.4381771, 19.6731262, -27.3091831, 26.2767029, -46.7148819, 46.9823074
7: -21.8581924, 18.8887234, -29.1731853, 25.2266026, -47.0847855, 48.0619011
8: -26.0306473, 17.2440929, -34.9918060, 23.1664467, -49.1970863, 52.2358971
9: -20.1597729, 19.5271244, -26.9766388, 26.1724625, -46.3322296, 46.5037613

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7364786, upper bound: 106.7364786
time: 8.02 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7364786, upper bound: 106.7364786
time: 7.77 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -37.8471146, 30.1299877, -29.5187664, 23.4870911, -61.3342018, 59.6487465
1: -30.4349766, 25.9966240, -23.5490704, 20.2099152, -50.6448860, 49.5456848
2: -40.6121483, 26.3152657, -31.5047932, 20.5330448, -61.1451874, 57.8200531
3: -42.9920540, 22.7105942, -33.3346443, 17.6447906, -60.6368446, 56.0452385
4: -40.1298027, 30.3276749, -31.2796803, 23.6343670, -63.7641678, 61.6073532
5: -36.2888680, 28.6771679, -28.0859947, 22.3803577, -58.6692276, 56.7631607
6: -34.4116020, 33.0819054, -26.8713856, 25.8427086, -60.2543030, 59.9532890
7: -36.7924232, 31.7752075, -28.7505798, 24.8096333, -61.6020546, 60.5257797
8: -44.3611984, 29.4308472, -34.4302559, 22.7473793, -67.1085739, 63.8610878
9: -33.9962540, 33.0737152, -26.5568848, 25.7704887, -59.7667389, 59.6306000

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 168

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7345154, upper bound: 106.7336190
time: 8.30 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7345154, upper bound: 106.7336190
time: 9.43 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -30.0552025, 23.9169464, -22.9962101, 18.2312794, -48.2864838, 46.9131546
1: -23.9647007, 20.5707531, -17.9922199, 15.6463699, -39.6110611, 38.5629730
2: -32.0421410, 20.8775845, -24.2844887, 15.9405718, -47.9827118, 45.1620674
3: -33.9282684, 17.9947720, -25.7163887, 13.6907778, -47.6190453, 43.7111588
4: -31.7673378, 24.0477104, -24.1710072, 18.3016148, -50.0689507, 48.2187157
5: -28.6590309, 22.8299713, -21.6525860, 17.3900032, -46.0490341, 44.4825554
6: -27.3091831, 26.2767029, -20.8454704, 20.0681534, -47.3773346, 47.1221733
7: -29.1731853, 25.2266026, -22.3190098, 19.2652969, -48.4384804, 47.5456123
8: -34.9918060, 23.1664467, -26.5726395, 17.5822601, -52.5740662, 49.7390785
9: -26.9766388, 26.1724625, -20.5671253, 19.9276485, -46.9042892, 46.7395859

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7345154, upper bound: 106.7336189
time: 7.90 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7345154, upper bound: 106.7336190
time: 8.02 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -29.5187664, 23.4870911, -37.8471146, 30.1299877, -59.6487465, 61.3342018
1: -23.5490704, 20.2099152, -30.4349766, 25.9966240, -49.5456848, 50.6448860
2: -31.5047932, 20.5330448, -40.6121483, 26.3152657, -57.8200531, 61.1451874
3: -33.3346443, 17.6447906, -42.9920540, 22.7105942, -56.0452385, 60.6368446
4: -31.2796803, 23.6343670, -40.1298027, 30.3276749, -61.6073532, 63.7641678
5: -28.0859947, 22.3803577, -36.2888680, 28.6771679, -56.7631607, 58.6692276
6: -26.8713856, 25.8427086, -34.4116020, 33.0819054, -59.9532890, 60.2543030
7: -28.7505798, 24.8096333, -36.7924232, 31.7752075, -60.5257797, 61.6020546
8: -34.4302559, 22.7473793, -44.3611984, 29.4308472, -63.8610916, 67.1085739
9: -26.5568848, 25.7704887, -33.9962540, 33.0737152, -59.6306000, 59.7667389

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 168

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7336190, upper bound: 106.7345154
time: 7.95 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7336190, upper bound: 106.7345154
time: 8.88 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -22.9962101, 18.2312794, -30.0552025, 23.9169464, -46.9131546, 48.2864838
1: -17.9922199, 15.6463699, -23.9647007, 20.5707531, -38.5629730, 39.6110611
2: -24.2844887, 15.9405718, -32.0421410, 20.8775845, -45.1620674, 47.9827118
3: -25.7163887, 13.6907778, -33.9282684, 17.9947720, -43.7111588, 47.6190453
4: -24.1710072, 18.3016148, -31.7673378, 24.0477104, -48.2187157, 50.0689507
5: -21.6525860, 17.3900032, -28.6590309, 22.8299713, -44.4825516, 46.0490341
6: -20.8454704, 20.0681534, -27.3091831, 26.2767029, -47.1221733, 47.3773346
7: -22.3190098, 19.2652969, -29.1731853, 25.2266026, -47.5456123, 48.4384804
8: -26.5726395, 17.5822601, -34.9918060, 23.1664467, -49.7390785, 52.5740662
9: -20.5671253, 19.9276485, -26.9766388, 26.1724625, -46.7395859, 46.9042892

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7336190, upper bound: 106.7345154
time: 8.63 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7336190, upper bound: 106.7345154
time: 8.53 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -29.5187664, 23.4870911, -40.5106430, 32.2322578, -61.7510185, 63.9977226
1: -23.5490704, 20.2099152, -32.6535645, 27.8397751, -51.3888435, 52.8634796
2: -31.5047932, 20.5330448, -43.5276222, 28.1626759, -59.6674576, 64.0606613
3: -33.3346443, 17.6447906, -46.0933571, 24.3177814, -57.6524277, 63.7381439
4: -31.2796803, 23.6343670, -43.0172424, 32.4772720, -63.7569504, 66.6516037
5: -28.0859947, 22.3803577, -38.8574066, 30.6514111, -58.7374039, 61.2377625
6: -26.8713856, 25.8427086, -36.8641205, 35.4313316, -62.3027153, 62.7068291
7: -28.7505798, 24.8096333, -39.4506950, 34.0039062, -62.7544861, 64.2603302
8: -34.4302559, 22.7473793, -47.5513649, 31.5370960, -65.9673538, 70.2987442
9: -26.5568848, 25.7704887, -36.4209099, 35.4289551, -61.9858398, 62.1913948

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 168

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7334059, upper bound: 106.7334073
time: 7.53 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7334059, upper bound: 106.7334073
time: 8.12 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -22.9962101, 18.2312794, -31.5634136, 25.1066113, -48.1028175, 49.7946892
1: -17.9922199, 15.6463699, -25.2240715, 21.6079121, -39.6001244, 40.8704414
2: -24.2844887, 15.9405718, -33.6904984, 21.9086761, -46.1931610, 49.6310692
3: -25.7163887, 13.6907778, -35.6812286, 18.8970261, -44.6134071, 49.3720055
4: -24.1710072, 18.3016148, -33.3945770, 25.2608681, -49.4318657, 51.6961899
5: -21.6525860, 17.3900032, -30.1051502, 23.9419403, -45.5945282, 47.4951553
6: -20.8454704, 20.0681534, -28.6896553, 27.6141739, -48.4596443, 48.7578011
7: -22.3190098, 19.2652969, -30.6836796, 26.4995632, -48.8185730, 49.9489708
8: -26.5726395, 17.5822601, -36.7943916, 24.3335533, -50.9061928, 54.3766518
9: -20.5671253, 19.9276485, -28.3497181, 27.5052948, -48.0724182, 48.2773666

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7334059, upper bound: 106.7334073
time: 7.41 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7334059, upper bound: 106.7334073
time: 7.12 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 15.87 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.87
Output dim: 0, lower bound: -106.7364786, upper bound: 106.7364786
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.87
Output dim: 0, lower bound: -106.7364786, upper bound: 106.7364786
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.87
Output dim: 0, lower bound: -106.7364786, upper bound: 106.7364786
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.87
Output dim: 0, lower bound: -106.7364786, upper bound: 106.7364786
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 15.87
Output dim: 0, lower bound: -106.7345154, upper bound: 106.7336190
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 15.87
Output dim: 0, lower bound: -106.7345154, upper bound: 106.7336190
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 15.87
Output dim: 0, lower bound: -106.7345154, upper bound: 106.7336189
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 15.87
Output dim: 0, lower bound: -106.7345154, upper bound: 106.7336190
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.87
Output dim: 0, lower bound: -106.7336190, upper bound: 106.7345154
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.87
Output dim: 0, lower bound: -106.7336190, upper bound: 106.7345154
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.87
Output dim: 0, lower bound: -106.7336190, upper bound: 106.7345154
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.87
Output dim: 0, lower bound: -106.7336190, upper bound: 106.7345154
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.87
Output dim: 0, lower bound: -106.7334059, upper bound: 106.7334073
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.87
Output dim: 0, lower bound: -106.7334059, upper bound: 106.7334073
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.87
Output dim: 0, lower bound: -106.7334059, upper bound: 106.7334073
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.87
Output dim: 0, lower bound: -106.7334059, upper bound: 106.7334073

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -26.7422485, 21.2718410, -26.7422485, 21.2718410, -48.0140839, 48.0140839
1: -21.2280235, 18.2902088, -21.2280235, 18.2902088, -39.5182304, 39.5182304
2: -28.4548073, 18.6351051, -28.4548073, 18.6351051, -47.0899124, 47.0899124
3: -30.0922546, 15.9876270, -30.0922546, 15.9876270, -46.0798798, 46.0798798
4: -28.2785721, 21.3875751, -28.2785721, 21.3875751, -49.6661415, 49.6661415
5: -25.3948956, 20.3002529, -25.3948956, 20.3002529, -45.6951485, 45.6951485
6: -24.3348961, 23.3814526, -24.3348961, 23.3814526, -47.7163467, 47.7163467
7: -26.0041809, 22.4731750, -26.0041809, 22.4731750, -48.4773560, 48.4773560
8: -31.1208191, 20.5765305, -31.1208191, 20.5765305, -51.6973495, 51.6973495
9: -24.0275688, 23.3268414, -24.0275688, 23.3268414, -47.3544083, 47.3544083

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7279963, upper bound: 106.7258219
time: 10.00 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7277806, upper bound: 106.7249537
time: 8.66 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -26.7422485, 21.2718410, -22.5513592, 17.8861790, -44.6284142, 43.8232002
1: -21.2280235, 18.2902088, -17.6214981, 15.3477020, -36.5757217, 35.9117050
2: -28.4548073, 18.6351051, -23.7973385, 15.6399269, -44.0947342, 42.4324417
3: -30.0922546, 15.9876270, -25.1999550, 13.4318771, -43.5241318, 41.1875839
4: -28.2785721, 21.3875751, -23.6879063, 17.9321709, -46.2107430, 45.0754814
5: -25.3948956, 20.3002529, -21.2457676, 17.0805855, -42.4754753, 41.5460129
6: -24.3348961, 23.3814526, -20.4381771, 19.6731262, -44.0080223, 43.8196297
7: -26.0041809, 22.4731750, -21.8581924, 18.8887234, -44.8928986, 44.3313599
8: -31.1208191, 20.5765305, -26.0306473, 17.2440929, -48.3649101, 46.6071777
9: -24.0275688, 23.3268414, -20.1597729, 19.5271244, -43.5546951, 43.4866066

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7285212, upper bound: 106.7251571
time: 8.42 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7277806, upper bound: 106.7249537
time: 8.75 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -22.5513592, 17.8861790, -26.7422485, 21.2718410, -43.8232002, 44.6284142
1: -17.6214981, 15.3477020, -21.2280235, 18.2902088, -35.9117050, 36.5757217
2: -23.7973385, 15.6399269, -28.4548073, 18.6351051, -42.4324417, 44.0947342
3: -25.1999550, 13.4318771, -30.0922546, 15.9876270, -41.1875839, 43.5241318
4: -23.6879063, 17.9321709, -28.2785721, 21.3875751, -45.0754814, 46.2107430
5: -21.2457676, 17.0805855, -25.3948956, 20.3002529, -41.5460129, 42.4754753
6: -20.4381771, 19.6731262, -24.3348961, 23.3814526, -43.8196297, 44.0080223
7: -21.8581924, 18.8887234, -26.0041809, 22.4731750, -44.3313599, 44.8928986
8: -26.0306473, 17.2440929, -31.1208191, 20.5765305, -46.6071777, 48.3649101
9: -20.1597729, 19.5271244, -24.0275688, 23.3268414, -43.4866066, 43.5546951

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7244161, upper bound: 106.7250799
time: 10.26 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7242420, upper bound: 106.7242420
time: 7.42 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -22.5513592, 17.8861790, -22.5513592, 17.8861790, -40.4375305, 40.4375305
1: -17.6214981, 15.3477020, -17.6214981, 15.3477020, -32.9692001, 32.9692001
2: -23.7973385, 15.6399269, -23.7973385, 15.6399269, -39.4372635, 39.4372635
3: -25.1999550, 13.4318771, -25.1999550, 13.4318771, -38.6318321, 38.6318321
4: -23.6879063, 17.9321709, -23.6879063, 17.9321709, -41.6200790, 41.6200790
5: -21.2457676, 17.0805855, -21.2457676, 17.0805855, -38.3263397, 38.3263397
6: -20.4381771, 19.6731262, -20.4381771, 19.6731262, -40.1113052, 40.1113052
7: -21.8581924, 18.8887234, -21.8581924, 18.8887234, -40.7469025, 40.7469025
8: -26.0306473, 17.2440929, -26.0306473, 17.2440929, -43.2747383, 43.2747383
9: -20.1597729, 19.5271244, -20.1597729, 19.5271244, -39.6868896, 39.6868896

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7250799, upper bound: 106.7244161
time: 8.87 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7242420, upper bound: 106.7242420
time: 9.42 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -26.7422485, 21.2718410, -29.5187664, 23.4870911, -50.2293320, 50.7905998
1: -21.2280235, 18.2902088, -23.5490704, 20.2099152, -41.4379349, 41.8392792
2: -28.4548073, 18.6351051, -31.5047932, 20.5330448, -48.9878464, 50.1398964
3: -30.0922546, 15.9876270, -33.3346443, 17.6447906, -47.7370453, 49.3222733
4: -28.2785721, 21.3875751, -31.2796803, 23.6343670, -51.9129410, 52.6672478
5: -25.3948956, 20.3002529, -28.0859947, 22.3803577, -47.7752533, 48.3862457
6: -24.3348961, 23.3814526, -26.8713856, 25.8427086, -50.1776047, 50.2528343
7: -26.0041809, 22.4731750, -28.7505798, 24.8096333, -50.8138123, 51.2237511
8: -31.1208191, 20.5765305, -34.4302559, 22.7473793, -53.8681984, 55.0067825
9: -24.0275688, 23.3268414, -26.5568848, 25.7704887, -49.7980576, 49.8837280

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7236513, upper bound: 106.7268637
time: 9.25 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7233881, upper bound: 106.7261975
time: 9.20 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -22.5513592, 17.8861790, -29.5187664, 23.4870911, -46.0384483, 47.4049301
1: -17.6214981, 15.3477020, -23.5490704, 20.2099152, -37.8314133, 38.8967743
2: -23.7973385, 15.6399269, -31.5047932, 20.5330448, -44.3303795, 47.1447182
3: -25.1999550, 13.4318771, -33.3346443, 17.6447906, -42.8447456, 46.7665215
4: -23.6879063, 17.9321709, -31.2796803, 23.6343670, -47.3222733, 49.2118530
5: -21.2457676, 17.0805855, -28.0859947, 22.3803577, -43.6261253, 45.1665726
6: -20.4381771, 19.6731262, -26.8713856, 25.8427086, -46.2808838, 46.5445061
7: -21.8581924, 18.8887234, -28.7505798, 24.8096333, -46.6678238, 47.6392937
8: -26.0306473, 17.2440929, -34.4302559, 22.7473793, -48.7780266, 51.6743431
9: -20.1597729, 19.5271244, -26.5568848, 25.7704887, -45.9302521, 46.0840073

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7236513, upper bound: 106.7268638
time: 9.70 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7233881, upper bound: 106.7261975
time: 9.59 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -26.7422485, 21.2718410, -22.9962101, 18.2312794, -44.9735222, 44.2680473
1: -21.2280235, 18.2902088, -17.9922199, 15.6463699, -36.8743820, 36.2824287
2: -28.4548073, 18.6351051, -24.2844887, 15.9405718, -44.3953781, 42.9195938
3: -30.0922546, 15.9876270, -25.7163887, 13.6907778, -43.7830315, 41.7040100
4: -28.2785721, 21.3875751, -24.1710072, 18.3016148, -46.5801849, 45.5585709
5: -25.3948956, 20.3002529, -21.6525860, 17.3900032, -42.7848969, 41.9528389
6: -24.3348961, 23.3814526, -20.8454704, 20.0681534, -44.4030495, 44.2269211
7: -26.0041809, 22.4731750, -22.3190098, 19.2652969, -45.2694778, 44.7921829
8: -31.1208191, 20.5765305, -26.5726395, 17.5822601, -48.7030754, 47.1491699
9: -24.0275688, 23.3268414, -20.5671253, 19.9276485, -43.9552155, 43.8939667

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7227044, upper bound: 106.7212512
time: 9.53 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7219503, upper bound: 106.7210984
time: 9.39 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -22.5513592, 17.8861790, -22.9962101, 18.2312794, -40.7826385, 40.8823776
1: -17.6214981, 15.3477020, -17.9922199, 15.6463699, -33.2678680, 33.3399200
2: -23.7973385, 15.6399269, -24.2844887, 15.9405718, -39.7379112, 39.9244156
3: -25.1999550, 13.4318771, -25.7163887, 13.6907778, -38.8907318, 39.1482658
4: -23.6879063, 17.9321709, -24.1710072, 18.3016148, -41.9895210, 42.1031799
5: -21.2457676, 17.0805855, -21.6525860, 17.3900032, -38.6357689, 38.7331696
6: -20.4381771, 19.6731262, -20.8454704, 20.0681534, -40.5063324, 40.5185966
7: -21.8581924, 18.8887234, -22.3190098, 19.2652969, -41.1234856, 41.2077293
8: -26.0306473, 17.2440929, -26.5726395, 17.5822601, -43.6129036, 43.8167305
9: -20.1597729, 19.5271244, -20.5671253, 19.9276485, -40.0874138, 40.0942497

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7222155, upper bound: 106.7221259
time: 7.73 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7219503, upper bound: 106.7210983
time: 9.27 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -29.5187664, 23.4870911, -26.7422485, 21.2718410, -50.7905998, 50.2293320
1: -23.5490704, 20.2099152, -21.2280235, 18.2902088, -41.8392792, 41.4379349
2: -31.5047932, 20.5330448, -28.4548073, 18.6351051, -50.1398964, 48.9878464
3: -33.3346443, 17.6447906, -30.0922546, 15.9876270, -49.3222733, 47.7370453
4: -31.2796803, 23.6343670, -28.2785721, 21.3875751, -52.6672478, 51.9129410
5: -28.0859947, 22.3803577, -25.3948956, 20.3002529, -48.3862457, 47.7752533
6: -26.8713856, 25.8427086, -24.3348961, 23.3814526, -50.2528343, 50.1776047
7: -28.7505798, 24.8096333, -26.0041809, 22.4731750, -51.2237511, 50.8138123
8: -34.4302559, 22.7473793, -31.1208191, 20.5765305, -55.0067825, 53.8681984
9: -26.5568848, 25.7704887, -24.0275688, 23.3268414, -49.8837280, 49.7980576

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7268637, upper bound: 106.7236513
time: 9.03 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7261975, upper bound: 106.7233881
time: 9.37 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -29.5187664, 23.4870911, -22.5513592, 17.8861790, -47.4049301, 46.0384483
1: -23.5490704, 20.2099152, -17.6214981, 15.3477020, -38.8967743, 37.8314133
2: -31.5047932, 20.5330448, -23.7973385, 15.6399269, -47.1447182, 44.3303795
3: -33.3346443, 17.6447906, -25.1999550, 13.4318771, -46.7665215, 42.8447456
4: -31.2796803, 23.6343670, -23.6879063, 17.9321709, -49.2118530, 47.3222733
5: -28.0859947, 22.3803577, -21.2457676, 17.0805855, -45.1665726, 43.6261253
6: -26.8713856, 25.8427086, -20.4381771, 19.6731262, -46.5445061, 46.2808838
7: -28.7505798, 24.8096333, -21.8581924, 18.8887234, -47.6392937, 46.6678238
8: -34.4302559, 22.7473793, -26.0306473, 17.2440929, -51.6743431, 48.7780266
9: -26.5568848, 25.7704887, -20.1597729, 19.5271244, -46.0840073, 45.9302521

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7268637, upper bound: 106.7236513
time: 8.62 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7261975, upper bound: 106.7233881
time: 10.07 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -22.9962101, 18.2312794, -26.7422485, 21.2718410, -44.2680473, 44.9735222
1: -17.9922199, 15.6463699, -21.2280235, 18.2902088, -36.2824287, 36.8743820
2: -24.2844887, 15.9405718, -28.4548073, 18.6351051, -42.9195938, 44.3953781
3: -25.7163887, 13.6907778, -30.0922546, 15.9876270, -41.7040100, 43.7830315
4: -24.1710072, 18.3016148, -28.2785721, 21.3875751, -45.5585709, 46.5801849
5: -21.6525860, 17.3900032, -25.3948956, 20.3002529, -41.9528389, 42.7848969
6: -20.8454704, 20.0681534, -24.3348961, 23.3814526, -44.2269211, 44.4030495
7: -22.3190098, 19.2652969, -26.0041809, 22.4731750, -44.7921829, 45.2694778
8: -26.5726395, 17.5822601, -31.1208191, 20.5765305, -47.1491699, 48.7030754
9: -20.5671253, 19.9276485, -24.0275688, 23.3268414, -43.8939667, 43.9552155

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7212512, upper bound: 106.7227044
time: 8.50 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2

### Relational analysis result of IS_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7210983, upper bound: 106.7219504
time: 9.00 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -22.9962101, 18.2312794, -22.5513592, 17.8861790, -40.8823776, 40.7826385
1: -17.9922199, 15.6463699, -17.6214981, 15.3477020, -33.3399200, 33.2678680
2: -24.2844887, 15.9405718, -23.7973385, 15.6399269, -39.9244156, 39.7379112
3: -25.7163887, 13.6907778, -25.1999550, 13.4318771, -39.1482658, 38.8907318
4: -24.1710072, 18.3016148, -23.6879063, 17.9321709, -42.1031799, 41.9895210
5: -21.6525860, 17.3900032, -21.2457676, 17.0805855, -38.7331696, 38.6357689
6: -20.8454704, 20.0681534, -20.4381771, 19.6731262, -40.5185966, 40.5063324
7: -22.3190098, 19.2652969, -21.8581924, 18.8887234, -41.2077293, 41.1234856
8: -26.5726395, 17.5822601, -26.0306473, 17.2440929, -43.8167305, 43.6129036
9: -20.5671253, 19.9276485, -20.1597729, 19.5271244, -40.0942497, 40.0874138

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7221260, upper bound: 106.7222156
time: 9.55 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7210984, upper bound: 106.7219504
time: 8.98 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -29.5187664, 23.4870911, -29.5187664, 23.4870911, -53.0058479, 53.0058479
1: -23.5490704, 20.2099152, -23.5490704, 20.2099152, -43.7589874, 43.7589874
2: -31.5047932, 20.5330448, -31.5047932, 20.5330448, -52.0378304, 52.0378304
3: -33.3346443, 17.6447906, -33.3346443, 17.6447906, -50.9794350, 50.9794350
4: -31.2796803, 23.6343670, -31.2796803, 23.6343670, -54.9140472, 54.9140472
5: -28.0859947, 22.3803577, -28.0859947, 22.3803577, -50.4663544, 50.4663544
6: -26.8713856, 25.8427086, -26.8713856, 25.8427086, -52.7140884, 52.7140884
7: -28.7505798, 24.8096333, -28.7505798, 24.8096333, -53.5602112, 53.5602112
8: -34.4302559, 22.7473793, -34.4302559, 22.7473793, -57.1776314, 57.1776314
9: -26.5568848, 25.7704887, -26.5568848, 25.7704887, -52.3273735, 52.3273735

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7265818, upper bound: 106.7221501
time: 8.86 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7257399, upper bound: 106.7219436
time: 10.03 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -29.5187664, 23.4870911, -22.9962101, 18.2312794, -47.7500381, 46.4832954
1: -23.5490704, 20.2099152, -17.9922199, 15.6463699, -39.1954346, 38.2021332
2: -31.5047932, 20.5330448, -24.2844887, 15.9405718, -47.4453621, 44.8175278
3: -33.3346443, 17.6447906, -25.7163887, 13.6907778, -47.0254211, 43.3611755
4: -31.2796803, 23.6343670, -24.1710072, 18.3016148, -49.5812874, 47.8053741
5: -28.0859947, 22.3803577, -21.6525860, 17.3900032, -45.4759979, 44.0329437
6: -26.8713856, 25.8427086, -20.8454704, 20.0681534, -46.9395332, 46.6881790
7: -28.7505798, 24.8096333, -22.3190098, 19.2652969, -48.0158730, 47.1286430
8: -34.4302559, 22.7473793, -26.5726395, 17.5822601, -52.0125122, 49.3200188
9: -26.5568848, 25.7704887, -20.5671253, 19.9276485, -46.4845352, 46.3376122

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7265818, upper bound: 106.7221500
time: 9.39 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7257399, upper bound: 106.7219436
time: 9.47 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -22.9962101, 18.2312794, -29.5184345, 23.4868336, -46.4830437, 47.7497063
1: -17.9922199, 15.6463699, -23.5488224, 20.2096786, -38.2018967, 39.1951866
2: -24.2844887, 15.9405718, -31.5044842, 20.5327835, -44.8172722, 47.4450531
3: -25.7163887, 13.6907778, -33.3342934, 17.6444931, -43.3608780, 47.0250702
4: -24.1710072, 18.3016148, -31.2792587, 23.6340942, -47.8050995, 49.5808716
5: -21.6525860, 17.3900032, -28.0856876, 22.3801250, -44.0327110, 45.4756927
6: -20.8454704, 20.0681534, -26.8709679, 25.8424377, -46.6879082, 46.9391136
7: -22.3190098, 19.2652969, -28.7502499, 24.8094006, -47.1284103, 48.0155487
8: -26.5726395, 17.5822601, -34.4298973, 22.7471027, -49.3197403, 52.0121536
9: -20.5671253, 19.9276485, -26.5565567, 25.7702217, -46.3373489, 46.4842072

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7210479, upper bound: 106.7217888
time: 9.24 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7208772, upper bound: 106.7208795
time: 8.63 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -22.9962101, 18.2312794, -22.9962101, 18.2312794, -41.2274857, 41.2274857
1: -17.9922199, 15.6463699, -17.9922199, 15.6463699, -33.6385803, 33.6385803
2: -24.2844887, 15.9405718, -24.2844887, 15.9405718, -40.2250595, 40.2250595
3: -25.7163887, 13.6907778, -25.7163887, 13.6907778, -39.4071617, 39.4071617
4: -24.1710072, 18.3016148, -24.1710072, 18.3016148, -42.4726105, 42.4726105
5: -21.6525860, 17.3900032, -21.6525860, 17.3900032, -39.0425873, 39.0425873
6: -20.8454704, 20.0681534, -20.8454704, 20.0681534, -40.9136238, 40.9136238
7: -22.3190098, 19.2652969, -22.3190098, 19.2652969, -41.5843048, 41.5843048
8: -26.5726395, 17.5822601, -26.5726395, 17.5822601, -44.1548958, 44.1548958
9: -20.5671253, 19.9276485, -20.5671253, 19.9276485, -40.4947739, 40.4947739

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7218084, upper bound: 106.7210532
time: 9.03 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7208772, upper bound: 106.7208795
time: 7.76 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 18.25 seconds
IS_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 0, lower bound: -106.7279963, upper bound: 106.7258219
IS_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 0, lower bound: -106.7277806, upper bound: 106.7249537
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 0, lower bound: -106.7285212, upper bound: 106.7251571
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 0, lower bound: -106.7277806, upper bound: 106.7249537
IS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 0, lower bound: -106.7244161, upper bound: 106.7250799
IS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 0, lower bound: -106.7242420, upper bound: 106.7242420
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 0, lower bound: -106.7250799, upper bound: 106.7244161
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 0, lower bound: -106.7242420, upper bound: 106.7242420
IS_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 0, lower bound: -106.7236513, upper bound: 106.7268637
IS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 0, lower bound: -106.7233881, upper bound: 106.7261975
IS_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 0, lower bound: -106.7236513, upper bound: 106.7268638
IS_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 0, lower bound: -106.7233881, upper bound: 106.7261975
IS_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 0, lower bound: -106.7227044, upper bound: 106.7212512
IS_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 0, lower bound: -106.7219503, upper bound: 106.7210984
IS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 0, lower bound: -106.7222155, upper bound: 106.7221259
IS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 0, lower bound: -106.7219503, upper bound: 106.7210983
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 0, lower bound: -106.7268637, upper bound: 106.7236513
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 0, lower bound: -106.7261975, upper bound: 106.7233881
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 0, lower bound: -106.7268637, upper bound: 106.7236513
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 0, lower bound: -106.7261975, upper bound: 106.7233881
IS_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 0, lower bound: -106.7212512, upper bound: 106.7227044
IS_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 0, lower bound: -106.7210983, upper bound: 106.7219504
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 0, lower bound: -106.7221260, upper bound: 106.7222156
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 0, lower bound: -106.7210984, upper bound: 106.7219504
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 0, lower bound: -106.7265818, upper bound: 106.7221501
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 0, lower bound: -106.7257399, upper bound: 106.7219436
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 0, lower bound: -106.7265818, upper bound: 106.7221500
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 0, lower bound: -106.7257399, upper bound: 106.7219436
IS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 0, lower bound: -106.7210479, upper bound: 106.7217888
IS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 0, lower bound: -106.7208772, upper bound: 106.7208795
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 0, lower bound: -106.7218084, upper bound: 106.7210532
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.25
Output dim: 0, lower bound: -106.7208772, upper bound: 106.7208795
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=108.02017974853516
rel_dist={0: [-106.76706178863964, 106.76706178863964]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7551944, upper bound: 106.7552854
time: 8.78 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7562950, upper bound: 106.7562949
time: 8.72 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 17.64 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 17.64
Output dim: 0, lower bound: -106.7551944, upper bound: 106.7552854
IS_A2, status: Status.UNKNOWN, split count: 1, time: 17.64
Output dim: 0, lower bound: -106.7562950, upper bound: 106.7562949

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -40.4215355, 32.1656380, -44.9332466, 35.7543182, -76.1758423, 77.0988846
1: -32.5613441, 27.7784863, -36.3032570, 30.9133835, -63.4747276, 64.0817261
2: -43.4267426, 28.1031914, -48.3676224, 31.2706375, -74.6973572, 76.4707947
3: -45.9748688, 24.2641106, -51.1954308, 26.9846935, -72.9595642, 75.4595261
4: -42.8716888, 32.4032478, -47.7197227, 36.0675087, -78.9391937, 80.1229706
5: -38.8010483, 30.6035709, -43.1899834, 33.9767609, -72.7778091, 73.7935562
6: -36.7541809, 35.3119507, -40.8936920, 39.2422791, -75.9964523, 76.2056427
7: -39.3098488, 33.9188652, -43.7536049, 37.6771469, -76.9869995, 77.6724701
8: -47.4530792, 31.4979420, -52.9246407, 35.1719322, -82.6250153, 84.4225769
9: -36.3031998, 35.3297119, -40.3834152, 39.3370667, -75.6402664, 75.7131271

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7432882, upper bound: 106.7424070
time: 11.44 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7495587, upper bound: 106.7496116
time: 10.75 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -43.0184174, 34.2086563, -44.8011589, 35.6482353, -78.6666565, 79.0098114
1: -34.7182274, 29.5691071, -36.2056007, 30.8217335, -65.5399628, 65.7747040
2: -46.2603874, 29.9053726, -48.2140770, 31.1773186, -77.4377060, 78.1194382
3: -48.9885712, 25.8325176, -51.0355492, 26.9116516, -75.9002075, 76.8680573
4: -45.6809273, 34.5013237, -47.5927773, 35.9599495, -81.6408768, 82.0941010
5: -41.2985191, 32.5255585, -43.0581436, 33.8837395, -75.1822586, 75.5836792
6: -39.1464157, 37.5944214, -40.7849388, 39.1476402, -78.2940521, 78.3793411
7: -41.8991547, 36.0853615, -43.6360703, 37.5730515, -79.4722061, 79.7214279
8: -50.5748444, 33.5495453, -52.7523232, 35.0514107, -85.6262436, 86.3018494
9: -38.6610641, 37.6220551, -40.2807121, 39.2122688, -77.8733292, 77.9027557

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7438820, upper bound: 106.7429018
time: 12.13 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7504117, upper bound: 106.7504117
time: 8.52 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 22.01 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 22.01
Output dim: 0, lower bound: -106.7432882, upper bound: 106.7424070
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 22.01
Output dim: 0, lower bound: -106.7495587, upper bound: 106.7496116
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 22.01
Output dim: 0, lower bound: -106.7438820, upper bound: 106.7429018
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 22.01
Output dim: 0, lower bound: -106.7504117, upper bound: 106.7504117

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -30.3322830, 24.0456409, -31.5775948, 25.0205479, -55.3528252, 55.6232338
1: -24.2242393, 20.7744884, -25.2669449, 21.6487885, -45.8730240, 46.0414314
2: -32.2905159, 21.0174713, -33.6366768, 21.8671169, -54.1576233, 54.6541481
3: -34.3549347, 18.2669144, -35.8285866, 19.0395374, -53.3944664, 54.0954971
4: -32.0029411, 24.2712002, -33.3348808, 25.2806892, -57.2836266, 57.6060753
5: -28.9231129, 22.9761753, -30.1311989, 23.8872948, -52.8104057, 53.1073761
6: -27.5458508, 26.5443993, -28.6877937, 27.6526070, -55.1984558, 55.2321854
7: -29.4645634, 25.4641399, -30.6902599, 26.5091972, -55.9737587, 56.1543999
8: -35.3616257, 23.3404484, -36.8591805, 24.3317451, -59.6933708, 60.1996307
9: -27.2634487, 26.3855457, -28.4190617, 27.4866199, -54.7500687, 54.8045998

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7205022, upper bound: 106.7205121
time: 12.21 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7193227, upper bound: 106.7180626
time: 12.62 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -34.9778366, 27.7900620, -37.2738876, 29.6123848, -64.5902252, 65.0639420
1: -28.0509548, 23.9952755, -29.9578686, 25.5987625, -53.6497116, 53.9531441
2: -37.4210815, 24.2684364, -39.9359131, 25.8706856, -63.2917671, 64.2043457
3: -39.6929474, 21.0218506, -42.3693352, 22.4194164, -62.1123657, 63.3911858
4: -37.0040665, 28.0011826, -39.4691772, 29.8563938, -66.8604584, 67.4703598
5: -33.4692421, 26.4932137, -35.7040253, 28.1991043, -61.6683464, 62.1972389
6: -31.7650948, 30.5835476, -33.8741684, 32.6066856, -64.3717804, 64.4577179
7: -33.9718704, 29.3662529, -36.2232628, 31.2894897, -65.2613525, 65.5895157
8: -40.8880272, 27.0640316, -43.6526833, 28.9093933, -69.7974243, 70.7167130
9: -31.4247837, 30.4794998, -33.5155602, 32.5115623, -63.9363480, 63.9950600

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7281214, upper bound: 106.7289938
time: 11.05 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7274968, upper bound: 106.7270904
time: 12.22 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -32.8103714, 26.0119839, -31.6365433, 25.0706730, -57.8810425, 57.6485291
1: -26.2914276, 22.4904461, -25.3238583, 21.6914024, -47.9828300, 47.8143044
2: -35.0050621, 22.7212467, -33.7037697, 21.9033470, -56.9084091, 56.4250183
3: -37.2437172, 19.7571354, -35.8942642, 19.0776291, -56.3213425, 55.6513977
4: -34.6873398, 26.2709293, -33.4201469, 25.3324947, -60.0198364, 59.6910706
5: -31.3186512, 24.8200359, -30.1776276, 23.9343491, -55.2529984, 54.9976654
6: -29.8217735, 28.7344494, -28.7524567, 27.7164803, -57.5382538, 57.4869080
7: -31.9206200, 27.5539742, -30.7697525, 26.5613480, -58.4819679, 58.3237267
8: -38.3212509, 25.2811279, -36.9277115, 24.3722630, -62.6935120, 62.2088394
9: -29.5237370, 28.5666924, -28.4877090, 27.5406780, -57.0644150, 57.0544014

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7204193, upper bound: 106.7183628
time: 11.59 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7191175, upper bound: 106.7179875
time: 10.44 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -37.3566017, 29.6629601, -37.1039848, 29.4709682, -66.8275681, 66.7669449
1: -30.0321655, 25.6401367, -29.8319225, 25.4819622, -55.5141220, 55.4720612
2: -40.0254593, 25.9138641, -39.7440224, 25.7505550, -65.7760162, 65.6578827
3: -42.4586678, 22.4597511, -42.1639061, 22.3244305, -64.7830887, 64.6236572
4: -39.5738754, 29.9185333, -39.2942314, 29.7218552, -69.2957077, 69.2127609
5: -35.7596817, 28.2547569, -35.5305557, 28.0746632, -63.8343430, 63.7853127
6: -33.9568672, 32.6848373, -33.7293663, 32.4753952, -66.4322586, 66.4141998
7: -36.3379402, 31.3633270, -36.0767097, 31.1536293, -67.4915543, 67.4400330
8: -43.7245827, 28.9288235, -43.4331818, 28.7572422, -72.4818192, 72.3620071
9: -33.5910110, 32.5721397, -33.3806076, 32.3543243, -65.9453354, 65.9527435

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7273566, upper bound: 106.7286877
time: 10.61 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7270000, upper bound: 106.7270000
time: 9.15 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 21.10 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 21.10
Output dim: 0, lower bound: -106.7205022, upper bound: 106.7205121
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 21.10
Output dim: 0, lower bound: -106.7193227, upper bound: 106.7180626
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 21.10
Output dim: 0, lower bound: -106.7281214, upper bound: 106.7289938
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 21.10
Output dim: 0, lower bound: -106.7274968, upper bound: 106.7270904
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.10
Output dim: 0, lower bound: -106.7204193, upper bound: 106.7183628
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.10
Output dim: 0, lower bound: -106.7191175, upper bound: 106.7179875
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 21.10
Output dim: 0, lower bound: -106.7273566, upper bound: 106.7286877
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 21.10
Output dim: 0, lower bound: -106.7270000, upper bound: 106.7270000

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -22.4959831, 17.8016338, -20.4341068, 16.1547279, -38.6507034, 38.2357407
1: -17.6605396, 15.3633003, -15.9239969, 13.9498692, -31.6104050, 31.2872963
2: -23.7512722, 15.6029997, -21.5065994, 14.1627855, -37.9140549, 37.1095963
3: -25.2089996, 13.4911366, -22.8317432, 12.2560434, -37.4650421, 36.3228645
4: -23.6622791, 17.9572010, -21.4467430, 16.3198299, -39.9821091, 39.4039421
5: -21.2090340, 17.0418320, -19.1767330, 15.4341488, -36.6431808, 36.2185555
6: -20.4191513, 19.6659412, -18.5544434, 17.8836098, -38.3027611, 38.2203827
7: -21.8636703, 18.8730793, -19.8667583, 17.1374435, -39.0011101, 38.7398262
8: -26.0655003, 17.2050304, -23.6574974, 15.6431170, -41.7086182, 40.8625259
9: -20.2026615, 19.5235271, -18.3619862, 17.7320309, -37.9346924, 37.8855133

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7076697, upper bound: 106.7076344
time: 11.87 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7074146, upper bound: 106.7075839
time: 12.12 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -17.2890072, 13.6532831, -16.9940681, 13.4348125, -30.7238197, 30.6473503
1: -13.2356157, 11.7406673, -12.9836082, 11.5474358, -24.7830505, 24.7242737
2: -18.0346355, 11.9619980, -17.7279568, 11.7687073, -29.8033409, 29.6899529
3: -19.1549702, 10.3257236, -18.8834381, 10.1303730, -29.2853413, 29.2091618
4: -17.9804821, 13.7703915, -17.7153664, 13.5430107, -31.5234928, 31.4857559
5: -16.1377659, 13.0774956, -15.8289661, 12.7937841, -28.9315491, 28.9064617
6: -15.6513796, 15.0983849, -15.4247179, 14.8912163, -30.5425949, 30.5231018
7: -16.6695099, 14.4508142, -16.4373417, 14.2191181, -30.8886280, 30.8881531
8: -19.8626366, 13.1908674, -19.5787277, 13.0019217, -32.8645554, 32.7695885
9: -15.4257107, 14.8702860, -15.1858768, 14.6416636, -30.0673733, 30.0561638

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7065071, upper bound: 106.7048443
time: 12.58 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7061420, upper bound: 106.7047818
time: 11.67 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -26.1810932, 20.7781658, -24.5588112, 19.4736576, -45.6547508, 45.3369751
1: -20.7504921, 17.8980522, -19.3835964, 16.7909164, -37.5414085, 37.2816467
2: -27.8037186, 18.2004528, -26.0353031, 17.0752983, -44.8790169, 44.2357521
3: -29.4718552, 15.6818485, -27.5828190, 14.6972036, -44.1690598, 43.2646637
4: -27.6310940, 20.9207134, -25.9083328, 19.6179142, -47.2489929, 46.8290482
5: -24.8322563, 19.8481693, -23.2163639, 18.5901985, -43.4224510, 43.0645256
6: -23.7922688, 22.8916397, -22.3157864, 21.4764557, -45.2687225, 45.2074280
7: -25.4524002, 21.9775066, -23.8940430, 20.6194382, -46.0718384, 45.8715439
8: -30.4264832, 20.1010971, -28.5149212, 18.8408394, -49.2673187, 48.6160202
9: -23.5239162, 22.7806320, -22.0851078, 21.3677063, -44.8916206, 44.8657379

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7156752, upper bound: 106.7165271
time: 11.88 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7153590, upper bound: 106.7164708
time: 12.42 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -20.3449364, 16.1129417, -20.5126400, 16.2379417, -36.5828781, 36.6255798
1: -15.8021727, 13.8394203, -15.9116144, 13.9540329, -29.7562065, 29.7510338
2: -21.3834743, 14.0938358, -21.5674133, 14.2087860, -35.5922623, 35.6612473
3: -22.6542492, 12.1427536, -22.8706131, 12.2285185, -34.8827667, 35.0133629
4: -21.2904034, 16.1978569, -21.4843178, 16.3187160, -37.6091194, 37.6821747
5: -19.1177921, 15.4265795, -19.2345886, 15.4824810, -34.6002731, 34.6611595
6: -18.4154396, 17.7452126, -18.5786514, 17.9261246, -36.3415565, 36.3238640
7: -19.6782513, 17.0259247, -19.8834953, 17.1657867, -36.8440361, 36.9094200
8: -23.4289894, 15.5367823, -23.6519947, 15.6873693, -39.1163559, 39.1887779
9: -18.1856956, 17.5681801, -18.3485966, 17.7297077, -35.9154053, 35.9167786

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7149851, upper bound: 106.7142822
time: 10.87 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7061420, upper bound: 106.7142213
time: 11.00 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -21.9858227, 17.4039288, -23.9612064, 18.9604568, -40.9462738, 41.3651352
1: -17.2205296, 15.0100842, -18.8817787, 16.3840313, -33.6045609, 33.8918610
2: -23.2171211, 15.2301216, -25.3492680, 16.5954189, -39.8125381, 40.5793915
3: -24.6212101, 13.1661453, -26.9283905, 14.3901844, -39.0113945, 40.0945358
4: -23.1308136, 17.5687618, -25.2487316, 19.1598701, -42.2906837, 42.8174934
5: -20.6871624, 16.6157169, -22.6190777, 18.1210136, -38.8081741, 39.2347946
6: -19.9741020, 19.2390347, -21.7613468, 20.9809685, -40.9550705, 41.0003815
7: -21.4081631, 18.4499435, -23.3244438, 20.1066036, -41.5147667, 41.7743874
8: -25.4975243, 16.8218365, -27.8209038, 18.3585358, -43.8560600, 44.6427383
9: -19.7599888, 19.0906696, -21.5655289, 20.8200111, -40.5800018, 40.6561966

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7073152, upper bound: 106.7056889
time: 13.99 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7072083, upper bound: 106.7050643
time: 12.25 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -16.8995533, 13.3620043, -17.5305004, 13.8704910, -30.7700443, 30.8925056
1: -12.8928480, 11.4582090, -13.4610500, 11.9337626, -24.8266106, 24.9192581
2: -17.6293354, 11.6966848, -18.3236866, 12.1371641, -29.7664986, 30.0203705
3: -18.7508793, 10.0470800, -19.4701824, 10.4893122, -29.2401905, 29.5172615
4: -17.5927048, 13.4609127, -18.2924557, 14.0035095, -31.5962143, 31.7533665
5: -15.7308550, 12.7027779, -16.3723793, 13.2467422, -28.9775963, 29.0751572
6: -15.3370209, 14.7930126, -15.8965368, 15.3585157, -30.6955376, 30.6895485
7: -16.3333111, 14.1277180, -16.9583664, 14.6831732, -31.0164814, 31.0860806
8: -19.4588585, 12.9059639, -20.1992321, 13.4024696, -32.8613243, 33.1051941
9: -15.0642138, 14.5388193, -15.6914253, 15.1175699, -30.1817818, 30.2302437

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7059472, upper bound: 106.7052732
time: 11.06 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7058250, upper bound: 106.7046909
time: 10.24 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -28.6867085, 22.7687092, -24.9342766, 19.7788982, -48.4656067, 47.7029839
1: -22.8432465, 19.6254425, -19.6978798, 17.0535984, -39.8968430, 39.3233185
2: -30.5459251, 19.9137402, -26.4491367, 17.3233261, -47.8692513, 46.3628769
3: -32.3910332, 17.1880493, -28.0223560, 14.9205723, -47.3116074, 45.2104034
4: -30.3312569, 22.9419155, -26.3229256, 19.9337101, -50.2649689, 49.2648392
5: -27.2545471, 21.7186260, -23.5790806, 18.8770771, -46.1316223, 45.2977028
6: -26.0812626, 25.1094418, -22.6572266, 21.8165550, -47.8978157, 47.7666626
7: -27.9326267, 24.0860405, -24.2798138, 20.9439259, -48.8765488, 48.3658524
8: -33.4132080, 22.0554409, -28.9619007, 19.1310673, -52.5442734, 51.0173378
9: -25.8075085, 24.9868717, -22.4368973, 21.6978970, -47.5054016, 47.4237671

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7149532, upper bound: 106.7161854
time: 12.56 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7145019, upper bound: 106.7161341
time: 11.61 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -21.2253609, 16.8048515, -19.6973457, 15.5983095, -36.8236694, 36.5021973
1: -16.5350361, 14.4408808, -15.2335777, 13.3885670, -29.9236031, 29.6744576
2: -22.3503017, 14.6936388, -20.6898041, 13.6519842, -36.0022850, 35.3834419
3: -23.6751595, 12.6653624, -21.9361820, 11.7374725, -35.4126320, 34.6015434
4: -22.2472858, 16.9142914, -20.6156158, 15.6812239, -37.9285088, 37.5299072
5: -19.9480000, 16.0586815, -18.4333706, 14.8586121, -34.8066101, 34.4920502
6: -19.2276230, 18.5289383, -17.8453941, 17.2245598, -36.4521751, 36.3743210
7: -20.5785255, 17.7672462, -19.0904903, 16.4908867, -37.0694122, 36.8577347
8: -24.4922943, 16.2107143, -22.7077503, 15.0597639, -39.5520554, 38.9184456
9: -18.9873085, 18.3534107, -17.6143227, 17.0127888, -36.0000954, 35.9677353

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7145536, upper bound: 106.7141925
time: 10.14 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7141250, upper bound: 106.7141250
time: 10.11 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 21.59 seconds
IS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 21.59
Output dim: 0, lower bound: -106.7076697, upper bound: 106.7076344
IS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 21.59
Output dim: 0, lower bound: -106.7074146, upper bound: 106.7075839
IS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 21.59
Output dim: 0, lower bound: -106.7065071, upper bound: 106.7048443
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 21.59
Output dim: 0, lower bound: -106.7061420, upper bound: 106.7047818
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 21.59
Output dim: 0, lower bound: -106.7156752, upper bound: 106.7165271
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 21.59
Output dim: 0, lower bound: -106.7153590, upper bound: 106.7164708
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 21.59
Output dim: 0, lower bound: -106.7149851, upper bound: 106.7142822
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 21.59
Output dim: 0, lower bound: -106.7061420, upper bound: 106.7142213
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.59
Output dim: 0, lower bound: -106.7073152, upper bound: 106.7056889
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.59
Output dim: 0, lower bound: -106.7072083, upper bound: 106.7050643
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.59
Output dim: 0, lower bound: -106.7059472, upper bound: 106.7052732
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.59
Output dim: 0, lower bound: -106.7058250, upper bound: 106.7046909
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 21.59
Output dim: 0, lower bound: -106.7149532, upper bound: 106.7161854
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 21.59
Output dim: 0, lower bound: -106.7145019, upper bound: 106.7161341
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 21.59
Output dim: 0, lower bound: -106.7145536, upper bound: 106.7141925
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 21.59
Output dim: 0, lower bound: -106.7141250, upper bound: 106.7141250

## BFS IS instance: IS_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -14.7349501, 11.6636944, -15.1145563, 11.9726295, -26.7075806, 26.7782516
1: -11.1498489, 9.9975443, -11.4758778, 10.2781029, -21.4279518, 21.4734211
2: -15.3166704, 10.2055130, -15.7344589, 10.4873848, -25.8040543, 25.9399700
3: -16.2350006, 8.7722464, -16.7237015, 9.0138845, -25.2488804, 25.4959488
4: -15.3150072, 11.7334280, -15.7612057, 12.0685644, -27.3835697, 27.4946308
5: -13.7247581, 11.1363220, -14.0737371, 11.4031124, -25.1278706, 25.2100582
6: -13.4047747, 12.8794498, -13.7605038, 13.2435055, -26.6482811, 26.6399517
7: -14.2083626, 12.3546705, -14.6254597, 12.6941195, -26.9024773, 26.9801273
8: -16.9282475, 11.2053900, -17.4234543, 11.5387068, -28.4669514, 28.6288452
9: -13.1507111, 12.6370964, -13.5377665, 13.0170021, -26.1677132, 26.1748619

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 120

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_B1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6597303, upper bound: 106.6607179
time: 13.12 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6566884, upper bound: 106.6561712
time: 11.95 seconds

## BFS IS instance: IS_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -14.6274586, 11.5677967, -13.5454082, 10.7439928, -25.3714504, 25.1132050
1: -11.0160799, 9.9175634, -10.1382093, 9.1987038, -20.2147827, 20.0557728
2: -15.1946840, 10.1034422, -14.0193653, 9.3874989, -24.5821838, 24.1228065
3: -16.1321487, 8.6935959, -14.9219971, 8.0661421, -24.1982918, 23.6155930
4: -15.1780424, 11.6528692, -14.0604544, 10.8147030, -25.9927444, 25.7133198
5: -13.6061659, 11.0242281, -12.5691299, 10.2191381, -23.8253040, 23.5933571
6: -13.3050280, 12.8032951, -12.3351688, 11.8769464, -25.1819744, 25.1384640
7: -14.1036739, 12.2371788, -13.0669651, 11.3655233, -25.4691963, 25.3041439
8: -16.7855816, 11.1186666, -15.5562820, 10.3356514, -27.1212330, 26.6749496
9: -13.0454311, 12.5268526, -12.0989542, 11.6183777, -24.6638088, 24.6258068

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 165

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6587403, upper bound: 106.6572892
time: 11.55 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6566417, upper bound: 106.6561449
time: 12.82 seconds

## BFS IS instance: IS_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -10.8929424, 8.6552134, -12.3975334, 9.8505840, -20.7435265, 21.0527458
1: -7.8930068, 7.3218746, -9.1441278, 8.3695765, -16.2625809, 16.4660034
2: -11.1157045, 7.5257812, -12.7572832, 8.5802946, -19.6959972, 20.2830601
3: -11.7940950, 6.4433413, -13.6084309, 7.3469367, -19.1410275, 20.0517673
4: -11.1478691, 8.6356306, -12.8142166, 9.8691292, -21.0169983, 21.4498444
5: -10.0045080, 8.2078772, -11.4372492, 9.3005295, -19.3050365, 19.6451263
6: -9.9134216, 9.5327587, -11.3081770, 10.8921118, -20.8055344, 20.8409348
7: -10.3833685, 9.0869942, -11.9336452, 10.3733892, -20.7567577, 21.0206394
8: -12.3520212, 8.2689953, -14.1922674, 9.4673882, -21.8194084, 22.4612617
9: -9.6288652, 9.2413397, -11.0401249, 10.6053324, -20.2341976, 20.2814636

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6584090, upper bound: 106.6589913
time: 11.55 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6548451, upper bound: 106.6528383
time: 11.32 seconds

## BFS IS instance: IS_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -11.2148237, 8.8952417, -11.3963261, 9.0647087, -20.2795315, 20.2915688
1: -8.1332788, 7.5324721, -8.2937775, 7.6675749, -15.8008537, 15.8262501
2: -11.4537086, 7.7209024, -11.6586037, 7.8686371, -19.3223457, 19.3795052
3: -12.1760311, 6.6244669, -12.4398050, 6.7400374, -18.9160671, 19.0642700
4: -11.4752321, 8.8938837, -11.7168188, 9.0619564, -20.5371895, 20.6107006
5: -10.3039188, 8.4212208, -10.4685564, 8.5448151, -18.8487320, 18.8897743
6: -10.2058048, 9.8246202, -10.3907642, 10.0199480, -20.2257538, 20.2153835
7: -10.6993847, 9.3330164, -10.9253883, 9.5198231, -20.2192059, 20.2584019
8: -12.7168636, 8.5096388, -12.9902477, 8.7018642, -21.4187279, 21.4998837
9: -9.9144907, 9.5104532, -10.1197205, 9.7000666, -19.6145573, 19.6301727

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6580455, upper bound: 106.6588072
time: 11.89 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6546321, upper bound: 106.6527503
time: 12.74 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -17.3477001, 13.7560663, -18.3100624, 14.5202932, -31.8679924, 32.0661278
1: -13.3510523, 11.7945261, -14.1630201, 12.4750013, -25.8260536, 25.9575462
2: -18.1914024, 12.0253181, -19.2484226, 12.7109728, -30.9023743, 31.2737408
3: -19.2273426, 10.3278313, -20.3599758, 10.9201574, -30.1474991, 30.6878052
4: -18.1401024, 13.8025646, -19.1883545, 14.6035776, -32.7436752, 32.9909134
5: -16.2635689, 13.1701784, -17.1696949, 13.8701200, -30.1336880, 30.3398743
6: -15.7535400, 15.1422424, -16.6324921, 16.0065441, -31.7600842, 31.7747345
7: -16.7841873, 14.5634785, -17.7655907, 15.3871069, -32.1712914, 32.3290672
8: -19.9676399, 13.2119751, -21.1263084, 13.9939156, -33.9615555, 34.3382759
9: -15.5217247, 14.9355736, -16.4350758, 15.8322163, -31.3539410, 31.3706493

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6770753, upper bound: 106.6789497
time: 12.57 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6762035, upper bound: 106.6767366
time: 11.78 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -17.1316833, 13.5655289, -16.3977509, 13.0055780, -30.1372604, 29.9632797
1: -13.1208057, 11.6310863, -12.5264635, 11.1509781, -24.2717800, 24.1575470
2: -17.9330444, 11.8434200, -17.1502647, 11.3690987, -29.3021393, 28.9936810
3: -18.9852409, 10.1688566, -18.1464272, 9.7502375, -28.7354774, 28.3152828
4: -17.8653584, 13.6352310, -17.1003933, 13.0739040, -30.9392624, 30.7356205
5: -16.0398712, 12.9663820, -15.3259258, 12.4217901, -28.4616623, 28.2923050
6: -15.5455360, 14.9601488, -14.8812590, 14.3304367, -29.8759708, 29.8414078
7: -16.5559216, 14.3491077, -15.8435068, 13.7645397, -30.3204575, 30.1926136
8: -19.6904583, 13.0280390, -18.8515854, 12.5119371, -32.2023849, 31.8796234
9: -15.3044968, 14.7205238, -14.6704960, 14.1039162, -29.4084129, 29.3910179

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6587403, upper bound: 106.6779456
time: 11.61 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6756783, upper bound: 106.6765577
time: 11.40 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -13.0776367, 10.3839445, -15.1295433, 11.9970064, -25.0746422, 25.5134888
1: -9.7147369, 8.8331060, -11.4080801, 10.2427359, -19.9574718, 20.2411861
2: -13.5125742, 9.0458355, -15.7466936, 10.4765034, -23.9890785, 24.7925301
3: -14.2733727, 7.7412014, -16.6828365, 8.9563618, -23.2297306, 24.4240379
4: -13.4782324, 10.3649235, -15.7018518, 12.0151939, -25.4934273, 26.0667744
5: -12.1333218, 9.8859434, -14.0647354, 11.3889523, -23.5222740, 23.9506798
6: -11.8754339, 11.4139519, -13.7364693, 13.2386503, -25.1140842, 25.1504211
7: -12.5057163, 10.9255552, -14.5694132, 12.6552029, -25.1609192, 25.4949684
8: -14.8888531, 9.9244413, -17.3465939, 11.5352612, -26.4241142, 27.2710342
9: -11.5812120, 11.1447525, -13.4667034, 12.9707203, -24.5519333, 24.6114559

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6763928, upper bound: 106.6777558
time: 10.67 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6751718, upper bound: 106.6740917
time: 11.64 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -13.3558273, 10.5891304, -13.9503651, 11.0797434, -24.4355698, 24.5394955
1: -9.9163828, 9.0150909, -10.4191771, 9.4349442, -19.3513241, 19.4342690
2: -13.8042078, 9.2128935, -14.4629869, 9.6521034, -23.4563103, 23.6758804
3: -14.6024561, 7.9002204, -15.3267279, 8.2578669, -22.8603230, 23.2269459
4: -13.7542372, 10.5900364, -14.4262924, 11.0799799, -24.8342171, 25.0163288
5: -12.3805962, 10.0682669, -12.9417324, 10.5042248, -22.8848190, 23.0099983
6: -12.1241245, 11.6708727, -12.6722002, 12.2182770, -24.3424015, 24.3430691
7: -12.7841511, 11.1316872, -13.4068298, 11.6622009, -24.4463520, 24.5385170
8: -15.2041197, 10.1332636, -15.9495525, 10.6367788, -25.8408985, 26.0828114
9: -11.8259745, 11.3777504, -12.3901901, 11.9254799, -23.7514534, 23.7679405

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6756843, upper bound: 106.6774871
time: 12.92 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6746551, upper bound: 106.6739616
time: 11.80 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -16.3906155, 12.9895830, -15.8151655, 12.5393505, -28.9299603, 28.8047485
1: -12.5456915, 11.1512804, -12.0807505, 10.7818594, -23.3275509, 23.2320290
2: -17.1562519, 11.3600321, -16.5363960, 10.9566774, -28.1129246, 27.8964272
3: -18.1825180, 9.7643652, -17.5459538, 9.4407558, -27.6232719, 27.3103180
4: -17.1330948, 13.0978880, -16.5480309, 12.6552849, -29.7883797, 29.6459198
5: -15.2957401, 12.3670826, -14.7685871, 11.9575033, -27.2532425, 27.1356678
6: -14.9167061, 14.3597994, -14.4039640, 13.8818579, -28.7985649, 28.7637634
7: -15.9029951, 13.7715397, -15.3432589, 13.3037615, -29.2067566, 29.1147976
8: -18.9304314, 12.5077658, -18.2630157, 12.0651340, -30.9955616, 30.7707825
9: -14.6916466, 14.1330843, -14.1983957, 13.6287289, -28.3203735, 28.3314800

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6574059, upper bound: 106.6540763
time: 12.46 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6555572, upper bound: 106.6534094
time: 10.23 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -14.4942312, 11.4952803, -15.3914642, 12.1955090, -26.6897392, 26.8867435
1: -10.9269323, 9.8319950, -11.6874733, 10.4767103, -21.4036427, 21.5194683
2: -15.0751629, 10.0318670, -16.0576706, 10.6408758, -25.7160378, 26.0895386
3: -15.9829664, 8.6161394, -17.0630913, 9.1667671, -25.1497345, 25.6792297
4: -15.0620403, 11.5790586, -16.0632706, 12.3135509, -27.3755913, 27.6423264
5: -13.4771471, 10.9294405, -14.3509483, 11.6135950, -25.0907421, 25.2803879
6: -13.1964474, 12.7023039, -14.0188465, 13.5274391, -26.7238827, 26.7211475
7: -14.0050964, 12.1667633, -14.9184561, 12.9270077, -26.9321041, 27.0852165
8: -16.6707954, 11.0494661, -17.7443218, 11.7316313, -28.4024277, 28.7937851
9: -12.9460402, 12.4407625, -13.8036079, 13.2439671, -26.1900043, 26.2443695

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 120

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6590521, upper bound: 106.6592901
time: 10.88 seconds

## Relational analysis of IS_A2_B1_A1_B2_B2

### Relational analysis result of IS_A2_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6554822, upper bound: 106.6530475
time: 11.62 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 23.90 seconds
IS_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 23.90
Output dim: 0, lower bound: -106.6597303, upper bound: 106.6607179
IS_A1_B1_B1_A1_B2, status: Status.VERIFIED, split count: 5, time: 23.90
Output dim: 0, lower bound: -106.6566884, upper bound: 106.6561712
IS_A1_B1_B1_A2_A1, status: Status.VERIFIED, split count: 5, time: 23.90
Output dim: 0, lower bound: -106.6587403, upper bound: 106.6572892
IS_A1_B1_B1_A2_A2, status: Status.VERIFIED, split count: 5, time: 23.90
Output dim: 0, lower bound: -106.6566417, upper bound: 106.6561449
IS_A1_B1_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 23.90
Output dim: 0, lower bound: -106.6584090, upper bound: 106.6589913
IS_A1_B1_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 23.90
Output dim: 0, lower bound: -106.6548451, upper bound: 106.6528383
IS_A1_B1_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 23.90
Output dim: 0, lower bound: -106.6580455, upper bound: 106.6588072
IS_A1_B1_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 23.90
Output dim: 0, lower bound: -106.6546321, upper bound: 106.6527503
IS_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 23.90
Output dim: 0, lower bound: -106.6770753, upper bound: 106.6789497
IS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 23.90
Output dim: 0, lower bound: -106.6762035, upper bound: 106.6767366
IS_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 23.90
Output dim: 0, lower bound: -106.6587403, upper bound: 106.6779456
IS_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 23.90
Output dim: 0, lower bound: -106.6756783, upper bound: 106.6765577
IS_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 23.90
Output dim: 0, lower bound: -106.6763928, upper bound: 106.6777558
IS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 23.90
Output dim: 0, lower bound: -106.6751718, upper bound: 106.6740917
IS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 23.90
Output dim: 0, lower bound: -106.6756843, upper bound: 106.6774871
IS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 23.90
Output dim: 0, lower bound: -106.6746551, upper bound: 106.6739616
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 23.90
Output dim: 0, lower bound: -106.6574059, upper bound: 106.6540763
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 23.90
Output dim: 0, lower bound: -106.6555572, upper bound: 106.6534094
IS_A2_B1_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 23.90
Output dim: 0, lower bound: -106.6590521, upper bound: 106.6592901
IS_A2_B1_A1_B2_B2, status: Status.VERIFIED, split count: 5, time: 23.90
Output dim: 0, lower bound: -106.6554822, upper bound: 106.6530475
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.90
Output dim: 0, lower bound: -106.7059472, upper bound: 106.7052732
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.90
Output dim: 0, lower bound: -106.7058250, upper bound: 106.7046909
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 23.90
Output dim: 0, lower bound: -106.7149532, upper bound: 106.7161854
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 23.90
Output dim: 0, lower bound: -106.7145019, upper bound: 106.7161341
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 23.90
Output dim: 0, lower bound: -106.7145536, upper bound: 106.7141925
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 23.90
Output dim: 0, lower bound: -106.7141250, upper bound: 106.7141250
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=108.02017974853516
rel_dist={0: [-106.76692036843552, 106.76692036843548]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1829.69 seconds
