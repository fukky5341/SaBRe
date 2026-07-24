## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 264.612307261
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660)
1: (-120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653)
2: (-158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565)
3: (-169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952)
4: (-154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305)
5: (-139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240)
6: (-133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709)
7: (-144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966)
8: (-174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793)
9: (-131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962)

## BASE Result
execution time: IAR + LP analysis = 1.10 + 11.08 = 12.18 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -264.6227448, upper bound: 264.6227448


# Binary Search by BASE starts (time budget: 2687.82 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=266.34649658203125
rel_dist={7: [-264.62269597868163, 264.6226959786816]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=266.34649658203125
rel_dist={7: [-264.62266420921173, 264.6226641973501]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=266.34649658203125
rel_dist={7: [-264.6222950859965, 264.6222950868074]}

## Binary Search Result
Binary search time: 40.52 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 2647.30 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6211596, upper bound: 264.6213232
time: 8.19 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6226960, upper bound: 264.6226960
time: 7.87 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 16.19 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 16.19
Output dim: 7, lower bound: -264.6211596, upper bound: 264.6213232
IS_B2, status: Status.UNKNOWN, split count: 1, time: 16.19
Output dim: 7, lower bound: -264.6226960, upper bound: 264.6226960

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -143.7657471, 114.5897064, -142.5778351, 113.6492615, -257.4150085, 257.1675110
1: -120.3232193, 101.5888977, -119.3355713, 100.7713852, -221.0946045, 220.9244385
2: -158.2776794, 102.8611984, -156.9659576, 102.0101395, -260.2877808, 259.8271484
3: -168.5139923, 89.8324966, -167.1516571, 89.0801926, -257.5941772, 256.9841614
4: -154.1229706, 118.2580338, -152.8229523, 117.2858200, -271.4087830, 271.0809937
5: -138.5903625, 107.5695877, -137.4328613, 106.6576385, -245.2480011, 245.0024261
6: -132.4803314, 128.0473480, -131.3840332, 126.9968948, -259.4772339, 259.4313965
7: -144.1100159, 121.1366119, -142.9139709, 120.1200180, -264.2300110, 264.0505371
8: -173.7971191, 118.8616333, -172.3660583, 117.8782959, -291.6754150, 291.2276917
9: -130.9741974, 129.4356079, -129.8991852, 128.3625793, -259.3366699, 259.3347778

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5831956, upper bound: 264.5829520
time: 9.39 seconds

## Relational analysis of IS_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6053573, upper bound: 264.6045012
time: 8.52 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6200478, upper bound: 264.6202286
time: 7.72 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -144.3596649, 115.0605316, -143.5242462, 114.4032974, -258.7629089, 258.5847778
1: -120.8221283, 102.0048447, -120.1220856, 101.4233856, -222.2454987, 222.1269226
2: -158.9320831, 103.2858582, -158.0138855, 102.6919479, -261.6240234, 261.2997437
3: -169.2086487, 90.2014694, -168.2381287, 89.6814194, -258.8900757, 258.4395752
4: -154.7666321, 118.7446289, -153.8648071, 118.0614929, -272.8281250, 272.6093750
5: -139.1623535, 108.0147781, -138.3584747, 107.3881683, -246.5505219, 246.3732605
6: -133.0277405, 128.5737915, -132.2587128, 127.8387756, -260.8664551, 260.8324585
7: -144.7083588, 121.6381607, -143.8743134, 120.9328995, -265.6412659, 265.5124817
8: -174.5120850, 119.3504868, -173.5107269, 118.6608429, -293.1729126, 292.8612061
9: -131.5167694, 129.9706268, -130.7577209, 129.2200165, -260.7367554, 260.7283325

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5860628, upper bound: 264.5855951
time: 9.22 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5734158, upper bound: 264.5734158
time: 6.18 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 16.53 seconds
IS_B1_A1, status: Status.VERIFIED, split count: 2, time: 16.53
Output dim: 7, lower bound: -264.6053573, upper bound: 264.6045012
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 16.53
Output dim: 7, lower bound: -264.6200478, upper bound: 264.6202286
IS_B2_B1, status: Status.VERIFIED, split count: 2, time: 16.53
Output dim: 7, lower bound: -264.5860628, upper bound: 264.5855951
IS_B2_B2, status: Status.VERIFIED, split count: 2, time: 16.53
Output dim: 7, lower bound: -264.5734158, upper bound: 264.5734158

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -141.4506683, 112.7504807, -142.5778351, 113.6492615, -255.0998993, 255.3283081
1: -118.3714752, 99.9565887, -119.3355713, 100.7713852, -219.1428528, 219.2921295
2: -155.7315826, 101.2061081, -156.9659576, 102.0101395, -257.7416687, 258.1720581
3: -165.8179016, 88.3959579, -167.1516571, 89.0801926, -254.8981018, 255.5476074
4: -151.6474915, 116.3492661, -152.8229523, 117.2858200, -268.9333191, 269.1722107
5: -136.3596497, 105.8354187, -137.4328613, 106.6576385, -243.0172882, 243.2682800
6: -130.3527985, 125.9943619, -131.3840332, 126.9968948, -257.3497009, 257.3783875
7: -141.8003082, 119.1951675, -142.9139709, 120.1200180, -261.9203186, 262.1090393
8: -171.0036469, 116.9396973, -172.3660583, 117.8782959, -288.8819580, 289.3057556
9: -128.8707428, 127.3455811, -129.8991852, 128.3625793, -257.2333069, 257.2447510

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5814819, upper bound: 264.5812582
time: 8.14 seconds

## Relational analysis of IS_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6025652, upper bound: 264.6029637
time: 8.13 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5929485, upper bound: 264.5934942
time: 7.46 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 29.12 seconds
IS_B1_A2_B1, status: Status.VERIFIED, split count: 3, time: 29.12
Output dim: 7, lower bound: -264.6025652, upper bound: 264.6029637
IS_B1_A2_B2, status: Status.VERIFIED, split count: 3, time: 29.12
Output dim: 7, lower bound: -264.5929485, upper bound: 264.5934942
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=266.34649658203125
rel_dist={7: [-264.62269597868163, 264.6226959786816]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6215402, upper bound: 264.6213397
time: 8.72 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6227210, upper bound: 264.6227210
time: 6.98 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 15.81 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 15.81
Output dim: 7, lower bound: -264.6215402, upper bound: 264.6213397
IS_A2, status: Status.UNKNOWN, split count: 1, time: 15.81
Output dim: 7, lower bound: -264.6227210, upper bound: 264.6227210

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -142.5778351, 113.6492615, -144.2240601, 114.9530563, -257.5308838, 257.8732910
1: -119.3355713, 100.7713852, -120.7082214, 101.9098969, -221.2454529, 221.4796143
2: -156.9659576, 102.0101395, -158.7826538, 103.1889038, -260.1548462, 260.7927856
3: -167.1516571, 89.0801926, -169.0500946, 90.1172180, -257.2688599, 258.1302795
4: -152.8229523, 117.2858200, -154.6196899, 118.6335297, -271.4564819, 271.9054871
5: -137.4328613, 106.6576385, -139.0317383, 107.9131317, -245.3459930, 245.6893768
6: -131.3840332, 126.9968948, -132.9027710, 128.4535980, -259.8376160, 259.8996582
7: -142.9139709, 120.1200180, -144.5717621, 121.5236359, -264.4375916, 264.6917419
8: -172.3660583, 117.8782959, -174.3488770, 119.2388916, -291.6049500, 292.2271729
9: -129.8991852, 128.3625793, -131.3928833, 129.8484650, -259.7476196, 259.7553711

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5895429, upper bound: 264.5899696
time: 9.98 seconds

## Relational analysis of IS_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6072503, upper bound: 264.6082469
time: 7.78 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6204464, upper bound: 264.6202352
time: 7.10 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -143.5242462, 114.4032974, -144.3596649, 115.0605316, -258.5847778, 258.7629089
1: -120.1220856, 101.4233856, -120.8221283, 102.0048447, -222.1269226, 222.2454987
2: -158.0138855, 102.6919479, -158.9320831, 103.2858582, -261.2997437, 261.6240234
3: -168.2381287, 89.6814194, -169.2086487, 90.2014694, -258.4396057, 258.8900757
4: -153.8648071, 118.0614929, -154.7666321, 118.7446289, -272.6093750, 272.8281250
5: -138.3584747, 107.3881683, -139.1623535, 108.0147781, -246.3732605, 246.5505219
6: -132.2587128, 127.8387756, -133.0277405, 128.5737915, -260.8324585, 260.8664551
7: -143.8743134, 120.9328995, -144.7083588, 121.6381607, -265.5124817, 265.6412659
8: -173.5107269, 118.6608429, -174.5120850, 119.3504868, -292.8612061, 293.1729126
9: -130.7577209, 129.2200165, -131.5167694, 129.9706268, -260.7283325, 260.7367554

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5910583, upper bound: 264.5916867
time: 8.83 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5734228, upper bound: 264.5734228
time: 6.45 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 16.43 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 16.43
Output dim: 7, lower bound: -264.6072503, upper bound: 264.6082469
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 16.43
Output dim: 7, lower bound: -264.6204464, upper bound: 264.6202352
IS_A2_A1, status: Status.VERIFIED, split count: 2, time: 16.43
Output dim: 7, lower bound: -264.5910583, upper bound: 264.5916867
IS_A2_A2, status: Status.VERIFIED, split count: 2, time: 16.43
Output dim: 7, lower bound: -264.5734228, upper bound: 264.5734228

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -142.5778351, 113.6492615, -141.9085388, 113.1134491, -255.6912689, 255.5577545
1: -119.3355713, 100.7713852, -118.7560883, 100.2772522, -219.6127625, 219.5274658
2: -156.9659576, 102.0101395, -156.2360992, 101.5334854, -258.4994202, 258.2462463
3: -167.1516571, 89.0801926, -166.3533783, 88.6804199, -255.8320770, 255.4335632
4: -152.8229523, 117.2858200, -152.1436768, 116.7244110, -269.5473633, 269.4295044
5: -137.4328613, 106.6576385, -136.8006287, 106.1785965, -243.6114502, 243.4582672
6: -131.3840332, 126.9968948, -130.7747803, 126.4001923, -257.7842407, 257.7716675
7: -142.9139709, 120.1200180, -142.2615967, 119.5817947, -262.4956970, 262.3816223
8: -172.3660583, 117.8782959, -171.5547791, 117.3165894, -289.6826477, 289.4330750
9: -129.8991852, 128.3625793, -129.2890167, 127.7580338, -257.6571655, 257.6515503

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5879058, upper bound: 264.5883560
time: 7.67 seconds

## Relational analysis of IS_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6131021, upper bound: 264.6130608
time: 8.22 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6190878, upper bound: 264.6188327
time: 7.15 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 28.38 seconds
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 28.38
Output dim: 7, lower bound: -264.6131021, upper bound: 264.6130608
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 28.38
Output dim: 7, lower bound: -264.6190878, upper bound: 264.6188327

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -142.3280487, 113.4515076, -139.3210144, 111.0435638, -253.3715973, 252.7724915
1: -119.1264191, 100.5934143, -116.6052551, 98.3735733, -217.4999847, 217.1986694
2: -156.6913147, 101.8306885, -153.3917542, 99.6399841, -256.3312988, 255.2224426
3: -166.8561859, 88.9268112, -163.2350464, 87.1010742, -253.9572296, 252.1618500
4: -152.5537567, 117.0784836, -149.3079071, 114.5207291, -267.0744629, 266.3863831
5: -137.1930389, 106.4699402, -134.3096008, 104.1655807, -241.3586121, 240.7795258
6: -131.1530457, 126.7761841, -128.3410950, 124.1040268, -255.2570801, 255.1172638
7: -142.6614838, 119.9098969, -139.5906525, 117.3924408, -260.0539246, 259.5004883
8: -172.0656586, 117.6717606, -168.4444275, 115.1335449, -287.1992188, 286.1161804
9: -129.6697388, 128.1375275, -126.8532410, 125.3697662, -255.0394745, 254.9907684

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5952685, upper bound: 264.5951527
time: 8.19 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5837942, upper bound: 264.5830091
time: 7.00 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -142.5778351, 113.6492615, -140.3081970, 111.8435059, -254.4213409, 253.9574280
1: -119.3355713, 100.7713852, -117.4140091, 99.1282120, -218.4637604, 218.1853943
2: -156.9659576, 102.0101395, -154.4779816, 100.3751450, -257.3410645, 256.4880676
3: -167.1516571, 89.0801926, -164.4488678, 87.6963196, -254.8479767, 253.5290527
4: -152.8229523, 117.2858200, -150.4161835, 115.3878860, -268.2107849, 267.7019958
5: -137.4328613, 106.6576385, -135.2617340, 104.9642792, -242.3971405, 241.9193726
6: -131.3840332, 126.9968948, -129.2873993, 124.9851837, -256.3692017, 256.2843018
7: -142.9139709, 120.1200180, -140.6353912, 118.2310104, -261.1449280, 260.7554016
8: -172.3660583, 117.8782959, -169.6270752, 115.9895020, -288.3555603, 287.5053711
9: -129.8991852, 128.3625793, -127.8106995, 126.3078995, -256.2070312, 256.1732788

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5859966, upper bound: 264.5865386
time: 8.24 seconds

## Relational analysis of IS_A1_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6041132, upper bound: 264.6039367
time: 8.24 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5925067, upper bound: 264.5919457
time: 7.93 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 29.58 seconds
IS_A1_B2_B1_A1, status: Status.VERIFIED, split count: 4, time: 29.58
Output dim: 7, lower bound: -264.5952685, upper bound: 264.5951527
IS_A1_B2_B1_A2, status: Status.VERIFIED, split count: 4, time: 29.58
Output dim: 7, lower bound: -264.5837942, upper bound: 264.5830091
IS_A1_B2_B2_A1, status: Status.VERIFIED, split count: 4, time: 29.58
Output dim: 7, lower bound: -264.6041132, upper bound: 264.6039367
IS_A1_B2_B2_A2, status: Status.VERIFIED, split count: 4, time: 29.58
Output dim: 7, lower bound: -264.5925067, upper bound: 264.5919457
Binary search (step 1): status=Status.VERIFIED, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=266.34649658203125
rel_dist={7: [-264.6227209801136, 264.62272095543653]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6216638, upper bound: 264.6214339
time: 7.56 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6227372, upper bound: 264.6227372
time: 6.24 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 13.92 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 13.92
Output dim: 7, lower bound: -264.6216638, upper bound: 264.6214339
IS_A2, status: Status.UNKNOWN, split count: 1, time: 13.92
Output dim: 7, lower bound: -264.6227372, upper bound: 264.6227372

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -142.5778351, 113.6492615, -144.3596649, 115.0605316, -257.6383362, 258.0088806
1: -119.3355713, 100.7713852, -120.8221283, 102.0048447, -221.3403931, 221.5935059
2: -156.9659576, 102.0101395, -158.9320831, 103.2858582, -260.2518311, 260.9421997
3: -167.1516571, 89.0801926, -169.2086487, 90.2014694, -257.3531189, 258.2888489
4: -152.8229523, 117.2858200, -154.7666321, 118.7446289, -271.5675354, 272.0524292
5: -137.4328613, 106.6576385, -139.1623535, 108.0147781, -245.4476318, 245.8199921
6: -131.3840332, 126.9968948, -133.0277405, 128.5737915, -259.9577942, 260.0246277
7: -142.9139709, 120.1200180, -144.7083588, 121.6381607, -264.5520630, 264.8283386
8: -172.3660583, 117.8782959, -174.5120850, 119.3504868, -291.7165527, 292.3903809
9: -129.8991852, 128.3625793, -131.5167694, 129.9706268, -259.8698120, 259.8792725

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6197227, upper bound: 264.6192334
time: 7.41 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6197184, upper bound: 264.6191631
time: 7.92 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -143.5242462, 114.4032974, -144.3596649, 115.0605316, -258.5847778, 258.7629089
1: -120.1220856, 101.4233856, -120.8221283, 102.0048447, -222.1269226, 222.2454987
2: -158.0138855, 102.6919479, -158.9320831, 103.2858582, -261.2997437, 261.6240234
3: -168.2381287, 89.6814194, -169.2086487, 90.2014694, -258.4396057, 258.8900757
4: -153.8648071, 118.0614929, -154.7666321, 118.7446289, -272.6093750, 272.8281250
5: -138.3584747, 107.3881683, -139.1623535, 108.0147781, -246.3732605, 246.5505219
6: -132.2587128, 127.8387756, -133.0277405, 128.5737915, -260.8324585, 260.8664551
7: -143.8743134, 120.9328995, -144.7083588, 121.6381607, -265.5124817, 265.6412659
8: -173.5107269, 118.6608429, -174.5120850, 119.3504868, -292.8612061, 293.1729126
9: -130.7577209, 129.2200165, -131.5167694, 129.9706268, -260.7283325, 260.7367554

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6213728, upper bound: 264.6214309
time: 7.44 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6213584, upper bound: 264.6213584
time: 7.83 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 16.41 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 16.41
Output dim: 7, lower bound: -264.6197227, upper bound: 264.6192334
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 16.41
Output dim: 7, lower bound: -264.6197184, upper bound: 264.6191631
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 16.41
Output dim: 7, lower bound: -264.6213728, upper bound: 264.6214309
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 16.41
Output dim: 7, lower bound: -264.6213584, upper bound: 264.6213584

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -142.5778351, 113.6492615, -137.8412933, 109.8958969, -252.4737244, 251.4905243
1: -119.3355713, 100.7713852, -115.3550110, 97.4181595, -216.7536926, 216.1264038
2: -156.9659576, 102.0101395, -151.7102203, 98.6460876, -255.6120453, 253.7203674
3: -167.1516571, 89.0801926, -161.5674744, 86.2000275, -253.3516846, 250.6476746
4: -152.8229523, 117.2858200, -147.7318878, 113.3872681, -266.2102051, 265.0177002
5: -137.4328613, 106.6576385, -132.8794250, 103.1506424, -240.5834808, 239.5370636
6: -131.3840332, 126.9968948, -127.0270538, 122.7826920, -254.1667023, 254.0239563
7: -142.9139709, 120.1200180, -138.1151123, 116.1337585, -259.0476990, 258.2351074
8: -172.3660583, 117.8782959, -166.6094513, 114.0110855, -286.3771362, 284.4877319
9: -129.8991852, 128.3625793, -125.5774002, 124.1473160, -254.0464783, 253.9399719

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6117296, upper bound: 264.6100183
time: 8.17 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6184969, upper bound: 264.6181015
time: 7.12 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -142.5778351, 113.6492615, -142.3042603, 113.4523087, -256.0301208, 255.9535065
1: -119.3355713, 100.7713852, -119.0911560, 100.5554810, -219.8910370, 219.8625488
2: -156.9659576, 102.0101395, -156.6529388, 101.8062897, -258.7722168, 258.6629944
3: -167.1516571, 89.0801926, -166.8479462, 88.9353714, -256.0870056, 255.9281311
4: -152.8229523, 117.2858200, -152.5530548, 117.0456924, -269.8686523, 269.8388672
5: -137.4328613, 106.6576385, -137.2094727, 106.4746170, -243.9074554, 243.8671112
6: -131.3840332, 126.9968948, -131.1432648, 126.7484055, -258.1324463, 258.1401672
7: -142.9139709, 120.1200180, -142.6210785, 119.8799820, -262.7938538, 262.7410583
8: -172.3660583, 117.8782959, -172.0416260, 117.6300964, -289.9961548, 289.9199219
9: -129.8991852, 128.3625793, -129.6566162, 128.1073914, -258.0065308, 258.0191650

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6122872, upper bound: 264.6124054
time: 7.62 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6184978, upper bound: 264.6180665
time: 7.52 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -143.5242462, 114.4032974, -137.8412933, 109.8958969, -253.4201355, 252.2445526
1: -120.1220856, 101.4233856, -115.3550110, 97.4181595, -217.5402374, 216.7783661
2: -158.0138855, 102.6919479, -151.7102203, 98.6460876, -256.6599731, 254.4021606
3: -168.2381287, 89.6814194, -161.5674744, 86.2000275, -254.4381561, 251.2488861
4: -153.8648071, 118.0614929, -147.7318878, 113.3872681, -267.2520447, 265.7933655
5: -138.3584747, 107.3881683, -132.8794250, 103.1506424, -241.5091095, 240.2675934
6: -132.2587128, 127.8387756, -127.0270538, 122.7826920, -255.0414124, 254.8658295
7: -143.8743134, 120.9328995, -138.1151123, 116.1337585, -260.0080566, 259.0480042
8: -173.5107269, 118.6608429, -166.6094513, 114.0110855, -287.5218201, 285.2702637
9: -130.7577209, 129.2200165, -125.5774002, 124.1473160, -254.9049988, 254.7973785

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6147677, upper bound: 264.6143650
time: 8.35 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6203868, upper bound: 264.6204015
time: 8.49 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -143.5242462, 114.4032974, -142.3042603, 113.4523087, -256.9765320, 256.7075500
1: -120.1220856, 101.4233856, -119.0911560, 100.5554810, -220.6775665, 220.5145264
2: -158.0138855, 102.6919479, -156.6529388, 101.8062897, -259.8201599, 259.3447876
3: -168.2381287, 89.6814194, -166.8479462, 88.9353714, -257.1734924, 256.5293579
4: -153.8648071, 118.0614929, -152.5530548, 117.0456924, -270.9104919, 270.6145325
5: -138.3584747, 107.3881683, -137.2094727, 106.4746170, -244.8330841, 244.5976410
6: -132.2587128, 127.8387756, -131.1432648, 126.7484055, -259.0071106, 258.9820557
7: -143.8743134, 120.9328995, -142.6210785, 119.8799820, -263.7542725, 263.5539551
8: -173.5107269, 118.6608429, -172.0416260, 117.6300964, -291.1408081, 290.7024536
9: -130.7577209, 129.2200165, -129.6566162, 128.1073914, -258.8650513, 258.8766174

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6147324, upper bound: 264.6143360
time: 8.66 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6203678, upper bound: 264.6203678
time: 8.64 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 18.42 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 18.42
Output dim: 7, lower bound: -264.6117296, upper bound: 264.6100183
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.42
Output dim: 7, lower bound: -264.6184969, upper bound: 264.6181015
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 18.42
Output dim: 7, lower bound: -264.6122872, upper bound: 264.6124054
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 18.42
Output dim: 7, lower bound: -264.6184978, upper bound: 264.6180665
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 18.42
Output dim: 7, lower bound: -264.6147677, upper bound: 264.6143650
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.42
Output dim: 7, lower bound: -264.6203868, upper bound: 264.6204015
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 18.42
Output dim: 7, lower bound: -264.6147324, upper bound: 264.6143360
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 18.42
Output dim: 7, lower bound: -264.6203678, upper bound: 264.6203678

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -140.9888916, 112.3881683, -137.8412933, 109.8958969, -250.8847809, 250.2294312
1: -118.0026245, 99.6304703, -115.3550110, 97.4181595, -215.4207611, 214.9854736
2: -155.2196198, 100.8599701, -151.7102203, 98.6460876, -253.8657074, 252.5701752
3: -165.2615356, 88.1028061, -161.5674744, 86.2000275, -251.4615631, 249.6702728
4: -151.1075439, 115.9587555, -147.7318878, 113.3872681, -264.4948120, 263.6906433
5: -135.9048004, 105.4523926, -132.8794250, 103.1506424, -239.0554352, 238.3318176
6: -129.9068451, 125.5915451, -127.0270538, 122.7826920, -252.6895294, 252.6185913
7: -141.2996063, 118.7788544, -138.1151123, 116.1337585, -257.4333496, 256.8939209
8: -170.4513702, 116.5603333, -166.6094513, 114.0110855, -284.4624634, 283.1697083
9: -128.4313660, 126.9224777, -125.5774002, 124.1473160, -252.5786743, 252.4998322

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5890576, upper bound: 264.5888762
time: 7.54 seconds

## Relational analysis of IS_A1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6048552, upper bound: 264.6043826
time: 7.77 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5906560, upper bound: 264.5892129
time: 6.95 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -142.5778351, 113.6492615, -139.6922607, 111.3638840, -253.9417114, 253.3414764
1: -119.3355713, 100.7713852, -116.9220734, 98.6349640, -217.9705200, 217.6934509
2: -156.9659576, 102.0101395, -153.7808075, 99.8964310, -256.8623657, 255.7909546
3: -167.1516571, 89.0801926, -163.7015533, 87.3432388, -254.4948730, 252.7817383
4: -152.8229523, 117.2858200, -149.6882019, 114.8231201, -267.6460571, 266.9739685
5: -137.4328613, 106.6576385, -134.6964874, 104.4439697, -241.8768158, 241.3541260
6: -131.3840332, 126.9968948, -128.6870728, 124.4302673, -255.8142853, 255.6839600
7: -142.9139709, 120.1200180, -139.9255219, 117.6709061, -260.5848083, 260.0455322
8: -172.3660583, 117.8782959, -168.9018097, 115.4284821, -287.7945557, 286.7800903
9: -129.8991852, 128.3625793, -127.1981049, 125.6977005, -255.5968781, 255.5606842

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5944743, upper bound: 264.5933876
time: 7.63 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6112049, upper bound: 264.6113398
time: 8.34 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -142.5778351, 113.6492615, -140.7164917, 112.1922913, -254.7701263, 254.3657227
1: -119.3355713, 100.7713852, -117.7596741, 99.4155731, -218.7511139, 218.5310669
2: -156.9659576, 102.0101395, -154.9091492, 100.6572723, -257.6231995, 256.9192505
3: -167.1516571, 89.0801926, -164.9582825, 87.9594955, -255.1111298, 254.0384674
4: -152.8229523, 117.2858200, -150.8396149, 115.7194519, -268.5423584, 268.1254272
5: -137.4328613, 106.6576385, -135.6823883, 105.2698517, -242.7026978, 242.3400269
6: -131.3840332, 126.9968948, -129.6677246, 125.3447037, -256.7286987, 256.6646118
7: -142.9139709, 120.1200180, -141.0080261, 118.5400696, -261.4540100, 261.1280518
8: -172.3660583, 117.8782959, -170.1293640, 116.3136597, -288.6797180, 288.0076599
9: -129.8991852, 128.3625793, -128.1900787, 126.6688309, -256.5679932, 256.5525818

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049353, upper bound: 264.6044074
time: 7.64 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5907564, upper bound: 264.5892516
time: 7.19 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -140.9479370, 112.3419266, -137.8412933, 109.8958969, -250.8438263, 250.1832275
1: -117.9801941, 99.5271912, -115.3550110, 97.4181595, -215.3983307, 214.8821869
2: -155.1810455, 100.8065796, -151.7102203, 98.6460876, -253.8271332, 252.5167847
3: -165.1324463, 88.1090622, -161.5674744, 86.2000275, -251.3324738, 249.6765442
4: -151.0398712, 115.8673935, -147.7318878, 113.3872681, -264.4271240, 263.5992737
5: -135.8784790, 105.3837357, -132.8794250, 103.1506424, -239.0291138, 238.2631531
6: -129.8352509, 125.5515594, -127.0270538, 122.7826920, -252.6179504, 252.5786133
7: -141.2142334, 118.7529831, -138.1151123, 116.1337585, -257.3479919, 256.8681030
8: -170.4131317, 116.4869537, -166.6094513, 114.0110855, -284.4242249, 283.0963745
9: -128.3311615, 126.8410873, -125.5774002, 124.1473160, -252.4784698, 252.4184570

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6015154, upper bound: 264.6010123
time: 9.13 seconds

## Relational analysis of IS_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5900717, upper bound: 264.5898583
time: 7.10 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -141.9284210, 113.1368561, -137.8412933, 109.8958969, -251.8243103, 250.9781342
1: -118.7838364, 100.2775116, -115.3550110, 97.4181595, -216.2019958, 215.6325226
2: -156.2608643, 101.5368881, -151.7102203, 98.6460876, -254.9069519, 253.2471008
3: -166.3389130, 88.7001801, -161.5674744, 86.2000275, -252.5389404, 250.2676392
4: -152.1423950, 116.7285690, -147.7318878, 113.3872681, -265.5296631, 264.4604492
5: -136.8237305, 106.1772156, -132.8794250, 103.1506424, -239.9743652, 239.0566406
6: -130.7755127, 126.4277115, -127.0270538, 122.7826920, -253.5581970, 253.4547729
7: -142.2527618, 119.5860519, -138.1151123, 116.1337585, -258.3865356, 257.7011414
8: -171.5885162, 117.3375320, -166.6094513, 114.0110855, -285.5995789, 283.9469910
9: -129.2837830, 127.7739182, -125.5774002, 124.1473160, -253.4310913, 253.3513031

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6070420, upper bound: 264.6070180
time: 9.08 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5980522, upper bound: 264.5980999
time: 6.32 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -140.9479370, 112.3419266, -142.3042603, 113.4523087, -254.4002228, 254.6461792
1: -117.9801941, 99.5271912, -119.0911560, 100.5554810, -218.5356750, 218.6183472
2: -155.1810455, 100.8065796, -156.6529388, 101.8062897, -256.9873047, 257.4594421
3: -165.1324463, 88.1090622, -166.8479462, 88.9353714, -254.0678101, 254.9570007
4: -151.0398712, 115.8673935, -152.5530548, 117.0456924, -268.0855713, 268.4204407
5: -135.8784790, 105.3837357, -137.2094727, 106.4746170, -242.3530884, 242.5932007
6: -129.8352509, 125.5515594, -131.1432648, 126.7484055, -256.5836487, 256.6948242
7: -141.2142334, 118.7529831, -142.6210785, 119.8799820, -261.0941467, 261.3740540
8: -170.4131317, 116.4869537, -172.0416260, 117.6300964, -288.0432129, 288.5285645
9: -128.3311615, 126.8410873, -129.6566162, 128.1073914, -256.4384460, 256.4976807

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5987696, upper bound: 264.5985224
time: 8.49 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5901599, upper bound: 264.5898920
time: 8.00 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -141.9284210, 113.1368561, -142.3042603, 113.4523087, -255.3806915, 255.4411011
1: -118.7838364, 100.2775116, -119.0911560, 100.5554810, -219.3393250, 219.3686676
2: -156.2608643, 101.5368881, -156.6529388, 101.8062897, -258.0671387, 258.1897278
3: -166.3389130, 88.7001801, -166.8479462, 88.9353714, -255.2742615, 255.5481262
4: -152.1423950, 116.7285690, -152.5530548, 117.0456924, -269.1880798, 269.2816162
5: -136.8237305, 106.1772156, -137.2094727, 106.4746170, -243.2983398, 243.3866882
6: -130.7755127, 126.4277115, -131.1432648, 126.7484055, -257.5239258, 257.5709534
7: -142.2527618, 119.5860519, -142.6210785, 119.8799820, -262.1326904, 262.2071228
8: -171.5885162, 117.3375320, -172.0416260, 117.6300964, -289.2185669, 289.3791504
9: -129.2837830, 127.7739182, -129.6566162, 128.1073914, -257.3911743, 257.4305420

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6070367, upper bound: 264.6070917
time: 8.49 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5981466, upper bound: 264.5981466
time: 7.59 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 17.27 seconds
IS_A1_B1_A2_A1, status: Status.VERIFIED, split count: 4, time: 17.27
Output dim: 7, lower bound: -264.6048552, upper bound: 264.6043826
IS_A1_B1_A2_A2, status: Status.VERIFIED, split count: 4, time: 17.27
Output dim: 7, lower bound: -264.5906560, upper bound: 264.5892129
IS_A1_B2_B1_A1, status: Status.VERIFIED, split count: 4, time: 17.27
Output dim: 7, lower bound: -264.5944743, upper bound: 264.5933876
IS_A1_B2_B1_A2, status: Status.VERIFIED, split count: 4, time: 17.27
Output dim: 7, lower bound: -264.6112049, upper bound: 264.6113398
IS_A1_B2_B2_A1, status: Status.VERIFIED, split count: 4, time: 17.27
Output dim: 7, lower bound: -264.6049353, upper bound: 264.6044074
IS_A1_B2_B2_A2, status: Status.VERIFIED, split count: 4, time: 17.27
Output dim: 7, lower bound: -264.5907564, upper bound: 264.5892516
IS_A2_B1_A1_A1, status: Status.VERIFIED, split count: 4, time: 17.27
Output dim: 7, lower bound: -264.6015154, upper bound: 264.6010123
IS_A2_B1_A1_A2, status: Status.VERIFIED, split count: 4, time: 17.27
Output dim: 7, lower bound: -264.5900717, upper bound: 264.5898583
IS_A2_B1_A2_A1, status: Status.VERIFIED, split count: 4, time: 17.27
Output dim: 7, lower bound: -264.6070420, upper bound: 264.6070180
IS_A2_B1_A2_A2, status: Status.VERIFIED, split count: 4, time: 17.27
Output dim: 7, lower bound: -264.5980522, upper bound: 264.5980999
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 17.27
Output dim: 7, lower bound: -264.5987696, upper bound: 264.5985224
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 17.27
Output dim: 7, lower bound: -264.5901599, upper bound: 264.5898920
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 17.27
Output dim: 7, lower bound: -264.6070367, upper bound: 264.6070917
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 17.27
Output dim: 7, lower bound: -264.5981466, upper bound: 264.5981466
Binary search (step 2): status=Status.VERIFIED, k_low=10, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=266.34649658203125
rel_dist={7: [-264.6227371509656, 264.6227371509656]}

## Binary search (step 3) starts
Candidate k: 12, corresponding eps: 0.0468750


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6217181, upper bound: 264.6214771
time: 6.70 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6227448, upper bound: 264.6227448
time: 6.37 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 13.19 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 13.19
Output dim: 7, lower bound: -264.6217181, upper bound: 264.6214771
IS_A2, status: Status.UNKNOWN, split count: 1, time: 13.19
Output dim: 7, lower bound: -264.6227448, upper bound: 264.6227448

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -142.5778351, 113.6492615, -144.3596649, 115.0605316, -257.6383362, 258.0088806
1: -119.3355713, 100.7713852, -120.8221283, 102.0048447, -221.3403931, 221.5935059
2: -156.9659576, 102.0101395, -158.9320831, 103.2858582, -260.2518311, 260.9421997
3: -167.1516571, 89.0801926, -169.2086487, 90.2014694, -257.3531189, 258.2888489
4: -152.8229523, 117.2858200, -154.7666321, 118.7446289, -271.5675354, 272.0524292
5: -137.4328613, 106.6576385, -139.1623535, 108.0147781, -245.4476318, 245.8199921
6: -131.3840332, 126.9968948, -133.0277405, 128.5737915, -259.9577942, 260.0246277
7: -142.9139709, 120.1200180, -144.7083588, 121.6381607, -264.5520630, 264.8283386
8: -172.3660583, 117.8782959, -174.5120850, 119.3504868, -291.7165527, 292.3903809
9: -129.8991852, 128.3625793, -131.5167694, 129.9706268, -259.8698120, 259.8792725

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6198246, upper bound: 264.6193160
time: 6.30 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6198086, upper bound: 264.6192408
time: 7.07 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -143.5242462, 114.4032974, -144.3596649, 115.0605316, -258.5847778, 258.7629089
1: -120.1220856, 101.4233856, -120.8221283, 102.0048447, -222.1269226, 222.2454987
2: -158.0138855, 102.6919479, -158.9320831, 103.2858582, -261.2997437, 261.6240234
3: -168.2381287, 89.6814194, -169.2086487, 90.2014694, -258.4396057, 258.8900757
4: -153.8648071, 118.0614929, -154.7666321, 118.7446289, -272.6093750, 272.8281250
5: -138.3584747, 107.3881683, -139.1623535, 108.0147781, -246.3732605, 246.5505219
6: -132.2587128, 127.8387756, -133.0277405, 128.5737915, -260.8324585, 260.8664551
7: -143.8743134, 120.9328995, -144.7083588, 121.6381607, -265.5124817, 265.6412659
8: -173.5107269, 118.6608429, -174.5120850, 119.3504868, -292.8612061, 293.1729126
9: -130.7577209, 129.2200165, -131.5167694, 129.9706268, -260.7283325, 260.7367554

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6213876, upper bound: 264.6214474
time: 6.15 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6213695, upper bound: 264.6213695
time: 6.69 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 14.01 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 14.01
Output dim: 7, lower bound: -264.6198246, upper bound: 264.6193160
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 14.01
Output dim: 7, lower bound: -264.6198086, upper bound: 264.6192408
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 14.01
Output dim: 7, lower bound: -264.6213876, upper bound: 264.6214474
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 14.01
Output dim: 7, lower bound: -264.6213695, upper bound: 264.6213695

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -142.5778351, 113.6492615, -137.8412933, 109.8958969, -252.4737244, 251.4905243
1: -119.3355713, 100.7713852, -115.3550110, 97.4181595, -216.7536926, 216.1264038
2: -156.9659576, 102.0101395, -151.7102203, 98.6460876, -255.6120453, 253.7203674
3: -167.1516571, 89.0801926, -161.5674744, 86.2000275, -253.3516846, 250.6476746
4: -152.8229523, 117.2858200, -147.7318878, 113.3872681, -266.2102051, 265.0177002
5: -137.4328613, 106.6576385, -132.8794250, 103.1506424, -240.5834808, 239.5370636
6: -131.3840332, 126.9968948, -127.0270538, 122.7826920, -254.1667023, 254.0239563
7: -142.9139709, 120.1200180, -138.1151123, 116.1337585, -259.0476990, 258.2351074
8: -172.3660583, 117.8782959, -166.6094513, 114.0110855, -286.3771362, 284.4877319
9: -129.8991852, 128.3625793, -125.5774002, 124.1473160, -254.0464783, 253.9399719

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6121552, upper bound: 264.6104215
time: 6.47 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6186068, upper bound: 264.6181867
time: 6.69 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -142.5778351, 113.6492615, -142.3042603, 113.4523087, -256.0301208, 255.9535065
1: -119.3355713, 100.7713852, -119.0911560, 100.5554810, -219.8910370, 219.8625488
2: -156.9659576, 102.0101395, -156.6529388, 101.8062897, -258.7722168, 258.6629944
3: -167.1516571, 89.0801926, -166.8479462, 88.9353714, -256.0870056, 255.9281311
4: -152.8229523, 117.2858200, -152.5530548, 117.0456924, -269.8686523, 269.8388672
5: -137.4328613, 106.6576385, -137.2094727, 106.4746170, -243.9074554, 243.8671112
6: -131.3840332, 126.9968948, -131.1432648, 126.7484055, -258.1324463, 258.1401672
7: -142.9139709, 120.1200180, -142.6210785, 119.8799820, -262.7938538, 262.7410583
8: -172.3660583, 117.8782959, -172.0416260, 117.6300964, -289.9961548, 289.9199219
9: -129.8991852, 128.3625793, -129.6566162, 128.1073914, -258.0065308, 258.0191650

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6126286, upper bound: 264.6126901
time: 6.81 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6185943, upper bound: 264.6181474
time: 6.39 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -143.5242462, 114.4032974, -137.8412933, 109.8958969, -253.4201355, 252.2445526
1: -120.1220856, 101.4233856, -115.3550110, 97.4181595, -217.5402374, 216.7783661
2: -158.0138855, 102.6919479, -151.7102203, 98.6460876, -256.6599731, 254.4021606
3: -168.2381287, 89.6814194, -161.5674744, 86.2000275, -254.4381561, 251.2488861
4: -153.8648071, 118.0614929, -147.7318878, 113.3872681, -267.2520447, 265.7933655
5: -138.3584747, 107.3881683, -132.8794250, 103.1506424, -241.5091095, 240.2675934
6: -132.2587128, 127.8387756, -127.0270538, 122.7826920, -255.0414124, 254.8658295
7: -143.8743134, 120.9328995, -138.1151123, 116.1337585, -260.0080566, 259.0480042
8: -173.5107269, 118.6608429, -166.6094513, 114.0110855, -287.5218201, 285.2702637
9: -130.7577209, 129.2200165, -125.5774002, 124.1473160, -254.9049988, 254.7973785

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6149869, upper bound: 264.6145989
time: 6.47 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6204006, upper bound: 264.6204152
time: 6.57 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -143.5242462, 114.4032974, -142.3042603, 113.4523087, -256.9765320, 256.7075500
1: -120.1220856, 101.4233856, -119.0911560, 100.5554810, -220.6775665, 220.5145264
2: -158.0138855, 102.6919479, -156.6529388, 101.8062897, -259.8201599, 259.3447876
3: -168.2381287, 89.6814194, -166.8479462, 88.9353714, -257.1734924, 256.5293579
4: -153.8648071, 118.0614929, -152.5530548, 117.0456924, -270.9104919, 270.6145325
5: -138.3584747, 107.3881683, -137.2094727, 106.4746170, -244.8330841, 244.5976410
6: -132.2587128, 127.8387756, -131.1432648, 126.7484055, -259.0071106, 258.9820557
7: -143.8743134, 120.9328995, -142.6210785, 119.8799820, -263.7542725, 263.5539551
8: -173.5107269, 118.6608429, -172.0416260, 117.6300964, -291.1408081, 290.7024536
9: -130.7577209, 129.2200165, -129.6566162, 128.1073914, -258.8650513, 258.8766174

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6149572, upper bound: 264.6145607
time: 6.69 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6203789, upper bound: 264.6203789
time: 7.15 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 15.04 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 15.04
Output dim: 7, lower bound: -264.6121552, upper bound: 264.6104215
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 15.04
Output dim: 7, lower bound: -264.6186068, upper bound: 264.6181867
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 15.04
Output dim: 7, lower bound: -264.6126286, upper bound: 264.6126901
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 15.04
Output dim: 7, lower bound: -264.6185943, upper bound: 264.6181474
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 15.04
Output dim: 7, lower bound: -264.6149869, upper bound: 264.6145989
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 15.04
Output dim: 7, lower bound: -264.6204006, upper bound: 264.6204152
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 15.04
Output dim: 7, lower bound: -264.6149572, upper bound: 264.6145607
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 15.04
Output dim: 7, lower bound: -264.6203789, upper bound: 264.6203789

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -140.9888916, 112.3881683, -137.8412933, 109.8958969, -250.8847809, 250.2294312
1: -118.0026245, 99.6304703, -115.3550110, 97.4181595, -215.4207611, 214.9854736
2: -155.2196198, 100.8599701, -151.7102203, 98.6460876, -253.8657074, 252.5701752
3: -165.2615356, 88.1028061, -161.5674744, 86.2000275, -251.4615631, 249.6702728
4: -151.1075439, 115.9587555, -147.7318878, 113.3872681, -264.4948120, 263.6906433
5: -135.9048004, 105.4523926, -132.8794250, 103.1506424, -239.0554352, 238.3318176
6: -129.9068451, 125.5915451, -127.0270538, 122.7826920, -252.6895294, 252.6185913
7: -141.2996063, 118.7788544, -138.1151123, 116.1337585, -257.4333496, 256.8939209
8: -170.4513702, 116.5603333, -166.6094513, 114.0110855, -284.4624634, 283.1697083
9: -128.4313660, 126.9224777, -125.5774002, 124.1473160, -252.5786743, 252.4998322

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6022279, upper bound: 264.6006162
time: 6.75 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6175385, upper bound: 264.6170697
time: 6.19 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -142.5778351, 113.6492615, -139.6922607, 111.3638840, -253.9417114, 253.3414764
1: -119.3355713, 100.7713852, -116.9220734, 98.6349640, -217.9705200, 217.6934509
2: -156.9659576, 102.0101395, -153.7808075, 99.8964310, -256.8623657, 255.7909546
3: -167.1516571, 89.0801926, -163.7015533, 87.3432388, -254.4948730, 252.7817383
4: -152.8229523, 117.2858200, -149.6882019, 114.8231201, -267.6460571, 266.9739685
5: -137.4328613, 106.6576385, -134.6964874, 104.4439697, -241.8768158, 241.3541260
6: -131.3840332, 126.9968948, -128.6870728, 124.4302673, -255.8142853, 255.6839600
7: -142.9139709, 120.1200180, -139.9255219, 117.6709061, -260.5848083, 260.0455322
8: -172.3660583, 117.8782959, -168.9018097, 115.4284821, -287.7945557, 286.7800903
9: -129.8991852, 128.3625793, -127.1981049, 125.6977005, -255.5968781, 255.5606842

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5959516, upper bound: 264.5947312
time: 7.22 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6115454, upper bound: 264.6116377
time: 6.76 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -142.5778351, 113.6492615, -140.7164917, 112.1922913, -254.7701263, 254.3657227
1: -119.3355713, 100.7713852, -117.7596741, 99.4155731, -218.7511139, 218.5310669
2: -156.9659576, 102.0101395, -154.9091492, 100.6572723, -257.6231995, 256.9192505
3: -167.1516571, 89.0801926, -164.9582825, 87.9594955, -255.1111298, 254.0384674
4: -152.8229523, 117.2858200, -150.8396149, 115.7194519, -268.5423584, 268.1254272
5: -137.4328613, 106.6576385, -135.6823883, 105.2698517, -242.7026978, 242.3400269
6: -131.3840332, 126.9968948, -129.6677246, 125.3447037, -256.7286987, 256.6646118
7: -142.9139709, 120.1200180, -141.0080261, 118.5400696, -261.4540100, 261.1280518
8: -172.3660583, 117.8782959, -170.1293640, 116.3136597, -288.6797180, 288.0076599
9: -129.8991852, 128.3625793, -128.1900787, 126.6688309, -256.5679932, 256.5525818

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6056821, upper bound: 264.6052054
time: 8.67 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5910933, upper bound: 264.5896177
time: 7.05 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -140.9479370, 112.3419266, -137.8412933, 109.8958969, -250.8438263, 250.1832275
1: -117.9801941, 99.5271912, -115.3550110, 97.4181595, -215.3983307, 214.8821869
2: -155.1810455, 100.8065796, -151.7102203, 98.6460876, -253.8271332, 252.5167847
3: -165.1324463, 88.1090622, -161.5674744, 86.2000275, -251.3324738, 249.6765442
4: -151.0398712, 115.8673935, -147.7318878, 113.3872681, -264.4271240, 263.5992737
5: -135.8784790, 105.3837357, -132.8794250, 103.1506424, -239.0291138, 238.2631531
6: -129.8352509, 125.5515594, -127.0270538, 122.7826920, -252.6179504, 252.5786133
7: -141.2142334, 118.7529831, -138.1151123, 116.1337585, -257.3479919, 256.8681030
8: -170.4131317, 116.4869537, -166.6094513, 114.0110855, -284.4242249, 283.0963745
9: -128.3311615, 126.8410873, -125.5774002, 124.1473160, -252.4784698, 252.4184570

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6023357, upper bound: 264.6018658
time: 6.61 seconds

## Relational analysis of IS_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5904408, upper bound: 264.5902532
time: 7.08 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -141.9284210, 113.1368561, -137.8412933, 109.8958969, -251.8243103, 250.9781342
1: -118.7838364, 100.2775116, -115.3550110, 97.4181595, -216.2019958, 215.6325226
2: -156.2608643, 101.5368881, -151.7102203, 98.6460876, -254.9069519, 253.2471008
3: -166.3389130, 88.7001801, -161.5674744, 86.2000275, -252.5389404, 250.2676392
4: -152.1423950, 116.7285690, -147.7318878, 113.3872681, -265.5296631, 264.4604492
5: -136.8237305, 106.1772156, -132.8794250, 103.1506424, -239.9743652, 239.0566406
6: -130.7755127, 126.4277115, -127.0270538, 122.7826920, -253.5581970, 253.4547729
7: -142.2527618, 119.5860519, -138.1151123, 116.1337585, -258.3865356, 257.7011414
8: -171.5885162, 117.3375320, -166.6094513, 114.0110855, -285.5995789, 283.9469910
9: -129.2837830, 127.7739182, -125.5774002, 124.1473160, -253.4310913, 253.3513031

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6102475, upper bound: 264.6089306
time: 6.82 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6193413, upper bound: 264.6193305
time: 6.81 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -140.9479370, 112.3419266, -142.3042603, 113.4523087, -254.4002228, 254.6461792
1: -117.9801941, 99.5271912, -119.0911560, 100.5554810, -218.5356750, 218.6183472
2: -155.1810455, 100.8065796, -156.6529388, 101.8062897, -256.9873047, 257.4594421
3: -165.1324463, 88.1090622, -166.8479462, 88.9353714, -254.0678101, 254.9570007
4: -151.0398712, 115.8673935, -152.5530548, 117.0456924, -268.0855713, 268.4204407
5: -135.8784790, 105.3837357, -137.2094727, 106.4746170, -242.3530884, 242.5932007
6: -129.8352509, 125.5515594, -131.1432648, 126.7484055, -256.5836487, 256.6948242
7: -141.2142334, 118.7529831, -142.6210785, 119.8799820, -261.0941467, 261.3740540
8: -170.4131317, 116.4869537, -172.0416260, 117.6300964, -288.0432129, 288.5285645
9: -128.3311615, 126.8410873, -129.6566162, 128.1073914, -256.4384460, 256.4976807

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5997666, upper bound: 264.5995198
time: 7.11 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5905296, upper bound: 264.5902901
time: 6.88 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -141.9284210, 113.1368561, -142.3042603, 113.4523087, -255.3806915, 255.4411011
1: -118.7838364, 100.2775116, -119.0911560, 100.5554810, -219.3393250, 219.3686676
2: -156.2608643, 101.5368881, -156.6529388, 101.8062897, -258.0671387, 258.1897278
3: -166.3389130, 88.7001801, -166.8479462, 88.9353714, -255.2742615, 255.5481262
4: -152.1423950, 116.7285690, -152.5530548, 117.0456924, -269.1880798, 269.2816162
5: -136.8237305, 106.1772156, -137.2094727, 106.4746170, -243.2983398, 243.3866882
6: -130.7755127, 126.4277115, -131.1432648, 126.7484055, -257.5239258, 257.5709534
7: -142.2527618, 119.5860519, -142.6210785, 119.8799820, -262.1326904, 262.2071228
8: -171.5885162, 117.3375320, -172.0416260, 117.6300964, -289.2185669, 289.3791504
9: -129.2837830, 127.7739182, -129.6566162, 128.1073914, -257.3911743, 257.4305420

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6076833, upper bound: 264.6076814
time: 8.56 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5981531, upper bound: 264.5981531
time: 7.24 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 16.99 seconds
IS_A1_B1_A2_A1, status: Status.VERIFIED, split count: 4, time: 16.99
Output dim: 7, lower bound: -264.6022279, upper bound: 264.6006162
IS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 16.99
Output dim: 7, lower bound: -264.6175385, upper bound: 264.6170697
IS_A1_B2_B1_A1, status: Status.VERIFIED, split count: 4, time: 16.99
Output dim: 7, lower bound: -264.5959516, upper bound: 264.5947312
IS_A1_B2_B1_A2, status: Status.VERIFIED, split count: 4, time: 16.99
Output dim: 7, lower bound: -264.6115454, upper bound: 264.6116377
IS_A1_B2_B2_A1, status: Status.VERIFIED, split count: 4, time: 16.99
Output dim: 7, lower bound: -264.6056821, upper bound: 264.6052054
IS_A1_B2_B2_A2, status: Status.VERIFIED, split count: 4, time: 16.99
Output dim: 7, lower bound: -264.5910933, upper bound: 264.5896177
IS_A2_B1_A1_A1, status: Status.VERIFIED, split count: 4, time: 16.99
Output dim: 7, lower bound: -264.6023357, upper bound: 264.6018658
IS_A2_B1_A1_A2, status: Status.VERIFIED, split count: 4, time: 16.99
Output dim: 7, lower bound: -264.5904408, upper bound: 264.5902532
IS_A2_B1_A2_A1, status: Status.VERIFIED, split count: 4, time: 16.99
Output dim: 7, lower bound: -264.6102475, upper bound: 264.6089306
IS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 16.99
Output dim: 7, lower bound: -264.6193413, upper bound: 264.6193305
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 16.99
Output dim: 7, lower bound: -264.5997666, upper bound: 264.5995198
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 16.99
Output dim: 7, lower bound: -264.5905296, upper bound: 264.5902901
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 16.99
Output dim: 7, lower bound: -264.6076833, upper bound: 264.6076814
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 16.99
Output dim: 7, lower bound: -264.5981531, upper bound: 264.5981531

## BFS IS instance: IS_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -138.6731415, 110.5490341, -137.8412933, 109.8958969, -248.5690308, 248.3903198
1: -116.0508270, 97.9984207, -115.3550110, 97.4181595, -213.4689636, 213.3534241
2: -152.6731415, 99.2048492, -151.7102203, 98.6460876, -251.3192291, 250.9150543
3: -162.5656281, 86.6663742, -161.5674744, 86.2000275, -248.7656555, 248.2338409
4: -148.6316528, 114.0499268, -147.7318878, 113.3872681, -262.0189209, 261.7817688
5: -133.6744080, 103.7182083, -132.8794250, 103.1506424, -236.8250427, 236.5976257
6: -127.7793732, 123.5387573, -127.0270538, 122.7826920, -250.5620728, 250.5658112
7: -138.9894409, 116.8377838, -138.1151123, 116.1337585, -255.1231995, 254.9528961
8: -167.6577301, 114.6377945, -166.6094513, 114.0110855, -281.6687927, 281.2472229
9: -126.3273010, 124.8320465, -125.5774002, 124.1473160, -250.4746094, 250.4094238

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5993555, upper bound: 264.5978688
time: 6.92 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5898661, upper bound: 264.5884861
time: 6.05 seconds

## BFS IS instance: IS_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -139.6062317, 111.2921982, -137.8412933, 109.8958969, -249.5021210, 249.1334839
1: -116.8261566, 98.6403198, -115.3550110, 97.4181595, -214.2442932, 213.9953003
2: -153.7070007, 99.8767853, -151.7102203, 98.6460876, -252.3530884, 251.5870056
3: -163.6346588, 87.2593307, -161.5674744, 86.2000275, -249.8346863, 248.8267975
4: -149.6591644, 114.8141785, -147.7318878, 113.3872681, -263.0464478, 262.5460815
5: -134.5864716, 104.4378052, -132.8794250, 103.1506424, -237.7371063, 237.3172150
6: -128.6414795, 124.3686066, -127.0270538, 122.7826920, -251.4241638, 251.3956604
7: -139.9359894, 117.6386337, -138.1151123, 116.1337585, -256.0697327, 255.7537537
8: -168.7866058, 115.4098053, -166.6094513, 114.0110855, -282.7976685, 282.0191956
9: -127.1737366, 125.6776581, -125.5774002, 124.1473160, -251.3210297, 251.2550507

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A2_A2_A1

### Relational analysis result of IS_A2_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6065669, upper bound: 264.6066481
time: 8.00 seconds

## Relational analysis of IS_A2_B1_A2_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5970206, upper bound: 264.5970984
time: 7.06 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 16.21 seconds
IS_A1_B1_A2_A2_B1, status: Status.VERIFIED, split count: 5, time: 16.21
Output dim: 7, lower bound: -264.5993555, upper bound: 264.5978688
IS_A1_B1_A2_A2_B2, status: Status.VERIFIED, split count: 5, time: 16.21
Output dim: 7, lower bound: -264.5898661, upper bound: 264.5884861
IS_A2_B1_A2_A2_A1, status: Status.VERIFIED, split count: 5, time: 16.21
Output dim: 7, lower bound: -264.6065669, upper bound: 264.6066481
IS_A2_B1_A2_A2_A2, status: Status.VERIFIED, split count: 5, time: 16.21
Output dim: 7, lower bound: -264.5970206, upper bound: 264.5970984
Binary search (step 3): status=Status.VERIFIED, k_low=12, k_high=12, k_mid=12, eps_mid=0.0468750, abs_max=266.34649658203125
rel_dist={7: [-264.6227447629983, 264.6227447629983]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.046875
execution time: 765.22 seconds
