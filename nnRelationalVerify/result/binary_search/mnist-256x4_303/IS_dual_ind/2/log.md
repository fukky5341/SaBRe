## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 206.199692099
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-133.0105286, 105.4250412, -133.0105286, 105.4250412, -238.4355469, 238.4355469)
1: (-113.2106094, 94.0342941, -113.2106094, 94.0342941, -207.2449036, 207.2449036)
2: (-147.5505219, 95.8398438, -147.5505219, 95.8398438, -243.3903656, 243.3903656)
3: (-156.0251617, 82.8341141, -156.0251617, 82.8341141, -238.8592682, 238.8592682)
4: (-143.4313812, 109.8025894, -143.4313812, 109.8025894, -253.2339783, 253.2339783)
5: (-127.3003082, 99.2028809, -127.3003082, 99.2028809, -226.5031586, 226.5031586)
6: (-122.3499832, 118.8279724, -122.3499832, 118.8279724, -241.1779480, 241.1779480)
7: (-134.1060638, 112.8064041, -134.1060638, 112.8064041, -246.9124603, 246.9124603)
8: (-162.7028046, 111.3033218, -162.7028046, 111.3033218, -274.0060730, 274.0060730)
9: (-122.2542877, 119.9744720, -122.2542877, 119.9744720, -242.2287292, 242.2287292)

## BASE Result
execution time: IAR + LP analysis = 1.29 + 11.07 = 12.36 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -206.2537789, upper bound: 206.2537789


# Binary Search by BASE starts (time budget: 2687.64 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=207.24490356445312
rel_dist={1: [-206.25362701135504, 206.25362701135498]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=207.24490356445312
rel_dist={1: [-206.2534079160462, 206.25340791604617]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=207.24490356445312
rel_dist={1: [-206.25298865227455, 206.2529886525199]}

## Binary Search Result
Binary search time: 42.13 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 2645.51 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 131

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2526026, upper bound: 206.2525384
time: 8.36 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2524979, upper bound: 206.2524979
time: 7.74 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 16.24 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 16.24
Output dim: 1, lower bound: -206.2526026, upper bound: 206.2525384
IS_A2, status: Status.UNKNOWN, split count: 1, time: 16.24
Output dim: 1, lower bound: -206.2524979, upper bound: 206.2524979

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -122.3482819, 96.9952698, -132.6041260, 105.1036301, -227.4518890, 229.5993958
1: -104.3410645, 86.5778198, -112.8735886, 93.7501144, -198.0911865, 199.4513855
2: -135.7731781, 88.2661514, -147.1007843, 95.5508881, -231.3240662, 235.3669128
3: -143.5900574, 76.3189392, -155.5515442, 82.5865250, -226.1765747, 231.8704681
4: -132.0429077, 101.1205978, -142.9982910, 109.4720154, -241.5149231, 244.1188965
5: -117.0824127, 91.2991104, -126.9107895, 98.9014664, -215.9838867, 218.2098999
6: -112.6044006, 109.3904037, -121.9798965, 118.4683609, -231.0727539, 231.3703003
7: -123.3929977, 103.8461914, -133.6973114, 112.4643631, -235.8573456, 237.5435028
8: -149.9116974, 102.6137314, -162.2153778, 110.9731750, -260.8848267, 264.8291016
9: -112.5510635, 110.4543991, -121.8847504, 119.6126938, -232.1637573, 232.3391418

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2489883, upper bound: 206.2488192
time: 7.50 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2495983, upper bound: 206.2495594
time: 7.86 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -125.7769241, 99.7046509, -132.9416962, 105.3705750, -231.1474915, 232.6463318
1: -107.2242966, 88.9826508, -113.1536255, 93.9862061, -201.2105103, 202.1362305
2: -139.5509796, 90.6998215, -147.4743652, 95.7909393, -235.3419037, 238.1741486
3: -147.5962524, 78.4478607, -155.9450073, 82.7923660, -230.3886108, 234.3928680
4: -135.7267151, 103.9251862, -143.3580780, 109.7466354, -245.4733582, 247.2832642
5: -120.3754959, 93.8318710, -127.2343750, 99.1517792, -219.5272827, 221.0662384
6: -115.7748718, 112.4425888, -122.2873917, 118.7671738, -234.5420532, 234.7299805
7: -126.8394623, 106.7288284, -134.0368805, 112.7485352, -239.5879974, 240.7656860
8: -154.0409698, 105.4287491, -162.6203461, 111.2474060, -265.2883911, 268.0491028
9: -115.6958160, 113.5333557, -122.1918259, 119.9131699, -235.6089783, 235.7251892

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 131

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2524979, upper bound: 206.2524979
time: 6.81 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2524979, upper bound: 206.2524979
time: 7.27 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 15.42 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 15.42
Output dim: 1, lower bound: -206.2489883, upper bound: 206.2488192
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 15.42
Output dim: 1, lower bound: -206.2495983, upper bound: 206.2495594
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 15.42
Output dim: 1, lower bound: -206.2524979, upper bound: 206.2524979
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 15.42
Output dim: 1, lower bound: -206.2524979, upper bound: 206.2524979

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -121.4596024, 96.2935028, -119.2024078, 94.5251236, -215.9847260, 215.4959106
1: -103.5954514, 85.9529648, -101.6267624, 84.3304749, -187.9259338, 187.5797272
2: -134.7959290, 87.6291122, -132.3657990, 85.9544144, -220.7503357, 219.9949036
3: -142.5433350, 75.7659607, -139.7647247, 74.2557220, -216.7990570, 215.5306854
4: -131.0925446, 100.3924103, -128.6563568, 98.4950790, -229.5876160, 229.0487518
5: -116.2298126, 90.6339493, -114.0552139, 88.8776398, -205.1074524, 204.6891479
6: -111.7889328, 108.6047516, -109.6845551, 106.6196747, -218.4085999, 218.2892914
7: -122.5061340, 103.0966644, -120.3191605, 101.1741943, -223.6803131, 223.4158173
8: -148.8389740, 101.8802261, -146.0421295, 99.9212799, -248.7602539, 247.9223480
9: -111.7426682, 109.6566544, -109.7005920, 107.5858612, -219.3285217, 219.3572388

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2392796, upper bound: 206.2394664
time: 8.32 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2439160, upper bound: 206.2436772
time: 8.84 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -122.1250916, 96.8187408, -123.7001953, 98.0641403, -220.1892242, 220.5189362
1: -104.1538620, 86.4207687, -105.4045105, 87.4858398, -191.6396942, 191.8252563
2: -135.5279999, 88.1060333, -137.3198853, 89.1674500, -224.6954498, 225.4259186
3: -143.3273010, 76.1803207, -145.0686340, 77.0580292, -220.3853302, 221.2489471
4: -131.8040924, 100.9377518, -133.4700317, 102.1790771, -233.9831390, 234.4077759
5: -116.8683472, 91.1321411, -118.3704529, 92.2450562, -209.1134033, 209.5025635
6: -112.3993378, 109.1931534, -113.8009796, 110.5967178, -222.9960327, 222.9941254
7: -123.1702423, 103.6580734, -124.8098526, 104.9626923, -228.1329346, 228.4679108
8: -149.6422119, 102.4298325, -151.4635773, 103.6362305, -253.2784271, 253.8933868
9: -112.3481903, 110.2539597, -113.7917480, 111.6124802, -223.9606628, 224.0457001

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2489405, upper bound: 206.2489886
time: 7.58 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2489405, upper bound: 206.2495594
time: 7.62 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -125.7769241, 99.7046509, -122.3482819, 96.9952698, -222.7721863, 222.0529327
1: -107.2242966, 88.9826508, -104.3410645, 86.5778198, -193.8021240, 193.3237000
2: -139.5509796, 90.6998215, -135.7731781, 88.2661514, -227.8170929, 226.4729767
3: -147.5962524, 78.4478607, -143.5900574, 76.3189392, -223.9151917, 222.0379181
4: -135.7267151, 103.9251862, -132.0429077, 101.1205978, -236.8473206, 235.9680786
5: -120.3754959, 93.8318710, -117.0824127, 91.2991104, -211.6746063, 210.9142761
6: -115.7748718, 112.4425888, -112.6044006, 109.3904037, -225.1652527, 225.0469971
7: -126.8394623, 106.7288284, -123.3929977, 103.8461914, -230.6856537, 230.1218262
8: -154.0409698, 105.4287491, -149.9116974, 102.6137314, -256.6546936, 255.3404541
9: -115.6958160, 113.5333557, -112.5510635, 110.4543991, -226.1502075, 226.0844116

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2487915, upper bound: 206.2489213
time: 8.16 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2495358, upper bound: 206.2495371
time: 8.84 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -125.7769241, 99.7046509, -125.7769241, 99.7046509, -225.4815674, 225.4815674
1: -107.2242966, 88.9826508, -107.2242966, 88.9826508, -196.2069092, 196.2069092
2: -139.5509796, 90.6998215, -139.5509796, 90.6998215, -230.2507629, 230.2507629
3: -147.5962524, 78.4478607, -147.5962524, 78.4478607, -226.0441132, 226.0441132
4: -135.7267151, 103.9251862, -135.7267151, 103.9251862, -239.6519012, 239.6519012
5: -120.3754959, 93.8318710, -120.3754959, 93.8318710, -214.2073669, 214.2073669
6: -115.7748718, 112.4425888, -115.7748718, 112.4425888, -228.2174530, 228.2174530
7: -126.8394623, 106.7288284, -126.8394623, 106.7288284, -233.5682831, 233.5682831
8: -154.0409698, 105.4287491, -154.0409698, 105.4287491, -259.4697266, 259.4697266
9: -115.6958160, 113.5333557, -115.6958160, 113.5333557, -229.2291718, 229.2291718

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2487915, upper bound: 206.2489213
time: 8.40 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2495358, upper bound: 206.2495374
time: 8.62 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 18.36 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 18.36
Output dim: 1, lower bound: -206.2392796, upper bound: 206.2394664
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.36
Output dim: 1, lower bound: -206.2439160, upper bound: 206.2436772
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 18.36
Output dim: 1, lower bound: -206.2489405, upper bound: 206.2489886
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 18.36
Output dim: 1, lower bound: -206.2489405, upper bound: 206.2495594
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 18.36
Output dim: 1, lower bound: -206.2487915, upper bound: 206.2489213
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.36
Output dim: 1, lower bound: -206.2495358, upper bound: 206.2495371
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 18.36
Output dim: 1, lower bound: -206.2487915, upper bound: 206.2489213
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 18.36
Output dim: 1, lower bound: -206.2495358, upper bound: 206.2495374

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -108.0103455, 85.6319427, -117.6804199, 93.3190994, -201.3294373, 203.3123627
1: -92.3798447, 76.4791565, -100.3586197, 83.2589722, -175.6388245, 176.8377686
2: -119.9658890, 78.0220566, -130.6890564, 84.8666382, -204.8325195, 208.7111206
3: -126.6260605, 67.4555206, -137.9636993, 73.3163300, -199.9423828, 205.4192200
4: -116.5628357, 89.3285980, -127.0139008, 97.2423859, -213.8051910, 216.3424988
5: -103.2478638, 80.6075668, -112.5870743, 87.7429962, -190.9908600, 193.1946411
6: -99.4190292, 96.7337723, -108.2854385, 105.2776184, -204.6966400, 205.0191956
7: -108.9118576, 91.7300568, -118.7797470, 99.8884201, -208.8002777, 210.5097961
8: -132.6826019, 90.8103333, -144.2145691, 98.6690216, -231.3516235, 235.0249023
9: -99.3136063, 97.5021133, -108.2935257, 106.2110825, -205.5246887, 205.7956238

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2392796, upper bound: 206.2394664
time: 9.02 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2392796, upper bound: 206.2394664
time: 8.68 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -114.3268051, 90.6389008, -119.2024078, 94.5251236, -208.8518982, 209.8412933
1: -97.6500397, 80.9326172, -101.6267624, 84.3304749, -181.9805145, 182.5593719
2: -126.9412766, 82.5303421, -132.3657990, 85.9544144, -212.8956909, 214.8961334
3: -134.1013336, 71.3602371, -139.7647247, 74.2557220, -208.3570404, 211.1249390
4: -123.4076920, 94.5306702, -128.6563568, 98.4950790, -221.9027710, 223.1870117
5: -109.3397446, 85.3172455, -114.0552139, 88.8776398, -198.2173767, 199.3724670
6: -105.2270813, 102.3216019, -109.6845551, 106.6196747, -211.8467560, 212.0061646
7: -115.3027115, 97.0764313, -120.3191605, 101.1741943, -216.4768982, 217.3955994
8: -140.2858582, 96.0186081, -146.0421295, 99.9212799, -240.2071381, 242.0607300
9: -105.1639252, 103.2118835, -109.7005920, 107.5858612, -212.7497864, 212.9124451

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2439160, upper bound: 206.2436772
time: 7.68 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2439160, upper bound: 206.2436772
time: 7.77 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -109.1148529, 86.5476608, -123.7001953, 98.0641403, -207.1789398, 210.2478638
1: -93.2331390, 77.2766342, -105.4045105, 87.4858398, -180.7189789, 182.6811218
2: -121.2215118, 78.7858582, -137.3198853, 89.1674500, -210.3889313, 216.1057281
3: -128.0054321, 68.0912018, -145.0686340, 77.0580292, -205.0634613, 213.1598358
4: -117.8769302, 90.2837524, -133.4700317, 102.1790771, -220.0559540, 223.7537842
5: -104.3852692, 81.4001389, -118.3704529, 92.2450562, -196.6303101, 199.7705994
6: -100.4566116, 97.6931458, -113.8009796, 110.5967178, -211.0533295, 211.4941254
7: -110.1820145, 92.6948242, -124.8098526, 104.9626923, -215.1447144, 217.5046692
8: -133.9401855, 91.7017670, -151.4635773, 103.6362305, -237.5764160, 243.1653137
9: -100.5180054, 98.5779800, -113.7917480, 111.6124802, -212.1304932, 212.3697052

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 131

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2486766, upper bound: 206.2489886
time: 7.75 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2486766, upper bound: 206.2489886
time: 6.85 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -113.5742111, 90.0595093, -123.7001953, 98.0641403, -211.6383209, 213.7597046
1: -96.9846725, 80.4064331, -105.4045105, 87.4858398, -184.4705048, 185.8109436
2: -126.1373520, 81.9760895, -137.3198853, 89.1674500, -215.3047791, 219.2959747
3: -133.2654877, 70.8757324, -145.0686340, 77.0580292, -210.3235168, 215.9443512
4: -122.6568298, 93.9355316, -133.4700317, 102.1790771, -224.8358459, 227.4055634
5: -108.6680679, 84.7397003, -118.3704529, 92.2450562, -200.9131165, 203.1101379
6: -104.5434570, 101.6393051, -113.8009796, 110.5967178, -215.1401672, 215.4402618
7: -114.6364365, 96.4555588, -124.8098526, 104.9626923, -219.5991211, 221.2654114
8: -139.3198853, 95.3898849, -151.4635773, 103.6362305, -242.9561157, 246.8534393
9: -104.5781860, 102.5741577, -113.7917480, 111.6124802, -216.1906738, 216.3659058

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 131

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2486766, upper bound: 206.2495594
time: 8.35 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2486766, upper bound: 206.2495594
time: 8.02 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -112.4196854, 89.1544571, -121.4596024, 96.2935028, -208.7131958, 210.6140594
1: -96.0130920, 79.5914459, -103.5954514, 85.9529648, -181.9660645, 183.1868896
2: -124.8613510, 81.1314926, -134.7959290, 87.6291122, -212.4904633, 215.9274292
3: -131.8618164, 70.1385956, -142.5433350, 75.7659607, -207.6277466, 212.6819153
4: -121.4286575, 92.9871063, -131.0925446, 100.3924103, -221.8210602, 224.0796509
5: -107.5588989, 83.8368149, -116.2298126, 90.6339493, -198.1928406, 200.0665894
6: -103.5165024, 100.6313019, -111.7889328, 108.6047516, -212.1212463, 212.4202271
7: -113.5033035, 95.4717941, -122.5061340, 103.0966644, -216.5999756, 217.9779358
8: -137.9167480, 94.4083633, -148.8389740, 101.8802261, -239.7969666, 243.2473450
9: -103.5501328, 101.5435104, -111.7426682, 109.6566544, -213.2067871, 213.2861633

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2394664, upper bound: 206.2392796
time: 8.39 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2436772, upper bound: 206.2439160
time: 7.82 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -116.9376831, 92.7133331, -122.1250916, 96.8187408, -213.7564240, 214.8384247
1: -99.8078537, 82.7624893, -104.1538620, 86.4207687, -186.2286224, 186.9163361
2: -129.8388214, 84.3585815, -135.5279999, 88.1060333, -217.9448547, 219.8865814
3: -137.1866302, 72.9565353, -143.3273010, 76.1803207, -213.3669434, 216.2838440
4: -126.2632370, 96.6837234, -131.8040924, 100.9377518, -227.2009888, 228.4878235
5: -111.8945160, 87.2217484, -116.8683472, 91.1321411, -203.0266418, 204.0900879
6: -107.6527405, 104.6259460, -112.3993378, 109.1931534, -216.8458862, 217.0252686
7: -118.0156250, 99.2793121, -123.1702423, 103.6580734, -221.6736755, 222.4495544
8: -143.3639679, 98.1394196, -149.6422119, 102.4298325, -245.7937927, 247.7816315
9: -107.6582947, 105.5898743, -112.3481903, 110.2539597, -217.9122620, 217.9380646

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2489886, upper bound: 206.2489405
time: 6.92 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2489886, upper bound: 206.2495983
time: 8.26 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -112.4196854, 89.1544571, -124.8836212, 98.9992142, -211.4188843, 214.0380707
1: -96.0130920, 79.5914459, -106.4749069, 88.3545456, -184.3676453, 186.0663452
2: -124.8613510, 81.1314926, -138.5688171, 90.0596924, -214.9210510, 219.7003174
3: -131.8618164, 70.1385956, -146.5441589, 77.8919296, -209.7537537, 216.6827393
4: -121.4286575, 92.9871063, -134.7716827, 103.1933289, -224.6219788, 227.7587891
5: -107.5588989, 83.8368149, -119.5184631, 93.1631012, -200.7220001, 203.3552551
6: -103.5165024, 100.6313019, -114.9553680, 111.6529236, -215.1694183, 215.5866547
7: -113.5033035, 95.4717941, -125.9482269, 105.9754639, -219.4787598, 221.4200134
8: -137.9167480, 94.4083633, -152.9628143, 104.6914825, -242.6082001, 247.3711853
9: -103.5501328, 101.5435104, -114.8834839, 112.7315903, -216.2817230, 216.4270020

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2393936, upper bound: 206.2390601
time: 8.05 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2436484, upper bound: 206.2438374
time: 7.79 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -116.9376831, 92.7133331, -125.5540390, 99.5283890, -216.4660645, 218.2673645
1: -99.8078537, 82.7624893, -107.0373154, 88.8257751, -188.6336365, 189.7998047
2: -129.8388214, 84.3585815, -139.3060303, 90.5398865, -220.3787079, 223.6646118
3: -137.1866302, 72.9565353, -147.3337402, 78.3094101, -215.4960327, 220.2902832
4: -126.2632370, 96.6837234, -135.4881439, 103.7425385, -230.0057526, 232.1718750
5: -111.8945160, 87.2217484, -120.1616974, 93.6651688, -205.5596771, 207.3834076
6: -107.6527405, 104.6259460, -115.5700607, 112.2455673, -219.8983154, 220.1960144
7: -118.0156250, 99.2793121, -126.6169510, 106.5409393, -224.5565643, 225.8962708
8: -143.3639679, 98.1394196, -153.7718048, 105.2450180, -248.6089630, 251.9112244
9: -107.6582947, 105.5898743, -115.4931335, 113.3331375, -220.9914246, 221.0830078

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2489217, upper bound: 206.2487938
time: 8.93 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2489217, upper bound: 206.2495374
time: 8.80 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 19.09 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 19.09
Output dim: 1, lower bound: -206.2392796, upper bound: 206.2394664
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.09
Output dim: 1, lower bound: -206.2392796, upper bound: 206.2394664
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.09
Output dim: 1, lower bound: -206.2439160, upper bound: 206.2436772
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.09
Output dim: 1, lower bound: -206.2439160, upper bound: 206.2436772
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 19.09
Output dim: 1, lower bound: -206.2486766, upper bound: 206.2489886
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.09
Output dim: 1, lower bound: -206.2486766, upper bound: 206.2489886
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.09
Output dim: 1, lower bound: -206.2486766, upper bound: 206.2495594
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.09
Output dim: 1, lower bound: -206.2486766, upper bound: 206.2495594
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 19.09
Output dim: 1, lower bound: -206.2394664, upper bound: 206.2392796
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.09
Output dim: 1, lower bound: -206.2436772, upper bound: 206.2439160
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.09
Output dim: 1, lower bound: -206.2489886, upper bound: 206.2489405
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.09
Output dim: 1, lower bound: -206.2489886, upper bound: 206.2495983
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 19.09
Output dim: 1, lower bound: -206.2393936, upper bound: 206.2390601
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.09
Output dim: 1, lower bound: -206.2436484, upper bound: 206.2438374
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.09
Output dim: 1, lower bound: -206.2489217, upper bound: 206.2487938
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.09
Output dim: 1, lower bound: -206.2489217, upper bound: 206.2495374

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -108.0103455, 85.6319427, -107.6138840, 85.3565292, -193.3668823, 193.2458191
1: -92.3798447, 76.4791565, -91.9811935, 76.2188110, -168.5986633, 168.4603577
2: -119.9658890, 78.0220566, -119.5654907, 77.7121429, -197.6780396, 197.5875549
3: -126.6260605, 67.4555206, -126.2276535, 67.1623688, -193.7884216, 193.6831665
4: -116.5628357, 89.3285980, -116.2550812, 89.0485382, -205.6113586, 205.5836639
5: -103.2478638, 80.6075668, -102.9351349, 80.2800140, -183.5278778, 183.5426941
6: -99.4190292, 96.7337723, -99.0745621, 96.3687286, -195.7877350, 195.8083191
7: -108.9118576, 91.7300568, -108.6637344, 91.4244537, -200.3363037, 200.3937836
8: -132.6826019, 90.8103333, -132.1363678, 90.4651489, -223.1477509, 222.9467010
9: -99.3136063, 97.5021133, -99.1290054, 97.2201157, -196.5337067, 196.6311035

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2392756, upper bound: 206.2394651
time: 8.50 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2392756, upper bound: 206.2394664
time: 9.06 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -108.0103455, 85.6319427, -110.9160309, 87.9613571, -195.9716949, 196.5479736
1: -92.3798447, 76.4791565, -94.7590866, 78.5317154, -170.9115601, 171.2382507
2: -119.9658890, 78.0220566, -123.2023392, 80.0558472, -200.0217285, 201.2243958
3: -126.6260605, 67.4555206, -130.0810699, 69.2083054, -195.8343658, 197.5365906
4: -116.5628357, 89.3285980, -119.8040619, 91.7498245, -208.3126526, 209.1326294
5: -103.2478638, 80.6075668, -106.1065445, 82.7149353, -185.9627991, 186.7141113
6: -99.4190292, 96.7337723, -102.1324921, 99.3046722, -198.7236938, 198.8662567
7: -108.9118576, 91.7300568, -111.9823608, 94.1992874, -203.1111145, 203.7124176
8: -132.6826019, 90.8103333, -136.1098633, 93.1698532, -225.8524475, 226.9201965
9: -99.3136063, 97.5021133, -102.1591644, 100.1834717, -199.4970398, 199.6612549

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2392756, upper bound: 206.2394651
time: 9.43 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2392756, upper bound: 206.2394664
time: 8.78 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -114.3268051, 90.6389008, -109.1148529, 86.5476608, -200.8744659, 199.7537079
1: -97.6500397, 80.9326172, -93.2331390, 77.2766342, -174.9266663, 174.1657562
2: -126.9412766, 82.5303421, -121.2215118, 78.7858582, -205.7271271, 203.7518158
3: -134.1013336, 71.3602371, -128.0054321, 68.0912018, -202.1925354, 199.3656616
4: -123.4076920, 94.5306702, -117.8769302, 90.2837524, -213.6914368, 212.4075623
5: -109.3397446, 85.3172455, -104.3852692, 81.4001389, -190.7398834, 189.7025146
6: -105.2270813, 102.3216019, -100.4566116, 97.6931458, -202.9202271, 202.7782135
7: -115.3027115, 97.0764313, -110.1820145, 92.6948242, -207.9975281, 207.2584534
8: -140.2858582, 96.0186081, -133.9401855, 91.7017670, -231.9876099, 229.9588013
9: -105.1639252, 103.2118835, -100.5180054, 98.5779800, -203.7418976, 203.7298889

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2433364, upper bound: 206.2432640
time: 8.25 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2433364, upper bound: 206.2436772
time: 7.93 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -114.3268051, 90.6389008, -112.4196854, 89.1544571, -203.4812317, 203.0585785
1: -97.6500397, 80.9326172, -96.0130920, 79.5914459, -177.2414856, 176.9456787
2: -126.9412766, 82.5303421, -124.8613510, 81.1314926, -208.0727692, 207.3916779
3: -134.1013336, 71.3602371, -131.8618164, 70.1385956, -204.2398987, 203.2220306
4: -123.4076920, 94.5306702, -121.4286575, 92.9871063, -216.3948059, 215.9593201
5: -109.3397446, 85.3172455, -107.5588989, 83.8368149, -193.1765442, 192.8761444
6: -105.2270813, 102.3216019, -103.5165024, 100.6313019, -205.8583832, 205.8381042
7: -115.3027115, 97.0764313, -113.5033035, 95.4717941, -210.7745056, 210.5797424
8: -140.2858582, 96.0186081, -137.9167480, 94.4083633, -234.6942139, 233.9353638
9: -105.1639252, 103.2118835, -103.5501328, 101.5435104, -206.7074280, 206.7620239

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2433364, upper bound: 206.2432640
time: 8.79 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2433364, upper bound: 206.2436772
time: 8.24 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -109.1148529, 86.5476608, -113.5742111, 90.0595093, -199.1743164, 200.1218719
1: -93.2331390, 77.2766342, -96.9846725, 80.4064331, -173.6395721, 174.2613068
2: -121.2215118, 78.7858582, -126.1373520, 81.9760895, -203.1975708, 204.9231720
3: -128.0054321, 68.0912018, -133.2654877, 70.8757324, -198.8811646, 201.3566895
4: -117.8769302, 90.2837524, -122.6568298, 93.9355316, -211.8124542, 212.9405518
5: -104.3852692, 81.4001389, -108.6680679, 84.7397003, -189.1249542, 190.0682068
6: -100.4566116, 97.6931458, -104.5434570, 101.6393051, -202.0959167, 202.2366028
7: -110.1820145, 92.6948242, -114.6364365, 96.4555588, -206.6375732, 207.3312531
8: -133.9401855, 91.7017670, -139.3198853, 95.3898849, -229.3300629, 231.0216370
9: -100.5180054, 98.5779800, -104.5781860, 102.5741577, -203.0921631, 203.1561584

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2398033, upper bound: 206.2400982
time: 8.09 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2437556, upper bound: 206.2438912
time: 8.13 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -109.1148529, 86.5476608, -116.9376831, 92.7133331, -201.8281403, 203.4853516
1: -93.2331390, 77.2766342, -99.8078537, 82.7624893, -175.9956207, 177.0844879
2: -121.2215118, 78.7858582, -129.8388214, 84.3585815, -205.5800629, 208.6246643
3: -128.0054321, 68.0912018, -137.1866302, 72.9565353, -200.9619751, 205.2778320
4: -117.8769302, 90.2837524, -126.2632370, 96.6837234, -214.5606537, 216.5469971
5: -104.3852692, 81.4001389, -111.8945160, 87.2217484, -191.6069946, 193.2946472
6: -100.4566116, 97.6931458, -107.6527405, 104.6259460, -205.0825500, 205.3458862
7: -110.1820145, 92.6948242, -118.0156250, 99.2793121, -209.4613342, 210.7104492
8: -133.9401855, 91.7017670, -143.3639679, 98.1394196, -232.0796051, 235.0657196
9: -100.5180054, 98.5779800, -107.6582947, 105.5898743, -206.1078796, 206.2362671

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2398033, upper bound: 206.2400982
time: 8.30 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2437556, upper bound: 206.2438912
time: 7.62 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -113.5742111, 90.0595093, -113.5742111, 90.0595093, -203.6337128, 203.6337128
1: -96.9846725, 80.4064331, -96.9846725, 80.4064331, -177.3911133, 177.3911133
2: -126.1373520, 81.9760895, -126.1373520, 81.9760895, -208.1134186, 208.1134186
3: -133.2654877, 70.8757324, -133.2654877, 70.8757324, -204.1412201, 204.1412201
4: -122.6568298, 93.9355316, -122.6568298, 93.9355316, -216.5923462, 216.5923462
5: -108.6680679, 84.7397003, -108.6680679, 84.7397003, -193.4077606, 193.4077606
6: -104.5434570, 101.6393051, -104.5434570, 101.6393051, -206.1827545, 206.1827545
7: -114.6364365, 96.4555588, -114.6364365, 96.4555588, -211.0919952, 211.0919952
8: -139.3198853, 95.3898849, -139.3198853, 95.3898849, -234.7097626, 234.7097626
9: -104.5781860, 102.5741577, -104.5781860, 102.5741577, -207.1523438, 207.1523438

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2397101, upper bound: 206.2400814
time: 9.80 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2447343, upper bound: 206.2446917
time: 8.06 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -113.5742111, 90.0595093, -116.9376831, 92.7133331, -206.2875214, 206.9971924
1: -96.9846725, 80.4064331, -99.8078537, 82.7624893, -179.7471466, 180.2142944
2: -126.1373520, 81.9760895, -129.8388214, 84.3585815, -210.4959106, 211.8149109
3: -133.2654877, 70.8757324, -137.1866302, 72.9565353, -206.2220154, 208.0623627
4: -122.6568298, 93.9355316, -126.2632370, 96.6837234, -219.3405457, 220.1987610
5: -108.6680679, 84.7397003, -111.8945160, 87.2217484, -195.8897858, 196.6342163
6: -104.5434570, 101.6393051, -107.6527405, 104.6259460, -209.1694031, 209.2920532
7: -114.6364365, 96.4555588, -118.0156250, 99.2793121, -213.9157410, 214.4711914
8: -139.3198853, 95.3898849, -143.3639679, 98.1394196, -237.4593048, 238.7538452
9: -104.5781860, 102.5741577, -107.6582947, 105.5898743, -210.1680603, 210.2324524

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2397101, upper bound: 206.2400814
time: 10.18 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2447343, upper bound: 206.2446917
time: 7.99 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -110.9160309, 87.9613571, -108.0103455, 85.6319427, -196.5479736, 195.9716949
1: -94.7590866, 78.5317154, -92.3798447, 76.4791565, -171.2382507, 170.9115601
2: -123.2023392, 80.0558472, -119.9658890, 78.0220566, -201.2243958, 200.0217285
3: -130.0810699, 69.2083054, -126.6260605, 67.4555206, -197.5365906, 195.8343658
4: -119.8040619, 91.7498245, -116.5628357, 89.3285980, -209.1326294, 208.3126526
5: -106.1065445, 82.7149353, -103.2478638, 80.6075668, -186.7141113, 185.9627991
6: -102.1324921, 99.3046722, -99.4190292, 96.7337723, -198.8662567, 198.7236938
7: -111.9823608, 94.1992874, -108.9118576, 91.7300568, -203.7124176, 203.1111145
8: -136.1098633, 93.1698532, -132.6826019, 90.8103333, -226.9201965, 225.8524475
9: -102.1591644, 100.1834717, -99.3136063, 97.5021133, -199.6612549, 199.4970398

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2279287, upper bound: 206.2257366
time: 10.71 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2234732, upper bound: 206.2227694
time: 9.58 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -112.4196854, 89.1544571, -114.3268051, 90.6389008, -203.0585785, 203.4812317
1: -96.0130920, 79.5914459, -97.6500397, 80.9326172, -176.9456787, 177.2414856
2: -124.8613510, 81.1314926, -126.9412766, 82.5303421, -207.3916779, 208.0727692
3: -131.8618164, 70.1385956, -134.1013336, 71.3602371, -203.2220306, 204.2398987
4: -121.4286575, 92.9871063, -123.4076920, 94.5306702, -215.9593201, 216.3948059
5: -107.5588989, 83.8368149, -109.3397446, 85.3172455, -192.8761444, 193.1765442
6: -103.5165024, 100.6313019, -105.2270813, 102.3216019, -205.8381042, 205.8583832
7: -113.5033035, 95.4717941, -115.3027115, 97.0764313, -210.5797424, 210.7745056
8: -137.9167480, 94.4083633, -140.2858582, 96.0186081, -233.9353638, 234.6942139
9: -103.5501328, 101.5435104, -105.1639252, 103.2118835, -206.7620239, 206.7074280

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2396071, upper bound: 206.2400798
time: 9.48 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2396071, upper bound: 206.2439160
time: 8.48 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -116.9376831, 92.7133331, -109.1148529, 86.5476608, -203.4853516, 201.8281403
1: -99.8078537, 82.7624893, -93.2331390, 77.2766342, -177.0844879, 175.9956207
2: -129.8388214, 84.3585815, -121.2215118, 78.7858582, -208.6246643, 205.5800629
3: -137.1866302, 72.9565353, -128.0054321, 68.0912018, -205.2778320, 200.9619751
4: -126.2632370, 96.6837234, -117.8769302, 90.2837524, -216.5469971, 214.5606537
5: -111.8945160, 87.2217484, -104.3852692, 81.4001389, -193.2946472, 191.6069946
6: -107.6527405, 104.6259460, -100.4566116, 97.6931458, -205.3458862, 205.0825500
7: -118.0156250, 99.2793121, -110.1820145, 92.6948242, -210.7104492, 209.4613342
8: -143.3639679, 98.1394196, -133.9401855, 91.7017670, -235.0657196, 232.0796051
9: -107.6582947, 105.5898743, -100.5180054, 98.5779800, -206.2362671, 206.1078796

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2391314, upper bound: 206.2395513
time: 7.55 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2438912, upper bound: 206.2437556
time: 7.90 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -116.9376831, 92.7133331, -113.5742111, 90.0595093, -206.9971924, 206.2875214
1: -99.8078537, 82.7624893, -96.9846725, 80.4064331, -180.2142944, 179.7471466
2: -129.8388214, 84.3585815, -126.1373520, 81.9760895, -211.8149109, 210.4959106
3: -137.1866302, 72.9565353, -133.2654877, 70.8757324, -208.0623627, 206.2220154
4: -126.2632370, 96.6837234, -122.6568298, 93.9355316, -220.1987610, 219.3405457
5: -111.8945160, 87.2217484, -108.6680679, 84.7397003, -196.6342163, 195.8897858
6: -107.6527405, 104.6259460, -104.5434570, 101.6393051, -209.2920532, 209.1694031
7: -118.0156250, 99.2793121, -114.6364365, 96.4555588, -214.4711914, 213.9157410
8: -143.3639679, 98.1394196, -139.3198853, 95.3898849, -238.7538452, 237.4593048
9: -107.6582947, 105.5898743, -104.5781860, 102.5741577, -210.2324524, 210.1680603

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2391314, upper bound: 206.2401225
time: 8.55 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2438912, upper bound: 206.2447418
time: 8.21 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -110.9160309, 87.9613571, -111.3596039, 88.2755814, -199.1916046, 199.3209381
1: -94.7590866, 78.5317154, -95.1956406, 78.8251038, -173.5841827, 173.7273560
2: -123.2023392, 80.0558472, -123.6539688, 80.3955383, -203.5978394, 203.7097931
3: -130.0810699, 69.2083054, -130.5327301, 69.5304184, -199.6114807, 199.7410278
4: -119.8040619, 91.7498245, -120.1603546, 92.0681076, -211.8721619, 211.9101868
5: -106.1065445, 82.7149353, -106.4640121, 83.0782852, -189.1848145, 189.1789551
6: -102.1324921, 99.3046722, -102.5193710, 99.7125397, -201.8450317, 201.8240356
7: -111.9823608, 94.1992874, -112.2776794, 94.5408859, -206.5232239, 206.4769287
8: -136.1098633, 93.1698532, -136.7120514, 93.5549698, -229.6648254, 229.8818970
9: -102.1591644, 100.1834717, -102.3851089, 100.5074005, -202.6665497, 202.5685425

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2280120, upper bound: 206.2257454
time: 9.12 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2235904, upper bound: 206.2227766
time: 8.48 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -112.4196854, 89.1544571, -117.6990509, 93.2998734, -205.7195587, 206.8535156
1: -96.0130920, 79.5914459, -100.4815521, 83.2953491, -179.3084259, 180.0729980
2: -124.8613510, 81.1314926, -130.6545258, 84.9226379, -209.7839966, 211.7860107
3: -131.8618164, 70.1385956, -138.0332642, 73.4483795, -205.3101959, 208.1718292
4: -121.4286575, 92.9871063, -127.0264816, 97.2849426, -218.7135773, 220.0135803
5: -107.5588989, 83.8368149, -112.5747375, 87.8056564, -195.3645477, 196.4115448
6: -103.5165024, 100.6313019, -108.3435364, 105.3183670, -208.8348694, 208.9748383
7: -113.5033035, 95.4717941, -118.6932220, 99.9101257, -213.4134216, 214.1650085
8: -137.9167480, 94.4083633, -144.3425598, 98.7780457, -236.6947937, 238.7509155
9: -103.5501328, 101.5435104, -108.2556610, 106.2355652, -209.7856903, 209.7991638

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2395786, upper bound: 206.2399913
time: 8.46 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2395786, upper bound: 206.2438374
time: 8.89 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -116.9376831, 92.7133331, -112.4196854, 89.1544571, -206.0921326, 205.1330261
1: -99.8078537, 82.7624893, -96.0130920, 79.5914459, -179.3992920, 178.7755432
2: -129.8388214, 84.3585815, -124.8613510, 81.1314926, -210.9703064, 209.2199402
3: -137.1866302, 72.9565353, -131.8618164, 70.1385956, -207.3252106, 204.8183441
4: -126.2632370, 96.6837234, -121.4286575, 92.9871063, -219.2503357, 218.1123810
5: -111.8945160, 87.2217484, -107.5588989, 83.8368149, -195.7313080, 194.7806396
6: -107.6527405, 104.6259460, -103.5165024, 100.6313019, -208.2840424, 208.1424561
7: -118.0156250, 99.2793121, -113.5033035, 95.4717941, -213.4874268, 212.7826233
8: -143.3639679, 98.1394196, -137.9167480, 94.4083633, -237.7723389, 236.0561676
9: -107.6582947, 105.5898743, -103.5501328, 101.5435104, -209.2017975, 209.1400146

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2390600, upper bound: 206.2394000
time: 8.83 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2438374, upper bound: 206.2436484
time: 8.06 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -116.9376831, 92.7133331, -116.9376831, 92.7133331, -209.6510162, 209.6510162
1: -99.8078537, 82.7624893, -99.8078537, 82.7624893, -182.5703430, 182.5703430
2: -129.8388214, 84.3585815, -129.8388214, 84.3585815, -214.1974030, 214.1974030
3: -137.1866302, 72.9565353, -137.1866302, 72.9565353, -210.1431580, 210.1431580
4: -126.2632370, 96.6837234, -126.2632370, 96.6837234, -222.9469604, 222.9469604
5: -111.8945160, 87.2217484, -111.8945160, 87.2217484, -199.1162567, 199.1162567
6: -107.6527405, 104.6259460, -107.6527405, 104.6259460, -212.2786865, 212.2786865
7: -118.0156250, 99.2793121, -118.0156250, 99.2793121, -217.2949371, 217.2949371
8: -143.3639679, 98.1394196, -143.3639679, 98.1394196, -241.5033875, 241.5033875
9: -107.6582947, 105.5898743, -107.6582947, 105.5898743, -213.2481689, 213.2481689

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2390600, upper bound: 206.2400539
time: 10.60 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2438374, upper bound: 206.2446597
time: 8.47 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 20.45 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.45
Output dim: 1, lower bound: -206.2392756, upper bound: 206.2394651
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.45
Output dim: 1, lower bound: -206.2392756, upper bound: 206.2394664
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.45
Output dim: 1, lower bound: -206.2392756, upper bound: 206.2394651
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.45
Output dim: 1, lower bound: -206.2392756, upper bound: 206.2394664
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.45
Output dim: 1, lower bound: -206.2433364, upper bound: 206.2432640
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.45
Output dim: 1, lower bound: -206.2433364, upper bound: 206.2436772
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.45
Output dim: 1, lower bound: -206.2433364, upper bound: 206.2432640
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.45
Output dim: 1, lower bound: -206.2433364, upper bound: 206.2436772
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.45
Output dim: 1, lower bound: -206.2398033, upper bound: 206.2400982
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.45
Output dim: 1, lower bound: -206.2437556, upper bound: 206.2438912
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.45
Output dim: 1, lower bound: -206.2398033, upper bound: 206.2400982
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.45
Output dim: 1, lower bound: -206.2437556, upper bound: 206.2438912
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.45
Output dim: 1, lower bound: -206.2397101, upper bound: 206.2400814
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.45
Output dim: 1, lower bound: -206.2447343, upper bound: 206.2446917
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.45
Output dim: 1, lower bound: -206.2397101, upper bound: 206.2400814
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.45
Output dim: 1, lower bound: -206.2447343, upper bound: 206.2446917
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.45
Output dim: 1, lower bound: -206.2279287, upper bound: 206.2257366
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.45
Output dim: 1, lower bound: -206.2234732, upper bound: 206.2227694
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.45
Output dim: 1, lower bound: -206.2396071, upper bound: 206.2400798
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.45
Output dim: 1, lower bound: -206.2396071, upper bound: 206.2439160
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.45
Output dim: 1, lower bound: -206.2391314, upper bound: 206.2395513
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.45
Output dim: 1, lower bound: -206.2438912, upper bound: 206.2437556
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.45
Output dim: 1, lower bound: -206.2391314, upper bound: 206.2401225
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.45
Output dim: 1, lower bound: -206.2438912, upper bound: 206.2447418
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.45
Output dim: 1, lower bound: -206.2280120, upper bound: 206.2257454
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.45
Output dim: 1, lower bound: -206.2235904, upper bound: 206.2227766
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.45
Output dim: 1, lower bound: -206.2395786, upper bound: 206.2399913
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.45
Output dim: 1, lower bound: -206.2395786, upper bound: 206.2438374
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.45
Output dim: 1, lower bound: -206.2390600, upper bound: 206.2394000
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.45
Output dim: 1, lower bound: -206.2438374, upper bound: 206.2436484
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.45
Output dim: 1, lower bound: -206.2390600, upper bound: 206.2400539
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.45
Output dim: 1, lower bound: -206.2438374, upper bound: 206.2446597

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -96.1970367, 76.2896957, -107.6138840, 85.3565292, -181.5535431, 183.9035797
1: -82.4421844, 68.1800385, -91.9811935, 76.2188110, -158.6609802, 160.1612244
2: -106.9756241, 69.5616074, -119.5654907, 77.7121429, -184.6877747, 189.1271057
3: -112.7064514, 60.0917168, -126.2276535, 67.1623688, -179.8687897, 186.3193512
4: -103.8875809, 79.6567764, -116.2550812, 89.0485382, -192.9360962, 195.9118652
5: -91.8912964, 71.7457352, -102.9351349, 80.2800140, -172.1713104, 174.6808777
6: -88.5594330, 86.2875595, -99.0745621, 96.3687286, -184.9281616, 185.3621216
7: -97.1042099, 81.7657318, -108.6637344, 91.4244537, -188.5286560, 190.4294434
8: -118.3957596, 81.0611572, -132.1363678, 90.4651489, -208.8609009, 213.1975098
9: -88.5610733, 86.8885956, -99.1290054, 97.2201157, -185.7811890, 186.0175629

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2252975, upper bound: 206.2277365
time: 8.88 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2223894, upper bound: 206.2232461
time: 8.76 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -100.0702209, 79.3398972, -107.6138840, 85.3565292, -185.4267578, 186.9537506
1: -85.7068481, 70.8934631, -91.9811935, 76.2188110, -161.9256439, 162.8746643
2: -111.2416840, 72.3274307, -119.5654907, 77.7121429, -188.9538269, 191.8929138
3: -117.2720871, 62.5142860, -126.2276535, 67.1623688, -184.4344177, 188.7419281
4: -108.0441055, 82.8273544, -116.2550812, 89.0485382, -197.0925903, 199.0823975
5: -95.6116562, 74.6517944, -102.9351349, 80.2800140, -175.8916626, 177.5869293
6: -92.1094742, 89.7162399, -99.0745621, 96.3687286, -188.4781799, 188.7907715
7: -100.9752350, 85.0315704, -108.6637344, 91.4244537, -192.3996887, 193.6952972
8: -123.0779190, 84.2658463, -132.1363678, 90.4651489, -213.5430603, 216.4022217
9: -92.0855331, 90.3559341, -99.1290054, 97.2201157, -189.3056488, 189.4849091

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2252975, upper bound: 206.2280916
time: 9.50 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2223894, upper bound: 206.2234587
time: 8.42 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 19.32 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 19.32
Output dim: 1, lower bound: -206.2252975, upper bound: 206.2277365
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 19.32
Output dim: 1, lower bound: -206.2223894, upper bound: 206.2232461
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 19.32
Output dim: 1, lower bound: -206.2252975, upper bound: 206.2280916
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 19.32
Output dim: 1, lower bound: -206.2223894, upper bound: 206.2234587
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.32
Output dim: 1, lower bound: -206.2392756, upper bound: 206.2394651
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.32
Output dim: 1, lower bound: -206.2392756, upper bound: 206.2394664
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.32
Output dim: 1, lower bound: -206.2433364, upper bound: 206.2432640
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.32
Output dim: 1, lower bound: -206.2433364, upper bound: 206.2436772
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.32
Output dim: 1, lower bound: -206.2433364, upper bound: 206.2432640
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.32
Output dim: 1, lower bound: -206.2433364, upper bound: 206.2436772
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.32
Output dim: 1, lower bound: -206.2398033, upper bound: 206.2400982
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.32
Output dim: 1, lower bound: -206.2437556, upper bound: 206.2438912
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.32
Output dim: 1, lower bound: -206.2398033, upper bound: 206.2400982
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.32
Output dim: 1, lower bound: -206.2437556, upper bound: 206.2438912
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.32
Output dim: 1, lower bound: -206.2397101, upper bound: 206.2400814
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.32
Output dim: 1, lower bound: -206.2447343, upper bound: 206.2446917
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.32
Output dim: 1, lower bound: -206.2397101, upper bound: 206.2400814
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.32
Output dim: 1, lower bound: -206.2447343, upper bound: 206.2446917
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.32
Output dim: 1, lower bound: -206.2279287, upper bound: 206.2257366
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.32
Output dim: 1, lower bound: -206.2234732, upper bound: 206.2227694
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.32
Output dim: 1, lower bound: -206.2396071, upper bound: 206.2400798
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.32
Output dim: 1, lower bound: -206.2396071, upper bound: 206.2439160
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.32
Output dim: 1, lower bound: -206.2391314, upper bound: 206.2395513
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.32
Output dim: 1, lower bound: -206.2438912, upper bound: 206.2437556
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.32
Output dim: 1, lower bound: -206.2391314, upper bound: 206.2401225
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.32
Output dim: 1, lower bound: -206.2438912, upper bound: 206.2447418
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.32
Output dim: 1, lower bound: -206.2280120, upper bound: 206.2257454
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.32
Output dim: 1, lower bound: -206.2235904, upper bound: 206.2227766
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.32
Output dim: 1, lower bound: -206.2395786, upper bound: 206.2399913
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.32
Output dim: 1, lower bound: -206.2395786, upper bound: 206.2438374
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.32
Output dim: 1, lower bound: -206.2390600, upper bound: 206.2394000
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.32
Output dim: 1, lower bound: -206.2438374, upper bound: 206.2436484
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.32
Output dim: 1, lower bound: -206.2390600, upper bound: 206.2400539
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.32
Output dim: 1, lower bound: -206.2438374, upper bound: 206.2446597
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=207.24490356445312
rel_dist={1: [-206.25362701135504, 206.25362701135498]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 131

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2523802, upper bound: 206.2523343
time: 8.64 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2523083, upper bound: 206.2523083
time: 9.45 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 18.23 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 18.23
Output dim: 1, lower bound: -206.2523802, upper bound: 206.2523343
IS_A2, status: Status.UNKNOWN, split count: 1, time: 18.23
Output dim: 1, lower bound: -206.2523083, upper bound: 206.2523083

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -122.3482819, 96.9952698, -129.7214508, 102.8237381, -225.1720276, 226.7167053
1: -104.3410645, 86.5778198, -110.4830704, 91.7346649, -196.0757294, 197.0608826
2: -135.7731781, 88.2661514, -143.9118958, 93.5015945, -229.2747650, 232.1780243
3: -143.5900574, 76.3189392, -152.1921692, 80.8304214, -224.4204712, 228.5111084
4: -132.0429077, 101.1205978, -139.9265137, 107.1277618, -239.1706543, 241.0471191
5: -117.0824127, 91.2991104, -124.1479263, 96.7636185, -213.8460236, 215.4470367
6: -112.6044006, 109.3904037, -119.3549957, 115.9184189, -228.5228271, 228.7453918
7: -123.3929977, 103.8461914, -130.7988434, 110.0385513, -233.4315491, 234.6450348
8: -149.9116974, 102.6137314, -158.7585297, 108.6323013, -258.5439758, 261.3722534
9: -112.5510635, 110.4543991, -119.2643738, 117.0467377, -229.5978088, 229.7187805

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 194

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2486373, upper bound: 206.2484857
time: 8.60 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2493538, upper bound: 206.2493262
time: 9.23 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -125.7769241, 99.7046509, -130.3652191, 103.3318634, -229.1087952, 230.0698700
1: -107.2242966, 88.9826508, -111.0207062, 92.1865540, -199.4108582, 200.0033264
2: -139.5509796, 90.6998215, -144.6239929, 93.9601288, -233.5110779, 235.3237915
3: -147.5962524, 78.4478607, -152.9430389, 81.2296295, -228.8258667, 231.3908997
4: -135.7267151, 103.9251862, -140.6140594, 107.6524734, -243.3791656, 244.5392456
5: -120.3754959, 93.8318710, -124.7670822, 97.2389374, -217.6144409, 218.5989532
6: -115.7748718, 112.4425888, -119.9449387, 116.4920959, -232.2669678, 232.3875275
7: -126.8394623, 106.7288284, -131.4476471, 110.5827026, -237.4221649, 238.1764526
8: -154.0409698, 105.4287491, -159.5344086, 109.1551208, -263.1961060, 264.9631653
9: -115.6958160, 113.5333557, -119.8543930, 117.6185684, -233.3143616, 233.3877563

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2485886, upper bound: 206.2484751
time: 8.41 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2493117, upper bound: 206.2493117
time: 9.43 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 19.18 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 19.18
Output dim: 1, lower bound: -206.2486373, upper bound: 206.2484857
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 19.18
Output dim: 1, lower bound: -206.2493538, upper bound: 206.2493262
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 19.18
Output dim: 1, lower bound: -206.2485886, upper bound: 206.2484751
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 19.18
Output dim: 1, lower bound: -206.2493117, upper bound: 206.2493117

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -117.3877563, 93.0797348, -116.3470917, 92.2663879, -209.6540985, 209.4268188
1: -100.1794739, 83.0907135, -99.2581635, 82.3336945, -182.5131683, 182.3488617
2: -130.3195648, 84.7116547, -129.2062683, 83.9230576, -214.2426147, 213.9179230
3: -137.7486115, 73.2325287, -136.4368134, 72.5149155, -210.2635040, 209.6693420
4: -126.7389984, 97.0565033, -125.6113968, 96.1730728, -222.9120789, 222.6678619
5: -112.3240814, 87.5865250, -111.3181000, 86.7598419, -199.0839233, 198.9046173
6: -108.0533295, 105.0058517, -107.0834427, 104.0927353, -212.1460571, 212.0892792
7: -118.4432068, 99.6633377, -117.4471970, 98.7700119, -217.2131958, 217.1105347
8: -143.9253845, 98.5217972, -142.6161194, 97.6010437, -241.5264130, 241.1379089
9: -108.0393906, 106.0019226, -107.1035690, 105.0437317, -213.0831146, 213.1054688

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2382568, upper bound: 206.2383505
time: 9.79 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2434917, upper bound: 206.2433168
time: 8.27 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -119.0426407, 94.3829117, -120.8202057, 95.7860184, -214.8286285, 215.2031250
1: -101.5703506, 84.2529678, -103.0169296, 85.4720993, -187.0424500, 187.2698975
2: -132.1433411, 85.8964005, -134.1337891, 87.1198959, -219.2632141, 220.0301666
3: -139.7006683, 74.2683411, -141.7126312, 75.3033524, -215.0039978, 215.9809723
4: -128.5073395, 98.4135590, -130.4009857, 99.8370438, -228.3443909, 228.8145447
5: -113.9131546, 88.8276672, -115.6102524, 90.1088791, -204.0220337, 204.4379272
6: -109.5680389, 106.4712067, -111.1778412, 108.0491333, -217.6171112, 217.6490479
7: -120.0946884, 101.0618668, -121.9140320, 102.5390778, -222.6337433, 222.9758911
8: -145.9225159, 99.8929062, -148.0091400, 101.2973099, -247.2198181, 247.9020386
9: -109.5475922, 107.4868851, -111.1739807, 109.0490646, -218.5966492, 218.6608582

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2387605, upper bound: 206.2389458
time: 8.28 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2445190, upper bound: 206.2444927
time: 9.76 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -120.7770767, 95.7560806, -116.9716339, 92.7590103, -213.5360565, 212.7277069
1: -103.0286331, 85.4662704, -99.7810211, 82.7725449, -185.8011780, 185.2472839
2: -134.0534821, 87.1170120, -129.8980560, 84.3692169, -218.4226837, 217.0150757
3: -141.7067719, 75.3346405, -137.1672516, 72.9030304, -214.6098022, 212.5018768
4: -130.3808441, 99.8286209, -126.2808533, 96.6830521, -227.0638885, 226.1094360
5: -115.5783386, 90.0880737, -111.9196167, 87.2202988, -202.7986145, 202.0076752
6: -111.1880417, 108.0214691, -107.6572342, 104.6504059, -215.8384399, 215.6787109
7: -121.8502045, 102.5119858, -118.0774078, 99.2992630, -221.1494751, 220.5893860
8: -148.0055847, 101.3013535, -143.3709717, 98.1091385, -246.1147156, 244.6723175
9: -111.1482849, 109.0443802, -107.6776276, 105.5992355, -216.7475281, 216.7220001

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2380931, upper bound: 206.2383157
time: 10.75 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2434406, upper bound: 206.2432968
time: 11.22 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -122.4555054, 97.0775986, -121.4819260, 96.3080978, -218.7635498, 218.5595245
1: -104.4375000, 86.6449585, -103.5686951, 85.9363174, -190.3738098, 190.2136536
2: -135.9013977, 88.3164062, -134.8651886, 87.5903778, -223.4917755, 223.1815643
3: -143.6846924, 76.3840256, -142.4829254, 75.7130280, -219.3977203, 218.8669434
4: -132.1708374, 101.2036438, -131.1064301, 100.3756104, -232.5464478, 232.3100739
5: -117.1890106, 91.3478012, -116.2459259, 90.5974579, -207.7864685, 207.5937195
6: -112.7226715, 109.5058670, -111.7843018, 108.6379700, -221.3606415, 221.2901611
7: -123.5236435, 103.9290161, -122.5808334, 103.0978699, -226.6215210, 226.5098572
8: -150.0293274, 102.6900482, -148.8069916, 101.8331757, -251.8625031, 251.4970245
9: -112.6755447, 110.5485611, -111.7790680, 109.6365433, -222.3120880, 222.3276367

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2386477, upper bound: 206.2389143
time: 8.66 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2444665, upper bound: 206.2444665
time: 10.10 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 20.12 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 20.12
Output dim: 1, lower bound: -206.2382568, upper bound: 206.2383505
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 20.12
Output dim: 1, lower bound: -206.2434917, upper bound: 206.2433168
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 20.12
Output dim: 1, lower bound: -206.2387605, upper bound: 206.2389458
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 20.12
Output dim: 1, lower bound: -206.2445190, upper bound: 206.2444927
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 20.12
Output dim: 1, lower bound: -206.2380931, upper bound: 206.2383157
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 20.12
Output dim: 1, lower bound: -206.2434406, upper bound: 206.2432968
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 20.12
Output dim: 1, lower bound: -206.2386477, upper bound: 206.2389143
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 20.12
Output dim: 1, lower bound: -206.2444665, upper bound: 206.2444665

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -104.0418243, 82.4932861, -111.0324936, 88.0501938, -192.0919800, 193.5257874
1: -89.0424194, 73.6909027, -94.8257980, 78.5884476, -167.6308594, 168.5166931
2: -115.6018982, 75.1800385, -123.3443146, 80.1216354, -195.7235413, 198.5243378
3: -121.9503479, 64.9790726, -130.1445160, 69.2274399, -191.1777954, 195.1235504
4: -112.3084488, 86.0791245, -119.8704987, 91.7991486, -204.1076050, 205.9496155
5: -99.4325562, 77.6294327, -106.1854553, 82.7945557, -182.2271118, 183.8148804
6: -95.7717056, 93.2244797, -102.1925125, 99.4035263, -195.1752014, 195.4169922
7: -104.9472427, 88.3800201, -112.0709610, 94.2740173, -199.2212219, 200.4509888
8: -127.8831406, 87.5350952, -136.2301788, 93.2243958, -221.1075439, 223.7652740
9: -95.7005234, 93.9356537, -102.1875305, 100.2375259, -195.9380341, 196.1231842

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2230146, upper bound: 206.2244060
time: 9.66 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2214356, upper bound: 206.2219521
time: 10.63 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -110.2379761, 87.4067535, -114.0528259, 90.4457397, -200.6837158, 201.4595795
1: -94.2180481, 78.0555038, -97.3451996, 80.7168884, -174.9349365, 175.4006958
2: -122.4410324, 79.5977783, -126.6781464, 82.2817230, -204.7227325, 206.2759247
3: -129.2846832, 68.8108139, -133.7200470, 71.0956421, -200.3803253, 202.5308228
4: -119.0327759, 91.1818695, -123.1382599, 94.2868652, -213.3196411, 214.3201294
5: -105.4125977, 82.2548447, -109.1000290, 85.0488663, -190.4614563, 191.3548737
6: -101.4710312, 98.7070923, -104.9712067, 102.0711975, -203.5422363, 203.6782990
7: -111.2223282, 93.6248016, -115.1298218, 96.8326111, -208.0549316, 208.7546082
8: -135.3489380, 92.6423798, -139.8641205, 95.7134552, -231.0623932, 232.5064850
9: -101.4430389, 99.5377350, -104.9865189, 102.9684982, -204.4115295, 204.5242615

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 131

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2308177, upper bound: 206.2316203
time: 11.48 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2289478, upper bound: 206.2286507
time: 11.04 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -105.5596085, 83.6894913, -115.3445282, 91.4463272, -197.0058746, 199.0340271
1: -90.3205032, 74.7548904, -98.4510422, 81.6135483, -171.9340515, 173.2059326
2: -117.2739944, 76.2635727, -128.0989990, 83.2029343, -200.4769287, 204.3625793
3: -123.7389069, 65.9313202, -135.2301178, 71.9193649, -195.6582642, 201.1614227
4: -113.9337387, 87.3219223, -124.4905624, 95.3283768, -209.2621155, 211.8124847
5: -100.8914185, 78.7693558, -110.3248062, 86.0246277, -186.9160461, 189.0941620
6: -97.1628036, 94.5681152, -106.1411896, 103.2193756, -200.3821716, 200.7093048
7: -106.4625702, 89.6630173, -116.3778610, 97.9103775, -204.3729401, 206.0408783
8: -129.7188110, 88.7913513, -141.4337769, 96.7890320, -226.5078430, 230.2251129
9: -97.0831451, 95.2966766, -106.1105728, 104.0996475, -201.1827698, 201.4072571

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2242801, upper bound: 206.2257683
time: 15.52 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2230174, upper bound: 206.2239026
time: 10.70 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -111.9761887, 88.7785416, -118.5891800, 94.0171661, -205.9933167, 207.3676910
1: -95.6787491, 79.2788239, -101.1571350, 83.9021301, -179.5808716, 180.4359283
2: -124.3592453, 80.8447342, -131.6773376, 85.5257950, -209.8850403, 212.5220490
3: -131.3346558, 69.9013443, -139.0710449, 73.9254303, -205.2600861, 208.9723816
4: -120.8918228, 92.6070557, -127.9964371, 98.0027084, -218.8945312, 220.6034851
5: -107.0843048, 83.5601501, -113.4547424, 88.4462509, -195.5305481, 197.0148926
6: -103.0649414, 100.2452240, -109.1254730, 106.0835495, -209.1484985, 209.3706970
7: -112.9574051, 95.0955200, -119.6607056, 100.6564331, -213.6138153, 214.7562256
8: -137.4458313, 94.0838242, -145.3336334, 99.4630432, -236.9088745, 239.4174500
9: -103.0284119, 101.0993729, -109.1153336, 107.0329742, -210.0613403, 210.2147064

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2326756, upper bound: 206.2335958
time: 11.48 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2313037, upper bound: 206.2312582
time: 9.18 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -107.3616867, 85.1156616, -111.6523361, 88.5399399, -195.9016266, 196.7680054
1: -91.8392258, 76.0144348, -95.3454056, 79.0246353, -170.8638611, 171.3598175
2: -119.2559967, 77.5317307, -124.0312576, 80.5651703, -199.8211670, 201.5629730
3: -125.8245697, 67.0384674, -130.8694611, 69.6135483, -195.4380646, 197.9079285
4: -115.8818970, 88.7946854, -120.5349960, 92.3056030, -208.1875000, 209.3296509
5: -102.6259689, 80.0824356, -106.7833099, 83.2522964, -185.8782501, 186.8657379
6: -98.8482056, 96.1784134, -102.7626877, 99.9573975, -198.8056030, 198.9410706
7: -108.2879868, 91.1677704, -112.6972961, 94.8000107, -203.0879822, 203.8650208
8: -131.8825531, 90.2557068, -136.9800262, 93.7288132, -225.6113586, 227.2357178
9: -98.7496109, 96.9163742, -102.7577057, 100.7891693, -199.5387421, 199.6740723

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 131

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2230461, upper bound: 206.2246021
time: 10.59 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2214797, upper bound: 206.2221006
time: 11.81 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -113.5877914, 90.0500717, -114.6801758, 90.9406433, -204.5284271, 204.7302399
1: -97.0308228, 80.4023819, -97.8702927, 81.1579056, -178.1887207, 178.2726593
2: -126.1305389, 81.9746552, -127.3729935, 82.7299576, -208.8605042, 209.3476562
3: -133.1899261, 70.8852234, -134.4539185, 71.4857178, -204.6756439, 205.3391266
4: -122.6287537, 93.9176712, -123.8108063, 94.7988586, -217.4276123, 217.7284546
5: -108.6267166, 84.7262192, -109.7043228, 85.5112381, -194.1379547, 194.4305420
6: -104.5681305, 101.6836624, -105.5473328, 102.6315613, -207.1996765, 207.2309723
7: -114.5899277, 96.4396057, -115.7627411, 97.3643723, -211.9542999, 212.2023468
8: -139.3787537, 95.3840561, -140.6223145, 96.2239227, -235.6026611, 236.0063782
9: -104.5150528, 102.5418243, -105.5631332, 103.5265503, -208.0415955, 208.1049194

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 131

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2310094, upper bound: 206.2319116
time: 10.38 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2290683, upper bound: 206.2287655
time: 10.12 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -108.9011688, 86.3283920, -116.0023804, 91.9659500, -200.8671265, 202.3307648
1: -93.1333313, 77.0931549, -99.0001526, 82.0754318, -175.2086945, 176.0933075
2: -120.9520569, 78.6287537, -128.8264771, 83.6704636, -204.6225281, 207.4552307
3: -127.6371918, 68.0026321, -135.9969330, 72.3269882, -199.9641724, 203.9995575
4: -117.5271912, 90.0552597, -125.1922684, 95.8644485, -213.3916321, 215.2475281
5: -104.1030655, 81.2386093, -110.9578552, 86.5104523, -190.6134949, 192.1964722
6: -100.2568054, 97.5410538, -106.7449493, 103.8051300, -204.0619354, 204.2859650
7: -109.8231430, 92.4668732, -117.0408401, 98.4659271, -208.2890625, 209.5077209
8: -133.7433167, 91.5290070, -142.2269897, 97.3226242, -231.0659332, 233.7559967
9: -100.1487579, 98.2954941, -106.7124863, 104.6838989, -204.8326416, 205.0079803

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2243013, upper bound: 206.2258754
time: 9.88 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2230441, upper bound: 206.2240014
time: 9.71 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -115.3366089, 91.4302216, -119.2534409, 94.5412292, -209.8778381, 210.6836243
1: -98.4997025, 81.6329956, -101.7111816, 84.3682785, -182.8679810, 183.3441620
2: -128.0590973, 83.2271500, -132.4116058, 85.9982224, -214.0573120, 215.6387634
3: -135.2520447, 71.9816742, -139.8441162, 74.3367233, -209.5887756, 211.8257904
4: -124.4962769, 95.3506317, -128.7043762, 98.5432129, -223.0394897, 224.0549927
5: -110.3079758, 86.0395355, -114.0928421, 88.9368057, -199.2447510, 200.1323700
6: -106.1707153, 103.2304840, -109.7339401, 106.6748123, -212.8455200, 212.9644165
7: -116.3352966, 97.9187851, -120.3299866, 101.2175522, -217.5528259, 218.2487793
8: -141.4880219, 96.8311157, -146.1345062, 100.0009155, -241.4889221, 242.9656067
9: -106.1087189, 104.1124649, -109.7228241, 107.6227646, -213.7314606, 213.8352814

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2327552, upper bound: 206.2337672
time: 9.56 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2313332, upper bound: 206.2313332
time: 8.07 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 18.99 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.99
Output dim: 1, lower bound: -206.2230146, upper bound: 206.2244060
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.99
Output dim: 1, lower bound: -206.2214356, upper bound: 206.2219521
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.99
Output dim: 1, lower bound: -206.2308177, upper bound: 206.2316203
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.99
Output dim: 1, lower bound: -206.2289478, upper bound: 206.2286507
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.99
Output dim: 1, lower bound: -206.2242801, upper bound: 206.2257683
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.99
Output dim: 1, lower bound: -206.2230174, upper bound: 206.2239026
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.99
Output dim: 1, lower bound: -206.2326756, upper bound: 206.2335958
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.99
Output dim: 1, lower bound: -206.2313037, upper bound: 206.2312582
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.99
Output dim: 1, lower bound: -206.2230461, upper bound: 206.2246021
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.99
Output dim: 1, lower bound: -206.2214797, upper bound: 206.2221006
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.99
Output dim: 1, lower bound: -206.2310094, upper bound: 206.2319116
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.99
Output dim: 1, lower bound: -206.2290683, upper bound: 206.2287655
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.99
Output dim: 1, lower bound: -206.2243013, upper bound: 206.2258754
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.99
Output dim: 1, lower bound: -206.2230441, upper bound: 206.2240014
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.99
Output dim: 1, lower bound: -206.2327552, upper bound: 206.2337672
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.99
Output dim: 1, lower bound: -206.2313332, upper bound: 206.2313332

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -103.1315155, 81.7671051, -107.1707306, 84.9726257, -188.1041412, 188.9378357
1: -88.2779846, 73.0572510, -91.5916443, 75.8957520, -164.1737366, 164.6488953
2: -114.5916519, 74.5371857, -119.0553131, 77.3905258, -191.9821777, 193.5924988
3: -120.8659286, 64.4180603, -125.5473633, 66.8509445, -187.7168274, 189.9653931
4: -111.3119812, 85.3317719, -115.6560593, 88.6303940, -199.9423523, 200.9878235
5: -98.5477753, 76.9507828, -102.4408569, 79.9236221, -178.4713898, 179.3916321
6: -94.9234085, 92.4137802, -98.5999908, 95.9669800, -190.8903656, 191.0137634
7: -104.0281830, 87.6181107, -108.1788940, 91.0432739, -195.0714569, 195.7969971
8: -126.7662048, 86.7732925, -131.5023193, 89.9953232, -216.7615356, 218.2756042
9: -94.8637238, 93.1157455, -98.6437454, 96.7628021, -191.6264954, 191.7594757

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2212083, upper bound: 206.2224527
time: 10.21 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2211822, upper bound: 206.2225017
time: 10.84 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -101.0351410, 80.0886459, -109.6706085, 86.9139786, -187.9491119, 189.7592468
1: -86.5240479, 71.5968094, -93.7024384, 77.6519699, -164.1760101, 165.2992401
2: -112.2612076, 73.0533829, -121.7613525, 79.1419144, -191.4030762, 194.8146973
3: -118.3623428, 63.1321030, -128.4391327, 68.3704834, -186.7328186, 191.5712128
4: -109.0047302, 83.6070099, -118.2743301, 90.6346130, -199.6393433, 201.8813477
5: -96.5056458, 75.3854523, -104.8064804, 81.7437210, -178.2493591, 180.1919098
6: -92.9574509, 90.5478058, -100.8442764, 98.1460419, -191.1034851, 191.3920898
7: -101.9059677, 85.8681183, -110.6501541, 93.1543274, -195.0602875, 196.5182800
8: -124.1874847, 85.0102158, -134.4720154, 91.9154129, -216.1029053, 219.4822388
9: -92.9389496, 91.2285233, -100.8895416, 99.0187912, -191.9576874, 192.1180267

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2200128, upper bound: 206.2203847
time: 11.11 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2198600, upper bound: 206.2203334
time: 10.77 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -109.3077011, 86.6659470, -110.1540070, 87.3396912, -196.6473999, 196.8199310
1: -93.4398499, 77.4069519, -94.0822678, 77.9974365, -171.4372864, 171.4892273
2: -121.4074478, 78.9398499, -122.3472900, 79.5232010, -200.9306488, 201.2871399
3: -128.1772003, 68.2384567, -129.0796661, 68.6973877, -196.8745880, 197.3181152
4: -118.0179901, 90.4185181, -118.8857956, 91.0884476, -209.1064453, 209.3043060
5: -104.5105820, 81.5637512, -105.3213577, 82.1524582, -186.6630096, 186.8851013
6: -100.6051865, 97.8793182, -101.3449020, 98.6022034, -199.2073669, 199.2242126
7: -110.2851944, 92.8464279, -111.2017517, 93.5712891, -203.8564453, 204.0481873
8: -134.2108459, 91.8639145, -135.0941162, 92.4525681, -226.6634064, 226.9580078
9: -100.5890427, 98.7005768, -101.4095612, 99.4611053, -200.0501404, 200.1101379

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2287539, upper bound: 206.2292173
time: 11.16 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2285717, upper bound: 206.2292014
time: 11.00 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -107.2329865, 85.0068130, -112.6563263, 89.2856369, -196.5186157, 197.6631317
1: -91.7098389, 75.9589233, -96.2008286, 79.7534409, -171.4632874, 172.1597595
2: -119.0987091, 77.4688492, -125.0552139, 81.2744141, -200.3731232, 202.5240631
3: -125.7033920, 66.9682541, -131.9783173, 70.2210693, -195.9244690, 198.9465637
4: -115.7453613, 88.7125168, -121.5179062, 93.0956802, -208.8410187, 210.2304230
5: -102.4955292, 80.0207291, -107.6952667, 83.9801865, -186.4757080, 187.7159882
6: -98.6631851, 96.0333939, -103.5953293, 100.7856216, -199.4487457, 199.6286926
7: -108.1912308, 91.1144562, -113.6819229, 95.6842499, -203.8754730, 204.7963715
8: -131.6651917, 90.1200409, -138.0738831, 94.3796921, -226.0448761, 228.1939240
9: -98.6887283, 96.8352890, -103.6622314, 101.7224731, -200.4111938, 200.4975281

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2274366, upper bound: 206.2270000
time: 10.57 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2270414, upper bound: 206.2268355
time: 10.35 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -104.6409073, 82.9568024, -111.3807526, 88.2889481, -192.9298401, 194.3375397
1: -89.5491714, 74.1154404, -95.1345596, 78.8486099, -168.3977814, 169.2500000
2: -116.2544403, 75.6148911, -123.6971359, 80.3983688, -196.6528015, 199.3120270
3: -122.6443024, 65.3651505, -130.5113983, 69.4812775, -192.1255798, 195.8765564
4: -112.9281387, 86.5676956, -120.1679840, 92.0753937, -205.0035400, 206.7356567
5: -99.9984970, 78.0845795, -106.4833450, 83.0797348, -183.0782318, 184.5679321
6: -96.3067017, 93.7500763, -102.4536438, 99.6935349, -196.0001984, 196.2037048
7: -105.5351105, 88.8941498, -112.3857346, 94.5950928, -200.1302032, 201.2798767
8: -128.5914612, 88.0225601, -136.5843506, 93.4721985, -222.0636597, 224.6069031
9: -96.2386093, 94.4693375, -102.4746170, 100.5344849, -196.7731018, 196.9439392

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2224920, upper bound: 206.2238335
time: 10.54 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2224358, upper bound: 206.2238440
time: 9.98 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -102.5815430, 81.3073807, -114.1100159, 90.4143600, -192.9958954, 195.4173737
1: -87.8258514, 72.6804657, -97.4388809, 80.7631149, -168.5889587, 170.1193542
2: -113.9650650, 74.1571350, -126.6568222, 82.3079453, -196.2730103, 200.8139648
3: -120.1857910, 64.1020737, -133.6809387, 71.1437912, -191.3295746, 197.7829742
4: -110.6614075, 84.8732147, -123.0448151, 94.2692795, -204.9306946, 207.9180298
5: -97.9919891, 76.5467834, -109.0728912, 85.0746994, -183.0666809, 185.6196747
6: -94.3755798, 91.9164886, -104.9159393, 102.0755234, -196.4510803, 196.8324127
7: -103.4500198, 87.1745834, -115.0925064, 96.8934860, -200.3435059, 202.2670746
8: -126.0581207, 86.2907104, -139.8374939, 95.5929718, -221.6510925, 226.1282043
9: -94.3478012, 92.6148987, -104.9337616, 102.9974823, -197.3452759, 197.5486603

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2215994, upper bound: 206.2222767
time: 10.15 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2214186, upper bound: 206.2222143
time: 11.30 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -111.0400009, 88.0330429, -114.5970459, 90.8404541, -201.8804474, 202.6300812
1: -94.8957596, 78.6261978, -97.8186340, 81.1191406, -176.0148773, 176.4448242
2: -123.3192520, 80.1826324, -127.2473755, 82.7021942, -206.0214386, 207.4299774
3: -130.2202454, 69.3254700, -134.3205719, 71.4731598, -201.6934052, 203.6460266
4: -119.8710327, 91.8387833, -123.6455536, 94.7255478, -214.5965881, 215.4843445
5: -106.1766891, 82.8647690, -109.5888748, 85.4823151, -191.6589966, 192.4536438
6: -102.1935501, 99.4123230, -105.4144440, 102.5334091, -204.7269592, 204.8267670
7: -112.0144806, 94.3124084, -115.6408615, 97.3202438, -209.3347168, 209.9532471
8: -136.3006897, 93.3003540, -140.4526825, 96.1241913, -232.4248657, 233.7530365
9: -102.1691666, 100.2571487, -105.4541245, 103.4449158, -205.6140747, 205.7112732

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2306214, upper bound: 206.2311108
time: 9.40 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2304152, upper bound: 206.2310980
time: 8.87 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -108.9893951, 86.3934937, -117.3229523, 92.9642487, -201.9536285, 203.7164307
1: -93.1870270, 77.1944580, -100.1210022, 83.0314026, -176.2184143, 177.3154297
2: -121.0367889, 78.7280350, -130.2043457, 84.6087189, -205.6454773, 208.9323425
3: -127.7762070, 68.0705032, -137.4881287, 73.1348648, -200.9110260, 205.5586243
4: -117.6254120, 90.1529922, -126.5212250, 96.9182816, -214.5436707, 216.6742249
5: -104.1855164, 81.3410187, -112.1773605, 87.4759369, -191.6614380, 193.5183716
6: -100.2745895, 97.5875397, -107.8757935, 104.9137344, -205.1882935, 205.4633331
7: -109.9458084, 92.6002731, -118.3465271, 99.6160431, -209.5618439, 210.9468079
8: -133.7857971, 91.5773544, -143.7040405, 98.2441864, -232.0299835, 235.2814026
9: -100.2912674, 98.4133987, -107.9119720, 105.9055939, -206.1968536, 206.3253784

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2298081, upper bound: 206.2295715
time: 9.54 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2294177, upper bound: 206.2294158
time: 9.46 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -106.4469223, 84.3858109, -107.7700043, 85.4463196, -191.8932495, 192.1557922
1: -91.0710144, 75.3775635, -92.0951843, 76.3172531, -167.3882751, 167.4727478
2: -118.2407761, 76.8857498, -119.7188797, 77.8192825, -196.0600586, 196.6046295
3: -124.7345581, 66.4748230, -126.2475967, 67.2248764, -191.9594421, 192.7224121
4: -114.8804321, 88.0438766, -116.2993546, 89.1203766, -204.0008087, 204.3432312
5: -101.7368927, 79.4005432, -103.0194473, 80.3670654, -182.1039581, 182.4199829
6: -97.9962082, 95.3637772, -99.1514053, 96.5029144, -194.4990997, 194.5151520
7: -107.3645020, 90.4021225, -108.7851944, 91.5522766, -198.9167786, 199.1873169
8: -130.7602997, 89.4905319, -132.2287445, 90.4830627, -221.2433624, 221.7192688
9: -97.9090729, 96.0926895, -99.1956100, 97.2966537, -195.2057190, 195.2882996

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2212601, upper bound: 206.2226302
time: 9.47 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2212252, upper bound: 206.2226707
time: 9.62 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -104.3584061, 82.7137375, -110.2607956, 87.3802109, -191.7386017, 192.9745331
1: -89.3228760, 73.9220963, -94.1977158, 78.0665054, -167.3893738, 168.1198120
2: -115.9197006, 75.4060364, -122.4155502, 79.5629272, -195.4826355, 197.8215942
3: -122.2415466, 65.1926270, -129.1311188, 68.7385788, -190.9801178, 194.3237305
4: -112.5811081, 86.3245392, -118.9083252, 91.1160126, -203.6971130, 205.2328644
5: -99.7023926, 77.8408279, -105.3762817, 82.1802216, -181.8825989, 183.2171021
6: -96.0365753, 93.5041504, -101.3874283, 98.6736908, -194.7102509, 194.8915405
7: -105.2497864, 88.6586304, -111.2469482, 93.6555176, -198.9053040, 199.9055786
8: -128.1903381, 87.7322388, -135.1851196, 92.3950500, -220.5853882, 222.9173584
9: -95.9905853, 94.2112656, -101.4338608, 99.5436859, -195.5342560, 195.6451263

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2200580, upper bound: 206.2205504
time: 9.53 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2199189, upper bound: 206.2205118
time: 12.06 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -112.6512222, 89.3040695, -110.7624435, 87.8195724, -200.4707947, 200.0665131
1: -96.2473221, 79.7492142, -94.5918961, 78.4251709, -174.6724854, 174.3411102
2: -125.0903473, 81.3121033, -123.0210266, 79.9583130, -205.0486450, 204.3331299
3: -132.0747833, 70.3093414, -129.7905579, 69.0761414, -201.1509247, 200.0998993
4: -121.6071472, 93.1493301, -119.5380859, 91.5851212, -213.1922302, 212.6874084
5: -107.7188110, 84.0307083, -105.9073944, 82.6009674, -190.3197632, 189.9381104
6: -103.6969757, 100.8504562, -101.9038391, 99.1458969, -202.8428650, 202.7543030
7: -113.6465302, 95.6562195, -111.8158722, 94.0873947, -207.7339172, 207.4720764
8: -138.2332764, 94.6007462, -135.8296661, 92.9477386, -231.1810150, 230.4303741
9: -103.6558151, 101.6994934, -101.9691162, 100.0025711, -203.6583862, 203.6686096

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2289823, upper bound: 206.2295308
time: 11.47 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2288861, upper bound: 206.2295200
time: 10.31 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -110.5700836, 87.6406708, -113.2651901, 89.7660675, -200.3361511, 200.9058533
1: -94.5121231, 78.2956848, -96.7103500, 80.1813965, -174.6935120, 175.0060120
2: -122.7745972, 79.8349533, -125.7296677, 81.7090530, -204.4836426, 205.5646210
3: -129.5957947, 69.0348892, -132.6918182, 70.5995865, -200.1953583, 201.7267151
4: -119.3285217, 91.4372406, -122.1707764, 93.5922470, -212.9207764, 213.6080017
5: -105.6989670, 82.4831238, -108.2818527, 84.4292297, -190.1282043, 190.7649841
6: -101.7488861, 98.9988861, -104.1543579, 101.3293228, -203.0782013, 203.1532440
7: -111.5465088, 93.9187012, -114.2960587, 96.2004395, -207.7469482, 208.2147369
8: -135.6816864, 92.8505783, -138.8083038, 94.8743515, -230.5560303, 231.6588745
9: -101.7497864, 99.8279266, -104.2222290, 102.2634125, -204.0131989, 204.0501404

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2275994, upper bound: 206.2272170
time: 10.25 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2272966, upper bound: 206.2270878
time: 10.85 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -107.9752045, 85.5897598, -112.0210495, 88.7943878, -196.7695923, 197.6108093
1: -92.3560410, 76.4483490, -95.6689377, 79.2981720, -171.6542053, 172.1172791
2: -119.9246063, 77.9746094, -124.4049225, 80.8534546, -200.7780609, 202.3795166
3: -126.5338898, 67.4323120, -131.2570190, 69.8784256, -196.4123077, 198.6893311
4: -116.5139084, 89.2951736, -120.8507309, 92.5971298, -209.1110382, 210.1459045
5: -103.2034149, 80.5486526, -107.0993881, 83.5525055, -186.7559204, 187.6480408
6: -99.3944168, 96.7167053, -103.0411911, 100.2637253, -199.6581421, 199.7578888
7: -108.8886185, 91.6919861, -113.0309753, 95.1360168, -204.0246124, 204.7229614
8: -132.6077271, 90.7544708, -137.3560181, 93.9912109, -226.5989380, 228.1104889
9: -99.2981949, 97.4620438, -103.0606842, 101.1031799, -200.4013519, 200.5226746

Time for backsubstitution: 1.26 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=207.24490356445312
rel_dist={1: [-206.2534079160462, 206.25340791604617]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 131

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2518341, upper bound: 206.2518146
time: 9.98 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2518032, upper bound: 206.2518032
time: 10.37 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 20.49 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 20.49
Output dim: 1, lower bound: -206.2518341, upper bound: 206.2518146
IS_A2, status: Status.UNKNOWN, split count: 1, time: 20.49
Output dim: 1, lower bound: -206.2518032, upper bound: 206.2518032

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -122.3482819, 96.9952698, -126.5949173, 100.3520813, -222.7003632, 223.5901794
1: -104.3410645, 86.5778198, -107.8907013, 89.5489273, -193.8899841, 194.4685211
2: -135.7731781, 88.2661514, -140.4537964, 91.2789001, -227.0520782, 228.7199249
3: -143.5900574, 76.3189392, -148.5479889, 78.9259186, -222.5159760, 224.8669281
4: -132.0429077, 101.1205978, -136.5941925, 104.5856018, -236.6285095, 237.7147827
5: -117.0824127, 91.2991104, -121.1515732, 94.4449234, -211.5273438, 212.4506836
6: -112.6044006, 109.3904037, -116.5081177, 113.1530914, -225.7574921, 225.8985291
7: -123.3929977, 103.8461914, -127.6557465, 107.4076920, -230.8006744, 231.5019226
8: -149.9116974, 102.6137314, -155.0090485, 106.0935822, -256.0052490, 257.6227722
9: -112.5510635, 110.4543991, -116.4226303, 114.2636185, -226.8146820, 226.8770294

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2479754, upper bound: 206.2479404
time: 10.81 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2488111, upper bound: 206.2487959
time: 10.53 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -125.7769241, 99.7046509, -127.6625595, 101.1950150, -226.9719391, 227.3672180
1: -107.2242966, 88.9826508, -108.7840271, 90.2991791, -197.5234680, 197.7666321
2: -139.5509796, 90.6998215, -141.6354065, 92.0398636, -231.5908356, 232.3351898
3: -147.5962524, 78.4478607, -149.7935791, 79.5908813, -227.1871338, 228.2414398
4: -135.7267151, 103.9251862, -137.7357635, 105.4562607, -241.1829681, 241.6609344
5: -120.3754959, 93.8318710, -122.1799850, 95.2323608, -215.6078491, 216.0118561
6: -115.7748718, 112.4425888, -117.4884491, 114.1065216, -229.8813934, 229.9310303
7: -126.8394623, 106.7288284, -128.7333221, 108.3123322, -235.1517944, 235.4621429
8: -154.0409698, 105.4287491, -156.2984619, 106.9600525, -261.0010376, 261.7272034
9: -115.6958160, 113.5333557, -117.4040451, 115.2122726, -230.9080811, 230.9374084

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2479474, upper bound: 206.2479388
time: 10.99 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2487879, upper bound: 206.2487879
time: 10.61 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 22.94 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 22.94
Output dim: 1, lower bound: -206.2479754, upper bound: 206.2479404
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 22.94
Output dim: 1, lower bound: -206.2488111, upper bound: 206.2487959
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 22.94
Output dim: 1, lower bound: -206.2479474, upper bound: 206.2479388
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 22.94
Output dim: 1, lower bound: -206.2487879, upper bound: 206.2487879

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -112.9879150, 89.6069260, -113.2505188, 89.8158798, -202.8037872, 202.8574371
1: -96.4864960, 79.9982224, -96.6889572, 80.1676712, -176.6541748, 176.6871796
2: -125.4819489, 81.5581131, -125.7787476, 81.7191696, -207.2011108, 207.3368378
3: -132.5675812, 70.4941788, -132.8275146, 70.6257935, -203.1933746, 203.3216858
4: -122.0340576, 93.4530945, -122.3082886, 93.6558838, -215.6898804, 215.7613831
5: -108.1034698, 84.2929459, -108.3489151, 84.4615173, -192.5649872, 192.6418304
6: -104.0165253, 101.1167297, -104.2617416, 101.3526764, -205.3692017, 205.3784790
7: -114.0517731, 95.9538345, -114.3325806, 96.1616974, -210.2134552, 210.2864075
8: -138.6144257, 94.8927155, -138.9002838, 95.0843124, -233.6987305, 233.7929840
9: -104.0371323, 102.0511703, -104.2878265, 102.2859802, -206.3230743, 206.3389893

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2372937, upper bound: 206.2373451
time: 10.64 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2428707, upper bound: 206.2428112
time: 9.74 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -115.8106079, 91.8279343, -117.7019272, 93.3195877, -209.1301880, 209.5298309
1: -98.8606339, 81.9795685, -100.4316559, 83.2916870, -182.1522980, 182.4112244
2: -128.5938721, 83.5794525, -130.6844635, 84.9025879, -213.4964600, 214.2639160
3: -135.8973541, 72.2635040, -138.0779724, 73.4030228, -209.3003845, 210.3414764
4: -125.0498352, 95.7668991, -127.0765839, 97.3017883, -222.3516235, 222.8434753
5: -110.8135529, 86.4116592, -112.6209412, 87.7954559, -198.6089935, 199.0325928
6: -106.5986633, 103.6157150, -108.3374252, 105.2911758, -211.8898010, 211.9531403
7: -116.8689423, 98.3392944, -118.7790070, 99.9147415, -216.7836609, 217.1183014
8: -142.0207062, 97.2317886, -144.2691498, 98.7647324, -240.7854309, 241.5009460
9: -106.6105347, 104.5839920, -108.3398895, 106.2735825, -212.8841095, 212.9238892

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2376705, upper bound: 206.2377504
time: 10.85 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2441209, upper bound: 206.2441059
time: 11.12 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -116.3303070, 92.2439575, -114.2843933, 90.6315613, -206.9618683, 206.5283508
1: -99.2960510, 82.3392105, -97.5566559, 80.8949890, -180.1910400, 179.8958588
2: -129.1628571, 83.9300613, -126.9251404, 82.4585266, -211.6213837, 210.8551941
3: -136.4686432, 72.5648041, -134.0357056, 71.2715454, -207.7401886, 206.6005096
4: -125.6258850, 96.1863556, -123.4173584, 94.5003815, -220.1262512, 219.6036987
5: -111.3122482, 86.7579041, -109.3456192, 85.2234268, -196.5356445, 196.1035156
6: -107.1089554, 104.0885773, -105.2134094, 102.2775726, -209.3865204, 209.3019867
7: -117.4108963, 98.7623215, -115.3773041, 97.0401993, -214.4510956, 214.1396027
8: -142.6363678, 97.6304932, -140.1516418, 95.9248962, -238.5612640, 237.7821350
9: -107.1032333, 105.0505066, -105.2404785, 103.2055817, -210.3088074, 210.2909546

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2372289, upper bound: 206.2373369
time: 12.45 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2428523, upper bound: 206.2428079
time: 11.01 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -119.1940002, 94.4981079, -118.8017426, 94.1877136, -213.3817139, 213.2998505
1: -101.7013245, 84.3498764, -101.3502579, 84.0641327, -185.7654266, 185.7001343
2: -132.3179932, 85.9769516, -131.9001465, 85.6844788, -218.0024414, 217.8770752
3: -139.8437500, 74.3581161, -139.3586731, 74.0872345, -213.9309845, 213.7167664
4: -128.6790619, 98.5318527, -128.2502594, 98.1973267, -226.8763885, 226.7821045
5: -114.0596390, 88.9090042, -113.6791458, 88.6067886, -202.6664276, 202.5881348
6: -109.7259369, 106.6215363, -109.3473206, 106.2713852, -215.9973145, 215.9688568
7: -120.2679672, 101.1804733, -119.8883896, 100.8452835, -221.1132355, 221.0688324
8: -146.0902557, 100.0003357, -145.5968781, 99.6544571, -245.7447052, 245.5971985
9: -109.7099304, 107.6178894, -109.3479080, 107.2497177, -216.9596558, 216.9657898

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2376301, upper bound: 206.2377310
time: 11.34 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2440971, upper bound: 206.2440971
time: 9.56 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 22.24 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.24
Output dim: 1, lower bound: -206.2372937, upper bound: 206.2373451
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.24
Output dim: 1, lower bound: -206.2428707, upper bound: 206.2428112
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.24
Output dim: 1, lower bound: -206.2376705, upper bound: 206.2377504
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.24
Output dim: 1, lower bound: -206.2441209, upper bound: 206.2441059
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.24
Output dim: 1, lower bound: -206.2372289, upper bound: 206.2373369
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.24
Output dim: 1, lower bound: -206.2428523, upper bound: 206.2428079
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.24
Output dim: 1, lower bound: -206.2376301, upper bound: 206.2377310
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.24
Output dim: 1, lower bound: -206.2440971, upper bound: 206.2440971

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -99.7578888, 79.1059113, -103.5863800, 82.1442413, -181.9021301, 182.6922607
1: -85.4390182, 70.6809006, -88.6225815, 73.3584290, -158.7974091, 159.3034515
2: -110.8908844, 72.1111755, -115.1184158, 74.8086243, -185.6995087, 187.2295837
3: -116.9035645, 62.3056908, -121.3805313, 64.6428604, -181.5464172, 183.6861877
4: -107.7158585, 82.5725250, -111.8579483, 85.7032089, -193.4190674, 194.4304657
5: -95.3151932, 74.4139786, -99.0084076, 77.2447433, -172.5598907, 173.4223938
6: -91.8350906, 89.4362030, -95.3619614, 92.8243103, -184.6593933, 184.7981567
7: -100.6668091, 84.7649307, -104.5513992, 87.9832687, -188.6500854, 189.3162994
8: -122.7018280, 83.9988708, -127.2797775, 87.1240616, -209.8258972, 211.2786560
9: -91.8006210, 90.0846252, -95.3450394, 93.5413361, -185.3419495, 185.4296417

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2207747, upper bound: 206.2213488
time: 14.61 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2202155, upper bound: 206.2204686
time: 13.11 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -105.8257141, 83.9190750, -108.0295639, 85.6688385, -191.4945526, 191.9486389
1: -90.5131378, 74.9507370, -92.3337326, 76.4868317, -166.9999542, 167.2844543
2: -117.5836334, 76.4326477, -120.0213470, 77.9818192, -195.5654602, 196.4539948
3: -124.0859070, 66.0586319, -126.6433411, 67.3920135, -191.4779205, 192.7019653
4: -114.3102341, 87.5687408, -116.6763916, 89.3648300, -203.6750641, 204.2451172
5: -101.1750946, 78.9490356, -103.2980728, 80.5658722, -181.7409668, 182.2471008
6: -97.4171600, 94.8057632, -99.4505768, 96.7515259, -194.1686859, 194.2563324
7: -106.8172836, 89.9002838, -109.0576782, 91.7489243, -198.5661774, 198.9579620
8: -130.0188904, 88.9984512, -132.6344147, 90.7868805, -220.8057556, 221.6328583
9: -97.4264069, 95.5704651, -99.4684525, 97.5603485, -194.9867096, 195.0389099

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2288868, upper bound: 206.2291295
time: 11.72 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2281796, upper bound: 206.2280511
time: 11.44 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -102.3126297, 81.1171646, -107.7142487, 85.3962936, -187.7089233, 188.8314209
1: -87.5917358, 72.4711609, -92.0995102, 76.2504959, -163.8422089, 164.5706787
2: -113.7061768, 73.9354935, -119.6679382, 77.7548370, -191.4609985, 193.6034241
3: -119.9141159, 63.9104156, -126.2491379, 67.2227783, -187.1368866, 190.1595459
4: -110.4505310, 84.6633759, -116.2900848, 89.0802612, -199.5307770, 200.9534607
5: -97.7686234, 76.3339157, -102.9732437, 80.3409653, -178.1095886, 179.3071594
6: -94.1743011, 91.6984787, -99.1432495, 96.4788208, -190.6531067, 190.8417358
7: -103.2172165, 86.9235535, -108.6792068, 91.4636765, -194.6808929, 195.6027527
8: -125.7916107, 86.1148224, -132.2684631, 90.5367661, -216.3283539, 218.3832703
9: -94.1274490, 92.3750381, -99.1018219, 97.2388153, -191.3662415, 191.4768219

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2207747, upper bound: 206.2227400
time: 13.02 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2216835, upper bound: 206.2220826
time: 11.49 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -108.8116531, 86.2736206, -112.6211014, 89.2883530, -198.0999908, 198.8947144
1: -93.0237961, 77.0520782, -96.1943130, 79.7149353, -172.7387390, 173.2463989
2: -120.8806763, 78.5754318, -125.0869598, 81.2703094, -202.1509857, 203.6623840
3: -127.6096497, 67.9346161, -132.0612793, 70.2615280, -197.8711700, 199.9958649
4: -117.5040436, 90.0167160, -121.5984497, 93.1259842, -210.6300049, 211.6151581
5: -104.0463181, 81.1931992, -107.7094269, 84.0077362, -188.0540466, 188.9026031
6: -100.1545334, 97.4482956, -103.6607590, 100.8143234, -200.9688568, 201.1090546
7: -109.7992554, 92.4271545, -113.6468506, 95.6239166, -205.4231720, 206.0740051
8: -133.6225891, 91.4758377, -138.1738586, 94.5862961, -228.2088928, 229.6496887
9: -100.1512909, 98.2546158, -103.6510010, 101.6795044, -201.8307953, 201.9056091

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2314157, upper bound: 206.2317434
time: 13.22 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2309187, upper bound: 206.2309070
time: 10.06 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -103.0541611, 81.7093353, -104.5876617, 82.9355774, -185.9897156, 186.2969818
1: -88.2173538, 72.9882889, -89.4654694, 74.0632706, -162.2806091, 162.4537659
2: -114.5189667, 74.4474335, -116.2281647, 75.5245743, -190.0435028, 190.6755981
3: -120.7505035, 64.3514633, -122.5521317, 65.2703247, -186.0208130, 186.9035950
4: -111.2650757, 85.2690201, -112.9340897, 86.5210724, -197.7860870, 198.2031097
5: -98.4867706, 76.8495026, -99.9767990, 77.9838028, -176.4705658, 176.8262787
6: -94.8914185, 92.3693085, -96.2859573, 93.7211304, -188.6125336, 188.6552582
7: -103.9843826, 87.5341187, -105.5651779, 88.8349457, -192.8192902, 193.0993042
8: -126.6729660, 86.7004318, -128.4941864, 87.9380646, -214.6110229, 215.1945953
9: -94.8296051, 93.0445328, -96.2695007, 94.4325867, -189.2621613, 189.3140106

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2207927, upper bound: 206.2214409
time: 14.12 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2202358, upper bound: 206.2205548
time: 12.01 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -109.1480713, 86.5398712, -109.0778885, 86.4964066, -195.6444702, 195.6177673
1: -93.3034744, 77.2783203, -93.2129135, 77.2251205, -170.5285797, 170.4912415
2: -121.2440186, 78.7903214, -121.1840820, 78.7322540, -199.9762421, 199.9743958
3: -127.9603195, 68.1164017, -127.8679962, 68.0467453, -196.0070648, 195.9843903
4: -117.8786621, 90.2826843, -117.8019257, 90.2207336, -208.0993652, 208.0846100
5: -104.3634796, 81.3992386, -104.3082581, 81.3384476, -185.7019196, 185.7074585
6: -100.4903793, 97.7585068, -100.4152222, 97.6895447, -198.1799011, 198.1737061
7: -110.1570206, 92.6926651, -110.1180496, 92.6402206, -202.7972412, 202.8107147
8: -134.0164642, 91.7192993, -133.9035645, 91.6397171, -225.6561584, 225.6228638
9: -100.4749527, 98.5507050, -100.4347763, 98.4929810, -198.9679260, 198.9854736

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2289829, upper bound: 206.2292811
time: 13.16 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2282660, upper bound: 206.2281427
time: 10.86 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -105.6503754, 83.7531738, -108.8019714, 86.2558212, -191.9061890, 192.5551453
1: -90.4008713, 74.8061981, -93.0094986, 77.0148392, -167.4157104, 167.8157043
2: -117.3799973, 76.2966690, -120.8716507, 78.5279388, -195.9079132, 197.1683197
3: -123.8065567, 65.9788437, -127.5181732, 67.9001541, -191.7067108, 193.4970093
4: -114.0380554, 87.3931808, -117.4518585, 89.9677277, -204.0057678, 204.8450317
5: -100.9772415, 78.8006668, -104.0215378, 81.1434555, -182.1206818, 182.8222046
6: -97.2640610, 94.6669159, -100.1434097, 97.4495316, -194.7135925, 194.8103333
7: -106.5732956, 89.7243042, -109.7763519, 92.3842697, -198.9575653, 199.5006561
8: -129.8106842, 88.8483658, -133.5820465, 91.4188614, -221.2295532, 222.4304199
9: -97.1895905, 95.3699265, -100.1004105, 98.2050934, -195.3946838, 195.4703064

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2221262, upper bound: 206.2228004
time: 10.29 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2216936, upper bound: 206.2221148
time: 13.45 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -112.1572800, 88.9143829, -113.7265701, 90.1615295, -202.3188019, 202.6409607
1: -95.8324814, 79.3960114, -97.1184006, 80.4921951, -176.3246765, 176.5143585
2: -124.5643768, 80.9466324, -126.3095551, 82.0572205, -206.6215973, 207.2561340
3: -131.5095978, 70.0057297, -133.3488770, 70.9499207, -202.4595032, 203.3545837
4: -121.0922318, 92.7479553, -122.7789154, 94.0252533, -215.1174774, 215.5268707
5: -107.2563705, 83.6617126, -108.7730331, 84.8228607, -192.0792236, 192.4347229
6: -103.2473221, 100.4206696, -104.6754990, 101.8005676, -205.0478821, 205.0961609
7: -113.1623535, 95.2385025, -114.7631378, 96.5607834, -209.7231445, 210.0016022
8: -137.6477051, 94.2105179, -139.5093384, 95.4803543, -233.1280518, 233.7198486
9: -103.2183609, 101.2547684, -104.6656113, 102.6611404, -205.8794708, 205.9203644

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2314595, upper bound: 206.2318269
time: 11.52 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2309646, upper bound: 206.2309646
time: 9.58 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 22.46 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.46
Output dim: 1, lower bound: -206.2207747, upper bound: 206.2213488
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.46
Output dim: 1, lower bound: -206.2202155, upper bound: 206.2204686
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.46
Output dim: 1, lower bound: -206.2288868, upper bound: 206.2291295
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.46
Output dim: 1, lower bound: -206.2281796, upper bound: 206.2280511
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.46
Output dim: 1, lower bound: -206.2207747, upper bound: 206.2227400
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.46
Output dim: 1, lower bound: -206.2216835, upper bound: 206.2220826
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.46
Output dim: 1, lower bound: -206.2314157, upper bound: 206.2317434
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.46
Output dim: 1, lower bound: -206.2309187, upper bound: 206.2309070
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.46
Output dim: 1, lower bound: -206.2207927, upper bound: 206.2214409
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.46
Output dim: 1, lower bound: -206.2202358, upper bound: 206.2205548
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.46
Output dim: 1, lower bound: -206.2289829, upper bound: 206.2292811
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.46
Output dim: 1, lower bound: -206.2282660, upper bound: 206.2281427
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.46
Output dim: 1, lower bound: -206.2221262, upper bound: 206.2228004
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.46
Output dim: 1, lower bound: -206.2216936, upper bound: 206.2221148
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.46
Output dim: 1, lower bound: -206.2314595, upper bound: 206.2318269
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.46
Output dim: 1, lower bound: -206.2309646, upper bound: 206.2309646

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -97.0833282, 76.9727631, -99.8054504, 79.1279526, -176.2112427, 176.7781982
1: -83.1926651, 68.8196182, -85.4476700, 70.7265625, -153.9192200, 154.2672882
2: -107.9224167, 70.2227249, -110.9218597, 72.1393814, -180.0617981, 181.1445923
3: -113.7176514, 60.6572342, -116.8754349, 62.3124352, -176.0300598, 177.5326538
4: -104.7881470, 80.3770752, -107.7193069, 82.5998764, -187.3880005, 188.0963745
5: -92.7160950, 72.4194489, -95.3339996, 74.4258270, -167.1418915, 167.7534485
6: -89.3431320, 87.0543671, -91.8396988, 89.4578476, -178.8009644, 178.8940582
7: -97.9663086, 82.5265656, -100.7337646, 84.8197784, -182.7860870, 183.2603302
8: -119.4194183, 81.7616196, -122.6395264, 83.9611664, -203.3805847, 204.4011536
9: -89.3421631, 87.6754303, -91.8698349, 90.1360092, -179.4781647, 179.5452576

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2190614, upper bound: 206.2196207
time: 12.42 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2190558, upper bound: 206.2196638
time: 11.73 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -94.3668137, 74.7928391, -102.3180237, 81.0780869, -175.4448853, 177.1108704
1: -80.9218140, 66.9269638, -87.5675583, 72.4914551, -153.4132690, 154.4945068
2: -104.8973312, 68.2984314, -113.6382370, 73.8987198, -178.7960510, 181.9366608
3: -110.4687805, 58.9988976, -119.7794495, 63.8388176, -174.3076019, 178.7783203
4: -101.7938080, 78.1402817, -110.3492203, 84.6121292, -186.4059448, 188.4894867
5: -90.0665054, 70.3883133, -97.7099991, 76.2542038, -166.3207092, 168.0983124
6: -86.7945251, 84.6366501, -94.0928268, 91.6463928, -178.4409180, 178.7294769
7: -95.2136154, 80.2588806, -103.2168045, 86.9398193, -182.1534271, 183.4756775
8: -116.0710373, 79.4689865, -125.6221619, 85.8870239, -201.9580688, 205.0911560
9: -86.8500061, 85.2281723, -94.1248856, 92.4023438, -179.2523346, 179.3530273

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2186505, upper bound: 206.2188773
time: 10.96 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2185982, upper bound: 206.2188687
time: 12.32 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -103.1072159, 81.7518234, -104.1675339, 82.5906677, -185.6978760, 185.9193573
1: -88.2328491, 73.0578079, -89.0977020, 73.7963181, -162.0291595, 162.1554871
2: -114.5653458, 74.5120010, -115.7322388, 75.2523346, -189.8176880, 190.2442322
3: -120.8476257, 64.3837967, -122.0435333, 65.0142365, -185.8618622, 186.4273224
4: -111.3368835, 85.3375854, -112.4569473, 86.1962433, -197.5331268, 197.7945251
5: -98.5346527, 76.9241104, -99.5500946, 77.6923752, -176.2270203, 176.4742126
6: -94.8847809, 92.3856201, -95.8553238, 93.3141403, -188.1988983, 188.2409363
7: -104.0737534, 87.6255798, -105.1629639, 88.5180359, -192.5917816, 192.7885284
8: -126.6857758, 86.7236786, -127.9025269, 87.5554199, -214.2411652, 214.6262054
9: -94.9278717, 93.1219635, -95.9216614, 94.0833282, -189.0112000, 189.0436249

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2270055, upper bound: 206.2272049
time: 11.05 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2269555, upper bound: 206.2272112
time: 10.85 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -100.4694138, 79.6335297, -106.7452316, 84.5932846, -185.0626984, 186.3787537
1: -86.0266495, 71.2187424, -91.2717590, 75.6072693, -161.6339111, 162.4905090
2: -111.6301498, 72.6425781, -118.5235901, 77.0590210, -188.6891785, 191.1661682
3: -117.6951675, 62.7690392, -125.0282288, 66.5815353, -184.2766724, 187.7972717
4: -108.4289627, 83.1648178, -115.1634445, 88.2628708, -196.6918335, 198.3282471
5: -95.9621048, 74.9524231, -101.9910126, 79.5710068, -175.5331116, 176.9434357
6: -92.4044266, 90.0370026, -98.1721649, 95.5624084, -187.9667969, 188.2091675
7: -101.4018555, 85.4250717, -107.7153168, 90.6920547, -192.0938873, 193.1403809
8: -123.4332352, 84.4986420, -130.9664612, 89.5419769, -212.9752197, 215.4650726
9: -92.5091019, 90.7481995, -98.2392197, 96.4105759, -188.9196320, 188.9874268

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2186505, upper bound: 206.2188773
time: 11.11 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2263777, upper bound: 206.2262668
time: 10.47 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -99.5743256, 78.9330139, -103.8437271, 82.3092194, -181.8835297, 182.7767029
1: -85.2927856, 70.5650864, -88.8502808, 73.5562744, -158.8490601, 159.4153442
2: -110.6673508, 72.0017395, -115.3728409, 75.0214386, -185.6887665, 187.3745422
3: -116.6513977, 62.2225342, -121.6358795, 64.8369522, -181.4883423, 183.8584137
4: -107.4528503, 82.4151382, -112.0532150, 85.9029617, -193.3557892, 194.4683533
5: -95.1072464, 74.2924042, -99.2120132, 77.4553833, -172.5626221, 173.5044250
6: -91.6223526, 89.2603302, -95.5368729, 93.0332108, -184.6555481, 184.7971802
7: -100.4529800, 84.6317368, -104.7721252, 88.2255249, -188.6784821, 189.4038696
8: -122.4311981, 83.8236847, -127.5180893, 87.2967377, -209.7279358, 211.3417358
9: -91.6103897, 89.9088669, -95.5443802, 93.7530823, -185.3634338, 185.4532318

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2204290, upper bound: 206.2209957
time: 13.17 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2204177, upper bound: 206.2210159
time: 12.44 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -96.9939499, 76.8614349, -106.6141663, 84.4643784, -181.4582825, 183.4755859
1: -83.1359100, 68.7668152, -91.1839828, 75.5004349, -158.6363220, 159.9507751
2: -107.7950974, 70.1735153, -118.3769760, 76.9620514, -184.7571411, 188.5504913
3: -113.5672607, 60.6459122, -124.8500290, 66.5223846, -180.0896454, 185.4959412
4: -104.6066895, 80.2896576, -114.9686203, 88.1273727, -192.7340698, 195.2582703
5: -92.5899811, 72.3632126, -101.8366776, 79.4771271, -172.0670929, 174.1998901
6: -89.1981812, 86.9629745, -98.0329742, 95.4499512, -184.6481323, 184.9959259
7: -97.8367004, 82.4783707, -107.5161133, 90.5579071, -188.3946075, 189.9944763
8: -119.2512894, 81.6480560, -130.8155975, 89.4462128, -208.6974792, 212.4636536
9: -89.2430573, 87.5841904, -98.0373840, 96.2510147, -185.4940491, 185.6215668

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2201433, upper bound: 206.2204784
time: 11.69 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2200942, upper bound: 206.2204626
time: 12.32 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -106.0291443, 84.0571289, -108.6677856, 86.1395798, -192.1687317, 192.7249146
1: -90.6952820, 75.1125107, -92.8875122, 76.9582291, -167.6534882, 168.0000305
2: -117.7896957, 76.6077042, -120.6952133, 78.4736557, -196.2633514, 197.3029022
3: -124.2964172, 66.2222137, -127.3543243, 67.8294678, -192.1258698, 193.5765076
4: -114.4677582, 87.7328415, -117.2867050, 89.8816147, -204.3493652, 205.0195312
5: -101.3476868, 79.1248550, -103.8772202, 81.0709000, -182.4185791, 183.0020752
6: -97.5637970, 94.9726181, -99.9812927, 97.2975082, -194.8612976, 194.9538879
7: -106.9955597, 90.0992126, -109.6647568, 92.3174973, -199.3130493, 199.7639618
8: -130.2166290, 89.1468887, -133.3358917, 91.2772293, -221.4938660, 222.4827881
9: -97.5968018, 95.7505188, -100.0235138, 98.1226349, -195.7194214, 195.7740326

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2295231, upper bound: 206.2296830
time: 11.17 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2294571, upper bound: 206.2296798
time: 12.04 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -103.5065918, 82.0316162, -111.4458847, 88.3028107, -191.8094025, 193.4775085
1: -88.5851440, 73.3544464, -95.2294540, 78.9080582, -167.4931946, 168.5838928
2: -114.9833679, 74.8194885, -123.7098007, 80.4186020, -195.4019470, 198.5292969
3: -121.2837753, 64.6787491, -130.5814972, 69.5211868, -190.8049469, 195.2602081
4: -111.6851120, 85.6556473, -120.2169037, 92.1151581, -203.8002625, 205.8725586
5: -98.8872757, 77.2398682, -106.5134277, 83.1010056, -181.9882507, 183.7532654
6: -95.1926804, 92.7253723, -102.4893265, 99.7229462, -194.9156189, 195.2146912
7: -104.4398727, 87.9943848, -112.4219513, 94.6562805, -199.0961609, 200.4163361
8: -127.1062927, 87.0211182, -136.6469574, 93.4376144, -220.5438995, 223.6680450
9: -95.2836914, 93.4793549, -102.5272446, 100.6289444, -195.9125977, 196.0065918

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2292468, upper bound: 206.2291437
time: 11.80 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2290882, upper bound: 206.2290821
time: 11.13 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -100.3786392, 79.5749054, -100.7601547, 79.8819046, -180.2605438, 180.3350220
1: -85.9706726, 71.1259766, -86.2513885, 71.3985596, -157.3692322, 157.3773651
2: -111.5487366, 72.5586777, -111.9794693, 72.8220901, -184.3708191, 184.5381470
3: -117.5616150, 62.7025146, -117.9906616, 62.9114342, -180.4730377, 180.6931610
4: -108.3361664, 83.0732422, -108.7445755, 83.3796310, -191.7157898, 191.8177948
5: -95.8861465, 74.8550873, -96.2569351, 75.1308060, -171.0169373, 171.1119995
6: -92.3990021, 89.9865799, -92.7210083, 90.3131332, -182.7121277, 182.7075806
7: -101.2834320, 85.2950363, -101.7005081, 85.6324005, -186.9158020, 186.9955444
8: -123.3896713, 84.4628525, -123.7980652, 84.7375641, -208.1271973, 208.2608795
9: -92.3706589, 90.6349182, -92.7522125, 90.9858856, -183.3565216, 183.3871307

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2191397, upper bound: 206.2197283
time: 12.74 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2191209, upper bound: 206.2197589
time: 11.19 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 25.30 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -206.2190614, upper bound: 206.2196207
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -206.2190558, upper bound: 206.2196638
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -206.2186505, upper bound: 206.2188773
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -206.2185982, upper bound: 206.2188687
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -206.2270055, upper bound: 206.2272049
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -206.2269555, upper bound: 206.2272112
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -206.2186505, upper bound: 206.2188773
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -206.2263777, upper bound: 206.2262668
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -206.2204290, upper bound: 206.2209957
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -206.2204177, upper bound: 206.2210159
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -206.2201433, upper bound: 206.2204784
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -206.2200942, upper bound: 206.2204626
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -206.2295231, upper bound: 206.2296830
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -206.2294571, upper bound: 206.2296798
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -206.2292468, upper bound: 206.2291437
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -206.2290882, upper bound: 206.2290821
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -206.2191397, upper bound: 206.2197283
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -206.2191209, upper bound: 206.2197589
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.30
Output dim: 1, lower bound: -206.2202358, upper bound: 206.2205548
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.30
Output dim: 1, lower bound: -206.2289829, upper bound: 206.2292811
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.30
Output dim: 1, lower bound: -206.2282660, upper bound: 206.2281427
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.30
Output dim: 1, lower bound: -206.2221262, upper bound: 206.2228004
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.30
Output dim: 1, lower bound: -206.2216936, upper bound: 206.2221148
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.30
Output dim: 1, lower bound: -206.2314595, upper bound: 206.2318269
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.30
Output dim: 1, lower bound: -206.2309646, upper bound: 206.2309646
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=207.24490356445312
rel_dist={1: [-206.25298865227455, 206.2529886525199]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1812.49 seconds
