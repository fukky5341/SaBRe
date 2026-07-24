## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 195.3388952653
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864)
1: (-94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567)
2: (-123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544)
3: (-135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521)
4: (-123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344)
5: (-109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042)
6: (-104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754)
7: (-116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424)
8: (-134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655)
9: (-103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908)

## BASE Result
execution time: IAR + LP analysis = 1.25 + 10.45 = 11.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -195.3645330, upper bound: 195.3645330


# Binary Search by BASE starts (time budget: 2688.30 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=207.82974243164062
rel_dist={7: [-195.364420828844, 195.364420828844]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=207.82974243164062
rel_dist={7: [-195.3643469135574, 195.36434691615057]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=207.82974243164062
rel_dist={7: [-195.36410562008308, 195.3641056191758]}

## Binary Search Result
Binary search time: 49.13 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 2639.17 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3007692, upper bound: 195.3121514
time: 11.66 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.2840079, upper bound: 195.2840079
time: 8.12 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 19.92 seconds
IS_A1, status: Status.VERIFIED, split count: 1, time: 19.92
Output dim: 7, lower bound: -195.3007692, upper bound: 195.3121514
IS_A2, status: Status.VERIFIED, split count: 1, time: 19.92
Output dim: 7, lower bound: -195.2840079, upper bound: 195.2840079
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=207.82974243164062
rel_dist={7: [-195.364420828844, 195.364420828844]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3075024, upper bound: 195.3222935
time: 10.53 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.2840550, upper bound: 195.2840550
time: 8.49 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 19.16 seconds
IS_A1, status: Status.VERIFIED, split count: 1, time: 19.16
Output dim: 7, lower bound: -195.3075024, upper bound: 195.3222935
IS_A2, status: Status.VERIFIED, split count: 1, time: 19.16
Output dim: 7, lower bound: -195.2840550, upper bound: 195.2840550
Binary search (step 1): status=Status.VERIFIED, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=207.82974243164062
rel_dist={7: [-195.36447750727785, 195.36447750727785]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3112943, upper bound: 195.3275753
time: 10.51 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.2840830, upper bound: 195.2840830
time: 8.37 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 19.03 seconds
IS_A1, status: Status.VERIFIED, split count: 1, time: 19.03
Output dim: 7, lower bound: -195.3112943, upper bound: 195.3275753
IS_A2, status: Status.VERIFIED, split count: 1, time: 19.03
Output dim: 7, lower bound: -195.2840830, upper bound: 195.2840830
Binary search (step 2): status=Status.VERIFIED, k_low=10, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=207.82974243164062
rel_dist={7: [-195.36451453067917, 195.36451453067923]}

## Binary search (step 3) starts
Candidate k: 12, corresponding eps: 0.0468750


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3131198, upper bound: 195.3300265
time: 8.70 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.2840970, upper bound: 195.2840970
time: 6.03 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 14.86 seconds
IS_A1, status: Status.VERIFIED, split count: 1, time: 14.86
Output dim: 7, lower bound: -195.3131198, upper bound: 195.3300265
IS_A2, status: Status.VERIFIED, split count: 1, time: 14.86
Output dim: 7, lower bound: -195.2840970, upper bound: 195.2840970
Binary search (step 3): status=Status.VERIFIED, k_low=12, k_high=12, k_mid=12, eps_mid=0.0468750, abs_max=207.82974243164062
rel_dist={7: [-195.36453304237992, 195.3645330423799]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.046875
execution time: 126.75 seconds
