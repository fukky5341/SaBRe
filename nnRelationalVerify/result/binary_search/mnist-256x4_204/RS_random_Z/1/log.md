## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 36.489946449
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748)
1: (-19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108)
2: (-26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393)
3: (-30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846)
4: (-31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904)
5: (-27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109)
6: (-31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243)
7: (-23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175)
8: (-34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087)
9: (-22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285)

## BASE Result
execution time: IAR + LP analysis = 1.09 + 9.01 = 10.10 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -36.5032747, upper bound: 36.5032750


# Binary Search by BASE starts (time budget: 2689.90 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=47.04602813720703
rel_dist={6: [-36.503194103113984, 36.50319409863076]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=47.04602813720703
rel_dist={6: [-36.50308478066641, 36.50308478066641]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=47.04602813720703
rel_dist={6: [-36.50297363759564, 36.50297363759566]}

## Binary Search Result
Binary search time: 35.36 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 2654.54 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5030562, upper bound: 36.5030562
time: 9.25 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5030562, upper bound: 36.5030562
time: 8.88 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 18.14 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 18.14
Output dim: 6, lower bound: -36.5030562, upper bound: 36.5030562
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 18.14
Output dim: 6, lower bound: -36.5030562, upper bound: 36.5030562

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5030560, upper bound: 36.5030462
time: 6.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5030462, upper bound: 36.5030562
time: 5.66 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4671086, upper bound: 36.4671299
time: 6.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4671086, upper bound: 36.4671299
time: 6.14 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 13.28 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.28
Output dim: 6, lower bound: -36.5030560, upper bound: 36.5030462
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.28
Output dim: 6, lower bound: -36.5030462, upper bound: 36.5030562
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 13.28
Output dim: 6, lower bound: -36.4671086, upper bound: 36.4671299
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 13.28
Output dim: 6, lower bound: -36.4671086, upper bound: 36.4671299

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5030555, upper bound: 36.5030462
time: 11.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5030562, upper bound: 36.5030458
time: 6.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5024485, upper bound: 36.5024523
time: 6.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5024482, upper bound: 36.5024530
time: 5.11 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 12.12 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.12
Output dim: 6, lower bound: -36.5030555, upper bound: 36.5030462
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.12
Output dim: 6, lower bound: -36.5030562, upper bound: 36.5030458
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.12
Output dim: 6, lower bound: -36.5024485, upper bound: 36.5024523
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.12
Output dim: 6, lower bound: -36.5024482, upper bound: 36.5024530

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4422662, upper bound: 36.4422534
time: 5.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4422662, upper bound: 36.4422534
time: 5.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 139

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5027284, upper bound: 36.5027138
time: 6.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5027237, upper bound: 36.5027149
time: 7.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 139

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5009177, upper bound: 36.5009187
time: 8.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5009151, upper bound: 36.5009208
time: 10.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4950995, upper bound: 36.4951178
time: 13.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4950995, upper bound: 36.4951178
time: 7.97 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 22.63 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 22.63
Output dim: 6, lower bound: -36.4422662, upper bound: 36.4422534
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 22.63
Output dim: 6, lower bound: -36.4422662, upper bound: 36.4422534
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 6, lower bound: -36.5027284, upper bound: 36.5027138
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 6, lower bound: -36.5027237, upper bound: 36.5027149
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 6, lower bound: -36.5009177, upper bound: 36.5009187
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 6, lower bound: -36.5009151, upper bound: 36.5009208
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 6, lower bound: -36.4950995, upper bound: 36.4951178
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 6, lower bound: -36.4950995, upper bound: 36.4951178

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4993095, upper bound: 36.4992827
time: 7.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4992990, upper bound: 36.4992899
time: 7.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4941293, upper bound: 36.4941327
time: 7.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4941293, upper bound: 36.4941327
time: 7.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 163

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4993987, upper bound: 36.4993875
time: 8.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4993987, upper bound: 36.4993875
time: 6.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4406699, upper bound: 36.4406628
time: 11.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4406699, upper bound: 36.4406628
time: 12.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 130

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4733731, upper bound: 36.4733815
time: 6.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4733731, upper bound: 36.4733815
time: 6.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 163

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4826058, upper bound: 36.4826101
time: 8.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4826058, upper bound: 36.4826101
time: 5.57 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 20.39 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.39
Output dim: 6, lower bound: -36.4993095, upper bound: 36.4992827
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.39
Output dim: 6, lower bound: -36.4992990, upper bound: 36.4992899
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.39
Output dim: 6, lower bound: -36.4941293, upper bound: 36.4941327
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.39
Output dim: 6, lower bound: -36.4941293, upper bound: 36.4941327
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.39
Output dim: 6, lower bound: -36.4993987, upper bound: 36.4993875
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.39
Output dim: 6, lower bound: -36.4993987, upper bound: 36.4993875
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 20.39
Output dim: 6, lower bound: -36.4406699, upper bound: 36.4406628
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 20.39
Output dim: 6, lower bound: -36.4406699, upper bound: 36.4406628
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 20.39
Output dim: 6, lower bound: -36.4733731, upper bound: 36.4733815
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 20.39
Output dim: 6, lower bound: -36.4733731, upper bound: 36.4733815
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 20.39
Output dim: 6, lower bound: -36.4826058, upper bound: 36.4826101
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 20.39
Output dim: 6, lower bound: -36.4826058, upper bound: 36.4826101

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4732456, upper bound: 36.4732384
time: 8.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4732456, upper bound: 36.4732384
time: 11.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4987271, upper bound: 36.4987116
time: 5.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4987197, upper bound: 36.4987142
time: 7.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 163

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4374445, upper bound: 36.4374394
time: 7.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4374445, upper bound: 36.4374394
time: 7.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4699076, upper bound: 36.4699021
time: 10.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4699076, upper bound: 36.4699021
time: 9.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4981115, upper bound: 36.4981009
time: 6.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4981023, upper bound: 36.4981102
time: 11.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4976725, upper bound: 36.4976682
time: 5.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4976725, upper bound: 36.4976682
time: 6.15 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 12.93 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 12.93
Output dim: 6, lower bound: -36.4732456, upper bound: 36.4732384
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 12.93
Output dim: 6, lower bound: -36.4732456, upper bound: 36.4732384
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.93
Output dim: 6, lower bound: -36.4987271, upper bound: 36.4987116
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.93
Output dim: 6, lower bound: -36.4987197, upper bound: 36.4987142
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 12.93
Output dim: 6, lower bound: -36.4374445, upper bound: 36.4374394
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 12.93
Output dim: 6, lower bound: -36.4374445, upper bound: 36.4374394
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 12.93
Output dim: 6, lower bound: -36.4699076, upper bound: 36.4699021
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 12.93
Output dim: 6, lower bound: -36.4699076, upper bound: 36.4699021
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.93
Output dim: 6, lower bound: -36.4981115, upper bound: 36.4981009
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.93
Output dim: 6, lower bound: -36.4981023, upper bound: 36.4981102
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.93
Output dim: 6, lower bound: -36.4976725, upper bound: 36.4976682
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.93
Output dim: 6, lower bound: -36.4976725, upper bound: 36.4976682

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4897968, upper bound: 36.4897925
time: 5.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4897931, upper bound: 36.4897925
time: 6.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4971223, upper bound: 36.4971409
time: 14.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4971223, upper bound: 36.4971409
time: 29.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4900309, upper bound: 36.4900050
time: 7.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4900309, upper bound: 36.4900050
time: 9.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4686899, upper bound: 36.4687015
time: 23.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4686899, upper bound: 36.4687015
time: 23.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4882164, upper bound: 36.4882045
time: 5.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4882164, upper bound: 36.4882045
time: 8.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4942731, upper bound: 36.4942724
time: 6.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4942697, upper bound: 36.4942785
time: 6.53 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 14.07 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 14.07
Output dim: 6, lower bound: -36.4897968, upper bound: 36.4897925
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 14.07
Output dim: 6, lower bound: -36.4897931, upper bound: 36.4897925
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.07
Output dim: 6, lower bound: -36.4971223, upper bound: 36.4971409
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.07
Output dim: 6, lower bound: -36.4971223, upper bound: 36.4971409
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.07
Output dim: 6, lower bound: -36.4900309, upper bound: 36.4900050
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.07
Output dim: 6, lower bound: -36.4900309, upper bound: 36.4900050
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 14.07
Output dim: 6, lower bound: -36.4686899, upper bound: 36.4687015
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 14.07
Output dim: 6, lower bound: -36.4686899, upper bound: 36.4687015
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 14.07
Output dim: 6, lower bound: -36.4882164, upper bound: 36.4882045
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 14.07
Output dim: 6, lower bound: -36.4882164, upper bound: 36.4882045
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.07
Output dim: 6, lower bound: -36.4942731, upper bound: 36.4942724
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.07
Output dim: 6, lower bound: -36.4942697, upper bound: 36.4942785

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4799891, upper bound: 36.4799773
time: 6.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4799891, upper bound: 36.4799773
time: 5.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4685030, upper bound: 36.4685167
time: 15.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4685030, upper bound: 36.4685167
time: 6.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 139

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4897472, upper bound: 36.4897166
time: 5.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4897444, upper bound: 36.4897164
time: 5.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4880674, upper bound: 36.4880345
time: 6.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4880674, upper bound: 36.4880345
time: 43.72 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 54.29 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 54.29
Output dim: 6, lower bound: -36.4799891, upper bound: 36.4799773
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 54.29
Output dim: 6, lower bound: -36.4799891, upper bound: 36.4799773
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 54.29
Output dim: 6, lower bound: -36.4685030, upper bound: 36.4685167
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 54.29
Output dim: 6, lower bound: -36.4685030, upper bound: 36.4685167
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 54.29
Output dim: 6, lower bound: -36.4897472, upper bound: 36.4897166
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 54.29
Output dim: 6, lower bound: -36.4897444, upper bound: 36.4897164
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 54.29
Output dim: 6, lower bound: -36.4880674, upper bound: 36.4880345
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 54.29
Output dim: 6, lower bound: -36.4880674, upper bound: 36.4880345
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 54.29
Output dim: 6, lower bound: -36.4942731, upper bound: 36.4942724
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 54.29
Output dim: 6, lower bound: -36.4942697, upper bound: 36.4942785
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=47.04602813720703
rel_dist={6: [-36.503194103113984, 36.50319409863076]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5019004, upper bound: 36.5018910
time: 7.60 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5018910, upper bound: 36.5019004
time: 6.81 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.43 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 14.43
Output dim: 6, lower bound: -36.5019004, upper bound: 36.5018910
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 14.43
Output dim: 6, lower bound: -36.5018910, upper bound: 36.5019004

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 139

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5019004, upper bound: 36.5018910
time: 7.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5019004, upper bound: 36.5018910
time: 6.29 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4944669, upper bound: 36.4944746
time: 8.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4944669, upper bound: 36.4944746
time: 7.99 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 17.04 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 17.04
Output dim: 6, lower bound: -36.5019004, upper bound: 36.5018910
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 17.04
Output dim: 6, lower bound: -36.5019004, upper bound: 36.5018910
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 17.04
Output dim: 6, lower bound: -36.4944669, upper bound: 36.4944746
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 17.04
Output dim: 6, lower bound: -36.4944669, upper bound: 36.4944746

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4654022, upper bound: 36.4654027
time: 5.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4654022, upper bound: 36.4654027
time: 5.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4995079, upper bound: 36.4995116
time: 9.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4995233, upper bound: 36.4994965
time: 7.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4712376, upper bound: 36.4712419
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4712376, upper bound: 36.4712419
time: 4.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4918116, upper bound: 36.4918163
time: 16.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4918116, upper bound: 36.4918163
time: 9.59 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 26.68 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 26.68
Output dim: 6, lower bound: -36.4654022, upper bound: 36.4654027
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 26.68
Output dim: 6, lower bound: -36.4654022, upper bound: 36.4654027
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.68
Output dim: 6, lower bound: -36.4995079, upper bound: 36.4995116
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.68
Output dim: 6, lower bound: -36.4995233, upper bound: 36.4994965
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 26.68
Output dim: 6, lower bound: -36.4712376, upper bound: 36.4712419
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 26.68
Output dim: 6, lower bound: -36.4712376, upper bound: 36.4712419
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.68
Output dim: 6, lower bound: -36.4918116, upper bound: 36.4918163
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.68
Output dim: 6, lower bound: -36.4918116, upper bound: 36.4918163

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4982004, upper bound: 36.4981974
time: 7.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4982004, upper bound: 36.4981974
time: 8.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4982027, upper bound: 36.4981939
time: 7.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4982027, upper bound: 36.4981939
time: 43.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4856935, upper bound: 36.4856970
time: 7.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4856935, upper bound: 36.4856970
time: 7.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 201

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4509754, upper bound: 36.4509754
time: 7.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4509754, upper bound: 36.4509754
time: 7.06 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 15.08 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.08
Output dim: 6, lower bound: -36.4982004, upper bound: 36.4981974
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.08
Output dim: 6, lower bound: -36.4982004, upper bound: 36.4981974
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.08
Output dim: 6, lower bound: -36.4982027, upper bound: 36.4981939
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.08
Output dim: 6, lower bound: -36.4982027, upper bound: 36.4981939
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 15.08
Output dim: 6, lower bound: -36.4856935, upper bound: 36.4856970
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 15.08
Output dim: 6, lower bound: -36.4856935, upper bound: 36.4856970
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 15.08
Output dim: 6, lower bound: -36.4509754, upper bound: 36.4509754
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 15.08
Output dim: 6, lower bound: -36.4509754, upper bound: 36.4509754

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4907356, upper bound: 36.4907299
time: 7.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4907356, upper bound: 36.4907299
time: 6.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4656996, upper bound: 36.4656999
time: 16.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4656996, upper bound: 36.4656999
time: 16.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 139

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4978918, upper bound: 36.4978782
time: 6.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4978917, upper bound: 36.4978781
time: 5.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4918455, upper bound: 36.4918357
time: 27.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4918455, upper bound: 36.4918357
time: 34.04 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 62.94 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 62.94
Output dim: 6, lower bound: -36.4907356, upper bound: 36.4907299
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 62.94
Output dim: 6, lower bound: -36.4907356, upper bound: 36.4907299
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 62.94
Output dim: 6, lower bound: -36.4656996, upper bound: 36.4656999
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 62.94
Output dim: 6, lower bound: -36.4656996, upper bound: 36.4656999
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 62.94
Output dim: 6, lower bound: -36.4978918, upper bound: 36.4978782
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 62.94
Output dim: 6, lower bound: -36.4978917, upper bound: 36.4978781
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 62.94
Output dim: 6, lower bound: -36.4918455, upper bound: 36.4918357
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 62.94
Output dim: 6, lower bound: -36.4918455, upper bound: 36.4918357

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4907356, upper bound: 36.4907295
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4907354, upper bound: 36.4907299
time: 12.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4840714, upper bound: 36.4840708
time: 10.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4840714, upper bound: 36.4840708
time: 23.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 201

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4679684, upper bound: 36.4679641
time: 6.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4679684, upper bound: 36.4679649
time: 6.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4961090, upper bound: 36.4960999
time: 6.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4960995, upper bound: 36.4961014
time: 17.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 193

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4796455, upper bound: 36.4796131
time: 6.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4796455, upper bound: 36.4796131
time: 5.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4847038, upper bound: 36.4846879
time: 16.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4847038, upper bound: 36.4846879
time: 8.02 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 25.32 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 25.32
Output dim: 6, lower bound: -36.4907356, upper bound: 36.4907295
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 25.32
Output dim: 6, lower bound: -36.4907354, upper bound: 36.4907299
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 25.32
Output dim: 6, lower bound: -36.4840714, upper bound: 36.4840708
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 25.32
Output dim: 6, lower bound: -36.4840714, upper bound: 36.4840708
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 25.32
Output dim: 6, lower bound: -36.4679684, upper bound: 36.4679641
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 25.32
Output dim: 6, lower bound: -36.4679684, upper bound: 36.4679649
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 25.32
Output dim: 6, lower bound: -36.4961090, upper bound: 36.4960999
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 25.32
Output dim: 6, lower bound: -36.4960995, upper bound: 36.4961014
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 25.32
Output dim: 6, lower bound: -36.4796455, upper bound: 36.4796131
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 25.32
Output dim: 6, lower bound: -36.4796455, upper bound: 36.4796131
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 25.32
Output dim: 6, lower bound: -36.4847038, upper bound: 36.4846879
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 25.32
Output dim: 6, lower bound: -36.4847038, upper bound: 36.4846879

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4845683, upper bound: 36.4845565
time: 6.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4845683, upper bound: 36.4845565
time: 9.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4673121, upper bound: 36.4673114
time: 6.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4673121, upper bound: 36.4673114
time: 6.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 201

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4660657, upper bound: 36.4660597
time: 8.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4660657, upper bound: 36.4660597
time: 7.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4945925, upper bound: 36.4945803
time: 8.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4945925, upper bound: 36.4945803
time: 7.11 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 18.19 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 18.19
Output dim: 6, lower bound: -36.4845683, upper bound: 36.4845565
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 18.19
Output dim: 6, lower bound: -36.4845683, upper bound: 36.4845565
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 18.19
Output dim: 6, lower bound: -36.4673121, upper bound: 36.4673114
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 18.19
Output dim: 6, lower bound: -36.4673121, upper bound: 36.4673114
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 18.19
Output dim: 6, lower bound: -36.4660657, upper bound: 36.4660597
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 18.19
Output dim: 6, lower bound: -36.4660657, upper bound: 36.4660597
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 18.19
Output dim: 6, lower bound: -36.4945925, upper bound: 36.4945803
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 18.19
Output dim: 6, lower bound: -36.4945925, upper bound: 36.4945803

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 163

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4451935, upper bound: 36.4451870
time: 13.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4451935, upper bound: 36.4451870
time: 13.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4877422, upper bound: 36.4877268
time: 9.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4877423, upper bound: 36.4877268
time: 16.09 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 27.00 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 27.00
Output dim: 6, lower bound: -36.4451935, upper bound: 36.4451870
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 27.00
Output dim: 6, lower bound: -36.4451935, upper bound: 36.4451870
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 27.00
Output dim: 6, lower bound: -36.4877422, upper bound: 36.4877268
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 27.00
Output dim: 6, lower bound: -36.4877423, upper bound: 36.4877268
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=47.04602813720703
rel_dist={6: [-36.50308451104452, 36.50308451104452]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4670979, upper bound: 36.4670979
time: 7.35 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4670979, upper bound: 36.4670979
time: 8.82 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 16.18 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 16.18
Output dim: 6, lower bound: -36.4670979, upper bound: 36.4670979
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 16.18
Output dim: 6, lower bound: -36.4670979, upper bound: 36.4670979
Binary search (step 2): status=Status.VERIFIED, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=47.04602813720703
rel_dist={6: [-36.50313378314387, 36.50313405923052]}

## Binary search (step 3) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5011796, upper bound: 36.5011796
time: 6.52 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5011796, upper bound: 36.5011796
time: 5.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.82 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.82
Output dim: 6, lower bound: -36.5011796, upper bound: 36.5011796
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.82
Output dim: 6, lower bound: -36.5011796, upper bound: 36.5011796

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4820397, upper bound: 36.4820397
time: 7.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4820397, upper bound: 36.4820397
time: 13.19 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4939579, upper bound: 36.4939579
time: 5.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4939579, upper bound: 36.4939579
time: 7.46 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 13.73 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 13.73
Output dim: 6, lower bound: -36.4820397, upper bound: 36.4820397
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 13.73
Output dim: 6, lower bound: -36.4820397, upper bound: 36.4820397
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.73
Output dim: 6, lower bound: -36.4939579, upper bound: 36.4939579
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.73
Output dim: 6, lower bound: -36.4939579, upper bound: 36.4939579

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 130

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4736232, upper bound: 36.4736232
time: 6.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4736232, upper bound: 36.4736232
time: 5.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748
1: -19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108
2: -26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393
3: -30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846
4: -31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904
5: -27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109
6: -31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243
7: -23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175
8: -34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087
9: -22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4694045, upper bound: 36.4694045
time: 7.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4694045, upper bound: 36.4694045
time: 6.68 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 14.90 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 14.90
Output dim: 6, lower bound: -36.4736232, upper bound: 36.4736232
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 14.90
Output dim: 6, lower bound: -36.4736232, upper bound: 36.4736232
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 14.90
Output dim: 6, lower bound: -36.4694045, upper bound: 36.4694045
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 14.90
Output dim: 6, lower bound: -36.4694045, upper bound: 36.4694045
Binary search (step 3): status=Status.VERIFIED, k_low=5, k_high=5, k_mid=5, eps_mid=0.0195312, abs_max=47.04602813720703
rel_dist={6: [-36.5031791348119, 36.50317887973097]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01953125
execution time: 1383.47 seconds
