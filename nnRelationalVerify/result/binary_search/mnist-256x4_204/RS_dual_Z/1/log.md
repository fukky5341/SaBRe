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
execution time: IAR + LP analysis = 1.06 + 8.97 = 10.03 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -36.5032747, upper bound: 36.5032750


# Binary Search by BASE starts (time budget: 2689.97 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=47.04602813720703
rel_dist={6: [-36.503194103113984, 36.50319409863076]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=47.04602813720703
rel_dist={6: [-36.50308451104452, 36.50308451104452]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=47.04602813720703
rel_dist={6: [-36.50297363759564, 36.50297363759566]}

## Binary Search Result
Binary search time: 36.58 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 2653.39 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5031941, upper bound: 36.5031808
time: 6.61 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5031805, upper bound: 36.5031944
time: 6.39 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 13.11 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 13.11
Output dim: 6, lower bound: -36.5031941, upper bound: 36.5031808
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 13.11
Output dim: 6, lower bound: -36.5031805, upper bound: 36.5031944

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
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4998232, upper bound: 36.4998225
time: 7.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4998251, upper bound: 36.4998198
time: 7.01 seconds

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
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4998198, upper bound: 36.4998251
time: 3.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4998225, upper bound: 36.4998232
time: 4.20 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 8.76 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 8.76
Output dim: 6, lower bound: -36.4998232, upper bound: 36.4998225
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 8.76
Output dim: 6, lower bound: -36.4998251, upper bound: 36.4998198
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 8.76
Output dim: 6, lower bound: -36.4998198, upper bound: 36.4998251
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 8.76
Output dim: 6, lower bound: -36.4998225, upper bound: 36.4998232

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
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4869757, upper bound: 36.4869665
time: 4.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4869757, upper bound: 36.4869665
time: 3.65 seconds

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

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4869774, upper bound: 36.4869638
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4869774, upper bound: 36.4869638
time: 5.12 seconds

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
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4869638, upper bound: 36.4869774
time: 6.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4869638, upper bound: 36.4869774
time: 13.09 seconds

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
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4869665, upper bound: 36.4869757
time: 8.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4869665, upper bound: 36.4869757
time: 5.35 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 15.10 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 15.10
Output dim: 6, lower bound: -36.4869757, upper bound: 36.4869665
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 15.10
Output dim: 6, lower bound: -36.4869757, upper bound: 36.4869665
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 15.10
Output dim: 6, lower bound: -36.4869774, upper bound: 36.4869638
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 15.10
Output dim: 6, lower bound: -36.4869774, upper bound: 36.4869638
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 15.10
Output dim: 6, lower bound: -36.4869638, upper bound: 36.4869774
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 15.10
Output dim: 6, lower bound: -36.4869638, upper bound: 36.4869774
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 15.10
Output dim: 6, lower bound: -36.4869665, upper bound: 36.4869757
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 15.10
Output dim: 6, lower bound: -36.4869665, upper bound: 36.4869757
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=47.04602813720703
rel_dist={6: [-36.503194103113984, 36.50319409863076]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5032365, upper bound: 36.5032191
time: 6.42 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5032192, upper bound: 36.5032366
time: 8.87 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 15.40 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 15.40
Output dim: 6, lower bound: -36.5032365, upper bound: 36.5032191
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 15.40
Output dim: 6, lower bound: -36.5032192, upper bound: 36.5032366

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
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4998681, upper bound: 36.4998663
time: 5.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4998703, upper bound: 36.4998634
time: 4.46 seconds

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
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4998634, upper bound: 36.4998703
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4998663, upper bound: 36.4998681
time: 7.89 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 14.16 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.16
Output dim: 6, lower bound: -36.4998681, upper bound: 36.4998663
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.16
Output dim: 6, lower bound: -36.4998703, upper bound: 36.4998634
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.16
Output dim: 6, lower bound: -36.4998634, upper bound: 36.4998703
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.16
Output dim: 6, lower bound: -36.4998663, upper bound: 36.4998681

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
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4870176, upper bound: 36.4870027
time: 5.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4870176, upper bound: 36.4870027
time: 6.79 seconds

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
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4870198, upper bound: 36.4870000
time: 7.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4870198, upper bound: 36.4870000
time: 8.40 seconds

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
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4870000, upper bound: 36.4870198
time: 19.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4870000, upper bound: 36.4870198
time: 20.67 seconds

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
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4870027, upper bound: 36.4870176
time: 6.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4870027, upper bound: 36.4870176
time: 4.19 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 12.16 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 12.16
Output dim: 6, lower bound: -36.4870176, upper bound: 36.4870027
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 12.16
Output dim: 6, lower bound: -36.4870176, upper bound: 36.4870027
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 12.16
Output dim: 6, lower bound: -36.4870198, upper bound: 36.4870000
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 12.16
Output dim: 6, lower bound: -36.4870198, upper bound: 36.4870000
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 12.16
Output dim: 6, lower bound: -36.4870000, upper bound: 36.4870198
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 12.16
Output dim: 6, lower bound: -36.4870000, upper bound: 36.4870198
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 12.16
Output dim: 6, lower bound: -36.4870027, upper bound: 36.4870176
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 12.16
Output dim: 6, lower bound: -36.4870027, upper bound: 36.4870176
Binary search (step 1): status=Status.VERIFIED, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=47.04602813720703
rel_dist={6: [-36.50323657019504, 36.50323656135174]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5032622, upper bound: 36.5032440
time: 7.52 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5032440, upper bound: 36.5032624
time: 35.66 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 43.28 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 43.28
Output dim: 6, lower bound: -36.5032622, upper bound: 36.5032440
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 43.28
Output dim: 6, lower bound: -36.5032440, upper bound: 36.5032624

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

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4998977, upper bound: 36.4998946
time: 6.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4998994, upper bound: 36.4998917
time: 4.76 seconds

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

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4998917, upper bound: 36.4998994
time: 4.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4998946, upper bound: 36.4998977
time: 8.01 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 13.54 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.54
Output dim: 6, lower bound: -36.4998977, upper bound: 36.4998946
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.54
Output dim: 6, lower bound: -36.4998994, upper bound: 36.4998917
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.54
Output dim: 6, lower bound: -36.4998917, upper bound: 36.4998994
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.54
Output dim: 6, lower bound: -36.4998946, upper bound: 36.4998977

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

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4870452, upper bound: 36.4870257
time: 6.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4870452, upper bound: 36.4870257
time: 6.12 seconds

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
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4870476, upper bound: 36.4870228
time: 5.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4870476, upper bound: 36.4870228
time: 4.53 seconds

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

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4870228, upper bound: 36.4870476
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4870228, upper bound: 36.4870476
time: 6.56 seconds

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
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4870257, upper bound: 36.4870452
time: 6.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4870257, upper bound: 36.4870452
time: 6.23 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 13.40 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 13.40
Output dim: 6, lower bound: -36.4870452, upper bound: 36.4870257
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 13.40
Output dim: 6, lower bound: -36.4870452, upper bound: 36.4870257
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 13.40
Output dim: 6, lower bound: -36.4870476, upper bound: 36.4870228
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 13.40
Output dim: 6, lower bound: -36.4870476, upper bound: 36.4870228
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 13.40
Output dim: 6, lower bound: -36.4870228, upper bound: 36.4870476
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 13.40
Output dim: 6, lower bound: -36.4870228, upper bound: 36.4870476
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 13.40
Output dim: 6, lower bound: -36.4870257, upper bound: 36.4870452
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 13.40
Output dim: 6, lower bound: -36.4870257, upper bound: 36.4870452
Binary search (step 2): status=Status.VERIFIED, k_low=10, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=47.04602813720703
rel_dist={6: [-36.503262448646396, 36.50326217370076]}

## Binary search (step 3) starts
Candidate k: 12, corresponding eps: 0.0468750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5032747, upper bound: 36.5032566
time: 6.31 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5032566, upper bound: 36.5032747
time: 5.09 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.51 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.51
Output dim: 6, lower bound: -36.5032747, upper bound: 36.5032566
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.51
Output dim: 6, lower bound: -36.5032566, upper bound: 36.5032747

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
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4999125, upper bound: 36.4999085
time: 5.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4999136, upper bound: 36.4999055
time: 5.06 seconds

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
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4999055, upper bound: 36.4999136
time: 6.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4999085, upper bound: 36.4999124
time: 5.11 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 12.54 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 12.54
Output dim: 6, lower bound: -36.4999125, upper bound: 36.4999085
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 12.54
Output dim: 6, lower bound: -36.4999136, upper bound: 36.4999055
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 12.54
Output dim: 6, lower bound: -36.4999055, upper bound: 36.4999136
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 12.54
Output dim: 6, lower bound: -36.4999085, upper bound: 36.4999124

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
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4870589, upper bound: 36.4870372
time: 5.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4870589, upper bound: 36.4870372
time: 6.92 seconds

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
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4870614, upper bound: 36.4870342
time: 14.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4870614, upper bound: 36.4870342
time: 18.97 seconds

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

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4870342, upper bound: 36.4870614
time: 11.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4870342, upper bound: 36.4870614
time: 9.30 seconds

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
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4870372, upper bound: 36.4870589
time: 5.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4870372, upper bound: 36.4870589
time: 6.61 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 12.94 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 12.94
Output dim: 6, lower bound: -36.4870589, upper bound: 36.4870372
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 12.94
Output dim: 6, lower bound: -36.4870589, upper bound: 36.4870372
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 12.94
Output dim: 6, lower bound: -36.4870614, upper bound: 36.4870342
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 12.94
Output dim: 6, lower bound: -36.4870614, upper bound: 36.4870342
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 12.94
Output dim: 6, lower bound: -36.4870342, upper bound: 36.4870614
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 12.94
Output dim: 6, lower bound: -36.4870342, upper bound: 36.4870614
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 12.94
Output dim: 6, lower bound: -36.4870372, upper bound: 36.4870589
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 12.94
Output dim: 6, lower bound: -36.4870372, upper bound: 36.4870589
Binary search (step 3): status=Status.VERIFIED, k_low=12, k_high=12, k_mid=12, eps_mid=0.0468750, abs_max=47.04602813720703
rel_dist={6: [-36.503274688156715, 36.50327495642699]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.046875
execution time: 507.85 seconds
