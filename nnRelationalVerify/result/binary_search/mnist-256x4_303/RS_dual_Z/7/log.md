## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 202.485480649
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138)
1: (-111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612)
2: (-144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239)
3: (-152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125)
4: (-140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590)
5: (-125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873)
6: (-120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172)
7: (-131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144)
8: (-159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220)
9: (-119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713)

## BASE Result
execution time: IAR + LP analysis = 1.04 + 9.76 = 10.80 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -202.6091898, upper bound: 202.6091898


# Binary Search by BASE starts (time budget: 2689.20 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=203.5233612060547
rel_dist={1: [-202.60902678108835, 202.60902678108835]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=203.5233612060547
rel_dist={1: [-202.60871310808878, 202.60871310808875]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=203.5233612060547
rel_dist={1: [-202.608202498089, 202.60820249808899]}

## Binary Search Result
Binary search time: 37.51 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 2651.68 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5559891, upper bound: 202.5559891
time: 5.62 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5559891, upper bound: 202.5559891
time: 5.68 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.41 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.41
Output dim: 1, lower bound: -202.5559891, upper bound: 202.5559891
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.41
Output dim: 1, lower bound: -202.5559891, upper bound: 202.5559891

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4960197, upper bound: 202.4960197
time: 4.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4960197, upper bound: 202.4960197
time: 4.25 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4960197, upper bound: 202.4960197
time: 4.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4960197, upper bound: 202.4960197
time: 4.22 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 11.58 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 11.58
Output dim: 1, lower bound: -202.4960197, upper bound: 202.4960197
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 11.58
Output dim: 1, lower bound: -202.4960197, upper bound: 202.4960197
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 11.58
Output dim: 1, lower bound: -202.4960197, upper bound: 202.4960197
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 11.58
Output dim: 1, lower bound: -202.4960197, upper bound: 202.4960197

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4930079, upper bound: 202.4930095
time: 7.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4930095, upper bound: 202.4930079
time: 6.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4930079, upper bound: 202.4930095
time: 6.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4930095, upper bound: 202.4930079
time: 6.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4930079, upper bound: 202.4930095
time: 6.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4930095, upper bound: 202.4930079
time: 6.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4930079, upper bound: 202.4930095
time: 7.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4930095, upper bound: 202.4930079
time: 6.14 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 16.41 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.41
Output dim: 1, lower bound: -202.4930079, upper bound: 202.4930095
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.41
Output dim: 1, lower bound: -202.4930095, upper bound: 202.4930079
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.41
Output dim: 1, lower bound: -202.4930079, upper bound: 202.4930095
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.41
Output dim: 1, lower bound: -202.4930095, upper bound: 202.4930079
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.41
Output dim: 1, lower bound: -202.4930079, upper bound: 202.4930095
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.41
Output dim: 1, lower bound: -202.4930095, upper bound: 202.4930079
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.41
Output dim: 1, lower bound: -202.4930079, upper bound: 202.4930095
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.41
Output dim: 1, lower bound: -202.4930095, upper bound: 202.4930079

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4894948, upper bound: 202.4895045
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4894949, upper bound: 202.4895047
time: 5.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895047, upper bound: 202.4894949
time: 5.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895045, upper bound: 202.4894948
time: 5.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4894948, upper bound: 202.4895045
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4894949, upper bound: 202.4895047
time: 4.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895047, upper bound: 202.4894949
time: 5.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895045, upper bound: 202.4894948
time: 4.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4894948, upper bound: 202.4895045
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4894949, upper bound: 202.4895047
time: 5.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895047, upper bound: 202.4894949
time: 5.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895045, upper bound: 202.4894948
time: 5.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4894948, upper bound: 202.4895045
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4894949, upper bound: 202.4895047
time: 4.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895047, upper bound: 202.4894949
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895045, upper bound: 202.4894948
time: 5.02 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 15.06 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.06
Output dim: 1, lower bound: -202.4894948, upper bound: 202.4895045
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.06
Output dim: 1, lower bound: -202.4894949, upper bound: 202.4895047
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.06
Output dim: 1, lower bound: -202.4895047, upper bound: 202.4894949
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.06
Output dim: 1, lower bound: -202.4895045, upper bound: 202.4894948
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.06
Output dim: 1, lower bound: -202.4894948, upper bound: 202.4895045
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.06
Output dim: 1, lower bound: -202.4894949, upper bound: 202.4895047
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.06
Output dim: 1, lower bound: -202.4895047, upper bound: 202.4894949
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.06
Output dim: 1, lower bound: -202.4895045, upper bound: 202.4894948
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.06
Output dim: 1, lower bound: -202.4894948, upper bound: 202.4895045
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.06
Output dim: 1, lower bound: -202.4894949, upper bound: 202.4895047
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.06
Output dim: 1, lower bound: -202.4895047, upper bound: 202.4894949
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.06
Output dim: 1, lower bound: -202.4895045, upper bound: 202.4894948
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.06
Output dim: 1, lower bound: -202.4894948, upper bound: 202.4895045
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.06
Output dim: 1, lower bound: -202.4894949, upper bound: 202.4895047
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.06
Output dim: 1, lower bound: -202.4895047, upper bound: 202.4894949
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.06
Output dim: 1, lower bound: -202.4895045, upper bound: 202.4894948

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854625, upper bound: 202.4854641
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854667, upper bound: 202.4854618
time: 5.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854625, upper bound: 202.4854674
time: 4.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854634, upper bound: 202.4854604
time: 4.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854604, upper bound: 202.4854634
time: 5.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854674, upper bound: 202.4854625
time: 4.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854618, upper bound: 202.4854667
time: 5.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854641, upper bound: 202.4854625
time: 4.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 158

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854625, upper bound: 202.4854641
time: 4.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854667, upper bound: 202.4854618
time: 5.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854625, upper bound: 202.4854674
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854634, upper bound: 202.4854604
time: 4.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854604, upper bound: 202.4854634
time: 5.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854674, upper bound: 202.4854625
time: 4.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854618, upper bound: 202.4854667
time: 5.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854641, upper bound: 202.4854625
time: 4.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854625, upper bound: 202.4854641
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854667, upper bound: 202.4854618
time: 5.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854625, upper bound: 202.4854674
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854634, upper bound: 202.4854604
time: 4.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854604, upper bound: 202.4854634
time: 5.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854674, upper bound: 202.4854625
time: 4.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854618, upper bound: 202.4854667
time: 5.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854641, upper bound: 202.4854625
time: 4.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854625, upper bound: 202.4854641
time: 4.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854667, upper bound: 202.4854618
time: 5.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854625, upper bound: 202.4854674
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854634, upper bound: 202.4854604
time: 4.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854604, upper bound: 202.4854634
time: 5.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854674, upper bound: 202.4854625
time: 4.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854618, upper bound: 202.4854667
time: 5.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854641, upper bound: 202.4854625
time: 4.86 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 17.33 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 17.33
Output dim: 1, lower bound: -202.4854625, upper bound: 202.4854641
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 17.33
Output dim: 1, lower bound: -202.4854667, upper bound: 202.4854618
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 17.33
Output dim: 1, lower bound: -202.4854625, upper bound: 202.4854674
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 17.33
Output dim: 1, lower bound: -202.4854634, upper bound: 202.4854604
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 17.33
Output dim: 1, lower bound: -202.4854604, upper bound: 202.4854634
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 17.33
Output dim: 1, lower bound: -202.4854674, upper bound: 202.4854625
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 17.33
Output dim: 1, lower bound: -202.4854618, upper bound: 202.4854667
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 17.33
Output dim: 1, lower bound: -202.4854641, upper bound: 202.4854625
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 17.33
Output dim: 1, lower bound: -202.4854625, upper bound: 202.4854641
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 17.33
Output dim: 1, lower bound: -202.4854667, upper bound: 202.4854618
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 17.33
Output dim: 1, lower bound: -202.4854625, upper bound: 202.4854674
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 17.33
Output dim: 1, lower bound: -202.4854634, upper bound: 202.4854604
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 17.33
Output dim: 1, lower bound: -202.4854604, upper bound: 202.4854634
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 17.33
Output dim: 1, lower bound: -202.4854674, upper bound: 202.4854625
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 17.33
Output dim: 1, lower bound: -202.4854618, upper bound: 202.4854667
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 17.33
Output dim: 1, lower bound: -202.4854641, upper bound: 202.4854625
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 17.33
Output dim: 1, lower bound: -202.4854625, upper bound: 202.4854641
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 17.33
Output dim: 1, lower bound: -202.4854667, upper bound: 202.4854618
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 17.33
Output dim: 1, lower bound: -202.4854625, upper bound: 202.4854674
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 17.33
Output dim: 1, lower bound: -202.4854634, upper bound: 202.4854604
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 17.33
Output dim: 1, lower bound: -202.4854604, upper bound: 202.4854634
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 17.33
Output dim: 1, lower bound: -202.4854674, upper bound: 202.4854625
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 17.33
Output dim: 1, lower bound: -202.4854618, upper bound: 202.4854667
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 17.33
Output dim: 1, lower bound: -202.4854641, upper bound: 202.4854625
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 17.33
Output dim: 1, lower bound: -202.4854625, upper bound: 202.4854641
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 17.33
Output dim: 1, lower bound: -202.4854667, upper bound: 202.4854618
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 17.33
Output dim: 1, lower bound: -202.4854625, upper bound: 202.4854674
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 17.33
Output dim: 1, lower bound: -202.4854634, upper bound: 202.4854604
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 17.33
Output dim: 1, lower bound: -202.4854604, upper bound: 202.4854634
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 17.33
Output dim: 1, lower bound: -202.4854674, upper bound: 202.4854625
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 17.33
Output dim: 1, lower bound: -202.4854618, upper bound: 202.4854667
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 17.33
Output dim: 1, lower bound: -202.4854641, upper bound: 202.4854625
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=203.5233612060547
rel_dist={1: [-202.60902678108835, 202.60902678108835]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5560228, upper bound: 202.5560228
time: 4.98 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5560228, upper bound: 202.5560228
time: 4.97 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.06 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.06
Output dim: 1, lower bound: -202.5560228, upper bound: 202.5560228
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.06
Output dim: 1, lower bound: -202.5560228, upper bound: 202.5560228

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4960342, upper bound: 202.4960342
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4960342, upper bound: 202.4960342
time: 5.42 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4960342, upper bound: 202.4960342
time: 5.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4960342, upper bound: 202.4960342
time: 5.44 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 13.90 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.90
Output dim: 1, lower bound: -202.4960342, upper bound: 202.4960342
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.90
Output dim: 1, lower bound: -202.4960342, upper bound: 202.4960342
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.90
Output dim: 1, lower bound: -202.4960342, upper bound: 202.4960342
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.90
Output dim: 1, lower bound: -202.4960342, upper bound: 202.4960342

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4930195, upper bound: 202.4930218
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4930218, upper bound: 202.4930195
time: 5.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4930195, upper bound: 202.4930218
time: 5.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4930218, upper bound: 202.4930195
time: 5.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4930195, upper bound: 202.4930218
time: 5.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4930218, upper bound: 202.4930195
time: 5.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4930195, upper bound: 202.4930218
time: 5.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4930218, upper bound: 202.4930195
time: 5.04 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 13.08 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.08
Output dim: 1, lower bound: -202.4930195, upper bound: 202.4930218
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.08
Output dim: 1, lower bound: -202.4930218, upper bound: 202.4930195
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.08
Output dim: 1, lower bound: -202.4930195, upper bound: 202.4930218
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.08
Output dim: 1, lower bound: -202.4930218, upper bound: 202.4930195
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.08
Output dim: 1, lower bound: -202.4930195, upper bound: 202.4930218
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.08
Output dim: 1, lower bound: -202.4930218, upper bound: 202.4930195
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.08
Output dim: 1, lower bound: -202.4930195, upper bound: 202.4930218
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.08
Output dim: 1, lower bound: -202.4930218, upper bound: 202.4930195

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895029, upper bound: 202.4895167
time: 5.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895032, upper bound: 202.4895163
time: 5.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895163, upper bound: 202.4895032
time: 4.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895167, upper bound: 202.4895029
time: 4.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895029, upper bound: 202.4895167
time: 5.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895032, upper bound: 202.4895163
time: 5.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895163, upper bound: 202.4895032
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895167, upper bound: 202.4895029
time: 4.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895029, upper bound: 202.4895167
time: 5.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895032, upper bound: 202.4895163
time: 5.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895163, upper bound: 202.4895032
time: 4.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895167, upper bound: 202.4895029
time: 4.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895029, upper bound: 202.4895167
time: 5.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895032, upper bound: 202.4895163
time: 5.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895163, upper bound: 202.4895032
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895167, upper bound: 202.4895029
time: 4.46 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 14.50 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.50
Output dim: 1, lower bound: -202.4895029, upper bound: 202.4895167
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.50
Output dim: 1, lower bound: -202.4895032, upper bound: 202.4895163
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.50
Output dim: 1, lower bound: -202.4895163, upper bound: 202.4895032
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.50
Output dim: 1, lower bound: -202.4895167, upper bound: 202.4895029
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.50
Output dim: 1, lower bound: -202.4895029, upper bound: 202.4895167
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.50
Output dim: 1, lower bound: -202.4895032, upper bound: 202.4895163
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.50
Output dim: 1, lower bound: -202.4895163, upper bound: 202.4895032
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.50
Output dim: 1, lower bound: -202.4895167, upper bound: 202.4895029
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.50
Output dim: 1, lower bound: -202.4895029, upper bound: 202.4895167
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.50
Output dim: 1, lower bound: -202.4895032, upper bound: 202.4895163
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.50
Output dim: 1, lower bound: -202.4895163, upper bound: 202.4895032
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.50
Output dim: 1, lower bound: -202.4895167, upper bound: 202.4895029
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.50
Output dim: 1, lower bound: -202.4895029, upper bound: 202.4895167
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.50
Output dim: 1, lower bound: -202.4895032, upper bound: 202.4895163
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.50
Output dim: 1, lower bound: -202.4895163, upper bound: 202.4895032
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.50
Output dim: 1, lower bound: -202.4895167, upper bound: 202.4895029

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854698, upper bound: 202.4854727
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854760, upper bound: 202.4854681
time: 4.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854695, upper bound: 202.4854765
time: 5.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854715, upper bound: 202.4854671
time: 4.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854671, upper bound: 202.4854715
time: 4.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854765, upper bound: 202.4854695
time: 5.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854681, upper bound: 202.4854760
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854727, upper bound: 202.4854698
time: 5.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854698, upper bound: 202.4854727
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854760, upper bound: 202.4854681
time: 4.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854695, upper bound: 202.4854765
time: 5.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854715, upper bound: 202.4854671
time: 5.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854671, upper bound: 202.4854715
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854765, upper bound: 202.4854695
time: 5.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854681, upper bound: 202.4854760
time: 4.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854727, upper bound: 202.4854698
time: 5.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854698, upper bound: 202.4854727
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854760, upper bound: 202.4854681
time: 4.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854695, upper bound: 202.4854765
time: 5.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854715, upper bound: 202.4854671
time: 4.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854671, upper bound: 202.4854715
time: 4.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854765, upper bound: 202.4854695
time: 5.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854681, upper bound: 202.4854760
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854727, upper bound: 202.4854698
time: 5.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854698, upper bound: 202.4854727
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854760, upper bound: 202.4854681
time: 4.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854695, upper bound: 202.4854765
time: 5.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854715, upper bound: 202.4854671
time: 5.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854671, upper bound: 202.4854715
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854765, upper bound: 202.4854695
time: 5.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854681, upper bound: 202.4854760
time: 4.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854727, upper bound: 202.4854698
time: 5.04 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 16.98 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 1, lower bound: -202.4854698, upper bound: 202.4854727
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 1, lower bound: -202.4854760, upper bound: 202.4854681
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 1, lower bound: -202.4854695, upper bound: 202.4854765
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 1, lower bound: -202.4854715, upper bound: 202.4854671
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 1, lower bound: -202.4854671, upper bound: 202.4854715
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 1, lower bound: -202.4854765, upper bound: 202.4854695
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 1, lower bound: -202.4854681, upper bound: 202.4854760
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 1, lower bound: -202.4854727, upper bound: 202.4854698
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 1, lower bound: -202.4854698, upper bound: 202.4854727
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 1, lower bound: -202.4854760, upper bound: 202.4854681
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 1, lower bound: -202.4854695, upper bound: 202.4854765
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 1, lower bound: -202.4854715, upper bound: 202.4854671
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 1, lower bound: -202.4854671, upper bound: 202.4854715
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 1, lower bound: -202.4854765, upper bound: 202.4854695
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 1, lower bound: -202.4854681, upper bound: 202.4854760
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 1, lower bound: -202.4854727, upper bound: 202.4854698
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 1, lower bound: -202.4854698, upper bound: 202.4854727
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 1, lower bound: -202.4854760, upper bound: 202.4854681
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 1, lower bound: -202.4854695, upper bound: 202.4854765
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 1, lower bound: -202.4854715, upper bound: 202.4854671
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 1, lower bound: -202.4854671, upper bound: 202.4854715
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 1, lower bound: -202.4854765, upper bound: 202.4854695
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 1, lower bound: -202.4854681, upper bound: 202.4854760
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 1, lower bound: -202.4854727, upper bound: 202.4854698
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 1, lower bound: -202.4854698, upper bound: 202.4854727
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 1, lower bound: -202.4854760, upper bound: 202.4854681
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 1, lower bound: -202.4854695, upper bound: 202.4854765
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 1, lower bound: -202.4854715, upper bound: 202.4854671
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 1, lower bound: -202.4854671, upper bound: 202.4854715
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 1, lower bound: -202.4854765, upper bound: 202.4854695
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 1, lower bound: -202.4854681, upper bound: 202.4854760
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.98
Output dim: 1, lower bound: -202.4854727, upper bound: 202.4854698
Binary search (step 1): status=Status.VERIFIED, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=203.5233612060547
rel_dist={1: [-202.60911179232386, 202.60911178727918]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5560429, upper bound: 202.5560429
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5560429, upper bound: 202.5560429
time: 4.76 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 9.78 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 9.78
Output dim: 1, lower bound: -202.5560429, upper bound: 202.5560429
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 9.78
Output dim: 1, lower bound: -202.5560429, upper bound: 202.5560429

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4960439, upper bound: 202.4960439
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4960439, upper bound: 202.4960439
time: 5.12 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4960439, upper bound: 202.4960439
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4960439, upper bound: 202.4960439
time: 5.11 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 12.96 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 12.96
Output dim: 1, lower bound: -202.4960439, upper bound: 202.4960439
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 12.96
Output dim: 1, lower bound: -202.4960439, upper bound: 202.4960439
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 12.96
Output dim: 1, lower bound: -202.4960439, upper bound: 202.4960439
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 12.96
Output dim: 1, lower bound: -202.4960439, upper bound: 202.4960439

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4930271, upper bound: 202.4930299
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4930299, upper bound: 202.4930271
time: 4.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4930271, upper bound: 202.4930299
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4930299, upper bound: 202.4930271
time: 5.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4930271, upper bound: 202.4930299
time: 4.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4930299, upper bound: 202.4930271
time: 5.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4930271, upper bound: 202.4930299
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4930299, upper bound: 202.4930271
time: 5.08 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 12.92 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.92
Output dim: 1, lower bound: -202.4930271, upper bound: 202.4930299
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.92
Output dim: 1, lower bound: -202.4930299, upper bound: 202.4930271
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.92
Output dim: 1, lower bound: -202.4930271, upper bound: 202.4930299
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.92
Output dim: 1, lower bound: -202.4930299, upper bound: 202.4930271
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.92
Output dim: 1, lower bound: -202.4930271, upper bound: 202.4930299
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.92
Output dim: 1, lower bound: -202.4930299, upper bound: 202.4930271
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.92
Output dim: 1, lower bound: -202.4930271, upper bound: 202.4930299
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.92
Output dim: 1, lower bound: -202.4930299, upper bound: 202.4930271

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895080, upper bound: 202.4895244
time: 5.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895085, upper bound: 202.4895235
time: 5.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895235, upper bound: 202.4895085
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895244, upper bound: 202.4895080
time: 5.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895080, upper bound: 202.4895244
time: 5.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895085, upper bound: 202.4895235
time: 5.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895235, upper bound: 202.4895085
time: 5.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895244, upper bound: 202.4895080
time: 5.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895080, upper bound: 202.4895244
time: 5.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895085, upper bound: 202.4895235
time: 5.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895235, upper bound: 202.4895085
time: 5.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895244, upper bound: 202.4895080
time: 5.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895080, upper bound: 202.4895244
time: 5.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895085, upper bound: 202.4895235
time: 5.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895235, upper bound: 202.4895085
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895244, upper bound: 202.4895080
time: 5.75 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 16.02 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.02
Output dim: 1, lower bound: -202.4895080, upper bound: 202.4895244
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.02
Output dim: 1, lower bound: -202.4895085, upper bound: 202.4895235
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.02
Output dim: 1, lower bound: -202.4895235, upper bound: 202.4895085
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.02
Output dim: 1, lower bound: -202.4895244, upper bound: 202.4895080
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.02
Output dim: 1, lower bound: -202.4895080, upper bound: 202.4895244
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.02
Output dim: 1, lower bound: -202.4895085, upper bound: 202.4895235
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.02
Output dim: 1, lower bound: -202.4895235, upper bound: 202.4895085
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.02
Output dim: 1, lower bound: -202.4895244, upper bound: 202.4895080
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.02
Output dim: 1, lower bound: -202.4895080, upper bound: 202.4895244
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.02
Output dim: 1, lower bound: -202.4895085, upper bound: 202.4895235
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.02
Output dim: 1, lower bound: -202.4895235, upper bound: 202.4895085
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.02
Output dim: 1, lower bound: -202.4895244, upper bound: 202.4895080
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.02
Output dim: 1, lower bound: -202.4895080, upper bound: 202.4895244
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.02
Output dim: 1, lower bound: -202.4895085, upper bound: 202.4895235
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.02
Output dim: 1, lower bound: -202.4895235, upper bound: 202.4895085
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.02
Output dim: 1, lower bound: -202.4895244, upper bound: 202.4895080

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854741, upper bound: 202.4854784
time: 5.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4854820, upper bound: 202.4854721
time: 5.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4854733, upper bound: 202.4854825
time: 5.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854770, upper bound: 202.4854716
time: 4.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854716, upper bound: 202.4854770
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4854825, upper bound: 202.4854733
time: 4.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4854721, upper bound: 202.4854820
time: 4.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854784, upper bound: 202.4854741
time: 5.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854741, upper bound: 202.4854784
time: 5.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4854820, upper bound: 202.4854721
time: 5.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4854733, upper bound: 202.4854825
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854770, upper bound: 202.4854716
time: 4.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854716, upper bound: 202.4854770
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4854825, upper bound: 202.4854733
time: 5.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4854721, upper bound: 202.4854820
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854784, upper bound: 202.4854741
time: 5.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854741, upper bound: 202.4854784
time: 5.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4854820, upper bound: 202.4854721
time: 5.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4854733, upper bound: 202.4854825
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854770, upper bound: 202.4854716
time: 4.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854716, upper bound: 202.4854770
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4854825, upper bound: 202.4854733
time: 4.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4854721, upper bound: 202.4854820
time: 4.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854784, upper bound: 202.4854741
time: 5.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854741, upper bound: 202.4854784
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4854820, upper bound: 202.4854721
time: 5.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4854733, upper bound: 202.4854825
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854770, upper bound: 202.4854716
time: 4.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854716, upper bound: 202.4854770
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4854825, upper bound: 202.4854733
time: 4.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4854721, upper bound: 202.4854820
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854784, upper bound: 202.4854741
time: 5.70 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 17.83 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 17.83
Output dim: 1, lower bound: -202.4854741, upper bound: 202.4854784
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.83
Output dim: 1, lower bound: -202.4854820, upper bound: 202.4854721
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.83
Output dim: 1, lower bound: -202.4854733, upper bound: 202.4854825
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 17.83
Output dim: 1, lower bound: -202.4854770, upper bound: 202.4854716
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 17.83
Output dim: 1, lower bound: -202.4854716, upper bound: 202.4854770
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.83
Output dim: 1, lower bound: -202.4854825, upper bound: 202.4854733
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.83
Output dim: 1, lower bound: -202.4854721, upper bound: 202.4854820
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 17.83
Output dim: 1, lower bound: -202.4854784, upper bound: 202.4854741
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 17.83
Output dim: 1, lower bound: -202.4854741, upper bound: 202.4854784
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.83
Output dim: 1, lower bound: -202.4854820, upper bound: 202.4854721
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.83
Output dim: 1, lower bound: -202.4854733, upper bound: 202.4854825
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 17.83
Output dim: 1, lower bound: -202.4854770, upper bound: 202.4854716
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 17.83
Output dim: 1, lower bound: -202.4854716, upper bound: 202.4854770
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.83
Output dim: 1, lower bound: -202.4854825, upper bound: 202.4854733
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.83
Output dim: 1, lower bound: -202.4854721, upper bound: 202.4854820
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 17.83
Output dim: 1, lower bound: -202.4854784, upper bound: 202.4854741
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 17.83
Output dim: 1, lower bound: -202.4854741, upper bound: 202.4854784
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.83
Output dim: 1, lower bound: -202.4854820, upper bound: 202.4854721
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.83
Output dim: 1, lower bound: -202.4854733, upper bound: 202.4854825
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 17.83
Output dim: 1, lower bound: -202.4854770, upper bound: 202.4854716
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 17.83
Output dim: 1, lower bound: -202.4854716, upper bound: 202.4854770
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.83
Output dim: 1, lower bound: -202.4854825, upper bound: 202.4854733
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.83
Output dim: 1, lower bound: -202.4854721, upper bound: 202.4854820
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 17.83
Output dim: 1, lower bound: -202.4854784, upper bound: 202.4854741
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 17.83
Output dim: 1, lower bound: -202.4854741, upper bound: 202.4854784
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.83
Output dim: 1, lower bound: -202.4854820, upper bound: 202.4854721
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.83
Output dim: 1, lower bound: -202.4854733, upper bound: 202.4854825
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 17.83
Output dim: 1, lower bound: -202.4854770, upper bound: 202.4854716
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 17.83
Output dim: 1, lower bound: -202.4854716, upper bound: 202.4854770
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.83
Output dim: 1, lower bound: -202.4854825, upper bound: 202.4854733
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.83
Output dim: 1, lower bound: -202.4854721, upper bound: 202.4854820
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 17.83
Output dim: 1, lower bound: -202.4854784, upper bound: 202.4854741

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4841829, upper bound: 202.4841769
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4841830, upper bound: 202.4841769
time: 4.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4841789, upper bound: 202.4841825
time: 5.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4841800, upper bound: 202.4841824
time: 5.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4841824, upper bound: 202.4841800
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4841825, upper bound: 202.4841789
time: 4.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4841769, upper bound: 202.4841830
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4841769, upper bound: 202.4841829
time: 5.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4841829, upper bound: 202.4841769
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4841830, upper bound: 202.4841769
time: 4.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4841789, upper bound: 202.4841825
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4841800, upper bound: 202.4841824
time: 5.32 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 17.58 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 17.58
Output dim: 1, lower bound: -202.4841829, upper bound: 202.4841769
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 17.58
Output dim: 1, lower bound: -202.4841830, upper bound: 202.4841769
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 17.58
Output dim: 1, lower bound: -202.4841789, upper bound: 202.4841825
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 17.58
Output dim: 1, lower bound: -202.4841800, upper bound: 202.4841824
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 17.58
Output dim: 1, lower bound: -202.4841824, upper bound: 202.4841800
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 17.58
Output dim: 1, lower bound: -202.4841825, upper bound: 202.4841789
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 17.58
Output dim: 1, lower bound: -202.4841769, upper bound: 202.4841830
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 17.58
Output dim: 1, lower bound: -202.4841769, upper bound: 202.4841829
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 17.58
Output dim: 1, lower bound: -202.4841829, upper bound: 202.4841769
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 17.58
Output dim: 1, lower bound: -202.4841830, upper bound: 202.4841769
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 17.58
Output dim: 1, lower bound: -202.4841789, upper bound: 202.4841825
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 17.58
Output dim: 1, lower bound: -202.4841800, upper bound: 202.4841824
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.58
Output dim: 1, lower bound: -202.4854825, upper bound: 202.4854733
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.58
Output dim: 1, lower bound: -202.4854721, upper bound: 202.4854820
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.58
Output dim: 1, lower bound: -202.4854820, upper bound: 202.4854721
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.58
Output dim: 1, lower bound: -202.4854733, upper bound: 202.4854825
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.58
Output dim: 1, lower bound: -202.4854825, upper bound: 202.4854733
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.58
Output dim: 1, lower bound: -202.4854721, upper bound: 202.4854820
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.58
Output dim: 1, lower bound: -202.4854820, upper bound: 202.4854721
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.58
Output dim: 1, lower bound: -202.4854733, upper bound: 202.4854825
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.58
Output dim: 1, lower bound: -202.4854825, upper bound: 202.4854733
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.58
Output dim: 1, lower bound: -202.4854721, upper bound: 202.4854820
Binary search (step 2): status=Status.UNKNOWN, k_low=10, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=203.5233612060547
rel_dist={1: [-202.6091655097684, 202.6091655097684]}

## Binary search (step 3) starts
Candidate k: 10, corresponding eps: 0.0390625


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5560329, upper bound: 202.5560329
time: 6.11 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5560329, upper bound: 202.5560329
time: 6.11 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 12.32 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 12.32
Output dim: 1, lower bound: -202.5560329, upper bound: 202.5560329
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 12.32
Output dim: 1, lower bound: -202.5560329, upper bound: 202.5560329

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4960390, upper bound: 202.4960390
time: 5.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4960390, upper bound: 202.4960390
time: 5.23 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4960390, upper bound: 202.4960390
time: 5.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4960390, upper bound: 202.4960390
time: 5.24 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 13.53 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.53
Output dim: 1, lower bound: -202.4960390, upper bound: 202.4960390
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.53
Output dim: 1, lower bound: -202.4960390, upper bound: 202.4960390
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.53
Output dim: 1, lower bound: -202.4960390, upper bound: 202.4960390
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.53
Output dim: 1, lower bound: -202.4960390, upper bound: 202.4960390

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4930233, upper bound: 202.4930258
time: 5.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4930258, upper bound: 202.4930233
time: 5.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4930233, upper bound: 202.4930258
time: 5.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4930258, upper bound: 202.4930233
time: 5.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4930233, upper bound: 202.4930258
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4930258, upper bound: 202.4930233
time: 5.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4930233, upper bound: 202.4930258
time: 5.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4930258, upper bound: 202.4930233
time: 5.39 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 13.71 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.71
Output dim: 1, lower bound: -202.4930233, upper bound: 202.4930258
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.71
Output dim: 1, lower bound: -202.4930258, upper bound: 202.4930233
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.71
Output dim: 1, lower bound: -202.4930233, upper bound: 202.4930258
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.71
Output dim: 1, lower bound: -202.4930258, upper bound: 202.4930233
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.71
Output dim: 1, lower bound: -202.4930233, upper bound: 202.4930258
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.71
Output dim: 1, lower bound: -202.4930258, upper bound: 202.4930233
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.71
Output dim: 1, lower bound: -202.4930233, upper bound: 202.4930258
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.71
Output dim: 1, lower bound: -202.4930258, upper bound: 202.4930233

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895055, upper bound: 202.4895206
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895058, upper bound: 202.4895200
time: 4.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895200, upper bound: 202.4895058
time: 5.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895206, upper bound: 202.4895055
time: 5.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895055, upper bound: 202.4895206
time: 5.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895058, upper bound: 202.4895200
time: 4.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895200, upper bound: 202.4895058
time: 5.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895206, upper bound: 202.4895055
time: 5.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895055, upper bound: 202.4895206
time: 5.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895058, upper bound: 202.4895200
time: 4.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895200, upper bound: 202.4895058
time: 5.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895206, upper bound: 202.4895055
time: 5.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895055, upper bound: 202.4895206
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895058, upper bound: 202.4895200
time: 4.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895200, upper bound: 202.4895058
time: 5.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4895206, upper bound: 202.4895055
time: 5.25 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 15.42 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.42
Output dim: 1, lower bound: -202.4895055, upper bound: 202.4895206
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.42
Output dim: 1, lower bound: -202.4895058, upper bound: 202.4895200
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.42
Output dim: 1, lower bound: -202.4895200, upper bound: 202.4895058
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.42
Output dim: 1, lower bound: -202.4895206, upper bound: 202.4895055
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.42
Output dim: 1, lower bound: -202.4895055, upper bound: 202.4895206
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.42
Output dim: 1, lower bound: -202.4895058, upper bound: 202.4895200
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.42
Output dim: 1, lower bound: -202.4895200, upper bound: 202.4895058
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.42
Output dim: 1, lower bound: -202.4895206, upper bound: 202.4895055
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.42
Output dim: 1, lower bound: -202.4895055, upper bound: 202.4895206
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.42
Output dim: 1, lower bound: -202.4895058, upper bound: 202.4895200
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.42
Output dim: 1, lower bound: -202.4895200, upper bound: 202.4895058
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.42
Output dim: 1, lower bound: -202.4895206, upper bound: 202.4895055
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.42
Output dim: 1, lower bound: -202.4895055, upper bound: 202.4895206
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.42
Output dim: 1, lower bound: -202.4895058, upper bound: 202.4895200
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.42
Output dim: 1, lower bound: -202.4895200, upper bound: 202.4895058
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.42
Output dim: 1, lower bound: -202.4895206, upper bound: 202.4895055

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854720, upper bound: 202.4854755
time: 6.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854790, upper bound: 202.4854701
time: 5.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854714, upper bound: 202.4854795
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854743, upper bound: 202.4854693
time: 4.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854693, upper bound: 202.4854743
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854795, upper bound: 202.4854714
time: 5.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854701, upper bound: 202.4854790
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854755, upper bound: 202.4854720
time: 5.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854720, upper bound: 202.4854755
time: 6.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854790, upper bound: 202.4854701
time: 4.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854714, upper bound: 202.4854795
time: 5.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854743, upper bound: 202.4854693
time: 4.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854693, upper bound: 202.4854743
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854795, upper bound: 202.4854714
time: 4.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854701, upper bound: 202.4854790
time: 4.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854755, upper bound: 202.4854720
time: 5.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854720, upper bound: 202.4854755
time: 6.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854790, upper bound: 202.4854701
time: 4.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854714, upper bound: 202.4854795
time: 5.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854743, upper bound: 202.4854693
time: 4.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854693, upper bound: 202.4854743
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854795, upper bound: 202.4854714
time: 5.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854701, upper bound: 202.4854790
time: 4.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854755, upper bound: 202.4854720
time: 5.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854720, upper bound: 202.4854755
time: 6.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854790, upper bound: 202.4854701
time: 4.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854714, upper bound: 202.4854795
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854743, upper bound: 202.4854693
time: 4.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854693, upper bound: 202.4854743
time: 4.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854795, upper bound: 202.4854714
time: 4.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138
1: -111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612
2: -144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239
3: -152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125
4: -140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590
5: -125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873
6: -120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172
7: -131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144
8: -159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220
9: -119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854701, upper bound: 202.4854790
time: 4.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4854755, upper bound: 202.4854720
time: 5.16 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 16.84 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.84
Output dim: 1, lower bound: -202.4854720, upper bound: 202.4854755
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.84
Output dim: 1, lower bound: -202.4854790, upper bound: 202.4854701
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.84
Output dim: 1, lower bound: -202.4854714, upper bound: 202.4854795
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.84
Output dim: 1, lower bound: -202.4854743, upper bound: 202.4854693
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.84
Output dim: 1, lower bound: -202.4854693, upper bound: 202.4854743
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.84
Output dim: 1, lower bound: -202.4854795, upper bound: 202.4854714
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.84
Output dim: 1, lower bound: -202.4854701, upper bound: 202.4854790
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.84
Output dim: 1, lower bound: -202.4854755, upper bound: 202.4854720
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.84
Output dim: 1, lower bound: -202.4854720, upper bound: 202.4854755
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.84
Output dim: 1, lower bound: -202.4854790, upper bound: 202.4854701
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.84
Output dim: 1, lower bound: -202.4854714, upper bound: 202.4854795
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.84
Output dim: 1, lower bound: -202.4854743, upper bound: 202.4854693
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.84
Output dim: 1, lower bound: -202.4854693, upper bound: 202.4854743
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.84
Output dim: 1, lower bound: -202.4854795, upper bound: 202.4854714
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.84
Output dim: 1, lower bound: -202.4854701, upper bound: 202.4854790
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.84
Output dim: 1, lower bound: -202.4854755, upper bound: 202.4854720
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.84
Output dim: 1, lower bound: -202.4854720, upper bound: 202.4854755
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.84
Output dim: 1, lower bound: -202.4854790, upper bound: 202.4854701
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.84
Output dim: 1, lower bound: -202.4854714, upper bound: 202.4854795
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.84
Output dim: 1, lower bound: -202.4854743, upper bound: 202.4854693
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.84
Output dim: 1, lower bound: -202.4854693, upper bound: 202.4854743
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.84
Output dim: 1, lower bound: -202.4854795, upper bound: 202.4854714
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.84
Output dim: 1, lower bound: -202.4854701, upper bound: 202.4854790
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.84
Output dim: 1, lower bound: -202.4854755, upper bound: 202.4854720
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.84
Output dim: 1, lower bound: -202.4854720, upper bound: 202.4854755
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.84
Output dim: 1, lower bound: -202.4854790, upper bound: 202.4854701
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.84
Output dim: 1, lower bound: -202.4854714, upper bound: 202.4854795
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.84
Output dim: 1, lower bound: -202.4854743, upper bound: 202.4854693
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.84
Output dim: 1, lower bound: -202.4854693, upper bound: 202.4854743
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.84
Output dim: 1, lower bound: -202.4854795, upper bound: 202.4854714
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.84
Output dim: 1, lower bound: -202.4854701, upper bound: 202.4854790
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.84
Output dim: 1, lower bound: -202.4854755, upper bound: 202.4854720
Binary search (step 3): status=Status.VERIFIED, k_low=10, k_high=10, k_mid=10, eps_mid=0.0390625, abs_max=203.5233612060547
rel_dist={1: [-202.60913898263854, 202.60913897847803]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0390625
execution time: 2110.73 seconds
