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
execution time: IAR + LP analysis = 1.02 + 9.73 = 10.75 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -202.6091898, upper bound: 202.6091898


# Binary Search by BASE starts (time budget: 2689.25 seconds, max iter: 100)

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
Binary search time: 37.34 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 2651.91 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5944836, upper bound: 202.5944836
time: 7.36 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5944836, upper bound: 202.5944836
time: 7.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.67 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 14.67
Output dim: 1, lower bound: -202.5944836, upper bound: 202.5944836
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 14.67
Output dim: 1, lower bound: -202.5944836, upper bound: 202.5944836

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

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5666305, upper bound: 202.5666305
time: 6.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5666305, upper bound: 202.5666305
time: 6.32 seconds

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

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5891219, upper bound: 202.5891219
time: 5.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5891219, upper bound: 202.5891219
time: 5.88 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 12.72 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 12.72
Output dim: 1, lower bound: -202.5666305, upper bound: 202.5666305
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 12.72
Output dim: 1, lower bound: -202.5666305, upper bound: 202.5666305
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 12.72
Output dim: 1, lower bound: -202.5891219, upper bound: 202.5891219
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 12.72
Output dim: 1, lower bound: -202.5891219, upper bound: 202.5891219

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

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 239

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5665366, upper bound: 202.5665366
time: 5.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5665366, upper bound: 202.5665366
time: 6.47 seconds

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

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5666290, upper bound: 202.5666304
time: 7.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5666304, upper bound: 202.5666290
time: 6.01 seconds

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

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5488663, upper bound: 202.5488663
time: 5.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5488663, upper bound: 202.5488663
time: 5.81 seconds

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

Time for backsubstitution: 0.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5861003, upper bound: 202.5860275
time: 7.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5860275, upper bound: 202.5861003
time: 6.84 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 15.04 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.04
Output dim: 1, lower bound: -202.5665366, upper bound: 202.5665366
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.04
Output dim: 1, lower bound: -202.5665366, upper bound: 202.5665366
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.04
Output dim: 1, lower bound: -202.5666290, upper bound: 202.5666304
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.04
Output dim: 1, lower bound: -202.5666304, upper bound: 202.5666290
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.04
Output dim: 1, lower bound: -202.5488663, upper bound: 202.5488663
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.04
Output dim: 1, lower bound: -202.5488663, upper bound: 202.5488663
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.04
Output dim: 1, lower bound: -202.5861003, upper bound: 202.5860275
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.04
Output dim: 1, lower bound: -202.5860275, upper bound: 202.5861003

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

Time for backsubstitution: 0.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5653935, upper bound: 202.5653935
time: 5.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5653935, upper bound: 202.5653935
time: 6.51 seconds

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

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5634891, upper bound: 202.5634859
time: 6.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5634891, upper bound: 202.5634859
time: 6.30 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 169

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 158

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5601047, upper bound: 202.5601228
time: 6.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5601047, upper bound: 202.5601228
time: 6.41 seconds

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

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5557430, upper bound: 202.5557456
time: 6.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5557430, upper bound: 202.5557456
time: 6.64 seconds

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

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5484873, upper bound: 202.5484873
time: 5.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5484873, upper bound: 202.5484873
time: 4.89 seconds

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

Time for backsubstitution: 0.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5488646, upper bound: 202.5488663
time: 5.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5488663, upper bound: 202.5488646
time: 5.89 seconds

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
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 99

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4804873, upper bound: 202.4804895
time: 5.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4804873, upper bound: 202.4804895
time: 5.03 seconds

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
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5848545, upper bound: 202.5849357
time: 8.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5848545, upper bound: 202.5849357
time: 7.47 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 17.29 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.29
Output dim: 1, lower bound: -202.5653935, upper bound: 202.5653935
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.29
Output dim: 1, lower bound: -202.5653935, upper bound: 202.5653935
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.29
Output dim: 1, lower bound: -202.5634891, upper bound: 202.5634859
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.29
Output dim: 1, lower bound: -202.5634891, upper bound: 202.5634859
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.29
Output dim: 1, lower bound: -202.5601047, upper bound: 202.5601228
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.29
Output dim: 1, lower bound: -202.5601047, upper bound: 202.5601228
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.29
Output dim: 1, lower bound: -202.5557430, upper bound: 202.5557456
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.29
Output dim: 1, lower bound: -202.5557430, upper bound: 202.5557456
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.29
Output dim: 1, lower bound: -202.5484873, upper bound: 202.5484873
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.29
Output dim: 1, lower bound: -202.5484873, upper bound: 202.5484873
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.29
Output dim: 1, lower bound: -202.5488646, upper bound: 202.5488663
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.29
Output dim: 1, lower bound: -202.5488663, upper bound: 202.5488646
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 17.29
Output dim: 1, lower bound: -202.4804873, upper bound: 202.4804895
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 17.29
Output dim: 1, lower bound: -202.4804873, upper bound: 202.4804895
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.29
Output dim: 1, lower bound: -202.5848545, upper bound: 202.5849357
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.29
Output dim: 1, lower bound: -202.5848545, upper bound: 202.5849357

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
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5099067, upper bound: 202.5099076
time: 5.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5099067, upper bound: 202.5099076
time: 5.52 seconds

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

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5587811, upper bound: 202.5587856
time: 7.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5587855, upper bound: 202.5587812
time: 6.69 seconds

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

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 121

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5475135, upper bound: 202.5474849
time: 6.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5475135, upper bound: 202.5474849
time: 6.17 seconds

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
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5599398, upper bound: 202.5599371
time: 6.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5599379, upper bound: 202.5599393
time: 6.11 seconds

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

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5057804, upper bound: 202.5058037
time: 5.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5057804, upper bound: 202.5058037
time: 5.35 seconds

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
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5533535, upper bound: 202.5533490
time: 6.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5533535, upper bound: 202.5533490
time: 5.73 seconds

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
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5545482, upper bound: 202.5545455
time: 6.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5545430, upper bound: 202.5545510
time: 6.21 seconds

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
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5557414, upper bound: 202.5557456
time: 6.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5557430, upper bound: 202.5557450
time: 6.56 seconds

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
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 99

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5484812, upper bound: 202.5484872
time: 5.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5484872, upper bound: 202.5484812
time: 5.84 seconds

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

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 158

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5400722, upper bound: 202.5400722
time: 5.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5400722, upper bound: 202.5400722
time: 5.64 seconds

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

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5488548, upper bound: 202.5488663
time: 6.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5488646, upper bound: 202.5488639
time: 5.76 seconds

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
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5465751, upper bound: 202.5465518
time: 5.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5465622, upper bound: 202.5465589
time: 5.96 seconds

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
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5848545, upper bound: 202.5849016
time: 6.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5848388, upper bound: 202.5849357
time: 7.31 seconds

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

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5848545, upper bound: 202.5849357
time: 8.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5848545, upper bound: 202.5849357
time: 8.89 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 17.96 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 1, lower bound: -202.5099067, upper bound: 202.5099076
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 1, lower bound: -202.5099067, upper bound: 202.5099076
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 1, lower bound: -202.5587811, upper bound: 202.5587856
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 1, lower bound: -202.5587855, upper bound: 202.5587812
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 1, lower bound: -202.5475135, upper bound: 202.5474849
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 1, lower bound: -202.5475135, upper bound: 202.5474849
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 1, lower bound: -202.5599398, upper bound: 202.5599371
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 1, lower bound: -202.5599379, upper bound: 202.5599393
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 1, lower bound: -202.5057804, upper bound: 202.5058037
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 1, lower bound: -202.5057804, upper bound: 202.5058037
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 1, lower bound: -202.5533535, upper bound: 202.5533490
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 1, lower bound: -202.5533535, upper bound: 202.5533490
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 1, lower bound: -202.5545482, upper bound: 202.5545455
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 1, lower bound: -202.5545430, upper bound: 202.5545510
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 1, lower bound: -202.5557414, upper bound: 202.5557456
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 1, lower bound: -202.5557430, upper bound: 202.5557450
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 1, lower bound: -202.5484812, upper bound: 202.5484872
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 1, lower bound: -202.5484872, upper bound: 202.5484812
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 1, lower bound: -202.5400722, upper bound: 202.5400722
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 1, lower bound: -202.5400722, upper bound: 202.5400722
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 1, lower bound: -202.5488548, upper bound: 202.5488663
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 1, lower bound: -202.5488646, upper bound: 202.5488639
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 1, lower bound: -202.5465751, upper bound: 202.5465518
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 1, lower bound: -202.5465622, upper bound: 202.5465589
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 1, lower bound: -202.5848545, upper bound: 202.5849016
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 1, lower bound: -202.5848388, upper bound: 202.5849357
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 1, lower bound: -202.5848545, upper bound: 202.5849357
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 1, lower bound: -202.5848545, upper bound: 202.5849357

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5099067, upper bound: 202.5099053
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5099032, upper bound: 202.5099076
time: 5.80 seconds

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

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5086380, upper bound: 202.5086385
time: 5.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5086369, upper bound: 202.5086390
time: 5.94 seconds

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

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5250341, upper bound: 202.5250677
time: 6.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5250341, upper bound: 202.5250677
time: 6.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5586542, upper bound: 202.5586538
time: 6.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5586542, upper bound: 202.5586538
time: 6.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5464331, upper bound: 202.5464167
time: 7.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5464429, upper bound: 202.5464158
time: 5.71 seconds

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

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 221

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5473768, upper bound: 202.5473466
time: 6.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5473777, upper bound: 202.5473465
time: 6.19 seconds

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
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5529255, upper bound: 202.5529195
time: 6.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5529204, upper bound: 202.5529221
time: 7.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5422472, upper bound: 202.5422206
time: 6.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5422472, upper bound: 202.5422206
time: 6.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 158

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5056202, upper bound: 202.5056158
time: 5.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5056149, upper bound: 202.5056198
time: 5.88 seconds

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

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 151

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5057742, upper bound: 202.5058037
time: 5.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5057804, upper bound: 202.5057991
time: 5.69 seconds

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

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5533534, upper bound: 202.5533490
time: 6.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5533535, upper bound: 202.5533490
time: 5.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 239

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5533453, upper bound: 202.5533425
time: 6.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5533453, upper bound: 202.5533425
time: 6.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5545482, upper bound: 202.5545452
time: 6.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5545471, upper bound: 202.5545455
time: 5.74 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 12.83 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.83
Output dim: 1, lower bound: -202.5099067, upper bound: 202.5099053
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.83
Output dim: 1, lower bound: -202.5099032, upper bound: 202.5099076
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.83
Output dim: 1, lower bound: -202.5086380, upper bound: 202.5086385
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.83
Output dim: 1, lower bound: -202.5086369, upper bound: 202.5086390
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.83
Output dim: 1, lower bound: -202.5250341, upper bound: 202.5250677
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.83
Output dim: 1, lower bound: -202.5250341, upper bound: 202.5250677
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.83
Output dim: 1, lower bound: -202.5586542, upper bound: 202.5586538
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.83
Output dim: 1, lower bound: -202.5586542, upper bound: 202.5586538
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.83
Output dim: 1, lower bound: -202.5464331, upper bound: 202.5464167
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.83
Output dim: 1, lower bound: -202.5464429, upper bound: 202.5464158
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.83
Output dim: 1, lower bound: -202.5473768, upper bound: 202.5473466
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.83
Output dim: 1, lower bound: -202.5473777, upper bound: 202.5473465
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.83
Output dim: 1, lower bound: -202.5529255, upper bound: 202.5529195
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.83
Output dim: 1, lower bound: -202.5529204, upper bound: 202.5529221
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.83
Output dim: 1, lower bound: -202.5422472, upper bound: 202.5422206
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.83
Output dim: 1, lower bound: -202.5422472, upper bound: 202.5422206
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.83
Output dim: 1, lower bound: -202.5056202, upper bound: 202.5056158
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.83
Output dim: 1, lower bound: -202.5056149, upper bound: 202.5056198
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.83
Output dim: 1, lower bound: -202.5057742, upper bound: 202.5058037
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.83
Output dim: 1, lower bound: -202.5057804, upper bound: 202.5057991
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.83
Output dim: 1, lower bound: -202.5533534, upper bound: 202.5533490
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.83
Output dim: 1, lower bound: -202.5533535, upper bound: 202.5533490
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.83
Output dim: 1, lower bound: -202.5533453, upper bound: 202.5533425
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.83
Output dim: 1, lower bound: -202.5533453, upper bound: 202.5533425
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.83
Output dim: 1, lower bound: -202.5545482, upper bound: 202.5545452
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.83
Output dim: 1, lower bound: -202.5545471, upper bound: 202.5545455
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.83
Output dim: 1, lower bound: -202.5545430, upper bound: 202.5545510
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.83
Output dim: 1, lower bound: -202.5557414, upper bound: 202.5557456
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.83
Output dim: 1, lower bound: -202.5557430, upper bound: 202.5557450
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.83
Output dim: 1, lower bound: -202.5484812, upper bound: 202.5484872
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.83
Output dim: 1, lower bound: -202.5484872, upper bound: 202.5484812
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.83
Output dim: 1, lower bound: -202.5400722, upper bound: 202.5400722
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.83
Output dim: 1, lower bound: -202.5400722, upper bound: 202.5400722
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.83
Output dim: 1, lower bound: -202.5488548, upper bound: 202.5488663
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.83
Output dim: 1, lower bound: -202.5488646, upper bound: 202.5488639
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.83
Output dim: 1, lower bound: -202.5465751, upper bound: 202.5465518
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.83
Output dim: 1, lower bound: -202.5465622, upper bound: 202.5465589
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.83
Output dim: 1, lower bound: -202.5848545, upper bound: 202.5849016
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.83
Output dim: 1, lower bound: -202.5848388, upper bound: 202.5849357
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.83
Output dim: 1, lower bound: -202.5848545, upper bound: 202.5849357
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.83
Output dim: 1, lower bound: -202.5848545, upper bound: 202.5849357
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=203.5233612060547
rel_dist={1: [-202.60902678108835, 202.60902678108835]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 121

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6056209, upper bound: 202.6056209
time: 7.34 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6056209, upper bound: 202.6056209
time: 7.20 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.55 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 14.55
Output dim: 1, lower bound: -202.6056209, upper bound: 202.6056209
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 14.55
Output dim: 1, lower bound: -202.6056209, upper bound: 202.6056209

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

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 99

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5779997, upper bound: 202.5779997
time: 7.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5779997, upper bound: 202.5779997
time: 7.35 seconds

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
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5982615, upper bound: 202.5982615
time: 6.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5982615, upper bound: 202.5982615
time: 6.46 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 13.88 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.88
Output dim: 1, lower bound: -202.5779997, upper bound: 202.5779997
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.88
Output dim: 1, lower bound: -202.5779997, upper bound: 202.5779997
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.88
Output dim: 1, lower bound: -202.5982615, upper bound: 202.5982615
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.88
Output dim: 1, lower bound: -202.5982615, upper bound: 202.5982615

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

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5779709, upper bound: 202.5779997
time: 6.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5779997, upper bound: 202.5779709
time: 7.11 seconds

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 158

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4951583, upper bound: 202.4951583
time: 5.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4951583, upper bound: 202.4951583
time: 7.00 seconds

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
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5982615, upper bound: 202.5982590
time: 8.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5982590, upper bound: 202.5982615
time: 9.38 seconds

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

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5955157, upper bound: 202.5955157
time: 8.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5955157, upper bound: 202.5955157
time: 8.09 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 17.52 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.52
Output dim: 1, lower bound: -202.5779709, upper bound: 202.5779997
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.52
Output dim: 1, lower bound: -202.5779997, upper bound: 202.5779709
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.52
Output dim: 1, lower bound: -202.4951583, upper bound: 202.4951583
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.52
Output dim: 1, lower bound: -202.4951583, upper bound: 202.4951583
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.52
Output dim: 1, lower bound: -202.5982615, upper bound: 202.5982590
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.52
Output dim: 1, lower bound: -202.5982590, upper bound: 202.5982615
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.52
Output dim: 1, lower bound: -202.5955157, upper bound: 202.5955157
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.52
Output dim: 1, lower bound: -202.5955157, upper bound: 202.5955157

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

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 99

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5236630, upper bound: 202.5236733
time: 5.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5236630, upper bound: 202.5236733
time: 5.39 seconds

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

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5498032, upper bound: 202.5497946
time: 7.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5498032, upper bound: 202.5497946
time: 7.13 seconds

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

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4924048, upper bound: 202.4924116
time: 5.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4924116, upper bound: 202.4924048
time: 6.18 seconds

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

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4951575, upper bound: 202.4951583
time: 5.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4951583, upper bound: 202.4951575
time: 6.58 seconds

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
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5977371, upper bound: 202.5977371
time: 7.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5977371, upper bound: 202.5977371
time: 8.15 seconds

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

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5639395, upper bound: 202.5639492
time: 6.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5639395, upper bound: 202.5639492
time: 6.80 seconds

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

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 151

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5903136, upper bound: 202.5903136
time: 9.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5903136, upper bound: 202.5903136
time: 9.62 seconds

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

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5614947, upper bound: 202.5614947
time: 7.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5614947, upper bound: 202.5614947
time: 7.51 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 15.98 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.98
Output dim: 1, lower bound: -202.5236630, upper bound: 202.5236733
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.98
Output dim: 1, lower bound: -202.5236630, upper bound: 202.5236733
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.98
Output dim: 1, lower bound: -202.5498032, upper bound: 202.5497946
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.98
Output dim: 1, lower bound: -202.5498032, upper bound: 202.5497946
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.98
Output dim: 1, lower bound: -202.4924048, upper bound: 202.4924116
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.98
Output dim: 1, lower bound: -202.4924116, upper bound: 202.4924048
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.98
Output dim: 1, lower bound: -202.4951575, upper bound: 202.4951583
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.98
Output dim: 1, lower bound: -202.4951583, upper bound: 202.4951575
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.98
Output dim: 1, lower bound: -202.5977371, upper bound: 202.5977371
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.98
Output dim: 1, lower bound: -202.5977371, upper bound: 202.5977371
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.98
Output dim: 1, lower bound: -202.5639395, upper bound: 202.5639492
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.98
Output dim: 1, lower bound: -202.5639395, upper bound: 202.5639492
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.98
Output dim: 1, lower bound: -202.5903136, upper bound: 202.5903136
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.98
Output dim: 1, lower bound: -202.5903136, upper bound: 202.5903136
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.98
Output dim: 1, lower bound: -202.5614947, upper bound: 202.5614947
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.98
Output dim: 1, lower bound: -202.5614947, upper bound: 202.5614947

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

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 239

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5236626, upper bound: 202.5236733
time: 6.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5236631, upper bound: 202.5236675
time: 5.33 seconds

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

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5236630, upper bound: 202.5236733
time: 6.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5236626, upper bound: 202.5236730
time: 6.85 seconds

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

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5256757, upper bound: 202.5256757
time: 6.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5256757, upper bound: 202.5256757
time: 6.54 seconds

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

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5346547, upper bound: 202.5346486
time: 7.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5346547, upper bound: 202.5346486
time: 7.36 seconds

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

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4924048, upper bound: 202.4924077
time: 6.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4924022, upper bound: 202.4924116
time: 5.61 seconds

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

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4845324, upper bound: 202.4845260
time: 6.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4845293, upper bound: 202.4845264
time: 6.30 seconds

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
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4924046, upper bound: 202.4924116
time: 6.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4924116, upper bound: 202.4924048
time: 6.83 seconds

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
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 151

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4950983, upper bound: 202.4950982
time: 6.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4950983, upper bound: 202.4950982
time: 6.41 seconds

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

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5964028, upper bound: 202.5963991
time: 9.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5964028, upper bound: 202.5963991
time: 8.90 seconds

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5913857, upper bound: 202.5913714
time: 8.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5913857, upper bound: 202.5913714
time: 8.29 seconds

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5278351, upper bound: 202.5278362
time: 6.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5278351, upper bound: 202.5278362
time: 6.90 seconds

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
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5639395, upper bound: 202.5639473
time: 7.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5639386, upper bound: 202.5639492
time: 6.09 seconds

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5827558, upper bound: 202.5827558
time: 7.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5827558, upper bound: 202.5827558
time: 8.05 seconds

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

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5766205, upper bound: 202.5766205
time: 8.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5766205, upper bound: 202.5766205
time: 8.56 seconds

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
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5614947, upper bound: 202.5614947
time: 6.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5614947, upper bound: 202.5614947
time: 6.10 seconds

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
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5588948, upper bound: 202.5588962
time: 6.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5588962, upper bound: 202.5588948
time: 6.89 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 14.29 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.29
Output dim: 1, lower bound: -202.5236626, upper bound: 202.5236733
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.29
Output dim: 1, lower bound: -202.5236631, upper bound: 202.5236675
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.29
Output dim: 1, lower bound: -202.5236630, upper bound: 202.5236733
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.29
Output dim: 1, lower bound: -202.5236626, upper bound: 202.5236730
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.29
Output dim: 1, lower bound: -202.5256757, upper bound: 202.5256757
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.29
Output dim: 1, lower bound: -202.5256757, upper bound: 202.5256757
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.29
Output dim: 1, lower bound: -202.5346547, upper bound: 202.5346486
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.29
Output dim: 1, lower bound: -202.5346547, upper bound: 202.5346486
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.29
Output dim: 1, lower bound: -202.4924048, upper bound: 202.4924077
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.29
Output dim: 1, lower bound: -202.4924022, upper bound: 202.4924116
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.29
Output dim: 1, lower bound: -202.4845324, upper bound: 202.4845260
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.29
Output dim: 1, lower bound: -202.4845293, upper bound: 202.4845264
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.29
Output dim: 1, lower bound: -202.4924046, upper bound: 202.4924116
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.29
Output dim: 1, lower bound: -202.4924116, upper bound: 202.4924048
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.29
Output dim: 1, lower bound: -202.4950983, upper bound: 202.4950982
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.29
Output dim: 1, lower bound: -202.4950983, upper bound: 202.4950982
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.29
Output dim: 1, lower bound: -202.5964028, upper bound: 202.5963991
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.29
Output dim: 1, lower bound: -202.5964028, upper bound: 202.5963991
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.29
Output dim: 1, lower bound: -202.5913857, upper bound: 202.5913714
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.29
Output dim: 1, lower bound: -202.5913857, upper bound: 202.5913714
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.29
Output dim: 1, lower bound: -202.5278351, upper bound: 202.5278362
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.29
Output dim: 1, lower bound: -202.5278351, upper bound: 202.5278362
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.29
Output dim: 1, lower bound: -202.5639395, upper bound: 202.5639473
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.29
Output dim: 1, lower bound: -202.5639386, upper bound: 202.5639492
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.29
Output dim: 1, lower bound: -202.5827558, upper bound: 202.5827558
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.29
Output dim: 1, lower bound: -202.5827558, upper bound: 202.5827558
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.29
Output dim: 1, lower bound: -202.5766205, upper bound: 202.5766205
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.29
Output dim: 1, lower bound: -202.5766205, upper bound: 202.5766205
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.29
Output dim: 1, lower bound: -202.5614947, upper bound: 202.5614947
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.29
Output dim: 1, lower bound: -202.5614947, upper bound: 202.5614947
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.29
Output dim: 1, lower bound: -202.5588948, upper bound: 202.5588962
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.29
Output dim: 1, lower bound: -202.5588962, upper bound: 202.5588948

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5056884, upper bound: 202.5056961
time: 6.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5056884, upper bound: 202.5056961
time: 6.93 seconds

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

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5206692, upper bound: 202.5206639
time: 6.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5206528, upper bound: 202.5206820
time: 6.33 seconds

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

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5027732, upper bound: 202.5027780
time: 6.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5027732, upper bound: 202.5027780
time: 6.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 239

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5191096, upper bound: 202.5191149
time: 5.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5191083, upper bound: 202.5191157
time: 6.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5223254, upper bound: 202.5223234
time: 7.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5223243, upper bound: 202.5223247
time: 6.39 seconds

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

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5119419, upper bound: 202.5119325
time: 6.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5119419, upper bound: 202.5119325
time: 5.67 seconds

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5219221, upper bound: 202.5219169
time: 6.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5219221, upper bound: 202.5219169
time: 6.83 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 14.48 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.48
Output dim: 1, lower bound: -202.5056884, upper bound: 202.5056961
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.48
Output dim: 1, lower bound: -202.5056884, upper bound: 202.5056961
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.48
Output dim: 1, lower bound: -202.5206692, upper bound: 202.5206639
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.48
Output dim: 1, lower bound: -202.5206528, upper bound: 202.5206820
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.48
Output dim: 1, lower bound: -202.5027732, upper bound: 202.5027780
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.48
Output dim: 1, lower bound: -202.5027732, upper bound: 202.5027780
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.48
Output dim: 1, lower bound: -202.5191096, upper bound: 202.5191149
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.48
Output dim: 1, lower bound: -202.5191083, upper bound: 202.5191157
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.48
Output dim: 1, lower bound: -202.5223254, upper bound: 202.5223234
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.48
Output dim: 1, lower bound: -202.5223243, upper bound: 202.5223247
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.48
Output dim: 1, lower bound: -202.5119419, upper bound: 202.5119325
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.48
Output dim: 1, lower bound: -202.5119419, upper bound: 202.5119325
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.48
Output dim: 1, lower bound: -202.5219221, upper bound: 202.5219169
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.48
Output dim: 1, lower bound: -202.5219221, upper bound: 202.5219169
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -202.5346547, upper bound: 202.5346486
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -202.4924048, upper bound: 202.4924077
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -202.4924022, upper bound: 202.4924116
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -202.4924046, upper bound: 202.4924116
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -202.4924116, upper bound: 202.4924048
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -202.4950983, upper bound: 202.4950982
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -202.4950983, upper bound: 202.4950982
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -202.5964028, upper bound: 202.5963991
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -202.5964028, upper bound: 202.5963991
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -202.5913857, upper bound: 202.5913714
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -202.5913857, upper bound: 202.5913714
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -202.5278351, upper bound: 202.5278362
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -202.5278351, upper bound: 202.5278362
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -202.5639395, upper bound: 202.5639473
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -202.5639386, upper bound: 202.5639492
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -202.5827558, upper bound: 202.5827558
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -202.5827558, upper bound: 202.5827558
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -202.5766205, upper bound: 202.5766205
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -202.5766205, upper bound: 202.5766205
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -202.5614947, upper bound: 202.5614947
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -202.5614947, upper bound: 202.5614947
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -202.5588948, upper bound: 202.5588962
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -202.5588962, upper bound: 202.5588948
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=203.5233612060547
rel_dist={1: [-202.60871310808878, 202.60871310808875]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6082025, upper bound: 202.6082003
time: 8.71 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6082003, upper bound: 202.6082025
time: 8.64 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 17.36 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 17.36
Output dim: 1, lower bound: -202.6082025, upper bound: 202.6082003
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 17.36
Output dim: 1, lower bound: -202.6082003, upper bound: 202.6082025

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

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6065373, upper bound: 202.6065453
time: 9.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6065453, upper bound: 202.6065364
time: 9.40 seconds

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
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6082003, upper bound: 202.6082025
time: 9.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6082003, upper bound: 202.6082025
time: 10.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 21.14 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.14
Output dim: 1, lower bound: -202.6065373, upper bound: 202.6065453
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.14
Output dim: 1, lower bound: -202.6065453, upper bound: 202.6065364
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.14
Output dim: 1, lower bound: -202.6082003, upper bound: 202.6082025
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.14
Output dim: 1, lower bound: -202.6082003, upper bound: 202.6082025

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
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 158

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5482249, upper bound: 202.5482237
time: 8.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5482249, upper bound: 202.5482237
time: 8.24 seconds

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
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6065453, upper bound: 202.6065329
time: 9.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6065342, upper bound: 202.6065364
time: 9.41 seconds

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
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6047756, upper bound: 202.6047749
time: 8.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6047744, upper bound: 202.6047765
time: 9.96 seconds

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

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 158

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5536171, upper bound: 202.5536186
time: 9.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5536171, upper bound: 202.5536186
time: 9.24 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 19.49 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.49
Output dim: 1, lower bound: -202.5482249, upper bound: 202.5482237
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.49
Output dim: 1, lower bound: -202.5482249, upper bound: 202.5482237
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.49
Output dim: 1, lower bound: -202.6065453, upper bound: 202.6065329
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.49
Output dim: 1, lower bound: -202.6065342, upper bound: 202.6065364
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.49
Output dim: 1, lower bound: -202.6047756, upper bound: 202.6047749
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.49
Output dim: 1, lower bound: -202.6047744, upper bound: 202.6047765
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.49
Output dim: 1, lower bound: -202.5536171, upper bound: 202.5536186
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.49
Output dim: 1, lower bound: -202.5536171, upper bound: 202.5536186

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

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 151

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5409126, upper bound: 202.5409104
time: 10.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5409126, upper bound: 202.5409104
time: 8.69 seconds

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

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5464400, upper bound: 202.5464390
time: 6.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5464400, upper bound: 202.5464390
time: 7.44 seconds

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
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6065335, upper bound: 202.6065329
time: 10.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6065453, upper bound: 202.6065291
time: 9.92 seconds

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

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6050947, upper bound: 202.6050941
time: 10.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6050971, upper bound: 202.6050941
time: 10.15 seconds

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

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5527336, upper bound: 202.5527288
time: 7.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5527336, upper bound: 202.5527288
time: 7.15 seconds

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

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 169

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6045780, upper bound: 202.6045766
time: 9.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6045756, upper bound: 202.6045793
time: 8.41 seconds

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

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5523713, upper bound: 202.5523743
time: 7.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5523713, upper bound: 202.5523743
time: 6.88 seconds

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
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5524559, upper bound: 202.5524595
time: 8.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5524559, upper bound: 202.5524595
time: 8.22 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 17.90 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.90
Output dim: 1, lower bound: -202.5409126, upper bound: 202.5409104
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.90
Output dim: 1, lower bound: -202.5409126, upper bound: 202.5409104
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.90
Output dim: 1, lower bound: -202.5464400, upper bound: 202.5464390
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.90
Output dim: 1, lower bound: -202.5464400, upper bound: 202.5464390
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.90
Output dim: 1, lower bound: -202.6065335, upper bound: 202.6065329
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.90
Output dim: 1, lower bound: -202.6065453, upper bound: 202.6065291
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.90
Output dim: 1, lower bound: -202.6050947, upper bound: 202.6050941
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.90
Output dim: 1, lower bound: -202.6050971, upper bound: 202.6050941
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.90
Output dim: 1, lower bound: -202.5527336, upper bound: 202.5527288
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.90
Output dim: 1, lower bound: -202.5527336, upper bound: 202.5527288
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.90
Output dim: 1, lower bound: -202.6045780, upper bound: 202.6045766
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.90
Output dim: 1, lower bound: -202.6045756, upper bound: 202.6045793
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.90
Output dim: 1, lower bound: -202.5523713, upper bound: 202.5523743
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.90
Output dim: 1, lower bound: -202.5523713, upper bound: 202.5523743
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.90
Output dim: 1, lower bound: -202.5524559, upper bound: 202.5524595
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.90
Output dim: 1, lower bound: -202.5524559, upper bound: 202.5524595

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

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5063526, upper bound: 202.5063511
time: 6.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5063526, upper bound: 202.5063511
time: 6.58 seconds

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

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 151

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5379406, upper bound: 202.5379383
time: 8.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5379406, upper bound: 202.5379383
time: 8.81 seconds

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

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5256334, upper bound: 202.5256327
time: 8.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5256334, upper bound: 202.5256327
time: 8.04 seconds

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
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5464400, upper bound: 202.5464390
time: 7.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5464400, upper bound: 202.5464390
time: 7.19 seconds

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
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 158

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6016610, upper bound: 202.6016626
time: 10.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6016592, upper bound: 202.6016645
time: 9.47 seconds

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
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6065453, upper bound: 202.6065291
time: 10.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6065335, upper bound: 202.6065291
time: 9.80 seconds

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
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 169

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6050774, upper bound: 202.6050941
time: 13.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6050774, upper bound: 202.6050796
time: 10.61 seconds

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

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5909058, upper bound: 202.5909030
time: 11.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5909058, upper bound: 202.5909030
time: 11.50 seconds

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

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5419880, upper bound: 202.5419885
time: 6.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5419880, upper bound: 202.5419885
time: 7.31 seconds

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
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5527312, upper bound: 202.5527263
time: 6.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5527305, upper bound: 202.5527263
time: 8.99 seconds

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

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6045780, upper bound: 202.6045672
time: 8.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6045670, upper bound: 202.6045766
time: 9.97 seconds

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

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6045756, upper bound: 202.6045793
time: 9.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6045756, upper bound: 202.6045793
time: 8.96 seconds

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
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5400877, upper bound: 202.5400895
time: 8.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5400877, upper bound: 202.5400895
time: 8.29 seconds

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

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5489552, upper bound: 202.5489598
time: 8.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5489552, upper bound: 202.5489598
time: 9.25 seconds

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
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5517577, upper bound: 202.5517641
time: 8.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5517604, upper bound: 202.5517607
time: 7.61 seconds

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5049541, upper bound: 202.5049544
time: 5.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5049541, upper bound: 202.5049544
time: 5.88 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 16.89 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 1, lower bound: -202.5063526, upper bound: 202.5063511
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 1, lower bound: -202.5063526, upper bound: 202.5063511
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 1, lower bound: -202.5379406, upper bound: 202.5379383
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 1, lower bound: -202.5379406, upper bound: 202.5379383
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 1, lower bound: -202.5256334, upper bound: 202.5256327
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 1, lower bound: -202.5256334, upper bound: 202.5256327
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 1, lower bound: -202.5464400, upper bound: 202.5464390
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 1, lower bound: -202.5464400, upper bound: 202.5464390
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 1, lower bound: -202.6016610, upper bound: 202.6016626
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 1, lower bound: -202.6016592, upper bound: 202.6016645
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 1, lower bound: -202.6065453, upper bound: 202.6065291
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 1, lower bound: -202.6065335, upper bound: 202.6065291
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 1, lower bound: -202.6050774, upper bound: 202.6050941
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 1, lower bound: -202.6050774, upper bound: 202.6050796
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 1, lower bound: -202.5909058, upper bound: 202.5909030
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 1, lower bound: -202.5909058, upper bound: 202.5909030
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 1, lower bound: -202.5419880, upper bound: 202.5419885
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 1, lower bound: -202.5419880, upper bound: 202.5419885
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 1, lower bound: -202.5527312, upper bound: 202.5527263
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 1, lower bound: -202.5527305, upper bound: 202.5527263
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 1, lower bound: -202.6045780, upper bound: 202.6045672
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 1, lower bound: -202.6045670, upper bound: 202.6045766
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 1, lower bound: -202.6045756, upper bound: 202.6045793
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 1, lower bound: -202.6045756, upper bound: 202.6045793
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 1, lower bound: -202.5400877, upper bound: 202.5400895
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 1, lower bound: -202.5400877, upper bound: 202.5400895
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 1, lower bound: -202.5489552, upper bound: 202.5489598
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 1, lower bound: -202.5489552, upper bound: 202.5489598
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 1, lower bound: -202.5517577, upper bound: 202.5517641
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 1, lower bound: -202.5517604, upper bound: 202.5517607
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 1, lower bound: -202.5049541, upper bound: 202.5049544
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.89
Output dim: 1, lower bound: -202.5049541, upper bound: 202.5049544

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=203.5233612060547
rel_dist={1: [-202.608202498089, 202.60820249808899]}

## Binary Search with RS_random_Z Result
status: None
Maximum delta epsilon: None
execution time: 1811.15 seconds
